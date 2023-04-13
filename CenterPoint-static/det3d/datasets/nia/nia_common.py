import numpy as np
import pickle
import os
import json
import glob
from pathlib import Path
from functools import reduce
from typing import List

from tqdm import tqdm
from pyquaternion import Quaternion

try:
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
except:
    print("nuScenes devkit not Found!")

general_to_detection = {
    "human.pedestrian.adult": "street_trees",
    "human.pedestrian.child": "street_trees",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "street_trees",
    "human.pedestrian.construction_worker": "street_trees",
    "animal": "ignore",
    "vehicle.car": "median_strip",
    "vehicle.motorcycle": "road_sign",
    "vehicle.bicycle": "ramp_sect",
    "vehicle.bus.bendy": "sound_barrier",
    "vehicle.bus.rigid": "sound_barrier",
    "vehicle.truck": "overpass",
    "vehicle.construction": "tunnel",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "ramp_sect": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "sound_barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "median_strip": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "tunnel": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "road_sign": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "street_trees": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "overpass": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}

def get_obj(path):
    if path[-4:] == 'json':
        with open(path, 'r') as f:
            obj = json.load(f)
        return obj
    else:
        with open(path, 'rb') as f:
                obj = pickle.load(f)
        return obj

def _second_det_to_nusc_box(detection):
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    box3d[:, -1] = -box3d[:, -1] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    try:
        s_record = nusc.get("sample", sample_token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = sample_token

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record["rotation"]))
        box.translate(np.array(pose_record["translation"]))
        box_list.append(box)
    return box_list

# def get_available_scenes(dir_path): # ver 1209
#     available_scenes = list(os.listdir(dir_path))
#     extreme_scenes = []
#
#     idxs = []
#     for idx, scenes in enumerate(available_scenes):
#         if scenes[:6] == 'S_Clip':
#             available_scenes[idx] = scenes.split('_')[2]#[7:12]
#             idxs.append(idx)
#
#     return np.sort(np.array(available_scenes)[idxs]).tolist()
#
#
# def get_available_frames(dir_path, available_scenes, subsample=1):
#     available_frames = []
#     for scenes in available_scenes:
#         frames = list(os.listdir(os.path.join(dir_path, f"S_Clip_{scenes}", 'Lidar')))
#         frames = frames[::subsample]
#         available_frames.extend(frames)
#
#     print("exist frame num:", len(available_frames))
#     return available_frames

def get_available_scenes(dir_path, ratio=5):
    def list_path(path):
        return list(np.sort(np.array(os.listdir(path))))

    available_scenes = []
    all_clips = []
    idxs = []

    for scene in list_path(dir_path):
        available_clips = []
        for clip in list_path(os.path.join(dir_path, scene, "source")):
        # for clip in list_path(os.path.join(dir_path, scene)):
            '''
            idx 0: Staria/Avante
            idx 1: "Clip"
            idx 2: clip number
            '''
            args = clip.split('_')
            print(args)
            if args[1] == 'Clip':
                available_clips.append(clip)
                all_clips.append(clip)
        available_scenes.append(available_clips)

    assert len(all_clips) == len(set(all_clips)), "duplicates detected"

    train_scenes, val_scenes, test_scenes = [], [], []

    for scenes in available_scenes:
        N = len(scenes) // ratio
        train = scenes[:-N]
        val = scenes[-N//2:]
        test = scenes[-N:-N//2]

        train_scenes.extend(train)
        val_scenes.extend(val)
        test_scenes.extend(test)
    return train_scenes, val_scenes, test_scenes



def get_available_frames(dir_path, available_clips, subsample=1):
    available_frames = []
    for clips in available_clips:
        vehicle = clips.split('_')[0]
        clip_num = clips.split('_')[2]
        scene_num = clips.split('_')[3]
        frames = os.listdir(os.path.join(dir_path, scene_num, "source", clips, "Lidar"))
        frames = [str(os.path.join(scene_num, "source", clips, "Lidar/"))+f for f in frames]
        # frames = os.listdir(os.path.join(dir_path, vehicle, "source", clips, "Lidar"))
        # frames = [str(os.path.join(vehicle, "source", clips, "Lidar/"))+f for f in frames]
        frames = frames[::subsample]
        available_frames.extend(frames)
    return available_frames

def _get_available_scenes(nia):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def get_sample_data(
    nusc, sample_data_token: str, selected_anntokens: List[str] = None
):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param selected_anntokens: If provided only return the selected annotation.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    else:
        cam_intrinsic = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic

CAM_CHANS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']


def get_lidar_to_image_transform(nusc, pointsensor,  camera_sensor):
    tms = []
    intrinsics = []  
    cam_paths = [] 
    for chan in CAM_CHANS:
        cam = camera_sensor[chan]

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        lidar_cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        car_from_lidar = transform_matrix(
            lidar_cs_record["translation"], Quaternion(lidar_cs_record["rotation"]), inverse=False
        )

        # Second step: transform to the global frame.
        lidar_poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        global_from_car = transform_matrix(
            lidar_poserecord["translation"],  Quaternion(lidar_poserecord["rotation"]), inverse=False,
        )

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        cam_poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        car_from_global = transform_matrix(
            cam_poserecord["translation"],
            Quaternion(cam_poserecord["rotation"]),
            inverse=True,
        )

        # Fourth step: transform into the camera.
        cam_cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        cam_from_car = transform_matrix(
            cam_cs_record["translation"], Quaternion(cam_cs_record["rotation"]), inverse=True
        )

        tm = reduce(
            np.dot,
            [cam_from_car, car_from_global, global_from_car, car_from_lidar],
        )

        cam_path, _, intrinsic = nusc.get_sample_data(cam['token'])

        tms.append(tm)
        intrinsics.append(intrinsic)
        cam_paths.append(cam_path )

    return tms, intrinsics, cam_paths  

def find_closet_camera_tokens(nusc, pointsensor, ref_sample):
    lidar_timestamp = pointsensor["timestamp"]

    min_cams = {} 

    for chan in CAM_CHANS:
        camera_token = ref_sample['data'][chan]

        cam = nusc.get('sample_data', camera_token)
        min_diff = abs(lidar_timestamp - cam['timestamp'])
        min_cam = cam

        for i in range(6):  # nusc allows at most 6 previous camera frames 
            if cam['prev'] == "":
                break 

            cam = nusc.get('sample_data', cam['prev'])
            cam_timestamp = cam['timestamp']

            diff = abs(lidar_timestamp-cam_timestamp)

            if (diff < min_diff):
                min_diff = diff 
                min_cam = cam 
            
        min_cams[chan] = min_cam 

    return min_cams     


def _fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, nsweeps=10, filter_zero=True):
    from nuscenes.utils.geometry_utils import transform_matrix

    train_nusc_infos = []
    val_nusc_infos = []

    ref_chan = "LIDAR_TOP"  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for sample in tqdm(nusc.sample):
        """ Manual save info["sweeps"] """        
        # Get reference pose and timestamp
        # ref_chan == "LIDAR_TOP"
        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        ref_cs_rec = nusc.get(
            "calibrated_sensor", ref_sd_rec["calibrated_sensor_token"]
        )
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample["data"]["CAM_FRONT"]
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )

        ref_cams = {}
        # get all camera sensor data
        for cam_chan in CAM_CHANS:
            camera_token = sample['data'][cam_chan]
            cam = nusc.get('sample_data', camera_token)

            ref_cams[cam_chan] = cam 

        # get camera info for point painting 
        all_cams_from_lidar, all_cams_intrinsic, all_cams_path = get_lidar_to_image_transform(nusc, pointsensor=ref_sd_rec, camera_sensor=ref_cams)    

        info = {
            "lidar_path": ref_lidar_path,
            "cam_front_path": ref_cam_path,
            "cam_intrinsic": ref_cam_intrinsic,
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
            "all_cams_from_lidar": all_cams_from_lidar,
            "all_cams_intrinsic": all_cams_intrinsic,
            "all_cams_path": all_cams_path
        }

        sample_data_token = sample["data"][chan]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        sweeps = []
        while len(sweeps) < nsweeps - 1:
            if curr_sd_rec["prev"] == "":
                if len(sweeps) == 0:
                    sweep = {
                        "lidar_path": ref_lidar_path,
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": None,
                        "time_lag": curr_sd_rec["timestamp"] * 0,
                        "all_cams_from_lidar": all_cams_from_lidar,
                        "all_cams_intrinsic": all_cams_intrinsic,
                        "all_cams_path": all_cams_path
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

                # get nearest camera frame data 
                cam_data = find_closet_camera_tokens(nusc, curr_sd_rec, ref_sample=sample)
                cur_cams_from_lidar, cur_cams_intrinsic, cur_cams_path = get_lidar_to_image_transform(nusc, pointsensor=curr_sd_rec, camera_sensor=cam_data)   

                # Get past pose
                current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
                global_from_car = transform_matrix(
                    current_pose_rec["translation"],
                    Quaternion(current_pose_rec["rotation"]),
                    inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"]
                )
                car_from_current = transform_matrix(
                    current_cs_rec["translation"],
                    Quaternion(current_cs_rec["rotation"]),
                    inverse=False,
                )

                tm = reduce(
                    np.dot,
                    [ref_from_car, car_from_global, global_from_car, car_from_current],
                )

                lidar_path = nusc.get_sample_data_path(curr_sd_rec["token"])

                time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": curr_sd_rec["token"],
                    "transform_matrix": tm,
                    "global_from_car": global_from_car,
                    "car_from_current": car_from_current,
                    "time_lag": time_lag,
                    "all_cams_from_lidar": cur_cams_from_lidar,
                    "all_cams_intrinsic": cur_cams_intrinsic,
                    "all_cams_path": cur_cams_path
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        assert (
            len(info["sweeps"]) == nsweeps - 1
        )
        
        if not test:
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]

            mask = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts'])>0 for anno in annotations], dtype=bool).reshape(-1)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
            # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(
                -1, 1
            )
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate(
                [locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1
            )
            # gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)

            if not filter_zero:
                info["gt_boxes"] = gt_boxes
                info["gt_boxes_velocity"] = velocity
                info["gt_names"] = np.array([general_to_detection[name] for name in names])
                info["gt_boxes_token"] = tokens
            else:
                info["gt_boxes"] = gt_boxes[mask, :]
                info["gt_boxes_velocity"] = velocity[mask, :]
                info["gt_names"] = np.array([general_to_detection[name] for name in names])[mask]
                info["gt_boxes_token"] = tokens[mask]

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def _fill_infos(root_path, frames, sensor='lidar'):
    # load all train infos
    infos = []
    for frame_name in tqdm(frames):  # global id
        if '.pcd' not in frame_name:
            continue

        ''' Point Cloud path '''
        if sensor == 'lidar':
            lidar_path = os.path.join(root_path, frame_name)
        elif sensor == 'radar':
            radar_frame_name = frame_name.replace('Lidar', 'Radar/RadarFront').replace('LR', 'RF')
            radar_path = os.path.join(root_path, radar_frame_name)
            lidar_path = radar_path

        ''' Annotation path '''
        ref_path = frame_name.replace('source', 'label').replace('Lidar', 'result').replace('LR', 'FC').replace('.pcd', '.json')

        ''' Image path '''
        cam_path = frame_name.replace('Lidar', 'Camera/CameraFront/blur').replace('LR', 'CF').replace('.pcd', '.png')

        ''' Calibration path '''
        scene_path = '/'.join(frame_name.split('/')[:-2])
        calib_path = glob.glob(f'{scene_path}/calib/Lidar_radar_calib/*.txt')[0]

        assert os.path.exists(lidar_path), f"Cannot find path: {lidar_path}"
        assert os.path.exists(ref_path), f"Cannot find path: {ref_path}"
        assert os.path.exists(cam_path), f"Cannot find path: {cam_path}"
        assert os.path.exists(calib_path), f"Cannot find path: {calib_path}"

        ref_obj = get_obj(ref_path)

        if sensor == 'radar':
            frame_name = radar_frame_name

        info = {
        "lidar_path": lidar_path,
        "cam_front_path": cam_path,
        "anno_path": ref_path,
        "calib_path": calib_path,
        # "cam_intrinsic": ref_cam_intrinsic,
        "token": frame_name[:-4], #sample["token"],
        "sweeps": [],
        # "ref_from_car": ref_from_car,
        # "car_from_global": car_from_global,
        # "timestamp": ref_time,
        # "all_cams_from_lidar": all_cams_from_lidar,
        # "all_cams_intrinsic": all_cams_intrinsic,
        # "all_cams_path": all_cams_path
        }

        annotations = ref_obj['annotation']
        if sensor == 'radar':
            remove_idx = []
            for idx, ann in enumerate(annotations):
                if ann['3d_box'][0]['radar_point_count'] < 3:
                    remove_idx.append(idx)
            remove_idx.reverse()
            for i in remove_idx:
                annotations.pop(i)


        # sub id 전부 사용
        ref_boxes = []
        names = []
        for anno in annotations:
            for box in anno['3d_box']:
                ref_boxes.append(box)
            name = [anno['category']]*len(anno['3d_box'])
            names.extend(name)

        # # sub id 첫번째만 사용
        # ref_boxes = [anno['3d_box'][0] for anno in annotations]
        # names = np.array([anno['category'] for anno in annotations])

        locs = np.array([b['location'] for b in ref_boxes]).reshape(-1, 3)
        dims = np.array([b['dimension'] for b in ref_boxes]).reshape(-1, 3)
        dims[:, [2, 1]] = dims[:, [1, 2]] # w/h/l(NIA) -> w/l/h (Nuscene)
        rots = np.array([b['rotation_y'] for b in ref_boxes]).reshape(-1, 1)
        # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
        velocity = np.zeros_like(locs)
        if 'ETC' in names:
            names = np.where(names == 'ETC', 'construction_vehicle', names)
        tokens = np.array([anno['id'] for anno in annotations])
        try:
            gt_boxes = np.concatenate(
                [locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1
            )
        except:
            print("ref_path:", ref_path)
        # gt_boxes = np.concatenate([locs, dims, rots], axis=1)

        # assert len(annotations) == len(gt_boxes) == len(velocity) # sub id 첫번째만 사용
        assert len(ref_boxes) == len(gt_boxes) == len(velocity) # sub id 전부 사용

        info["gt_boxes"] = gt_boxes
        info["gt_boxes_velocity"] = velocity
        info["gt_names"] = np.array([name.lower() for name in names])
        info["gt_boxes_token"] = tokens

        infos.append(info)

    return infos


def create_nia_infos(root_path, sensor='lidar', filter_zero=True, subsample=1):
    # root_path = Path(root_path)
    # normal_path = os.path.join(root_path, 'normal')
    # extreme_path = os.path.join(root_path, 'abnormal')

    ''' Get Scenes and divide into train/val set '''
    # train_scenes, val_scenes, test_scenes = get_available_scenes(normal_path, ratio=ratio)
    train_frames = sorted(glob.glob(f'{root_path}/train/source/*/*/*/Lidar/*'))
    val_frames = sorted(glob.glob(f'{root_path}/val/source/*/*/*/Lidar/*'))

    test_normal_frames = sorted(glob.glob(f'{root_path}/test/source/normal/*/*/Lidar/*'))
    test_abnormal_frames = sorted(glob.glob(f'{root_path}/test/source/abnormal/*/*/Lidar/*'))

    # train_frames = get_available_frames(normal_path, train_scenes, subsample=subsample)
    # val_frames = get_available_frames(normal_path, val_scenes, subsample=subsample)
    print("exist train frames:", len(train_frames), \
          "exist val frames:", len(val_frames), \
          "exist test_nomal frames:", len(test_normal_frames), \
          "exist test_abnormal frames:", len(test_abnormal_frames))

    train_infos = _fill_infos(root_path, train_frames, sensor)
    val_infos = _fill_infos(root_path, val_frames, sensor)

    test_normal_infos = _fill_infos(root_path, test_normal_frames, sensor)
    test_abnormal_infos = _fill_infos(root_path, test_abnormal_frames, sensor)

    # filter_zero = str(filter_zero)
    root_path = Path(root_path)
    
    with open(root_path / "infos_train_filter_{}_{}.pkl".format(filter_zero, sensor), "wb") as f:
        pickle.dump(train_infos, f)
    with open(root_path / "infos_val_filter_{}_{}.pkl".format(filter_zero, sensor), "wb") as f:
        pickle.dump(val_infos, f)

    with open(root_path / "infos_test_normal_filter_{}_{}.pkl".format(filter_zero, sensor), "wb") as f:
        pickle.dump(test_normal_infos, f)
    with open(root_path / "infos_test_abnormal_filter_{}_{}.pkl".format(filter_zero, sensor), "wb") as f:
        pickle.dump(test_abnormal_infos, f)


def eval_main(nusc, eval_version, res_path, eval_set, output_dir):
    # nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=10,)
