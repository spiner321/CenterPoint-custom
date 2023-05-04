import sys
import pickle
import json
import random
import operator
import numpy as np
from det3d.core import box_np_ops
from functools import reduce
from pathlib import Path
from copy import deepcopy

try:
    from tools.nuscenes.nuscenes import NuScenes
    from tools.nuscenes.eval.detection.config import config_factory
except:
    print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class NIADataset(PointCloudDataset):
    # NumPointFeatures = 5 # x, y, z, intensity, ring_index
    # NumPointFeatures = 4 # x, y, z, intensity

    def __init__(
        self,
        info_path,
        root_path,
        nsweeps=0, # here set to zero to catch unset nsweep
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        version="v1.0-trainval",
        load_interval=1,
        **kwargs,
    ):
        self.load_interval = load_interval 
        super(NIADataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self.nsweeps = nsweeps
        assert self.nsweeps > 0, "At least input one sweep please!"
        print(self.nsweeps)

        self._info_path = info_path
        
        print('------------------------------------')
        print(info_path)
        print('------------------------------------')

        self._class_names = class_names

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        if 'lidar' in str(info_path):
            self._num_point_features = 4
        elif 'radar' in str(info_path):
            self._num_point_features = 5
        # self._num_point_features = NIADataset.NumPointFeatures
        self._name_mapping = general_to_detection

        self.virtual = kwargs.get('virtual', False)
        if self.virtual:
            self._num_point_features = 16 

        self.version = version
        self.eval_version = "detection_cvpr_2019"

    def reset(self):
        self.logger.info(f"re-sample {self.frac} frames from full set")
        random.shuffle(self._nusc_infos_all)
        self._nusc_infos = self._nusc_infos_all[: self.frac]

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _nusc_infos_all = pickle.load(f)

        _nusc_infos_all = _nusc_infos_all[::self.load_interval]

        if not self.test_mode:  # if training
            self.frac = int(len(_nusc_infos_all) * 0.25)

            _cls_infos = {name: [] for name in self._class_names}
            for info in _nusc_infos_all:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / max(duplicated_samples, 1) for k, v in _cls_infos.items()}

            print(_cls_dist) # for test

            self._nusc_infos = []

            frac = 1.0 / len(self._class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._nusc_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()

            _cls_infos = {name: [] for name in self._class_names}
            for info in self._nusc_infos:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            _cls_dist = {
                k: len(v) / len(self._nusc_infos) for k, v in _cls_infos.items()
            }
        else:
            if isinstance(_nusc_infos_all, dict):
                self._nusc_infos = []
                for v in _nusc_infos_all.values():
                    self._nusc_infos.extend(v)
            else:
                self._nusc_infos = _nusc_infos_all

    def __len__(self):

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        cls_range_map = config_factory(self.eval_version).serialize()['class_range']
        gt_annos = []
        for info in self._nusc_infos:
            gt_names = np.array(info["gt_names"])
            gt_boxes = info["gt_boxes"]
            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            # det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append(
                {
                    "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                    "alpha": np.full(N, -10),
                    "occluded": np.zeros(N),
                    "truncated": np.zeros(N),
                    "name": gt_names[mask],
                    "location": gt_boxes[mask][:, :3],
                    "dimensions": gt_boxes[mask][:, 3:6],
                    "rotation_y": gt_boxes[mask][:, 6],
                    "token": info["token"],
                }
            )
        return gt_annos

    def get_sensor_data(self, idx):

        info = self._nusc_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "virtual": self.virtual 
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)


    def convert_detection_to_kitti_annos(self, detection):
        from ..kitti.kitti_common import get_start_result_anno, empty_result_anno

        class_names = self._class_names
        det_image_idxes = [k for k in detection.keys()]
        gt_image_idxes = [str(info["image"]["image_idx"]) for info in self._kitti_infos]
        # print(f"det_image_idxes: {det_image_idxes[:10]}")
        # print(f"gt_image_idxes: {gt_image_idxes[:10]}")
        annos = []
        # for i in range(len(detection)):
        # for det_idx in gt_image_idxes:
        for idx, (det_idx, det) in enumerate(detection.items()):
            # det = detection[det_idx]
            info = self._kitti_infos[idx]
            # info = self._kitti_infos[i]
            calib = info["calib"]
            rect = calib["R0_rect"]
            Trv2c = calib["Tr_velo_to_cam"]
            P2 = calib["P2"]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()

            anno = get_start_result_anno()
            num_example = 0
            box3d_camera = final_box_preds
            if final_box_preds.shape[0] != 0:
                final_box_preds[:, -1] = box_np_ops.limit_period(
                    final_box_preds[:, -1], offset=0.5, period=np.pi * 2,
                )
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2

                for j in range(det['label_preds'].shape[0]):
                    image_shape = info["image"]["image_shape"]

                    bbox = np.zeros((4,))
                    anno["bbox"].append(bbox)

                    anno["alpha"].append(
                        -np.arctan2(-final_box_preds[j, 1], final_box_preds[j, 0])
                        + box3d_camera[j, 6]
                    )
                    # anno["dimensions"].append(box3d_camera[j, [4, 5, 3]])
                    anno["dimensions"].append(box3d_camera[j, 3:6])
                    anno["location"].append(box3d_camera[j, :3])
                    anno["rotation_y"].append(box3d_camera[j, 6])
                    anno["name"].append(class_names[int(label_preds[j])])
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["score"].append(scores[j])

                    num_example += 1

                # # aim: x, y, z, w, l, h, r -> -y, -z, x, h, w, l, r
                # # (x, y, z, w, l, h r) in lidar -> (x', y', z', l, h, w, r) in camera
                # box3d_camera = box_np_ops.box_lidar_to_camera(
                #     final_box_preds, rect, Trv2c
                # )
                # camera_box_origin = [0.5, 1.0, 0.5]
                # box_corners = box_np_ops.center_to_corner_box3d(
                #     box3d_camera[:, :3],
                #     box3d_camera[:, 3:6],
                #     box3d_camera[:, 6],
                #     camera_box_origin,
                #     axis=1,
                # )
                # box_corners_in_image = box_np_ops.project_to_image(box_corners, P2)
                # # box_corners_in_image: [N, 8, 2]
                # minxy = np.min(box_corners_in_image, axis=1)
                # maxxy = np.max(box_corners_in_image, axis=1)
                # bbox = np.concatenate([minxy, maxxy], axis=1)
                #
                # for j in range(box3d_camera.shape[0]):
                #     image_shape = info["image"]["image_shape"]
                #     if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                #         continue
                #     if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                #         continue
                #     bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                #     bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                #     anno["bbox"].append(bbox[j])
                #
                #     anno["alpha"].append(
                #         -np.arctan2(-final_box_preds[j, 1], final_box_preds[j, 0])
                #         + box3d_camera[j, 6]
                #     )
                #     # anno["dimensions"].append(box3d_camera[j, [4, 5, 3]])
                #     anno["dimensions"].append(box3d_camera[j, 3:6])
                #     anno["location"].append(box3d_camera[j, :3])
                #     anno["rotation_y"].append(box3d_camera[j, 6])
                #     anno["name"].append(class_names[int(label_preds[j])])
                #     anno["truncated"].append(0.0)
                #     anno["occluded"].append(0)
                #     anno["score"].append(scores[j])
                #
                #     num_example += 1

            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def nia_to_kitti_anno(self, idx, info_n):
        anno_k = {}
        anno_k.update(
            {
                "name": [],
                "truncated": [],
                "occluded": [],
                "alpha": [],
                "bbox": [],
                "dimensions": [],
                "location": [],
                "rotation_y": [],
            }
        )
        length = len(info_n["gt_boxes_token"])
        anno_k["name"] = info_n["gt_names"]
        anno_k["truncated"] = np.zeros(length)
        anno_k["occluded"] = np.zeros(length)
        anno_k["alpha"] = info_n['gt_boxes'][:,8]
        anno_k["bbox"] = np.zeros((length,4))
        anno_k["dimensions"] = info_n['gt_boxes'][:,3:6]
        anno_k["location"] = info_n['gt_boxes'][:,:3]
        anno_k["rotation_y"] = info_n['gt_boxes'][:,8]
        anno_k["score"] = np.zeros((anno_k["bbox"].shape[0],))
        anno_k["index"] = np.array(idx, dtype=np.int32)
        anno_k["group_ids"] = np.arange(length, dtype=np.int32)



        return anno_k

    def nia_to_kitti_infos(self):
        from ..kitti.kitti_common import _extend_matrix
        # self._kitti_infos = get_start_result_anno()
        self._kitti_infos = []
        self._kitti_annos = []
        for idx, info_n in enumerate(self._nusc_infos):
            info_k = {}

            pc_info = {"num_features": self._num_point_features}
            pc_info["velodyne_path"] = info_n['lidar_path']
            calib_info = {}
            image_info = {"image_idx": idx}

            image_info["image_path"] = info_n['cam_front_path']
            image_info["image_shape"] = np.array((1920,1200), dtype=np.int32)
            annotations = self.nia_to_kitti_anno(idx, info_n)
            info_k["image"] = image_info
            info_k["point_cloud"] = pc_info

            calib_path = info_n['calib_path']
            with open(calib_path, "r") as f:
                lines = f.readlines()
                angles = [float(a) for a in lines[4].split(',')]
                trans = np.array([float(a) for a in lines[6].split(',')]).reshape(3,1)
                intrinsics = [float(a) for a in lines[8].split(',')]

            P = np.array((intrinsics[0],0,intrinsics[2],0,0,intrinsics[1],intrinsics[3],0,0,0,1,0)).reshape((3,4))
            calib_info["P0"] = P #P0
            calib_info["P1"] = P #P1
            calib_info["P2"] = P #P2
            calib_info["P3"] = P #np.identity(4) #P3
            calib_info["R0_rect"] = np.identity(3) #rect_4x4
            calib_info["Tr_velo_to_cam"] = _extend_matrix(np.concatenate((trans,trans,trans,trans),axis=1)) #Tr_velo_to_cam)
            calib_info["Tr_imu_to_velo"] = np.identity(4) #Tr_imu_to_velo
            info_k["calib"] = calib_info
            info_k["annos"] = annotations
            self._kitti_infos.append(info_k)
            self._kitti_annos.append(annotations)

    ''' KITTI EVAL '''
    # def evaluation(self, detections, output_dir=None, testset=False):
    #     from ..kitti.eval import get_official_eval_result
    #     from ..kitti.eval import get_coco_eval_result
    #     from ..kitti.eval import do_eval_nia
    #
    #     # gt_annos = self._nusc_infos
    #     self.nia_to_kitti_infos()
    #     gt_annos = self._kitti_annos
    #     dt_annos = self.convert_detection_to_kitti_annos(detections)
    #
    #     # firstly convert standard detection to kitti-format dt annos
    #     z_axis = 1  # KITTI camera format use y as regular "z" axis.
    #     z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
    #     # for regular raw lidar data, z_axis = 2, z_center = 0.5.
    #
    #     overlaps = np.array(
    #         [
    #             [0.7, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5],
    #             [0.7, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5],
    #             [0.7, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5],
    #         ]
    #     )
    #     class_to_name = {
    #         0: "car",
    #         1: "pedestrian",
    #         2: "bicycle",
    #         3: "truck",
    #         4: "bus",
    #         5: "trailer",
    #         6: "construction_vehicle",
    #         7: "motorcycle",
    #         8: "barrier",
    #         9: "traffic_cone",
    #         10: "cyclist",
    #     }
    #     name_to_class = {v: n for n, v in class_to_name.items()}
    #
    #     current_classes = self._class_names
    #     current_classes_int = []
    #     for curcls in current_classes:
    #         if isinstance(curcls, str):
    #             current_classes_int.append(name_to_class[curcls.lower()])
    #         else:
    #             current_classes_int.append(curcls)
    #     current_classes = current_classes_int
    #
    #     # do_eval_nia(gt_annos, dt_annos, current_classes, min_overlaps=overlaps, difficultys=(0,0,0))
    #
    #     result_official_dict = get_official_eval_result(
    #         gt_annos, dt_annos, self._class_names, z_axis=z_axis, z_center=z_center
    #     )
    #     result_coco_dict = get_coco_eval_result(
    #         gt_annos, dt_annos, self._class_names, z_axis=z_axis, z_center=z_center
    #     )
    #
    #     results = {
    #         "results": {
    #             "official": result_official_dict["result"],
    #             "coco": result_coco_dict["result"],
    #         },
    #         "detail": {
    #             "eval.kitti": {
    #                 "official": result_official_dict["detail"],
    #                 "coco": result_coco_dict["detail"],
    #             }
    #         },
    #     }
    #
    #     return results, dt_annos

    ''' NUSCENES EVAL '''
    def evaluation(self, detections, anno_path, output_dir=None, testset=False):
        from nuscenes.eval.detection.evaluate import DetectionEval
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        if not testset:
            dets = []
            gt_annos = self.ground_truth_annotations
            assert gt_annos is not None

            miss = 0
            for gt in gt_annos:
                try:
                    dets.append(detections[gt["token"]])
                except Exception:
                    miss += 1

            assert miss == 0
        else:
            dets = [v for _, v in detections.items()]
            assert len(detections) == 6008

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        # nusc = NuScenes(version=version, dataroot="/path/to/CenterPoint-NIA/data/nuScenes", verbose=True)
        # nusc = NuScenes(version=version, dataroot="/data/kimgh/CenterPoint-NIA/data", verbose=True)
        nusc = None

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            # boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"]["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({det["metadata"]["token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,
                self.eval_version,
                res_path,
                eval_set_map[self.version],
                output_dir,
                anno_path
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"],},
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"],},
            }
        else:
            res = None

        return res, None
