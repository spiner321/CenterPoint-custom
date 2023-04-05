import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
from tqdm import tqdm, trange
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import load_checkpoint
import pickle
import time
from matplotlib import pyplot as plt
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from tools.demo_utils import visual, visual_nia
from collections import defaultdict
from pathlib import Path


def convert_box(info):
    boxes = info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]

    assert len(boxes) == len(names)

    detection = {}

    detection['box3d_lidar'] = boxes

    # dummy value
    detection['label_preds'] = np.zeros(len(boxes))
    detection['scores'] = np.ones(len(boxes))

    return detection


def main(**kwargs):
    # cfg = Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo.py')
    base_path = "/path/to/CenterPoint-NIA"

    if 'sensor' in kwargs:
        sensor = kwargs.get('sensor', 'lidar')
    config_path = base_path + '/configs/nia/lidar/nia_centerpoint_voxelnet_01voxel.py'
    if sensor == 'radar':
        config_path = base_path + '/configs/nia/radar/nia_centerpoint_voxelnet_01voxel_radar.py'

    cfg = Config.fromfile(config_path)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    dataset = build_dataset(cfg.data.val)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_kitti,
        pin_memory=False,
    )

    # checkpoint = load_checkpoint(model, 'work_dirs/centerpoint_pillar_512_demo/latest.pth', map_location="cpu")
    work_dir = base_path + '/work_dirs/nia_centerpoint_voxelnet_01voxel/latest.pth'
    if sensor == 'radar':
        work_dir = base_path + '/work_dirs/nia_centerpoint_voxelnet_01voxel_radar/latest.pth'

    checkpoint = load_checkpoint(model, work_dir, map_location="cpu")
    model.eval()

    model = model.cuda()

    cpu_device = torch.device("cpu")

    points_list = []
    gt_annos = []
    detections = []
    images = []

    for i, data_batch in enumerate(tqdm(data_loader)):
        info = dataset._nusc_infos[i]
        gt_annos.append(convert_box(info))

        points = data_batch['points'][0][:, :3].cpu().numpy()
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=0,
            )
        for output in outputs:
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.append(output)

        # data_path = kwargs.get('data_path')
        scene = data_batch['metadata'][0]['token'].split('_')[1]
        frame = data_batch['metadata'][0]['token'].split('_')[3]
        images.append(info['cam_front_path'])
            # f'{base_path}/data/nia/S_Clip_{scene}/Camera/CameraFront/blur/2-048_{scene}_CF_{frame}.png')

        points_list.append(points.T)

    print('Done model inference. Please wait a minute, the matplotlib is a little slow...')

    for i in trange(len(points_list)):
        visual_nia(points_list[i], gt_annos[i], detections[i], i, cam_path=images[i])
        # print("Rendered Image {}".format(i))

    image_folder = 'demo'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda img_name: int(img_name.split('.')[0][4:]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))
    cv2_images = []

    for image in images:
        cv2_images.append(cv2.imread(os.path.join(image_folder, image)))

    for img in cv2_images:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    print("Successfully save video in the main folder")


if __name__ == "__main__":
    main(sensor='radar')
