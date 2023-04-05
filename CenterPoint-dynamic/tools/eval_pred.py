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
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle
import time

def save_pred(pred, root, sensor='lidar'):
    with open(os.path.join(root, f"prediction_{sensor}.pkl"), "wb") as f:
        pickle.dump(pred, f)

def load_pred(root, sensor='lidar'):
    with open(os.path.join(root, f"prediction_{sensor}.pkl"), "rb") as f:
        pred = pickle.load(f)
    return pred


def parse_args(**kwargs):
    if 'sensor' in kwargs:
        sensor = kwargs.get('sensor', 'lidar')
    config_path = '/path/to/CenterPoint-NIA/configs/nia/lidar/nia_centerpoint_voxelnet_01voxel.py'
    if sensor == 'radar':
        config_path = '/path/to/CenterPoint-NIA/configs/nia/radar/nia_centerpoint_voxelnet_01voxel_radar.py'

    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config",
                        default=config_path,
                        help="train config file path")
    parser.add_argument("--work_dir", required=False, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main(**kwargs):

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args(**kwargs)

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    else:
        args.work_dir = cfg.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)


    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # save_pred(predictions, args.work_dir)
    sensor = kwargs.get('sensor', 'lidar')
    predictions = load_pred(args.work_dir, sensor=sensor)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
