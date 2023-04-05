from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset
from .nia import NIADataset

dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "NIA": NIADataset
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
