from dataclasses import dataclass
from typing import Optional


@dataclass
class DataModuleConfig:
    batch_size: int = 64
    num_workers: int = 4
    test_perc: float = 0.01
    val_perc: float = 0.01
    reduce_dataset: Optional[int] = None
    resolution: int = 128


@dataclass
class PoseMaskDataModuleConfig(DataModuleConfig):
    _target_: str = "data.datasets.pose_mask.PoseMaskDataModule"


@dataclass
class ObjectsDataModuleConfig(DataModuleConfig):
    _target_: str = "data.datasets.objects.ObjectsDataModule"
