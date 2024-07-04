from abc import ABC, abstractmethod
from logging import getLogger
from math import ceil
from typing import Optional, Tuple

import torch
from numpy.random import Generator, default_rng
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import Tensor
from torch.utils.data import IterableDataset
from torch_geometric.data import Data, DataLoader

from utils.data import stage_seeds


class SyntheticDataset(IterableDataset, ABC):
    def __init__(
        self,
        resolution: int = 256,
        n_samples: int = 10000,
        stage: Optional[str] = None,
        buffer_size: int = 100,
    ) -> None:
        super().__init__()
        self._logger = getLogger(self.__class__.__name__)
        self.resolution = resolution
        self.n_samples = n_samples
        self.stage = stage
        self.buffer_size = buffer_size

    def __len__(self):
        worker_info = torch.utils.data.get_worker_info()
        return (
            self.n_samples
            if worker_info is None
            else int(ceil(self.n_samples / float(worker_info.num_workers)))
        )

    @abstractmethod
    def get_sample(self, rng: Generator) -> Tuple[Tensor, Data]:
        pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.n_samples
            rng = default_rng(stage_seeds[self.stage])
        else:
            num_workers = worker_info.num_workers
            per_worker = int(ceil(self.n_samples / float(num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.n_samples)
            rng = default_rng(stage_seeds[self.stage].spawn(num_workers)[worker_id])
        if self.stage == "train":
            if worker_info is None:
                shuffle_rng = default_rng(stage_seeds[self.stage].spawn(1)[0])
            else:
                shuffle_rng = default_rng(
                    stage_seeds[self.stage].spawn(num_workers)[worker_id]
                )
            buffer_size = min(self.buffer_size, iter_end - iter_start)
            shuffle_buffer = []
            for i in range(buffer_size):
                shuffle_buffer.append(self.get_sample(rng))

            def sample_from_buffer():
                replacement = self.get_sample(rng)
                evict_idx = shuffle_rng.integers(0, len(shuffle_buffer))
                sample = shuffle_buffer[evict_idx]
                shuffle_buffer[evict_idx] = replacement
                return sample

            return (sample_from_buffer() for _ in range(iter_start, iter_end))
        else:
            return (self.get_sample(rng) for _ in range(iter_start, iter_end))


class SyntheticDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        test_perc: float = 0.1,
        val_perc: float = 0.01,
        resolution: int = 256,
        reduce_dataset: Optional[int] = None,
        buffer_size: int = 100,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_perc = test_perc
        self.val_perc = val_perc
        self.resolution = resolution
        self.buffer_size = buffer_size
        self.n_samples = 100000 if reduce_dataset is None else reduce_dataset
        self.n_samples = {
            "validate": int(self.n_samples * self.val_perc),
            "test": int(self.n_samples * self.test_perc),
            "train": int(self.n_samples * (1 - self.val_perc - self.test_perc)),
            "predict": int(self.n_samples * self.test_perc),
        }
        self.dataloaders = {
            "validate": None,
            "test": None,
            "train": None,
            "predict": None,
        }

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            self.dataloaders["train"] = DataLoader(
                self.get_dataset(stage="train"),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
            self.dataloaders["validate"] = DataLoader(
                self.get_dataset(stage="validate"),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        elif stage == "test":
            self.dataloaders["test"] = DataLoader(
                self.get_dataset(stage=stage),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        elif stage == "predict":
            self.dataloaders["predict"] = DataLoader(
                self.get_dataset(stage=stage),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        elif stage == "validate":
            self.dataloaders["validate"] = DataLoader(
                self.get_dataset(stage=stage),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

    @abstractmethod
    def get_dataset(self, stage=None) -> SyntheticDataset:
        pass

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["validate"]

    def test_dataloader(self):
        return self.dataloaders["test"]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.dataloaders["predict"]
