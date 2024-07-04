from typing import Tuple

from numpy.random import Generator
from torch import Tensor
from torch_geometric.data import Data

from data.datasets.synthetic import SyntheticDataModule, SyntheticDataset
from data.datasets.utils import mask_from_graph, random_graph


class PoseMaskDataset(SyntheticDataset):
    def get_sample(self, rng: Generator) -> Tuple[Tensor, Data]:
        graph = random_graph(rng=rng)
        mask = mask_from_graph(graph, resolution=self.resolution)
        return mask, graph


class PoseMaskDataModule(SyntheticDataModule):
    def get_dataset(self, stage=None) -> SyntheticDataset:
        return PoseMaskDataset(
            resolution=self.resolution,
            n_samples=self.n_samples[stage],
            stage=stage,
            buffer_size=self.buffer_size,
        )
