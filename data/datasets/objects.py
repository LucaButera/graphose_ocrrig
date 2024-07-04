from typing import Optional, Tuple

import networkx as nx
import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from data.datasets.synthetic import SyntheticDataModule, SyntheticDataset
from data.synthetic.primitives import COLORS, str_to_rgb
from data.synthetic.sample import SyntheticObjectsSample

NODE_TYPE_TO_NUMERIC = {
    "robot_base": 0,
    "robot_arm": 1,
    "robot_prong": 2,
    "hand_wrist": 3,
    "hand_finger": 4,
    "pie_center": 5,
    "pie_tip": 6,
    "polygon_vertex": 7,
    "polygon_center": 8,
    "scissors_pivot": 9,
    "scissors_tip": 10,
    "scissors_handle": 11,
    "lattice_vertex": 12,
}

COLOR_TO_NUMERIC = {tuple(str_to_rgb(c)): i for i, c in enumerate(COLORS)}


class ObjectsDataset(SyntheticDataset):
    def __init__(
        self,
        resolution: int = 256,
        n_samples: int = 10000,
        stage: Optional[str] = None,
        buffer_size: int = 100,
        mask_prob: float = 0.1,
        max_retry: int = 10,
    ):
        assert 0 <= mask_prob <= 1
        super().__init__(
            resolution=resolution,
            n_samples=n_samples,
            stage=stage,
            buffer_size=buffer_size,
        )
        self.mask_prob = mask_prob
        self.max_retry = max_retry

    def get_sample(self, rng: Generator) -> Tuple[Tensor, Data]:
        for _ in range(self.max_retry):
            try:
                sample = SyntheticObjectsSample(self.resolution, rng=rng)
                graph, img = sample.get_sample()
                break
            except IndexError:
                continue
        else:
            raise RuntimeError(
                f"Unable to produce a valid sample after {self.max_retry} retries."
            )
        img = np.array(img, copy=True, dtype=np.uint8)[..., :3]
        img = torch.from_numpy(np.transpose(img.astype(np.float32) / 255, (2, 0, 1)))
        nx.set_node_attributes(
            graph,
            {k: NODE_TYPE_TO_NUMERIC[graph.nodes[k]["type"]] for k in graph},
            "type",
        )
        nx.set_node_attributes(
            graph,
            {k: COLOR_TO_NUMERIC[tuple(graph.nodes[k]["color"])] for k in graph},
            "color",
        )
        graph = from_networkx(graph, group_node_attrs=["color", "type"])
        graph.x = graph.x.to(torch.int32)
        graph.pos = graph.pos.to(torch.float32)
        # Masking occurs in 50% of samples with probability self.mask_prob
        if rng.random() > 0.5:
            graph.x_mask = torch.from_numpy(
                rng.uniform(size=graph.x.shape) > self.mask_prob
            ).to(graph.x.device)
            graph.pos_mask = torch.from_numpy(
                rng.uniform(size=(graph.pos.shape[0], 1)) > self.mask_prob
            ).to(graph.pos.device)
        else:
            graph.x_mask = torch.ones_like(
                graph.x, dtype=torch.bool, device=graph.x.device
            )
            graph.pos_mask = torch.ones(
                size=(graph.pos.shape[0], 1), dtype=torch.bool, device=graph.pos.device
            )
        return img, graph


class ObjectsDataModule(SyntheticDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        test_perc: float = 0.1,
        val_perc: float = 0.01,
        resolution: int = 256,
        reduce_dataset: Optional[int] = None,
        buffer_size: int = 100,
        mask_prob: float = 0.1,
        max_retry: int = 10,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            test_perc=test_perc,
            val_perc=val_perc,
            resolution=resolution,
            reduce_dataset=reduce_dataset,
            buffer_size=buffer_size,
        )
        self.mask_prob = mask_prob
        self.max_retry = max_retry

    def get_dataset(self, stage=None) -> SyntheticDataset:
        return ObjectsDataset(
            resolution=self.resolution,
            n_samples=self.n_samples[stage],
            mask_prob=self.mask_prob,
            max_retry=self.max_retry,
            stage=stage,
            buffer_size=self.buffer_size,
        )
