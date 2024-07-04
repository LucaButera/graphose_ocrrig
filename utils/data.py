import numpy as np
import torch
from numpy.random import SeedSequence
from torch import Tensor
from torch_scatter import scatter

from env import REPRODUCIBLE

seed = SeedSequence(42) if REPRODUCIBLE else SeedSequence()
graphose_rng = np.random.default_rng(seed)
stages = ["train", "validate", "test", "predict"]
stage_seeds = {stage: seed for stage, seed in zip(stages, seed.spawn(len(stages)))}


def directed_edges(edge_index: Tensor) -> Tensor:
    d_edges = []
    seen = set()
    for i, j in edge_index.T:
        t = (i.item(), j.item())
        s = frozenset(t)
        if s not in seen:
            seen.add(s)
            d_edges.append(t)
    return torch.tensor(d_edges, dtype=edge_index.dtype, device=edge_index.device).T


def reduce_by_batch_idx(features: Tensor, idx: Tensor, reduce: str = "sum"):
    # Use scatter_add to sum the features for each batch element
    return scatter(features, idx, dim=0, reduce=reduce)
