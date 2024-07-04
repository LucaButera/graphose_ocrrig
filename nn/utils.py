from typing import Optional

import torch
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Identity,
    InstanceNorm1d,
    InstanceNorm2d,
    LeakyReLU,
    ReLU,
    Sigmoid,
    Tanh,
)
from torch_geometric.utils import is_undirected
from torch_scatter import scatter

from data.datasets.utils import limb_masks
from env import BIG_ARCHITECTURES
from utils.data import directed_edges, reduce_by_batch_idx

KEEP_SIZE = "keep_size"


def get_normalization(norm_type: Optional[str], ndims=2):
    if norm_type == "batch":
        if ndims == 2:
            return BatchNorm2d
        elif ndims == 1:
            return BatchNorm1d
    elif norm_type == "instance":
        if ndims == 2:
            return InstanceNorm2d
        elif ndims == 1:
            return InstanceNorm1d
    elif norm_type is None:
        return Identity
    raise RuntimeError(
        f"Unknown norm type {norm_type}. "
        f'Viable options are "batch", "instance" or None.'
    )


conv_params = {
    KEEP_SIZE: {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
    },
}


def d_arch(base_features: int = 64, resolution: int = 64, slim: bool = False):
    architectures = (
        {
            128: {
                "channels": [i * base_features for i in [1, 2, 2, 4, 4, 4]],
                "downsample": [True] * 5 + [False],
                "attention": [res == 64 for res in [64, 32, 16, 8, 4, 4]],
            },
            64: {
                "channels": [i * base_features for i in [2, 2, 4, 4, 4]],
                "downsample": [True] * 4 + [False],
                "attention": [res == 32 for res in [32, 16, 8, 4, 4]],
            },
        }
        if not BIG_ARCHITECTURES
        else {
            64: {
                "channels": [i * base_features for i in [1, 2, 4, 8, 16]],
                "downsample": [True] * 4 + [False],
                "attention": [res == 32 for res in [32, 16, 8, 4, 4]],
            },
        }
    )
    slim_architectures = (
        {
            128: {
                "channels": [i * base_features for i in [1, 2, 2, 4, 4]],
                "downsample": [True] * 5,
                "attention": [res == 64 for res in [64, 32, 16, 8, 4]],
            },
            64: {
                "channels": [i * base_features for i in [2, 2, 4, 4]],
                "downsample": [True] * 4,
                "attention": [res == 32 for res in [32, 16, 8, 4]],
            },
        }
        if not BIG_ARCHITECTURES
        else {}
    )
    return slim_architectures[resolution] if slim else architectures[resolution]


def g_arch(base_features: int = 64, resolution: int = 64):
    architectures = (
        {
            128: {
                "channels": [i * base_features for i in [4, 4, 2, 2, 1]],
                "upsample": [True] * 5,
                "attention": [res == 64 for res in [8, 16, 32, 64, 128]],
            },
            64: {
                "channels": [i * base_features for i in [4, 4, 2, 2]],
                "upsample": [True] * 4,
                "attention": [res == 32 for res in [8, 16, 32, 64]],
            },
        }
        if not BIG_ARCHITECTURES
        else {
            64: {
                "channels": [i * base_features for i in [16, 8, 4, 2]],
                "upsample": [True] * 4,
                "attention": [res == 32 for res in [8, 16, 32, 64]],
            },
        }
    )
    return architectures[resolution]


def g_extractor_arch(mode: str, base_features: int = 64, resolution: int = 64):
    architectures = {
        "mask": {
            128: {
                "feature_channels": [i * base_features for i in [1, 4, 16]],
                "upsample_input_shape": (base_features, 4, 4),
                "upsample_channels": [i * base_features for i in [1, 2, 4, 2, 1]],
                "g_conv": [0, 2, 3],
            },
            64: {
                "feature_channels": [i * base_features for i in [1, 4, 16]],
                "upsample_input_shape": (base_features, 4, 4),
                "upsample_channels": [i * base_features for i in [1, 2, 4, 1]],
                "g_conv": [0, 1, 2],
            },
        },
        "features": {
            n: {
                "feature_channels": [i * base_features for i in [1, 2, 4]],
            }
            for n in [64, 128]
        },
    }
    return architectures[mode][resolution]


def get_activation(activation_type: str):
    if activation_type == "relu":
        return ReLU(inplace=False)
    elif activation_type == "lrelu":
        return LeakyReLU(inplace=False)
    elif activation_type == "tanh":
        return Tanh()
    elif activation_type == "sigmoid":
        return Sigmoid()
    else:
        raise RuntimeError(
            f"Unknown activation type {activation_type}. "
            f'Viable options are "relu" and "lrelu".'
        )


def get_fixed_mask(pos, edge_index, batch, resolution: int):
    dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze().fill_diagonal_(2.0)
    sigma = torch.min(dist, dim=1)[0]
    sigma = (
        torch.log10(1 + sigma.unsqueeze(-1)) * resolution + torch.finfo(sigma.dtype).eps
    )
    x, y = torch.arange(resolution, device=pos.device), torch.arange(
        resolution, device=pos.device
    )
    x = x.expand(pos.shape[0], *x.shape)
    y = y.expand(pos.shape[0], *y.shape)
    coords = (
        (pos[:, :2] * torch.tensor(resolution, dtype=pos.dtype, device=pos.device))
        .to(torch.int)
        .unsqueeze(-1)
    )
    gx = torch.exp(-((x - coords[:, 1]) ** 2) / (2 * sigma**2))
    gy = torch.exp(-((y - coords[:, 0]) ** 2) / (2 * sigma**2))
    node_masks = torch.einsum("nw,nh->nwh", gx, gy)
    node_mask = reduce_by_batch_idx(node_masks, batch).unsqueeze(1)
    if not bool(edge_index.shape[1]):
        node_masks_w_edges = node_masks.unsqueeze(1)
        edge_mask = torch.zeros(
            (1, resolution, resolution),
            dtype=torch.float32,
            device=pos.device,
        )
    else:
        dir_edges = (
            directed_edges(edge_index) if is_undirected(edge_index) else edge_index
        )
        edge_masks = limb_masks(
            pos[:, :2],
            dir_edges,
            (resolution, resolution),
            chi2_factor=1.5,
            ar=10,
            prepend_negative=False,
        )
        edge_mask = reduce_by_batch_idx(edge_masks, batch[dir_edges[0]]).unsqueeze(1)
        node_masks_w_edges = (
            node_masks
            + scatter(
                torch.cat((edge_masks, edge_masks), dim=0),
                torch.cat((dir_edges[0], dir_edges[1]), dim=0),
                dim=0,
                reduce="sum",
                out=torch.zeros_like(node_masks),
            )
        ).unsqueeze(1)
    node_masks_w_edges = node_masks_w_edges.clip(0.0, 1.0)
    total_mask = (edge_mask + node_mask).clip(0.0, 1.0)
    return total_mask, node_masks_w_edges
