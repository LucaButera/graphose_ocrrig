from typing import Tuple, Union

import networkx as nx
import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import (
    barabasi_albert_graph,
    erdos_renyi_graph,
    is_undirected,
    to_networkx,
    to_undirected,
)

from utils.data import directed_edges, graphose_rng, reduce_by_batch_idx


def gauss_from_2p(p1: Tensor, p2: Tensor, chi2_factor: float = 3.219, ar: float = 2.0):
    # P1 and P2 have shape [NPoints x 2]
    p_diff = p2 - p1
    phi = torch.atan2(p_diff[:, 1], p_diff[:, 0])
    maj_ax = torch.diagonal(torch.cdist(p1, p2))
    min_ax = maj_ax / ar

    var1 = (maj_ax / 2) ** 2 * torch.cos(phi) ** 2 + (min_ax / 2) ** 2 * torch.sin(
        phi
    ) ** 2
    var2 = (maj_ax / 2) ** 2 * torch.sin(phi) ** 2 + (min_ax / 2) ** 2 * torch.cos(
        phi
    ) ** 2
    cov12 = ((maj_ax / 2) ** 2 - (min_ax / 2) ** 2) * torch.sin(phi) * torch.cos(phi)

    mu = (p1 + p2) / 2
    sigma = (
        torch.stack(
            [torch.stack([var1, cov12], dim=1), torch.stack([cov12, var2], dim=1)],
            dim=1,
        )
        / chi2_factor
    )
    return mu, sigma


def multivariate_gaussian(img_size: Tuple[int, int], mu: Tensor, sigma: Tensor):
    x, y = torch.meshgrid(
        torch.arange(img_size[0], device=mu.device),
        torch.arange(img_size[1], device=mu.device),
        indexing="xy",
    )
    grid = torch.stack([x, y], dim=-1)
    n = mu.shape[1]
    sigma_det = torch.linalg.det(sigma)
    sigma_inv, info = torch.linalg.inv_ex(sigma)
    g = torch.sqrt((2 * torch.pi) ** n * sigma_det)[:, None, None]
    # Set g to 1 where sigma was not invertible
    g[info != 0] = 1
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    trans_grid = grid - mu[:, None, None, :]
    fac = torch.einsum("b...k,b...kl,b...l->b...", trans_grid, sigma_inv, trans_grid)
    result = torch.exp(-fac / 2) / g
    # Setting values to zero where sigma was not invertible
    # Sigma is not invertible where the two points are placed at the same location
    result[info != 0] = 0.0
    return result


def limb_masks(
    x: Tensor,
    edge_index: Tensor,
    img_size: Tuple[int, int],
    chi2_factor: float = 3.219,
    ar: float = 2.0,
    prepend_negative: bool = True,
) -> Tensor:
    # Point_index, Pair_index, Coord_index
    p = x[edge_index] * torch.tensor(img_size, dtype=x.dtype, device=x.device)
    mu, sigma = gauss_from_2p(p[0], p[1], chi2_factor=chi2_factor, ar=ar)
    g = multivariate_gaussian(img_size, mu, sigma)
    g -= g.amin(dim=(1, 2), keepdim=True)
    g /= g.amax(dim=(1, 2), keepdim=True) + 1e-6
    if prepend_negative:
        bg_mask = 1.0 - g.amax(dim=0, keepdim=True)
        masks = torch.cat([bg_mask, g], dim=0)
    else:
        masks = g
    return masks


def random_graph(
    num_nodes: Union[int, Tuple[int, int]] = (5, 30),
    layout_alg: str = "kamada",
    rng: Generator = graphose_rng,
) -> Data:
    if layout_alg == "kamada":
        alg = nx.kamada_kawai_layout
    elif layout_alg == "spring":
        alg = nx.spring_layout
    else:
        raise ValueError("layout_alg must be one of kamada or spring.")
    if isinstance(num_nodes, tuple):
        num_nodes = rng.integers(num_nodes[0], num_nodes[1], endpoint=True)
    if rng.random() > 0.5:
        num_edges = 2 if num_nodes < 10 or rng.random() > 0.9 else 1
        graph = Data(
            edge_index=barabasi_albert_graph(num_nodes=num_nodes, num_edges=num_edges),
            num_nodes=num_nodes,
        )
    else:
        graph = Data(
            edge_index=erdos_renyi_graph(
                num_nodes=num_nodes, edge_prob=np.exp(1 / num_nodes) - 0.95
            ),
            num_nodes=num_nodes,
        )
    nx_graph = to_networkx(graph).to_undirected()
    layout = alg(nx_graph, center=(0.5, 0.5), scale=0.5)
    pos = torch.zeros(len(layout), 2)
    random_scale = rng.uniform(0.3, 1.0, 2)
    random_translate = rng.uniform(0.0, 1 - random_scale)
    for node_id in layout:
        pos[node_id] = (
            torch.from_numpy(layout[node_id]) * random_scale + random_translate
        )
    graph.pos = pos
    graph.edge_index = to_undirected(edge_index=graph.edge_index)
    return graph


def node_mask_from_graph(
    graph: Data,
    resolution: int,
    dynamic_size: bool = True,
    pos_attr: str = "pos",
) -> Tensor:
    if dynamic_size:
        dist = (
            torch.cdist(graph[pos_attr].unsqueeze(0), graph[pos_attr].unsqueeze(0))
            .squeeze()
            .fill_diagonal_(2.0)
        )
        sigma = torch.min(dist, dim=1)[0]
        sigma = (
            torch.log10(1 + sigma.unsqueeze(-1)) * resolution
            + torch.finfo(sigma.dtype).eps
        )
    else:
        sigma = resolution / 16

    x, y = torch.arange(resolution, device=graph[pos_attr].device), torch.arange(
        resolution, device=graph[pos_attr].device
    )
    x = x.expand(graph[pos_attr].shape[0], *x.shape)
    y = y.expand(graph[pos_attr].shape[0], *y.shape)

    coords = (
        (
            graph[pos_attr][:, :2]
            * torch.tensor(
                resolution, dtype=graph[pos_attr].dtype, device=graph[pos_attr].device
            )
        )
        .to(torch.int)
        .unsqueeze(-1)
    )

    gx = torch.exp(-((x - coords[:, 1]) ** 2) / (2 * sigma**2))
    gy = torch.exp(-((y - coords[:, 0]) ** 2) / (2 * sigma**2))
    g = torch.einsum("nw,nh->nwh", gx, gy)

    if hasattr(graph, "batch") and graph.batch is not None:
        mask = reduce_by_batch_idx(g, graph.batch).unsqueeze(1)
    else:
        mask = g.sum(dim=0, keepdims=True).clip(0.0, 1.0)
    return mask


def limb_mask_from_graph(graph: Data, resolution: int, pos_attr: str = "pos") -> Tensor:
    if not bool(graph.edge_index.shape[1]):
        return torch.zeros(
            (1, resolution, resolution),
            dtype=torch.float32,
            device=graph[pos_attr].device,
        )
    dir_edges = (
        directed_edges(graph.edge_index)
        if is_undirected(graph.edge_index)
        else graph.edge_index
    )
    masks = limb_masks(
        graph[pos_attr][:, :2],
        dir_edges,
        (resolution, resolution),
        chi2_factor=1.5,
        ar=10,
        prepend_negative=False,
    )
    if hasattr(graph, "batch") and graph.batch is not None:
        mask = reduce_by_batch_idx(masks, graph.batch[dir_edges[0]]).unsqueeze(1)
    else:
        mask = masks.sum(dim=0, keepdims=True).clip(0.0, 1.0)
    return mask


def mask_from_graph(graph: Data, resolution: int, pos_attr: str = "pos") -> Tensor:
    return (
        limb_mask_from_graph(graph, resolution=resolution, pos_attr=pos_attr)
        + node_mask_from_graph(graph, resolution=resolution, pos_attr=pos_attr)
    ).clip(0.0, 1.0)
