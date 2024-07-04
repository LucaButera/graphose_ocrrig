from typing import Sequence, Union

import torch
from torch import sigmoid
from torch.nn import AvgPool2d, Module, ModuleList, Sequential
from torch_geometric.nn import global_add_pool

from nn.blocks import DBlock
from nn.generator import GraphLayoutExtractor
from nn.layers import Attention, linear_module
from nn.utils import d_arch, get_activation, get_normalization


class BGDiscriminator(Module):
    def __init__(
        self,
        num_channels: int = 3,
        base_features: int = 64,
        resolution: int = 64,
        activation: str = "relu",
        spectral_norm: bool = True,
        sigmoid_out: bool = True,
        attention: bool = True,
        wide: bool = True,
        slim: bool = False,
        simple: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.base_features = base_features
        self.wide = wide
        self.slim = slim
        self.resolution = resolution
        self.spectral_norm = spectral_norm
        self.attention = attention
        self.activation = get_activation(activation_type=activation)
        self.architecture = d_arch(
            base_features=self.base_features,
            resolution=self.resolution,
            slim=self.slim,
        )
        self.sigmoid_out = sigmoid_out
        self.simple = simple

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []
        for i in range(len(self.architecture["channels"])):
            self.blocks += [
                [
                    DBlock(
                        in_channels=(
                            self.num_channels
                            if i == 0
                            else self.architecture["channels"][i - 1]
                        ),
                        out_channels=self.architecture["channels"][i],
                        spectral_norm=self.spectral_norm,
                        wide=self.wide,
                        activation=self.activation,
                        preactivation=True,
                        downsample=(
                            AvgPool2d(2) if self.architecture["downsample"][i] else None
                        ),
                        simple=self.simple,
                    )
                ]
            ]
            if self.attention and self.architecture["attention"][i]:
                self.blocks[-1] += [
                    Attention(
                        ch=self.architecture["channels"][i],
                        spectral_norm=self.spectral_norm,
                    )
                ]
        self.blocks = ModuleList([ModuleList(block) for block in self.blocks])
        # Linear output layer
        self.linear = linear_module(
            spectral_norm=self.spectral_norm,
            in_features=self.architecture["channels"][-1],
            out_features=1,
        )

    def forward(self, x, truncated: bool = False):
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(input=self.activation(h), dim=(2, 3))
        if truncated:
            return h
        out = self.linear(h)
        return sigmoid(out) if self.sigmoid_out else out


class ImGraphDiscriminator(Module):
    def __init__(
        self,
        num_node_features: Union[Sequence[int], int] = 3,
        resolution: int = 64,
        norm: str = "batch",
        activation: str = "relu",
        node_embedding_dim: int = 64,
        base_features: int = 64,
        base_extractor_features: int = 32,
        spectral_norm: bool = True,
        num_channels: int = 3,
        wide: bool = True,
        slim: bool = False,
        attention: bool = True,
        sigmoid_out: bool = False,
        num_point_dimensions: int = 3,
        skip_connection: bool = True,
        embedding_aggr: str = "cat",
        simple: bool = False,
        use_graph: bool = True,
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.node_embedding_dim = node_embedding_dim
        self.resolution = resolution
        self.spectral_norm = spectral_norm
        self.activation = get_activation(activation_type=activation)
        self.norm_module_1d = get_normalization(norm_type=norm, ndims=1)
        self.norm_module_2d = get_normalization(norm_type=norm, ndims=2)
        self.num_channels = num_channels
        self.base_extractor_features = base_extractor_features
        self.base_features = base_features
        self.wide = wide
        self.slim = slim
        self.attention = attention
        self.architecture = d_arch(
            base_features=self.base_features, resolution=self.resolution
        )
        self.sigmoid_out = sigmoid_out
        self.num_point_dimensions = num_point_dimensions
        self.skip_connection = skip_connection
        self.embedding_aggr = embedding_aggr
        self.simple = simple
        self.use_graph = use_graph

        extractor = GraphLayoutExtractor(
            num_node_features=self.num_node_features,
            node_embedding_dim=self.node_embedding_dim,
            num_point_dimensions=self.num_point_dimensions,
            base_features=self.base_features,
            resolution=self.resolution,
            norm=norm,
            spectral_norm=self.spectral_norm,
            activation=activation,
            skip_connection=self.skip_connection,
            embedding_aggr=self.embedding_aggr,
            use_graph=self.use_graph,
        )
        self.feature_embedding = extractor.feature_embedding
        self.graph_feature_extractor = extractor.graph_feature_extractor

        self.img_feature_extractor = BGDiscriminator(
            num_channels=self.num_channels,
            base_features=self.base_features,
            resolution=self.resolution,
            activation=activation,
            spectral_norm=self.spectral_norm,
            sigmoid_out=self.sigmoid_out,
            attention=self.attention,
            wide=self.wide,
            slim=self.slim,
            simple=self.simple,
        )
        if self.use_graph:
            graph_out_features = (
                self.graph_feature_extractor[-2].pose_conv.global_nn[0].out_features
            )
        else:
            graph_out_features = self.graph_feature_extractor[-2].self_nn.out_features
        self.decision_blocks = Sequential(
            linear_module(
                spectral_norm=self.spectral_norm,
                in_features=self.img_feature_extractor.linear.in_features
                + graph_out_features,
                out_features=self.img_feature_extractor.linear.in_features,
            ),
            self.activation,
            linear_module(
                spectral_norm=self.spectral_norm,
                in_features=self.img_feature_extractor.linear.in_features,
                out_features=1,
            ),
        )

    def forward(self, x, pos, edge_index, image, batch, x_mask=None):
        embedding = self.feature_embedding(
            x.split(1, dim=-1),
            x_mask.split(1, dim=-1) if x_mask is not None else x_mask,
        )
        if self.use_graph:
            graph_features = global_add_pool(
                self.graph_feature_extractor(embedding, pos, edge_index), batch
            )
        else:
            graph_features = global_add_pool(
                self.graph_feature_extractor(embedding, pos), batch
            )
        img_features = self.img_feature_extractor(image, truncated=True)
        return self.decision_blocks(torch.cat([graph_features, img_features], dim=1))
