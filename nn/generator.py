from logging import getLogger
from typing import Optional, Sequence, Union

import torch
from torch.nn import Identity, Module, ModuleList, Sequential
from torch.nn.functional import avg_pool2d
from torch_geometric.nn import Sequential as GSequential

from nn.blocks import GBlock, PG2DBlock, PGGBlock
from nn.layers import (
    Attention,
    BaselineLayer,
    GPoseLayer,
    Interpolate,
    MultiMaskedEmbedding,
    conv_module,
    linear_module,
)
from nn.utils import (
    KEEP_SIZE,
    conv_params,
    g_arch,
    g_extractor_arch,
    get_activation,
    get_fixed_mask,
    get_normalization,
)
from utils.data import reduce_by_batch_idx


class GraphLayoutMaskExtractor(Module):
    def __init__(
        self,
        num_point_dimensions: int = 3,
        base_features: int = 64,
        mask_channels: int = 1,
        z_dim: int = 0,
        resolution: int = 64,
        norm: Optional[str] = "batch",
        spectral_norm: bool = True,
        activation: str = "relu",
        skip_connection: bool = True,
        mask_reduce: str = "sum",
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        self.num_point_dimensions = num_point_dimensions
        self.base_features = base_features
        self.mask_channels = mask_channels
        self.z_dim = z_dim
        self.resolution = resolution
        self.spectral_norm = spectral_norm
        self.activation = get_activation(activation_type=activation)
        self.output_activation = (
            get_activation(activation_type=output_activation)
            if output_activation is not None
            else Identity()
        )
        self.norm_module_1d = get_normalization(norm_type=norm, ndims=1)
        self.norm_module_2d = get_normalization(norm_type=norm, ndims=2)
        self.skip_connection = skip_connection
        self.mask_reduce = mask_reduce
        self.arch = g_extractor_arch(
            mode="mask", base_features=self.base_features, resolution=self.resolution
        )
        self.layout_channels = self.arch["feature_channels"][-1]

        self.graph_feature_extractor = []
        for i in range(len(self.arch["feature_channels"])):
            in_channels = (
                self.arch["feature_channels"][i - 1]
                if i > 0
                else self.num_point_dimensions + z_dim
            )
            out_channels = self.arch["feature_channels"][i]
            self.graph_feature_extractor.append(
                (
                    GPoseLayer(
                        in_channels=in_channels,
                        hidden_channels=out_channels,
                        out_channels=out_channels,
                        num_dimensions=self.num_point_dimensions,
                        spectral_norm=self.spectral_norm,
                        skip_connection=self.skip_connection,
                    ),
                    "x, pos, edge_index -> x",
                )
            )
            if i < len(self.arch["feature_channels"]) - 1:
                self.graph_feature_extractor.append(
                    (self.norm_module_1d(num_features=out_channels), "x -> x"),
                )
                self.graph_feature_extractor.append(
                    (self.activation, "x -> x"),
                )
        self.graph_feature_extractor = GSequential(
            "x, pos, edge_index", self.graph_feature_extractor
        )
        self.graph2masks = []
        for i in range(len(self.arch["upsample_channels"])):
            in_channels = self.arch["upsample_channels"][i]
            out_channels = self.arch["upsample_channels"][
                min(i + 1, len(self.arch["upsample_channels"]) - 1)
            ]
            if i in self.arch["g_conv"]:
                self.graph2masks.append(
                    (
                        PG2DBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            norm_module=self.norm_module_2d,
                            spectral_norm=self.spectral_norm,
                            activation=self.activation,
                        ),
                        "x, edge_index -> x",
                    ),
                )
            else:
                self.graph2masks.append(
                    (
                        GBlock(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            spectral_norm=self.spectral_norm,
                            norm_module=self.norm_module_2d,
                            activation=self.activation,
                            simple=True,
                        ),
                        "x -> x",
                    ),
                )
        last_out_channels = self.arch["upsample_channels"][-1]
        self.graph2masks.extend(
            [
                (
                    self.norm_module_2d(num_features=last_out_channels),
                    "x -> x",
                ),
                self.activation,
                (
                    conv_module(
                        in_channels=last_out_channels,
                        out_channels=self.mask_channels,
                        spectral_norm=self.spectral_norm,
                        **conv_params[KEEP_SIZE],
                    ),
                    "x -> x",
                ),
                self.output_activation,
            ]
        )
        self.graph2masks = GSequential("x, edge_index", self.graph2masks)

    def forward(
        self,
        pos,
        edge_index,
        z: Optional[torch.Tensor] = None,
        batch: Optional = None,
    ):
        if z is None:
            z = torch.Tensor().to(pos)
        features = self.graph_feature_extractor(
            torch.cat((pos, z), dim=1), pos, edge_index
        )
        square_features = features.view(
            features.shape[0], *self.arch["upsample_input_shape"]
        )
        masks = self.graph2masks(
            square_features,
            edge_index,
        )
        total_mask = reduce_by_batch_idx(masks, batch, reduce=self.mask_reduce)
        return total_mask, masks


class GraphLayoutExtractor(Module):
    def __init__(
        self,
        num_node_features: Union[Sequence[int], int] = 1,
        num_point_dimensions: int = 3,
        node_embedding_dim: int = 64,
        base_features: int = 64,
        mask_channels: int = 1,
        z_dim_mask: int = 0,
        z_dim_feature: int = 0,
        resolution: int = 64,
        norm: Optional[str] = "batch",
        spectral_norm: bool = True,
        activation: str = "relu",
        skip_connection: bool = True,
        mask_reduce: str = "sum",
        output_activation: Optional[str] = None,
        learn_masks: bool = True,
        use_graph: bool = True,
        embedding_aggr: str = "cat",
    ):
        super().__init__()
        self.num_node_features = (
            (num_node_features,)
            if isinstance(num_node_features, int)
            else num_node_features
        )
        self.num_point_dimensions = num_point_dimensions
        self.node_embedding_dim = node_embedding_dim
        self.base_features = base_features
        self.z_dim = z_dim_feature
        self.resolution = resolution
        self.spectral_norm = spectral_norm
        # Can learn masks only if using graph
        self.use_graph = use_graph
        self.learn_masks = learn_masks & self.use_graph
        if self.learn_masks != learn_masks:
            getLogger(self.__class__.__name__).warning(
                f"Set learn_masks to {self.learn_masks} "
                f"as use_graph is {self.use_graph}"
            )
        self.activation = get_activation(activation_type=activation)
        self.output_activation = (
            get_activation(activation_type=output_activation)
            if output_activation is not None
            else Identity()
        )
        self.embedding_aggr = embedding_aggr
        self.norm_module_1d = get_normalization(norm_type=norm, ndims=1)
        self.skip_connection = skip_connection
        self.mask_reduce = mask_reduce
        self.arch = g_extractor_arch(
            mode="features",
            base_features=self.base_features,
            resolution=self.resolution,
        )
        self.layout_channels = self.arch["feature_channels"][-1]

        self.feature_embedding = MultiMaskedEmbedding(
            num_embeddings=self.num_node_features,
            embedding_dim=self.node_embedding_dim,
            aggr=self.embedding_aggr,
        )

        self.graph_feature_extractor = []
        for i in range(len(self.arch["feature_channels"])):
            in_channels = (
                self.arch["feature_channels"][i - 1]
                if i > 0
                else self.feature_embedding.out_features + self.z_dim
            )
            out_channels = self.arch["feature_channels"][i]
            self.graph_feature_extractor.append(
                (
                    (
                        GPoseLayer(
                            in_channels=in_channels,
                            hidden_channels=out_channels,
                            out_channels=out_channels,
                            num_dimensions=self.num_point_dimensions,
                            spectral_norm=self.spectral_norm,
                            skip_connection=self.skip_connection,
                        )
                        if self.use_graph
                        else BaselineLayer(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            num_dimensions=self.num_point_dimensions,
                            spectral_norm=self.spectral_norm,
                            skip_connection=self.skip_connection,
                        )
                    ),
                    "x, pos, edge_index -> x" if self.use_graph else "x, pos -> x",
                )
            )
            if i < len(self.arch["feature_channels"]) - 1:
                self.graph_feature_extractor.append(
                    (self.norm_module_1d(num_features=out_channels), "x -> x"),
                )
                self.graph_feature_extractor.append(
                    (self.activation, "x -> x"),
                )
        self.graph_feature_extractor.append((self.output_activation, "x -> x"))
        self.graph_feature_extractor = GSequential(
            "x, pos, edge_index" if self.use_graph else "x, pos",
            self.graph_feature_extractor,
        )
        if self.learn_masks:
            self.mask_extractor = GraphLayoutMaskExtractor(
                num_point_dimensions=num_point_dimensions,
                base_features=base_features,
                mask_channels=mask_channels,
                z_dim=z_dim_mask,
                resolution=resolution,
                norm=norm,
                spectral_norm=spectral_norm,
                activation=activation,
                skip_connection=skip_connection,
                mask_reduce=mask_reduce,
                output_activation=output_activation,
            )

    def forward(
        self,
        x,
        pos,
        edge_index,
        z_mask: Optional[torch.Tensor] = None,
        z_feature: Optional[torch.Tensor] = None,
        batch: Optional = None,
        x_mask: Optional[torch.Tensor] = None,
    ):
        embedding = self.feature_embedding(
            x.split(1, dim=-1),
            x_mask.split(1, dim=-1) if x_mask is not None else x_mask,
        )
        if z_feature is None:
            z_feature = torch.Tensor().to(embedding)
        style = torch.cat((embedding, z_feature), dim=1)
        if self.use_graph:
            features = self.graph_feature_extractor(style, pos, edge_index)
        else:
            features = self.graph_feature_extractor(style, pos)
        if self.learn_masks:
            total_mask, masks = self.mask_extractor(pos, edge_index, z_mask, batch)
        else:
            total_mask, masks = get_fixed_mask(pos, edge_index, batch, self.resolution)
        rich_masks = torch.einsum("nc,nchw->nchw", features, masks)
        layout = reduce_by_batch_idx(rich_masks, batch, reduce=self.mask_reduce)
        return layout, total_mask, masks


class PGBGGenerator(Module):
    def __init__(
        self,
        layout_channels: int,
        num_channels: int = 3,
        base_image_features: int = 8,
        base_input_width: int = 4,
        resolution: int = 64,
        norm: Optional[str] = "batch",
        spectral_norm: bool = True,
        attention: bool = True,
        activation: str = "relu",
        latent_dim: int = 0,
        separate_layout_conv: bool = False,
        simple: bool = False,
    ):
        super().__init__()
        self.layout_channels = layout_channels
        self.num_channels = num_channels
        self.base_features = base_image_features
        self.base_input_width = base_input_width
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.attention = attention
        self.spectral_norm = spectral_norm
        self.separate_layout_conv = separate_layout_conv
        self.simple = simple
        self.activation = get_activation(activation_type=activation)
        self.norm_module = get_normalization(norm_type=norm)
        self.architecture = g_arch(
            base_features=self.base_features, resolution=self.resolution
        )

        if self.latent_dim > 0:
            self.linear = linear_module(
                spectral_norm=self.spectral_norm,
                in_features=self.latent_dim,
                out_features=self.base_input_width**2,
            )

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for i in range(len(self.architecture["channels"])):
            self.blocks += [
                [
                    (
                        GBlock(
                            in_channels=(
                                1 if i == 0 else self.architecture["channels"][i - 1]
                            )
                            + self.layout_channels,
                            out_channels=self.architecture["channels"][i],
                            spectral_norm=self.spectral_norm,
                            norm_module=self.norm_module,
                            activation=self.activation,
                            upsample=(
                                Interpolate(scale_factor=2)
                                if self.architecture["upsample"][i]
                                else None
                            ),
                            simple=self.simple,
                        )
                        if not self.separate_layout_conv
                        else PGGBlock(
                            in_channels=(
                                1 if i == 0 else self.architecture["channels"][i - 1]
                            ),
                            out_channels=self.architecture["channels"][i],
                            layout_channels=self.layout_channels,
                            spectral_norm=self.spectral_norm,
                            norm_module=self.norm_module,
                            activation=self.activation,
                            upsample=(
                                Interpolate(scale_factor=2)
                                if self.architecture["upsample"][i]
                                else None
                            ),
                        )
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

        # output layer: batch-norm-relu-conv.
        self.output_layer = Sequential(
            self.norm_module(num_features=self.architecture["channels"][-1]),
            self.activation,
            conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.architecture["channels"][-1],
                out_channels=self.num_channels,
                **conv_params[KEEP_SIZE],
            ),
        )

    def forward(self, layout, z):
        if self.latent_dim > 0:
            h = self.linear(z)
            h = h.view(z.size(0), -1, self.base_input_width, self.base_input_width)
        else:
            h = avg_pool2d(
                layout.mean(dim=1, keepdim=True),
                int(layout.shape[-1] / self.base_input_width),
            )
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                if isinstance(block, Attention):
                    h = block(h)
                elif not self.separate_layout_conv:
                    pooled_layout = avg_pool2d(
                        layout, int(layout.shape[-1] / h.shape[-1])
                    )
                    h = block(torch.concat((pooled_layout, h), dim=1))
                else:
                    h = block(layout, h)
        # Apply batch-norm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))


class PGNOUPGenerator(Module):
    def __init__(
        self,
        layout_channels: int,
        num_channels: int = 3,
        base_image_features: int = 8,
        resolution: int = 64,
        norm: Optional[str] = "batch",
        spectral_norm: bool = True,
        activation: str = "relu",
        simple: bool = False,
    ):
        super().__init__()
        self.layout_channels = layout_channels
        self.num_channels = num_channels
        self.base_features = base_image_features
        self.resolution = resolution
        self.spectral_norm = spectral_norm
        self.simple = simple
        self.activation = get_activation(activation_type=activation)
        self.norm_module = get_normalization(norm_type=norm)
        self.architecture = g_arch(
            base_features=self.base_features, resolution=self.resolution
        )

        self.blocks = []
        for i in range(len(self.architecture["channels"])):
            self.blocks += [
                [
                    GBlock(
                        in_channels=(
                            self.layout_channels
                            if i == 0
                            else self.architecture["channels"][i - 1]
                        ),
                        out_channels=self.architecture["channels"][i],
                        spectral_norm=self.spectral_norm,
                        norm_module=self.norm_module,
                        activation=self.activation,
                        upsample=None,
                        simple=self.simple,
                    )
                ]
            ]
        self.blocks = ModuleList([ModuleList(block) for block in self.blocks])

        # output layer: batch-norm-relu-conv.
        self.output_layer = Sequential(
            self.norm_module(num_features=self.architecture["channels"][-1]),
            self.activation,
            conv_module(
                spectral_norm=self.spectral_norm,
                in_channels=self.architecture["channels"][-1],
                out_channels=self.num_channels,
                **conv_params[KEEP_SIZE],
            ),
        )

    def forward(self, layout, z):
        # z is not used in the simple model.
        # It is just needed to avoid having two different forward calls.
        h = layout
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        return torch.tanh(self.output_layer(h))
