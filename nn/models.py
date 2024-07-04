from typing import Optional, Sequence, Tuple, Union

from torch import Tensor, randn
from torch.nn import Module

from nn.discriminator import ImGraphDiscriminator
from nn.generator import (
    GraphLayoutExtractor,
    GraphLayoutMaskExtractor,
    PGBGGenerator,
    PGNOUPGenerator,
)


class MaskGeneratorModel(Module):
    def __init__(
        self,
        num_pos_dims: int = 2,
        base_features: int = 8,
        mask_channels: int = 1,
        noise_channels_mask: int = 0,
        resolution: int = 64,
        norm: Optional[str] = "batch",
        spectral_norm: bool = True,
        activation: str = "relu",
        use_skip_connection: bool = True,
        mask_reduce: str = "sum",
        gen_output_activation: Optional[str] = "sigmoid",
    ):
        super().__init__()
        self.extractor = GraphLayoutMaskExtractor(
            num_point_dimensions=num_pos_dims,
            base_features=base_features,
            mask_channels=mask_channels,
            z_dim=noise_channels_mask,
            resolution=resolution,
            norm=norm,
            spectral_norm=spectral_norm,
            activation=activation,
            skip_connection=use_skip_connection,
            mask_reduce=mask_reduce,
            output_activation=gen_output_activation,
        )

    def forward(
        self,
        pos,
        edge_index,
        batch=None,
        z_mask=None,
    ) -> Tuple[Tensor, Tensor]:
        return self.extractor(pos, edge_index, z_mask, batch)

    def _sample_noise(self, graphs) -> Optional[Tensor]:
        z_mask = (
            randn(
                graphs.pos.shape[0],
                self.extractor.z_dim,
                device=graphs.pos.device,
                dtype=graphs.pos.dtype,
            )
            if self.extractor.z_dim > 0
            else None
        )
        return z_mask

    def sample(self, graphs) -> Tuple[Tensor, Tensor]:
        z_mask = self._sample_noise(graphs)
        return self(
            pos=graphs.pos,
            edge_index=graphs.edge_index,
            batch=graphs.batch,
            z_mask=z_mask,
        )


class GeneratorModel(Module):
    def __init__(
        self,
        num_node_features: Union[Sequence[int], int] = 16,
        num_pos_dims: int = 2,
        node_embedding_dim: int = 8,
        base_features: int = 8,
        base_extractor_features: int = 8,
        mask_channels: int = 1,
        resolution: int = 64,
        norm: Optional[str] = "batch",
        spectral_norm: bool = True,
        activation: str = "relu",
        gen_output_activation: Optional[str] = "sigmoid",
        attention: bool = True,
        noise_channels_mask: int = 0,
        noise_channels_feature: int = 0,
        use_skip_connection: bool = True,
        num_img_channels: int = 3,
        generator_latent_z_dim: int = 0,
        separate_layout_conv: bool = False,
        learn_masks: bool = True,
        use_graph: bool = True,
        embedding_aggr: str = "cat",
        simple_model: bool = False,
        upsampling_generator: bool = True,
    ):
        super().__init__()
        self.extractor = GraphLayoutExtractor(
            num_node_features=num_node_features,
            resolution=resolution,
            z_dim_mask=noise_channels_mask,
            z_dim_feature=noise_channels_feature,
            norm=norm,
            spectral_norm=spectral_norm,
            activation=activation,
            node_embedding_dim=node_embedding_dim,
            base_features=base_extractor_features,
            mask_channels=mask_channels,
            num_point_dimensions=num_pos_dims,
            skip_connection=use_skip_connection,
            output_activation=gen_output_activation,
            learn_masks=learn_masks,
            use_graph=use_graph,
            embedding_aggr=embedding_aggr,
        )
        if not upsampling_generator:
            self.generator = PGNOUPGenerator(
                layout_channels=self.extractor.layout_channels,
                num_channels=num_img_channels,
                base_image_features=base_features,
                resolution=resolution,
                norm=norm,
                spectral_norm=spectral_norm,
                activation=activation,
                simple=simple_model,
            )
        else:
            self.generator = PGBGGenerator(
                base_image_features=base_features,
                layout_channels=self.extractor.layout_channels,
                resolution=resolution,
                norm=norm,
                spectral_norm=spectral_norm,
                attention=attention,
                activation=activation,
                latent_dim=generator_latent_z_dim,
                separate_layout_conv=separate_layout_conv,
                simple=simple_model,
                num_channels=num_img_channels,
            )

    def forward(
        self,
        x,
        pos,
        edge_index,
        batch=None,
        z_mask=None,
        z_feature=None,
        z_generator=None,
        x_mask=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        layout, total_mask, masks = self.extractor(
            x, pos, edge_index, z_mask, z_feature, batch, x_mask
        )
        return self.generator(layout, z_generator), layout, total_mask, masks

    def _sample_noise(
        self, imgs, graphs
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        z_generator = (
            randn(
                imgs.shape[0],
                self.generator.latent_dim,
                device=imgs.device,
                dtype=imgs.dtype,
            )
            if hasattr(self.generator, "latent_dim") and self.generator.latent_dim > 0
            else None
        )
        z_mask = (
            randn(
                graphs.pos.shape[0],
                self.extractor.mask_extractor.z_dim,
                device=graphs.pos.device,
                dtype=graphs.pos.dtype,
            )
            if hasattr(self.extractor, "mask_extractor")
            and self.extractor.mask_extractor.z_dim > 0
            else None
        )
        z_feature = (
            randn(
                graphs.pos.shape[0],
                self.extractor.z_dim,
                device=graphs.pos.device,
                dtype=graphs.pos.dtype,
            )
            if self.extractor.z_dim > 0
            else None
        )
        return z_generator, z_mask, z_feature

    def sample(self, imgs, graphs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z_generator, z_mask, z_feature = self._sample_noise(imgs, graphs)
        return self(
            x=graphs.x,
            pos=graphs.pos,
            edge_index=graphs.edge_index,
            batch=graphs.batch,
            z_mask=z_mask,
            z_feature=z_feature,
            z_generator=z_generator,
            x_mask=graphs.x_mask if hasattr(graphs, "x_mask") else None,
        )


class GeneratorDiscriminatorModel(GeneratorModel):
    def __init__(
        self,
        num_node_features: Union[Sequence[int], int] = 16,
        num_pos_dims: int = 2,
        node_embedding_dim: int = 8,
        base_features: int = 8,
        base_extractor_features: int = 8,
        mask_channels: int = 1,
        resolution: int = 64,
        norm: Optional[str] = "batch",
        spectral_norm: bool = True,
        activation: str = "relu",
        gen_output_activation: Optional[str] = "sigmoid",
        attention: bool = True,
        noise_channels_mask: int = 0,
        noise_channels_feature: int = 0,
        use_skip_connection: bool = True,
        num_img_channels: int = 3,
        generator_latent_z_dim: int = 0,
        separate_layout_conv: bool = False,
        learn_masks: bool = True,
        use_graph: bool = True,
        embedding_aggr: str = "cat",
        simple_model: bool = False,
        upsampling_generator: bool = True,
        slim_discriminator: bool = False,
        discriminator_sigmoid_out: bool = False,
    ):
        super().__init__(
            num_node_features,
            num_pos_dims,
            node_embedding_dim,
            base_features,
            base_extractor_features,
            mask_channels,
            resolution,
            norm,
            spectral_norm,
            activation,
            gen_output_activation,
            attention,
            noise_channels_mask,
            noise_channels_feature,
            use_skip_connection,
            num_img_channels,
            generator_latent_z_dim,
            separate_layout_conv,
            learn_masks,
            use_graph,
            embedding_aggr,
            simple_model,
            upsampling_generator,
        )
        self.discriminator = ImGraphDiscriminator(
            num_node_features=num_node_features,
            resolution=resolution,
            norm=norm,
            spectral_norm=spectral_norm,
            activation=activation,
            node_embedding_dim=base_extractor_features,
            base_features=base_features,
            base_extractor_features=base_extractor_features,
            num_channels=num_img_channels,
            attention=attention,
            sigmoid_out=discriminator_sigmoid_out,
            num_point_dimensions=num_pos_dims,
            skip_connection=use_skip_connection,
            embedding_aggr=embedding_aggr,
            slim=slim_discriminator,
            simple=simple_model,
            use_graph=use_graph,
        )

    def discriminate(self, x, pos, edge_index, images, batch=None, x_mask=None):
        return self.discriminator(x, pos, edge_index, images, batch, x_mask)
