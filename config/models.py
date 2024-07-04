from dataclasses import dataclass, field

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("neq", lambda a, b: a != b)


@dataclass
class ModelConfig:
    resolution: int = "${datamodule.resolution}"


@dataclass
class DiscriminatorConfig:
    discriminator_sigmoid_out: bool = "${neq:${predictor.loss},Hinge}"


@dataclass
class NoisyConfig:
    noise_channels_feature: int = 4
    noise_channels_mask: int = 4


@dataclass
class MaskGeneratorConfig(ModelConfig):
    noise_channels_mask: int = 4


@dataclass
class SimpleParamsConfig:
    num_node_features: list = field(default_factory=lambda: [10, 13])
    attention: bool = False
    upsampling_generator: bool = False
    simple_model: bool = True
    slim_discriminator: bool = True


@dataclass
class BigParamsConfig:
    separate_layout_conv: bool = True
    generator_latent_z_dim: int = 8
    base_features: int = 32
    base_extractor_features: int = 16


@dataclass
class GANBaselineConfig(
    ModelConfig, NoisyConfig, DiscriminatorConfig, SimpleParamsConfig
):
    use_graph: bool = False


@dataclass
class GANGraphoseNoMaskConfig(
    ModelConfig, NoisyConfig, DiscriminatorConfig, SimpleParamsConfig
):
    learn_masks: bool = False


@dataclass
class GANGraphoseConfig(
    ModelConfig, NoisyConfig, DiscriminatorConfig, SimpleParamsConfig
):
    pass


@dataclass
class BIGGANBaselineConfig(
    ModelConfig, NoisyConfig, DiscriminatorConfig, BigParamsConfig
):
    use_graph: bool = False


@dataclass
class BIGGANGraphoseNoMaskConfig(
    ModelConfig, NoisyConfig, DiscriminatorConfig, BigParamsConfig
):
    learn_masks: bool = False


@dataclass
class BIGGANGraphoseConfig(
    ModelConfig, NoisyConfig, DiscriminatorConfig, BigParamsConfig
):
    pass
