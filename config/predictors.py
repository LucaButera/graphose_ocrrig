from dataclasses import dataclass

from config.models import ModelConfig


@dataclass
class PredictorConfig:
    model_kwargs: ModelConfig = "${model}"


@dataclass
class MaskPredictorConfig(PredictorConfig):
    _target_: str = "nn.predictors.MaskPredictor"
    loss: str = "BCEWL"


@dataclass
class GANPredictorConfig(PredictorConfig):
    _target_: str = "nn.predictors.GANPredictor"
    loss: str = "Hinge"
