from typing import Sequence, Union

from hydra.core.config_store import ConfigStore

from config.callbacks import (
    EarlyStoppingConfig,
    LearningRateMonitorConfig,
    ModelCheckpointConfig,
)
from config.datamodule import ObjectsDataModuleConfig, PoseMaskDataModuleConfig
from config.loggers import LoggerConfig
from config.models import (
    BIGGANBaselineConfig,
    BIGGANGraphoseConfig,
    BIGGANGraphoseNoMaskConfig,
    GANBaselineConfig,
    GANGraphoseConfig,
    GANGraphoseNoMaskConfig,
    MaskGeneratorConfig,
)
from config.predictors import GANPredictorConfig, MaskPredictorConfig

_groups = {
    "logger": {
        "dummy": LoggerConfig,
    },
    "callback": {
        "early": EarlyStoppingConfig,
        "lr": LearningRateMonitorConfig,
        "checkpoint": ModelCheckpointConfig,
    },
    "predictor": {
        "mask": MaskPredictorConfig,
        "gan": GANPredictorConfig,
    },
    "model": {
        "baseline_gan": GANBaselineConfig,
        "graphose_nomask_gan": GANGraphoseNoMaskConfig,
        "graphose_gan": GANGraphoseConfig,
        "baseline_biggan": BIGGANBaselineConfig,
        "graphose_nomask_biggan": BIGGANGraphoseNoMaskConfig,
        "graphose_biggan": BIGGANGraphoseConfig,
        "mask": MaskGeneratorConfig,
    },
    "datamodule": {
        "pose_mask": PoseMaskDataModuleConfig,
        "objects": ObjectsDataModuleConfig,
    },
}


def register(groups: Union[str, Sequence[str]]):
    cs = ConfigStore.instance()
    if isinstance(groups, str):
        groups = [groups]
    for group in groups:
        for name in _groups[group]:
            cs.store(group=group, name=name, node=_groups[group][name])
