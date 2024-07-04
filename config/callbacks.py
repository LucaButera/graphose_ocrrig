from dataclasses import dataclass
from typing import Optional


@dataclass
class CallbackConfig:
    pass


@dataclass
class LearningRateMonitorConfig(CallbackConfig):
    _target_: str = "pytorch_lightning.callbacks.LearningRateMonitor"
    logging_interval: Optional[str] = None
    log_momentum: bool = False


@dataclass
class ModelCheckpointConfig(CallbackConfig):
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    dirpath: str = "checkpoints"
    every_n_epochs: Optional[int] = None
    monitor: Optional[str] = "${monitor}"
    save_top_k: int = 1
    mode: str = "min"
    save_last: Optional[bool] = True
    save_weights_only: bool = False


@dataclass
class EarlyStoppingConfig(CallbackConfig):
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: Optional[str] = "${monitor}"
    min_delta: float = 0.0
    patience: int = 50
    verbose: bool = False
    mode: str = "min"
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    check_on_train_epoch_end: Optional[bool] = None
