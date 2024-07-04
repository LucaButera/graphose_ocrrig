from dataclasses import dataclass


@dataclass
class LoggerConfig:
    _target_: str = "pytorch_lightning.loggers.tensorboard.TensorBoardLogger"
    save_dir: str = "tensorboard_logs"
