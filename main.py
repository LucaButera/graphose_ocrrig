from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING

from config import register
from config.callbacks import CallbackConfig
from config.datamodule import DataModuleConfig
from config.loggers import LoggerConfig
from config.models import ModelConfig
from config.predictors import PredictorConfig


@dataclass
class TrainerConfig:
    _target_: str = "pytorch_lightning.Trainer"
    gpus: int = 0
    max_epochs: int = 50
    logger: LoggerConfig = "${logger}"
    log_every_n_steps: int = 1
    fast_dev_run: bool = False
    auto_select_gpus: bool = gpus > 0
    callbacks: List[CallbackConfig] = "${oc.dict.values: callbacks}"


@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"callback@callbacks.cb1": "checkpoint"},
            {"callback@callbacks.cb2": "lr"},
            {"callback@callbacks.cb3": "early"},
        ]
    )
    run_id: str = wandb.util.generate_id()
    checkpoint: Optional[str] = None
    monitor: str = "val/loss"
    mode: str = "fit"
    hydra: Any = field(
        default_factory=lambda: {
            "job": {"chdir": True},
            "run": {"dir": "outputs/${run_id}"},
            "sweep": {
                "dir": "multirun/${run_id}",
                "subdir": "${hydra.job.num}",
            },
        }
    )
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    datamodule: DataModuleConfig = MISSING
    predictor: PredictorConfig = MISSING
    model: ModelConfig = MISSING
    callbacks: Dict[str, CallbackConfig] = MISSING


cs = ConfigStore.instance()
cs.store(name="experiment", node=ExperimentConfig)
register(["logger", "callback", "predictor", "model", "datamodule"])


def setup_multiprocessing():
    from logging import getLogger

    from torch import multiprocessing

    start_method = "spawn"
    try:
        multiprocessing.set_start_method(start_method)
    except RuntimeError as e:
        if multiprocessing.get_start_method() == start_method:
            getLogger("graphose/setup").debug(e)
        else:
            raise e


@hydra.main(config_path=None, config_name="experiment", version_base="1.2")
def main(cfg: ExperimentConfig) -> None:
    setup_multiprocessing()
    datamodule = instantiate(cfg.datamodule)
    predictor = instantiate(cfg.predictor)
    trainer = instantiate(cfg.trainer)
    getattr(trainer, cfg.mode)(predictor, datamodule, ckpt_path=cfg.checkpoint)


if __name__ == "__main__":
    main()
