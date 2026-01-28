"""
Foundation Models Benchmark (FMB)

Module: fmb.models.base.__init__
Description: FMB module: fmb.models.base.__init__
"""

from fmb.models.base.config import BaseTrainingConfig
from fmb.models.base.trainer import BaseTrainer
from fmb.models.base.utils import set_seed, setup_amp

__all__ = [
    "BaseTrainer",
    "BaseTrainingConfig",
    "set_seed",
    "setup_amp",
]
