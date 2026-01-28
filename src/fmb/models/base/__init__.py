"""
2026-01-23
base/__init__.py


Description
-----------
Base module for shared training infrastructure across all FMB models.
Provides abstract base classes and utilities for standardized training.

Usage
-----
from fmb.models.base import BaseTrainer, BaseTrainingConfig
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
