"""
2026-01-23
base/config.py


Description
-----------
Base configuration dataclass for all FMB model trainers.
Provides common training hyperparameters and system settings.

Usage
-----
from fmb.models.base import BaseTrainingConfig

@dataclass
class MyModelConfig(BaseTrainingConfig):
    model_specific_param: int = 42
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fmb.paths import load_paths


@dataclass
class BaseTrainingConfig:
    """
    Base configuration for all FMB model trainers.
    
    Parameters
    ----------
    out_dir : Path
        Output directory for checkpoints and logs.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        Weight decay (L2 regularization).
    grad_clip : float
        Gradient clipping threshold (0 to disable).
    device : str
        Device to use ('cuda', 'cpu', etc.).
    seed : int
        Random seed for reproducibility.
    amp_dtype : str
        AMP dtype ('float16', 'bfloat16', or 'float32').
    log_interval : int
        Logging interval in steps.
    checkpoint_interval : int
        Checkpoint saving interval in epochs.
    num_workers : int
        Number of dataloader workers.
    """
    
    # Paths
    out_dir: Path = load_paths().retrained_weights
    
    # Training
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # System
    device: str = "cuda"
    seed: int = 42
    amp_dtype: str = "bfloat16"
    
    # Logging
    log_interval: int = 20
    checkpoint_interval: int = 1
    
    # Data
    num_workers: int = 0
    
    # Resume
    resume_checkpoint: Optional[str] = None
