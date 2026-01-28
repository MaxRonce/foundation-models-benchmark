"""
Foundation Models Benchmark (FMB)

Module: fmb.models.astropt.config
Description: AstroPT model configuration
"""

from dataclasses import dataclass
from pathlib import Path

from fmb.models.base.config import BaseTrainingConfig
from fmb.paths import load_paths


@dataclass
class AstroPTTrainingConfig(BaseTrainingConfig):
    """
    Configuration for AstroPT multimodal training.

    Parameters
    ----------
    block_size : int
        Maximum sequence length for transformer.
    image_patch_size : int
        Patch size for image tokenization.
    spectrum_patch_size : int
        Patch size for spectrum tokenization.
    n_layer : int
        Number of transformer layers.
    n_head : int
        Number of attention heads.
    n_embd : int
        Embedding dimension.
    n_chan : int
        Number of image channels (RGB).
    dropout : float
        Dropout probability.
    bias : bool
        Use bias in linear layers.
    image_size : int
        Input image size.
    spectrum_length : int
        Input spectrum length.
    warmup_iters : int
        Learning rate warmup iterations.
    lr_decay_iters : int
        Learning rate decay iterations.
    min_lr : float
        Minimum learning rate.
    eval_interval : int
        Evaluation interval in iterations.
    eval_iters : int
        Number of evaluation iterations.
    compile : bool
        Use torch.compile for model.
    """

    # Output
    out_dir: Path = load_paths().retrained_weights / "astropt"

    # Model architecture
    block_size: int = 1024
    image_patch_size: int = 16
    spectrum_patch_size: int = 10
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_chan: int = 3
    dropout: float = 0.0
    bias: bool = False

    # Data
    cache_dir: str = str(load_paths().dataset)
    train_split: str = "train"
    val_split: str = "test"
    image_size: int = 224
    spectrum_length: int = 7781

    # Training defaults (override base)
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 4
    amp_dtype: str = "bfloat16"

    # Learning rate schedule
    warmup_iters: int = 2000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5

    # Evaluation
    eval_interval: int = 100
    eval_iters: int = 50

    # System
    compile: bool = True
    max_iters: int = 30000
