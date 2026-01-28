"""
Foundation Models Benchmark (FMB)

Module: fmb.models.aion.config
Description: AION model configuration
"""

from dataclasses import dataclass
from pathlib import Path

from fmb.models.base.config import BaseTrainingConfig
from fmb.paths import load_paths


@dataclass
class AIONTrainingConfig(BaseTrainingConfig):
    """
    Configuration for AION adapter training.

    Parameters
    ----------
    hidden : int
        Hidden dimension for U-Net adapters.
    use_unet_checkpointing : bool
        Enable gradient checkpointing in U-Net blocks.
    codec_grad : str
        Gradient mode for codec: 'ste' (straight-through estimator) or 'full'.
    disable_codec_checkpointing : bool
        Disable gradient checkpointing in codec forward pass.
    resize : int
        Resize images to this size before cropping.
    crop_size : int
        Random crop size for training.
    max_abs : float
        Clamp absolute flux values (0 to disable).
    cpu_crop : bool
        Perform cropping on CPU before moving to GPU.
    max_entries : int
        Maximum dataset entries (0 for all).
    """

    # Output
    out_dir: Path = load_paths().retrained_weights / "aion"

    # Model (U-Net)
    hidden: int = 16
    use_unet_checkpointing: bool = False

    # Codec
    codec_grad: str = "ste"
    disable_codec_checkpointing: bool = False

    # Preprocessing
    resize: int = 96
    crop_size: int = 96
    max_abs: float = 100.0
    cpu_crop: bool = False

    # Data
    cache_dir: str = str(load_paths().dataset)
    split: str = "all"
    max_entries: int = 0

    # Training defaults (override base)
    epochs: int = 15
    learning_rate: float = 1e-4
    amp_dtype: str = "float16"
