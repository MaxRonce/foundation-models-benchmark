"""
2026-01-23
astroclip/config.py


Description
-----------
Configuration dataclass for AstroCLIP image encoder fine-tuning.

Usage
-----
from fmb.models.astroclip.config import AstroCLIPTrainingConfig

config = AstroCLIPTrainingConfig(epochs=5, batch_size=128)
"""

from dataclasses import dataclass
from pathlib import Path

from fmb.models.base.config import BaseTrainingConfig
from fmb.paths import load_paths


@dataclass
class AstroCLIPTrainingConfig(BaseTrainingConfig):
    """
    Configuration for AstroCLIP fine-tuning.
    
    Parameters
    ----------
    checkpoint : str
        Path to AstroCLIP Lightning checkpoint.
    slice_length : int
        Spectrum length after padding/trimming.
    image_size : int
        Image size.
    max_samples : int
        Maximum number of samples (None for all).
    val_ratio : float
        Validation split ratio.
    spectrum_norm : str
        Spectrum normalization: 'zscore', 'minmax', or 'none'.
    include_wavelength : bool
        Stack flux and wavelength as two channels.
    focus_high_z : bool
        Prioritize high-redshift galaxies in sampling.
    warmup_steps : int
        Learning rate warmup steps (0 for auto 10%).
    patience : int
        Early stopping patience (epochs).
    min_delta : float
        Minimum improvement for early stopping.
    accumulate_steps : int
        Gradient accumulation steps.
    unfreeze_backbone_blocks : int
        Number of DINO backbone blocks to unfreeze.
    learnable_scale : bool
        Make CLIP temperature learnable.
    finetune_spectrum : bool
        Fine-tune spectrum encoder (else frozen).
    use_arrow : bool
        Load from Arrow cache instead of parquet.
    split : str
        Dataset split to use ('train', 'test').
    """
    
    # Output
    out_dir: Path = load_paths().retrained_weights / "astroclip"
    
    # Model
    checkpoint: str = ""  # Required
    learnable_scale: bool = False
    finetune_spectrum: bool = False
    unfreeze_backbone_blocks: int = 0
    
    # Data
    cache_dir: str = str(load_paths().dataset)
    use_arrow: bool = True
    split: str = "train"
    slice_length: int = 7700
    image_size: int = 144
    max_samples: int = None
    val_ratio: float = 0.1
    spectrum_norm: str = "none"
    include_wavelength: bool = False
    focus_high_z: bool = False
    
    # Training defaults (override base)
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 3e-6
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    accumulate_steps: int = 1
    amp_dtype: str = "float16"
    
    # Learning rate schedule
    warmup_steps: int = 0  # 0 means auto (10% of total)
    
    # Early stopping
    patience: int = 3
    min_delta: float = 1e-4
