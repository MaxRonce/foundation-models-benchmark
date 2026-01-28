"""
Foundation Models Benchmark (FMB)

Module: fmb.models.base.utils
Description: Common model utilities
"""

import random
from contextlib import nullcontext
from typing import Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_amp(
    device: str,
    dtype: str = "bfloat16",
) -> Tuple[torch.amp.GradScaler, object]:
    """
    Setup Automatic Mixed Precision (AMP) training.

    Parameters
    ----------
    device : str
        Device type ('cuda', 'cpu', etc.).
    dtype : str
        AMP dtype: 'float16', 'bfloat16', or 'float32'.

    Returns
    -------
    scaler : torch.amp.GradScaler
        Gradient scaler for AMP (enabled only for float16 on CUDA).
    ctx : context manager
        Autocast context manager for forward passes.
    """
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    target_dtype = dtype_map.get(dtype, torch.float32)
    use_amp = target_dtype in {torch.bfloat16, torch.float16}

    # GradScaler only needed for float16
    scaler = torch.amp.GradScaler(
        device, enabled=(device == "cuda" and target_dtype == torch.float16)
    )

    # Autocast context
    ctx = (
        torch.amp.autocast(device_type=device, dtype=target_dtype)
        if use_amp
        else nullcontext()
    )

    return scaler, ctx


def format_memory(bytes_val: int) -> str:
    """
    Format memory size in bytes to human-readable string.

    Parameters
    ----------
    bytes_val : int
        Memory size in bytes.

    Returns
    -------
    str
        Formatted string (e.g., '1.5 GB').
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"
