"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.style
Description: Centralized visualization styling
"""

from pathlib import Path
from typing import Optional

from .utils import load_viz_style


def apply_style(config_path: Optional[Path] = None):
    """
    Apply the FMB publication style.
    Delegates to utils.load_viz_style which loads from standard config location.
    """
    load_viz_style()


# Alias for backward compatibility or preference
set_style = apply_style
