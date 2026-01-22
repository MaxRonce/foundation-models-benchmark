from pathlib import Path

import torch
from aion.model import AION
from .codec_manager import LocalCodecManager
from fmb.paths import load_paths

# Use lazy loading for default to allow config to be set before access if possible, or just load now.
# Since this is toplevel constant, it will load on import.
def _get_default_model_dir() -> Path:
    try:
        return load_paths().base_weights / "aion"
    except Exception:
        return Path("/pbs/throng/training/astroinfo2025/model")

def load_model_and_codec(
    model_dir: Path | None = None,
    device: torch.device | None = None,
    codec_dir: Path | None = None,
):
    """Load the pretrained model and its codec manager for the given directory.

    Args:
        model_dir: Directory containing the AION model checkpoint.
        device: Target torch.device.
        codec_dir: Optional directory containing codec weights (defaults to model_dir).
    """
    if model_dir is None:
        model_dir = _get_default_model_dir()
        
    print(f"Loading model from {model_dir}...")
    model_dir = Path(model_dir)
    codec_repo = Path(codec_dir) if codec_dir is not None else model_dir
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AION.from_pretrained(model_dir).to(device).eval()
    codec_manager = LocalCodecManager(repo=codec_repo, device=device)
    print(f"Model loaded from {model_dir}; codecs from {codec_repo}; device {device}")
    return model, codec_manager


if __name__ == "__main__":
    model, codec_manager = load_model_and_codec()
    print(f"Model and codec manager loaded")
