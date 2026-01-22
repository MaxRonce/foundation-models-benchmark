#!/usr/bin/env python3
"""Download the required AstroCLIP checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
src_path = Path(__file__).resolve().parents[2]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from huggingface_hub import hf_hub_download
from fmb.paths import load_paths

# Weights to download: (repo_id, filename)
ASTROCLIP_WEIGHTS = [
    ("polymathic-ai/astrodino", "astrodino.ckpt"),
    ("polymathic-ai/specformer", "specformer.ckpt"),
    ("polymathic-ai/astroclip", "astroclip.ckpt"),
]

def download_astroclip_weights(dest_dir: Path) -> None:
    """Download the 3 required AstroCLIP checkpoints."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AstroCLIP weights to {dest_dir}...")
    
    for repo_id, filename in ASTROCLIP_WEIGHTS:
        print(f"  Downloading {filename} from {repo_id}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
    print("AstroCLIP weights download complete.")

def main() -> None:
    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=paths.base_weights_astroclip,
        help="Destination directory for the checkpoints.",
    )
    args = parser.parse_args()
    
    download_astroclip_weights(args.dest)

if __name__ == "__main__":
    main()
