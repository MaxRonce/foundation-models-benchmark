#!/usr/bin/env python3
"""
Foundation Models Benchmark (FMB)

Module: fmb.setup.check_environment_astroclip
Description: Validate AstroCLIP environment and dependencies
"""

"""Environment sanity checks AstroCLIP model loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath so we can import 'fmb' package if not installed
src_path = Path(__file__).resolve().parents[2]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch

from fmb.paths import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for model loading test.",
    )

    # Default to configured path
    try:
        default_model_dir = load_paths().base_weights_astroclip
    except Exception:
        default_model_dir = None

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Directory containing AstroCLIP weights.",
    )
    return parser.parse_args()


def check_astroclip(model_dir: Path, device: str) -> None:
    print("\n== AstroCLIP Model ==")

    if not model_dir or not model_dir.exists():
        print(f"Error: Model directory {model_dir} does not exist.")
        sys.exit(1)

    # Check for the 3 expected files
    required_files = ["astrodino.ckpt", "specformer.ckpt", "astroclip.ckpt"]
    missing = [f for f in required_files if not (model_dir / f).exists()]

    if missing:
        print(f"Error: Missing required weight files in {model_dir}: {missing}")
        print("Please run `python src/fmb/setup/download_weights_astroclip.py` first.")
        sys.exit(1)

    print(f"Found all required weight files in {model_dir}")

    # Try loading the checkpoints simply with torch.load to verify integrity
    # We don't need to instantiate the full model here, just check if weights are readable
    try:
        for f in required_files:
            p = model_dir / f
            print(f"Verifying {f}...", end=" ", flush=True)
            # map_location=device to test device memory allocation too if needed,
            # but usually cpu is safer strict integrity check without OOM
            # weights_only=False is required because these checkpoints might contain
            # lightning globals (AttributeDict, etc).
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            print("OK")
            del ckpt

    except Exception as e:
        print(f"\nError check-loading weights: {e}")
        sys.exit(1)

    print("AstroCLIP weights verified successfully.")


def main() -> None:
    args = parse_args()
    check_astroclip(args.model_dir, args.device)


if __name__ == "__main__":
    sys.exit(main())
