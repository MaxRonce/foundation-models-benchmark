#!/usr/bin/env python3
"""
Foundation Models Benchmark (FMB)

Module: fmb.setup.download_weights_aion
Description: FMB module: fmb.setup.download_weights_aion
"""

"""
Script to download AION (and potentially other) model weights to the configured paths.
Consolidates logic from camels-aion scripts.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download

# Add src to pythonpath so we can import 'fmb' package if not installed
src_path = Path(__file__).resolve().parents[2]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fmb.paths import load_paths

# Try to import AION specifics, might fail if package not installed
try:
    from aion.codecs.config import MODALITY_CODEC_MAPPING
    from aion.modalities import LegacySurveyImage

    _AION_AVAILABLE = True
except ImportError:
    _AION_AVAILABLE = False
    print("Warning: 'aion' package not found. Codec priming will be skipped.")

# Constants
DEFAULT_AION_REPO = "polymathic-ai/aion-base"


def download_aion_model(
    repo_id: str,
    revision: Optional[str],
    dest_dir: Path,
) -> None:
    """Download AION model snapshot."""
    print(f"Downloading AION model from '{repo_id}' to '{dest_dir}'...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
    )
    print("AION model download complete.")


def prime_aion_codecs(
    repo_id: str,
    dest_dir: Path,
    device: str = "cpu",
) -> None:
    """Prime/Download AION codecs by instantiating them once."""
    if not _AION_AVAILABLE:
        print("Skipping codec priming (AION not installed).")
        return

    # For now, we only handle LegacySurveyImage as in the original script
    # Simplified logic: just ensure config is there and try to load

    print(
        f"Priming AION Codecs from '{repo_id}' (using local dir '{dest_dir}' if valid)..."
    )

    # Check if we can use local config
    repo_ref = str(dest_dir) if (dest_dir / "config.json").exists() else repo_id

    # Download codec config if needed
    try:
        if Path(repo_ref).exists():
            config_path = (
                Path(repo_ref) / "codecs" / LegacySurveyImage.name / "config.json"
            )
            if not config_path.exists():
                print(
                    "Local codec config missing, falling back to HF Hub download for config..."
                )
                config_path = Path(
                    hf_hub_download(
                        repo_id, f"codecs/{LegacySurveyImage.name}/config.json"
                    )
                )
        else:
            config_path = Path(
                hf_hub_download(repo_id, f"codecs/{LegacySurveyImage.name}/config.json")
            )

        with open(config_path, "r", encoding="utf-8") as fh:
            codec_config = json.load(fh)

        # Instantiate
        codec_cls = MODALITY_CODEC_MAPPING[LegacySurveyImage]
        init_params = inspect.signature(codec_cls.__init__).parameters
        init_kwargs = {
            name: codec_config[name]
            for name in init_params
            if name != "self" and name in codec_config
        }

        # Use local path if it exists, else repo_id
        load_repo = str(dest_dir) if (dest_dir / "config.json").exists() else repo_id

        print(f"Instantiating codec from {load_repo}...")
        codec = codec_cls.from_pretrained(
            load_repo,
            modality=LegacySurveyImage,
            **init_kwargs,
        )
        codec.to(device).eval()
        print("Codec instantiated successfully (weights should be cached).")

    except Exception as e:
        print(f"Error priming codecs: {e}")
        print(
            "You may need to run this again or check your AION installation/connection."
        )


def main() -> None:
    paths = load_paths()

    parser = argparse.ArgumentParser(description="Download AION model weights.")
    parser.add_argument(
        "--repo", default=DEFAULT_AION_REPO, help="HF repo ID for AION."
    )
    parser.add_argument("--revision", default=None, help="Revision for AION.")
    parser.add_argument(
        "--force-codecs",
        action="store_true",
        help="Force codec priming even if model download skipped.",
    )

    args = parser.parse_args()

    dest = paths.base_weights_aion
    if not dest.exists() or len(list(dest.glob("*"))) == 0:
        download_aion_model(args.repo, args.revision, dest)
        prime_aion_codecs(args.repo, dest)
    else:
        print(
            f"AION directory {dest} already exists and is not empty. Skipping download."
        )
        if args.force_codecs:
            prime_aion_codecs(args.repo, dest)

    print("AION weight setup process finished.")


if __name__ == "__main__":
    main()
