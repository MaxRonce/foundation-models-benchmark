#!/usr/bin/env python3
"""Environment sanity checks for CAMELS ↔︎ AION workflow on Jean-Zay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - should not happen if deps installed
    HfApi = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device for model loading test.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="aion-base",
        help="Hugging Face model identifier to load.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Load model weights from a local directory (avoids HF download on compute nodes).",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip loading the AION model (useful for lightweight checks).",
    )
    parser.add_argument(
        "--skip-codecs",
        action="store_true",
        help="Skip codec/tokenization check.",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip Hugging Face authentication check.",
    )
    return parser.parse_args()


def check_torch(device: str) -> None:
    print("== PyTorch ==")
    print(f"torch version      : {torch.__version__}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count  : {torch.cuda.device_count()}")
        idx = torch.cuda.current_device()
        print(f"Current CUDA device: {idx} ({torch.cuda.get_device_name(idx)})")
    else:
        print("CUDA not available in this environment.")

    try:
        test_tensor = torch.randn(4, 4, device=device)
        print(f"Allocated tensor on '{device}' with shape {tuple(test_tensor.shape)}")
        del test_tensor
    except Exception as exc:  # pragma: no cover
        print(f"[WARNING] Unable to allocate tensor on '{device}': {exc}")


def check_hf_auth() -> None:
    print("\n== Hugging Face ==")
    if HfApi is None:
        print("huggingface_hub not installed; run `pip install huggingface_hub`.")
        return
    try:
        info = HfApi().whoami()
        username = info.get("name") or info.get("fullname") or "<unknown>"
        print(username)
        print(f"Authenticated as   : {username}")
        print(f"Org memberships    : {info.get('orgs', [])}")
    except Exception as exc:  # pragma: no cover
        print("[WARNING] Hugging Face authentication failed.")
        print("          Run `huggingface-cli login --token <HF_TOKEN>` and retry.")
        print(f"          Details: {exc}")


def check_aion(model_name: str, model_dir: Path | None, device: str, skip_codecs: bool) -> None:
    print("\n== AION Model ==")
    import json
    from huggingface_hub import hf_hub_download
    from aion import AION  # Lazy import to provide clearer error if missing

    if model_dir is not None:
        model = AION.from_pretrained(model_dir)
        repo_id = str(model_dir)
        config = None
        codec_repo: str | Path = model_dir
    else:
        repo_id = model_name if "/" in model_name else f"polymathic-ai/{model_name}"
        config_path = hf_hub_download(repo_id, "config.json")
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        model = AION.from_pretrained(repo_id, config=config)
        codec_repo = repo_id
    model = model.to(device)
    model.eval()
    print(f"Loaded `{repo_id}` and moved to `{device}`.")

    if skip_codecs:
        return

    from aion.modalities import LegacySurveyImage
    from camels_aion.config import CAMELS_CODEC_BANDS
    from camels_aion.codec_manager import LocalCodecManager

    codec_manager = LocalCodecManager(repo=codec_repo, device="cpu")
    flux = torch.zeros(1, len(CAMELS_CODEC_BANDS), 128, 128, dtype=torch.float32)
    image = LegacySurveyImage(flux=flux, bands=list(CAMELS_CODEC_BANDS))
    tokens = codec_manager.encode(image)
    tokens = {key: tensor.to(device) for key, tensor in tokens.items()}
    print("Encoded synthetic CAMELS 4-channel image:")
    for key, value in tokens.items():
        print(f"  - {key}: shape={tuple(value.shape)}, dtype={value.dtype}")

    with torch.no_grad():
        embeddings = model.encode(tokens, num_encoder_tokens=10)
    print(f"Model produced embeddings with shape {tuple(embeddings.shape)}.")


def main() -> None:
    args = parse_args()
    check_torch(args.device)
    if not args.skip_hf:
        check_hf_auth()
    if not args.skip_model:
        check_aion(args.model_name, args.model_dir, args.device, args.skip_codecs)


if __name__ == "__main__":
    sys.exit(main())
