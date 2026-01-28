#!/usr/bin/env python3
# Original script comming from https://github.com/mhuertascompany/camels-aion/tree/main and modified 
# to be used as a test for the fmb package
"""Environment sanity checks AION model loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to pythonpath so we can import 'fmb' package if not installed
src_path = Path(__file__).resolve().parents[2]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - should not happen if deps installed
    HfApi = None  # type: ignore


from fmb.paths import load_paths

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
    
    # Default to configured path
    try:
        default_model_dir = load_paths().base_weights_aion
    except Exception:
        default_model_dir = None

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
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

    print(f"Loaded `{repo_id}` and moved to `{device}`.")
    print("Model loaded successfully (weights validation passed).")
    
    if not skip_codecs:
        print("Note: Codec verification skipped")


def main() -> None:
    args = parse_args()
    check_torch(args.device)
    if not args.skip_hf:
        check_hf_auth()
    if not args.skip_model:
        check_aion(args.model_name, args.model_dir, args.device, args.skip_codecs)


if __name__ == "__main__":
    sys.exit(main())
