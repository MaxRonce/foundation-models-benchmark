#!/usr/bin/env python3
"""
Run inference with the trained astroPT multimodal model and export embeddings.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys
import yaml
import os
import warnings

# Add src to path to allow direct execution
src_path = str(Path(__file__).resolve().parents[2])
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fmb.data.datasets import AstroPTDataset, FMBDataConfig
from fmb.paths import load_paths

# Add external/astroPT/src to path
astropt_path = Path(__file__).resolve().parents[3] / "external" / "astroPT" / "src"
if str(astropt_path) not in sys.path:
    sys.path.insert(0, str(astropt_path))

# Imports from external AstroPT
try:
    from astropt.model import GPT, GPTConfig, ModalityRegistry, ModalityConfig
    # Use internal path for dataloader which was migrated
    from fmb.models.astropt.euclid_desi_dataset.multimodal_dataloader import (
        multimodal_collate_fn,
        prepare_multimodal_batch,
    )
except ImportError as e:
    print(f"Error importing AstroPT components: {e}")
    print("Error importing AstroPT. Make sure external/astroPT/src is in PYTHONPATH.")
    sys.exit(1)


def parse_args_and_config() -> argparse.Namespace:
    paths = load_paths()
    
    # Defaults
    def_ckpt = paths.retrained_weights / "astropt" / "ckpt_final.pt"
    def_cache = paths.dataset
    def_out = paths.embeddings / "astropt"

    # Argument Parser
    p = argparse.ArgumentParser(description="Export astroPT embeddings for Euclid+DESI.")
    p.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    
    # Set defaults to None to detect presence
    p.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    p.add_argument("--split", type=str, default=None, help="Dataset split to process.")
    p.add_argument("--cache-dir", type=str, default=None, help="Path to dataset cache.")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save embeddings.")
    p.add_argument("--output-name", type=str, default=None, help="Filename for embeddings.")
    p.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    p.add_argument("--num-workers", type=int, default=None, help="Number of dataloader workers.")
    p.add_argument("--reduction", choices=["mean", "exp_decay", "last", "none"], default=None, help="Embedding reduction method.")
    p.add_argument("--device", type=str, default=None, help="Compute device (cuda/cpu).")
    
    args = p.parse_args()
    
    # 1. Load config
    cfg = {}
    if args.config:
         with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    # 2. Get Value Helper
    def get_val(arg_name, default, config_key=None):
        cli_val = getattr(args, arg_name)
        if cli_val is not None:
             return cli_val
        key = config_key or arg_name.replace("-", "_")
        if key in cfg and cfg[key] is not None:
             return cfg[key]
        return default

    # 3. Resolve
    # Checkpoint Smart Recovery
    checkpoint_str = get_val("checkpoint", None)
    
    if checkpoint_str:
        final_ckpt = Path(checkpoint_str)
    else:
        # Smart Recovery
        if def_ckpt.exists():
            print(f"[Auto-Recovery] Found retrained weights: {def_ckpt}")
            final_ckpt = def_ckpt
        else:
            # Fallback to failing if not found
            final_ckpt = def_ckpt 

    # Construct clean Namespace object
    args.checkpoint = str(final_ckpt)
    args.split = get_val("split", "all")
    args.cache_dir = get_val("cache_dir", str(def_cache))
    args.output_dir = get_val("output_dir", str(def_out))
    args.output_name = get_val("output_name", "astropt_embeddings.pt")
    args.batch_size = int(get_val("batch_size", 16))
    args.num_workers = int(get_val("num_workers", 2))
    args.reduction = get_val("reduction", "mean")
    args.device = get_val("device", None)
    
    return args


def create_modality_registry(config) -> ModalityRegistry:
    # Check if config is object or dict
    def get(k): return getattr(config, k) if hasattr(config, k) else config[k]
    
    modalities = [
        ModalityConfig(
            name="images",
            input_size=get('image_patch_size') * get('image_patch_size') * get('n_chan'),
            patch_size=get('image_patch_size'),
            loss_weight=1.0, 
            embed_pos=True,
            pos_input_size=1,
        ),
        ModalityConfig(
            name="spectra",
            input_size=get('spectrum_patch_size'),
            patch_size=get('spectrum_patch_size'),
            pos_input_size=1,
            loss_weight=1.0,
            embed_pos=True,
        ),
    ]
    return ModalityRegistry(modalities)

def load_model(checkpoint_path: Path, device: torch.device):
    # Force weights_only=False to support safe globals in older checkpoints or custom classes
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config_dict = ckpt["config"]
    
    # Create simple config object
    class SimpleConfig:
        def __init__(self, **entries): self.__dict__.update(entries)
    
    train_cfg = SimpleConfig(**config_dict)
    modality_registry = create_modality_registry(train_cfg)

    # Rebuild model
    gpt_conf = GPTConfig(attn_type="causal", **ckpt["model_args"])
    model = GPT(gpt_conf, modality_registry)
    
    msg = model.load_state_dict(ckpt["model"], strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"Model weights loaded with info: {msg}")
    else:
        print("Model weights loaded successfully.")

    model.to(device)
    model.eval()
    return model, modality_registry, train_cfg


def main() -> None:
    # Suppress warnings
    warnings.filterwarnings("ignore", message="xFormers is not available")
    warnings.filterwarnings("ignore", message=".*instance of `nn.Module` and is already saved.*")

    args = parse_args_and_config()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)
    
    print(f"Using device: {device}")
    print(f"Using checkpoint: {ckpt_path}")
    
    if not ckpt_path.is_file():
        # Clean error message
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"Loading model from {ckpt_path}...")
    model, modality_registry, train_cfg = load_model(ckpt_path, device)

    # Use FMBDataConfig
    data_config = FMBDataConfig(
        split=args.split,
        cache_dir=args.cache_dir,
        image_size=train_cfg.image_size,
        spectrum_length=train_cfg.spectrum_length
    )
    print(f"Loading dataset from {args.cache_dir} (split={args.split})...")
    dataset = AstroPTDataset(data_config)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=device.type == "cuda",
    )
    
    print(f"Starting inference on {len(dataset)} samples...")

    records: Dict[str, Dict] = {}
    progress = tqdm(loader, desc="Encoding", unit="batch")

    with torch.inference_mode():
        for batch in progress:
            inputs = prepare_multimodal_batch(
                batch,
                image_patch_size=train_cfg.image_patch_size,
                spectrum_patch_size=train_cfg.spectrum_patch_size,
                device=device,
                modality_registry=modality_registry,
            )
            if not inputs:
                continue

            embeds = model.generate_embeddings(inputs, reduction=args.reduction)

            if "images" in embeds and "image_object_ids" in batch:
                ids: List = batch["image_object_ids"]
                redshifts = batch.get("image_redshifts", [])
                for i, obj_id in enumerate(ids):
                    rec = records.setdefault(str(obj_id), {"object_id": obj_id})
                    if i < len(redshifts):
                        rec["redshift"] = float(redshifts[i])
                    rec["embedding_images"] = embeds["images"][i].detach().cpu()

            if "spectra" in embeds and "spectrum_object_ids" in batch:
                ids_spec: List = batch["spectrum_object_ids"]
                redshifts_spec = batch.get("spectrum_redshifts", [])
                for i, obj_id in enumerate(ids_spec):
                    rec = records.setdefault(str(obj_id), {"object_id": obj_id})
                    if i < len(redshifts_spec):
                        rec["redshift"] = float(redshifts_spec[i])
                    rec["embedding_spectra"] = embeds["spectra"][i].detach().cpu()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_name
    
    print(f"Saving {len(records)} embedding records to {out_path}...")
    torch.save(list(records.values()), out_path)

    print("Done!")


if __name__ == "__main__":
    main()
