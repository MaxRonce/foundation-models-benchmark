#!/usr/bin/env python3
"""
Foundation Models Benchmark (FMB)

Module: fmb.embeddings.generate_embeddings_astropt
Description: Generate embeddings using AstroPT
"""

"""
Run inference with the trained astroPT multimodal model and export embeddings.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imports from local package
from fmb.data.datasets import AstroPTDataset, FMBDataConfig
# Use fmb.paths
from fmb.paths import load_paths

# Add external/astroPT/src to path
astropt_path = Path(__file__).resolve().parents[3] / "external" / "astroPT" / "src"
if str(astropt_path) not in sys.path:
    sys.path.insert(0, str(astropt_path))

# Imports from external AstroPT
try:
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
    # Use internal path for dataloader which was migrated
    from fmb.models.astropt.euclid_desi_dataset.multimodal_dataloader import (
        multimodal_collate_fn, prepare_multimodal_batch)
except ImportError as e:
    print(f"Error importing AstroPT components: {e}")
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
    
    p.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    p.add_argument("--split", type=str, default=None, help="Dataset split to process.")
    p.add_argument("--cache-dir", type=str, default=None, help="Path to dataset cache.")
    p.add_argument("--output-dir", type=str, default=None, help="Directory to save embeddings.")
    p.add_argument("--output-name", type=str, default=None, help="Filename for embeddings.")
    p.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    p.add_argument("--num-workers", type=int, default=None, help="Number of dataloader workers.")
    p.add_argument("--reduction", choices=["mean", "exp_decay", "last", "none"], default=None, help="Embedding reduction method.")
    p.add_argument("--device", type=str, default=None, help="Compute device (cuda/cpu).")
    p.add_argument("--verbose", action="store_true", default=None, help="Enable verbose logging.")
    p.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process.")
    
    args = p.parse_args()
    
    cfg = {}
    if args.config:
         with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    def get_val(arg_name, default, config_key=None):
        cli_val = getattr(args, arg_name)
        if cli_val is not None:
             return cli_val
        key = config_key or arg_name.replace("-", "_")
        if key in cfg and cfg[key] is not None:
             return cfg[key]
        return default

    # Checkpoint Smart Recovery
    checkpoint_str = get_val("checkpoint", None)
    if checkpoint_str:
        final_ckpt = Path(checkpoint_str)
    else:
        candidates = [
            paths.retrained_weights / "models" / "astropt" / "ckpt_final.pt",
            paths.retrained_weights / "astropt" / "ckpt_final.pt",
            paths.retrained_weights / "astropt" / "ckpt.pt",
        ]
        final_ckpt = None
        for cand in candidates:
            if cand.exists():
                print(f"[Auto-Recovery] Found retrained weights: {cand}")
                final_ckpt = cand
                break
        if not final_ckpt:
            final_ckpt = def_ckpt 

    args.checkpoint = str(final_ckpt)
    args.split = get_val("split", "all")
    args.cache_dir = get_val("cache_dir", str(def_cache))
    args.output_dir = get_val("output_dir", str(def_out))
    args.output_name = get_val("output_name", "astropt_embeddings.pt")
    args.max_samples = get_val("max_samples", None)
    args.verbose = get_val("verbose", False)
    args.batch_size = int(get_val("batch_size", 16))
    args.num_workers = int(get_val("num_workers", 2))
    args.reduction = get_val("reduction", "mean")
    args.device = get_val("device", None)
    
    return args


def create_modality_registry(config) -> ModalityRegistry:
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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = ckpt["config"]
    class SimpleConfig:
        def __init__(self, **entries): self.__dict__.update(entries)
    train_cfg = SimpleConfig(**config_dict)
    modality_registry = create_modality_registry(train_cfg)
    gpt_conf = GPTConfig(attn_type="causal", **ckpt["model_args"])
    model = GPT(gpt_conf, modality_registry)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    return model, modality_registry, train_cfg


def main() -> None:
    warnings.filterwarnings("ignore", message="xFormers is not available")
    args = parse_args_and_config()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)
    
    print(f"Using device: {device}")
    print(f"Using checkpoint: {ckpt_path}")
    
    if not ckpt_path.is_file():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"Loading model from {ckpt_path}...")
    model, modality_registry, train_cfg = load_model(ckpt_path, device)

    data_config = FMBDataConfig(
        split=args.split,
        cache_dir=args.cache_dir,
        image_size=train_cfg.image_size,
        spectrum_length=train_cfg.spectrum_length,
        max_entries=args.max_samples,
    )
    print(f"Loading dataset split={args.split}...")
    dataset = AstroPTDataset(data_config, verbose=args.verbose)
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=device.type == "cuda",
    )
    
    print(f"Starting inference on {len(dataset)} samples...")
    records = []
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
            img_ids = batch.get("image_object_ids", [])
            spec_ids = batch.get("spectrum_object_ids", [])
            img_redshifts = batch.get("image_redshifts", [])
            spec_redshifts = batch.get("spectrum_redshifts", [])
            
            img_emb = embeds.get("images")
            spec_emb = embeds.get("spectra")
            
            batch_size = len(img_ids) if img_ids else len(spec_ids)
            
            for i in range(batch_size):
                obj_id = img_ids[i] if i < len(img_ids) else (spec_ids[i] if i < len(spec_ids) else None)
                redshift = img_redshifts[i] if i < len(img_redshifts) else (spec_redshifts[i] if i < len(spec_redshifts) else None)
                
                record = {
                    "object_id": obj_id,
                    "targetid": obj_id,
                    "redshift": float(redshift) if redshift is not None else None,
                }
                
                emb_i = None
                if img_emb is not None and i < len(img_emb):
                    emb_i = img_emb[i].detach().cpu()
                    record["embedding_images"] = emb_i
                
                emb_s = None
                if spec_emb is not None and i < len(spec_emb):
                    emb_s = spec_emb[i].detach().cpu()
                    record["embedding_spectra"] = emb_s
                    
                if emb_i is not None and emb_s is not None:
                    import torch.nn.functional as F
                    joint = F.normalize(F.normalize(emb_i, dim=0) + F.normalize(emb_s, dim=0), dim=0)
                    record["embedding_joint"] = joint
                elif emb_i is not None:
                    record["embedding_joint"] = emb_i
                elif emb_s is not None:
                    record["embedding_joint"] = emb_s
                    
                records.append(record)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_name
    
    print(f"Saving {len(records)} records to {out_path}...")
    torch.save(records, out_path)
    print("Done!")


if __name__ == "__main__":
    main()
