
"""
Retraining script for AstroCLIP using Euclid+DESI unified dataset and YAML configuration.
Adapted from finetune_image_encoder.py.
"""

import argparse
import math
import os
import time
import json
import random
# Suppress warnings early
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message=".*instance of `nn.Module` and is already saved.*")

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

# Ensure src is in pythonpath
src_path = Path(__file__).resolve().parents[3]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add external/AstroCLIP to path for legacy checkpoint loading
astroclip_ext_path = Path(__file__).resolve().parents[4] / "external" / "AstroCLIP"
if str(astroclip_ext_path) not in sys.path:
    sys.path.insert(0, str(astroclip_ext_path))

from fmb.data.load_display_data import EuclidDESIDataset
from fmb.paths import load_paths
from fmb.models.astroclip.core.astroclip import AstroClipModel, CLIPLoss

# Hack for unpickling old checkpoints safely
from contextlib import contextmanager
@contextmanager
def unsafe_torch_load_context():
    original_load = torch.load
    def unsafe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = unsafe_load
    try:
        yield
    finally:
        torch.load = original_load

@dataclass
class TrainingConfig:
    # Output
    out_dir: str = str(load_paths().retrained_weights / "astroclip")
    output_filename: str = "astroclip_ft.pt"
    save_ckpt: bool = True
    
    # Data
    train_split: str = "train"
    val_split: str = "test"
    batch_size: int = 128
    num_workers: int = 4
    image_size: int = 144
    max_samples: Optional[int] = None
    cache_dir: str = str(load_paths().dataset)
    
    # Preprocessing
    slice_length: int = 7700
    spectrum_norm: str = "zscore" # none, zscore, minmax
    include_wavelength: bool = False
    
    # Model
    checkpoint: str = "hackathon2025/data/astroclip.ckpt"
    learnable_scale: bool = True
    finetune_spectrum: bool = True
    unfreeze_backbone_blocks: int = 0
    
    # Training
    learning_rate: float = 3e-6
    weight_decay: float = 5e-4
    max_epochs: int = 30
    warmup_steps: int = 0
    accumulate_steps: int = 2
    grad_clip: float = 1.0
    patience: int = 3
    min_delta: float = 1e-4
    
    # System
    device: str = "cuda"
    amp: bool = True
    log_interval: int = 20
    seed: int = 42
    log_via_wandb: bool = False
    wandb_project: str = "astroclip-retrain"


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Retrain AstroCLIP on Euclid+DESI")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    
    # Add CLI overrides
    parser.add_argument("--out-dir", default=TrainingConfig.out_dir)
    parser.add_argument("--checkpoint", default=TrainingConfig.checkpoint)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--device", default=TrainingConfig.device)
    
    args = parser.parse_args()
    
    config_dict = TrainingConfig().__dict__.copy()
    
    # Load YAML
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for k, v in yaml_config.items():
                    if k in config_dict:
                        config_dict[k] = v
    
    # CLI Overrides
    import sys
    def is_arg_passed(name): return any(arg.startswith(f"--{name}") for arg in sys.argv)
    
    if is_arg_passed("out-dir"): config_dict["out_dir"] = args.out_dir
    if is_arg_passed("checkpoint"): config_dict["checkpoint"] = args.checkpoint
    if is_arg_passed("batch-size"): config_dict["batch_size"] = args.batch_size
    if is_arg_passed("epochs"): config_dict["max_epochs"] = args.epochs
    if is_arg_passed("device"): config_dict["device"] = args.device
    
    config = TrainingConfig(**config_dict)
    
    # Type casing
    try:
        config.learning_rate = float(config.learning_rate)
        config.weight_decay = float(config.weight_decay)
        config.batch_size = int(config.batch_size)
        config.max_epochs = int(config.max_epochs)
        config.accumulate_steps = int(config.accumulate_steps)
        config.image_size = int(config.image_size)
        config.slice_length = int(config.slice_length)
    except ValueError:
        pass
        
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from fmb.data.datasets import AstroClipDataset, FMBDataConfig

def _cosine_scheduler(total_steps: int, warmup_steps: int) -> List[float]:
    schedule = []
    for step in range(total_steps):
        if step < warmup_steps:
            schedule.append(step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            schedule.append(0.5 * (1.0 + math.cos(math.pi * progress)))
    return schedule


def evaluate(image_encoder, spectrum_encoder, loader, criterion, device, logit_scale):
    image_encoder.eval()
    spectrum_encoder.eval()
    total_loss = 0.0
    total_cosine = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            spects = batch["spectrum"].to(device, non_blocking=True)
            
            img_feats = image_encoder(images)
            spec_feats = spectrum_encoder(spects)
            
            loss = criterion(img_feats, spec_feats, logit_scale)
            
            # Cosine similarity
            cosine = F.cosine_similarity(
                F.normalize(img_feats, dim=-1),
                F.normalize(spec_feats, dim=-1),
                dim=-1,
            ).mean()
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_cosine += cosine.item() * batch_size
            total_samples += batch_size
            
    if total_samples == 0:
        return float('inf'), float('nan')
        
    return total_loss / total_samples, total_cosine / total_samples


def main():
    config = parse_args()
    set_seed(config.seed)
    
    device = torch.device(config.device)
    os.makedirs(config.out_dir, exist_ok=True)
    
    print("="*60)
    print("AstroCLIP Retraining Startup")
    print("="*60)
    print(f"Config loaded. Output dir: {config.out_dir}")
    print(f"Device: {device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.max_epochs}")
    print(f"Accumulate Steps: {config.accumulate_steps}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Checkpoint: {config.checkpoint}")
    print("="*60)
    
    print(f"Loading checkpoint: {config.checkpoint}")
    
    # Load Model
    with unsafe_torch_load_context():
        # Handle relative path for checkpoint if needed
        ckpt_path = config.checkpoint
        if not os.path.exists(ckpt_path):
             # Try relative to repo root
             repo_root = Path(__file__).resolve().parents[4]
             if (repo_root / ckpt_path).exists():
                 ckpt_path = str(repo_root / ckpt_path)
             else:
                 pass # Trust user
                 
        print(f"Resolving checkpoint to: {ckpt_path}")
        model = AstroClipModel.load_from_checkpoint(ckpt_path, map_location=device)

    image_encoder = model.image_encoder.to(device)
    spectrum_encoder = model.spectrum_encoder.to(device)
    
    # Freeze/Unfreeze
    if config.finetune_spectrum:
        spectrum_encoder.train()
        for p in spectrum_encoder.parameters(): p.requires_grad = True
    else:
        spectrum_encoder.eval()
        for p in spectrum_encoder.parameters(): p.requires_grad = False
        
    # Unfreeze backbone blocks (if using DINO backbone)
    if config.unfreeze_backbone_blocks > 0:
         pass 

    # Trainable params
    params = [p for p in image_encoder.parameters() if p.requires_grad]
    if config.finetune_spectrum:
        params += [p for p in spectrum_encoder.parameters() if p.requires_grad]
        
    if config.learnable_scale:
        if isinstance(model.logit_scale, torch.Tensor):
            model.logit_scale.requires_grad = True
        else:
            val = model.logit_scale.item() if isinstance(model.logit_scale, torch.Tensor) else model.logit_scale
            if isinstance(val, (int, float)):
                 model.logit_scale = nn.Parameter(torch.tensor(val, device=device))
        params.append(model.logit_scale)
        
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Datasets
    print("Loading datasets...")
    # Adapt TrainingConfig to FMBDataConfig
    def mk_config(split_name):
        return FMBDataConfig(
            split=split_name,
            cache_dir=config.cache_dir,
            image_size=config.image_size,
            max_entries=config.max_samples,
            spectrum_length=config.slice_length,
            spectrum_norm=config.spectrum_norm,
            include_wavelength=config.include_wavelength
        )

    train_dataset = AstroClipDataset(mk_config(config.train_split))
    val_dataset = None
    if config.val_split:
        val_dataset = AstroClipDataset(mk_config(config.val_split))
        
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers) if val_dataset else None
    
    total_steps = (len(train_loader) * config.max_epochs) // config.accumulate_steps
    scheduler_factors = _cosine_scheduler(total_steps, config.warmup_steps)
    
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    criterion = CLIPLoss()
    
    best_val_loss = float('inf')
    patience_curr = 0
    global_step = 0
    
    print(f"Starting training for {config.max_epochs} epochs...")
    
    for epoch in range(1, config.max_epochs + 1):
        image_encoder.train()
        epoch_loss = 0.0
        epoch_cosine = 0.0
        sample_count = 0
        
        # Add tqdm for progress tracking within epoch
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(train_iter, start=1):
            images = batch["image"].to(device, non_blocking=True)
            spects = batch["spectrum"].to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=config.amp):
                img_feats = image_encoder(images)
                spec_feats = spectrum_encoder(spects)
                scale = model.logit_scale
                loss = criterion(img_feats, spec_feats, scale)
                
                cosine = F.cosine_similarity(
                    F.normalize(img_feats, dim=-1),
                    F.normalize(spec_feats, dim=-1),
                    dim=-1,
                ).mean()
                
            batch_size = images.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_cosine += cosine.item() * batch_size
            sample_count += batch_size
            
            loss = loss / config.accumulate_steps
            scaler.scale(loss).backward()
            
            if i % config.accumulate_steps == 0:
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn_utils.clip_grad_norm_(params, config.grad_clip)
                    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Scheduler
                if global_step < len(scheduler_factors):
                    lr_scale = scheduler_factors[global_step]
                    for pg in optimizer.param_groups:
                        pg["lr"] = config.learning_rate * lr_scale
                        
                global_step += 1
                
            if i % config.log_interval == 0:
                 scale_val = model.logit_scale.item() if isinstance(model.logit_scale, torch.Tensor) else model.logit_scale
                 loss_val = loss.item() * config.accumulate_steps
                 tqdm.write(
                    f"[Epoch {epoch}] Batch {i}/{len(train_loader)} "
                    f"loss={loss_val:.4f} cosine={cosine.item():.4f} "
                    f"scale={scale_val:.2f}"
                 )
                     
        # Validation
        train_loss_avg = epoch_loss / max(1, sample_count)
        train_cosine_avg = epoch_cosine / max(1, sample_count)
        
        val_loss = float('inf')
        val_cosine = float('nan')
        
        if val_loader:
            val_loss, val_cosine = evaluate(image_encoder, spectrum_encoder, val_loader, criterion, device, model.logit_scale)
            print(f"Epoch {epoch}: Train Loss {train_loss_avg:.4f}, Train Cosine {train_cosine_avg:.4f}, Val Loss {val_loss:.4f}, Val Cosine {val_cosine:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_curr = 0
                # Save best
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "epoch": epoch
                    }, 
                    os.path.join(config.out_dir, config.output_filename)
                )
                print(f"[Epoch {epoch}] New best model saved to {os.path.join(config.out_dir, config.output_filename)}")
            else:
                patience_curr += 1
                if patience_curr >= config.patience:
                    print("Patience reached. Stopping.")
                    break
        else:
             # Just save every epoch if no validation
             torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "epoch": epoch
                },
                os.path.join(config.out_dir, config.output_filename)
             )
             
             
    # Detailed Summary Logging
    if config.log_via_wandb:
        wandb.finish()
        
    print("Training finished.")
    
    summary_path = os.path.join(config.out_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("AstroCLIP Training Summary\n")
        f.write("==========================\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Config: {config}\n")
        # Could add more details here
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
