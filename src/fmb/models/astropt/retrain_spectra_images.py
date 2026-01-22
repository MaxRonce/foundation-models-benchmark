"""Training script for multimodal AstroPT using Euclid images + DESI spectra.

This script adapts the multimodal training approach to work with the 
Euclid+DESI HuggingFace dataset, training on both RGB images and spectra
simultaneously.

USAGE EXAMPLES:
===============

Single GPU training:
-------------------
python models/astropt/retrain_spectra_images.py --batch-size 8 --compile

Multi-GPU training (2 GPUs):
---------------------------
torchrun --standalone --nproc_per_node=2 models/astropt/retrain_spectra_images.py \\
    --batch-size 16 \\
    --grad-accum 4 \\
    --compile
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# CHANGED: Import from fmb.data instead of scratch
from fmb.data.load_display_data_hsc import EuclidDESIDataset
from fmb.paths import load_paths

# CHANGED: Ensure astropt imports work (try/except block from existing file)
try:
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
except ImportError:
    import sys
    from pathlib import Path
    # Attempt to add external/astroPT to path if not installed
    # Assumption: external/astroPT is 4 levels up from this file if in src/fmb/models/astropt
    # But this file is now in src/fmb/models/astropt/retrain_spectra_images.py
    # So relative path: ../../../../external/astroPT
    external_path = Path(__file__).resolve().parents[4] / "external" / "astroPT"
    if external_path.exists():
        sys.path.append(str(external_path))
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry

# CHANGED: Relative import to local euclid_desi_dataset package
# Ensure src/fmb/models/astropt/euclid_desi_dataset exists and has multimodal_dataloader
from .euclid_desi_dataset.multimodal_dataloader import multimodal_collate_fn, prepare_multimodal_batch


class EuclidDESIMultimodalDataset(torch.utils.data.Dataset):
    """Wrapper around EuclidDESIDataset to emit tensors for multimodal training."""

    def __init__(
        self,
        split: str = "train",
        image_size: int = 224,
        spectrum_length: int = 7781,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        if cache_dir is None:
             cache_dir = str(load_paths().cache)
             
        normalized_split = split.replace("+", ",") if isinstance(split, str) else split
        self.base = EuclidDESIDataset(split=normalized_split, cache_dir=cache_dir, verbose=verbose)
        self.image_size = image_size
        self.spectrum_length = spectrum_length

    def __len__(self):
        return len(self.base)

    def _prepare_image(self, image: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if image is None:
            return None
        img = torch.as_tensor(image, dtype=torch.float32)
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        if img.ndim == 3 and img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1).contiguous()
        img = img.clamp(0.0, 1.0)
        if self.image_size and (img.shape[-1] != self.image_size or img.shape[-2] != self.image_size):
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return img

    def _prepare_spectrum(self, spectrum_dict: Optional[dict]) -> Optional[torch.Tensor]:
        if spectrum_dict is None or spectrum_dict.get("flux") is None:
            return None
        flux = torch.as_tensor(spectrum_dict["flux"], dtype=torch.float32)
        flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

        if flux.numel() < self.spectrum_length:
            flux = F.pad(flux, (0, self.spectrum_length - flux.numel()))
        elif flux.numel() > self.spectrum_length:
            flux = flux[:self.spectrum_length]

        if flux.std() > 0:
            flux = flux / (flux.std() + 1e-8)
        return flux

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        image = self._prepare_image(sample.get("rgb_image"))
        spectrum = self._prepare_spectrum(sample.get("spectrum"))

        targetid = sample.get("targetid")
        try:
            targetid_val = int(targetid) if targetid is not None else -1
        except Exception:
            targetid_val = -1
        redshift = sample.get("redshift")
        redshift_val = float(redshift) if redshift is not None else 0.0

        return {
            "object_id": sample.get("object_id") or targetid_val,
            "targetid": targetid_val,
            "redshift": redshift_val,
            "image": image,
            "spectrum": spectrum,
        }


@dataclass
class TrainingConfig:
    """Configuration for multimodal training."""
    
    # Output and logging
    out_dir: str = str(load_paths().retrained_weights / "astropt")
    eval_interval: int = 100  # More frequent evaluation for production
    eval_iters: int = 50  # Faster evaluation
    log_interval: int = 20  # More frequent logging
    checkpoint_interval: int = 5000
    always_save_checkpoint: bool = False
    
    # Data
    train_split: str = "train"
    val_split: str = "test"
    batch_size: int = 8  # Original batch size (patch_size was changed to 10, not batch_size)
    gradient_accumulation_steps: int = 4
    num_workers: int = 0  # Disable multiprocessing to avoid disk space issues
    image_size: int = 224
    spectrum_length: int = 7781
    cache_dir: str = str(load_paths().dataset)
    
    # Model architecture
    block_size: int = 1024  # Increased to handle longer sequences with small patch sizes
    image_patch_size: int = 16
    spectrum_patch_size: int = 10  # Decreased from 256, following DESI success with smaller patches
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_chan: int = 3  # RGB channels
    dropout: float = 0.0
    bias: bool = False
    
    # Training
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5
    max_iters: int = 3000  # Longer production run for proper convergence
    
    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    log_via_wandb: bool = False
    wandb_project: str = "astropt-multimodal"
    wandb_run_name: str = None


def parse_args() -> TrainingConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AstroPT on Euclid images + DESI spectra")
    
    parser.add_argument("--out-dir", default=TrainingConfig.out_dir)
    parser.add_argument("--train-split", default=TrainingConfig.train_split)
    parser.add_argument("--val-split", default=TrainingConfig.val_split)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--grad-accum", type=int, default=TrainingConfig.gradient_accumulation_steps)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--image-size", type=int, default=TrainingConfig.image_size)
    parser.add_argument("--max-iters", type=int, default=TrainingConfig.max_iters)
    parser.add_argument("--eval-interval", type=int, default=TrainingConfig.eval_interval)
    parser.add_argument("--eval-iters", type=int, default=TrainingConfig.eval_iters)
    parser.add_argument("--log-interval", type=int, default=TrainingConfig.log_interval)
    parser.add_argument("--device", default=TrainingConfig.device, help="Device to use (cpu, cuda, etc.)")
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--log-wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", default=TrainingConfig.wandb_project)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume-best", action="store_true", default=False, help="Resume from best checkpoint in out_dir")
    parser.add_argument("--cache-dir", default=TrainingConfig.cache_dir, help="Path to HuggingFace cache / dataset root")
    
    args = parser.parse_args()
    
    config = TrainingConfig()
    config.out_dir = args.out_dir
    config.train_split = args.train_split
    config.val_split = args.val_split
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.grad_accum
    config.num_workers = args.num_workers
    config.image_size = args.image_size
    config.max_iters = args.max_iters
    config.eval_interval = args.eval_interval
    config.eval_iters = args.eval_iters
    config.log_interval = args.log_interval
    config.device = args.device
    config.compile = args.compile
    config.log_via_wandb = args.log_wandb and _WANDB_AVAILABLE
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_run_name
    config.cache_dir = args.cache_dir
    
    # Handle resume arguments
    resume_path = None
    if args.resume_best:
        resume_path = os.path.join(config.out_dir, "ckpt_best.pt")
    elif args.resume:
        resume_path = args.resume
    
    return config, resume_path


def setup_ddp(config: TrainingConfig):
    """Setup distributed data parallel."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        # Ensure gradient accumulation is compatible with world size
        assert config.gradient_accumulation_steps % ddp_world_size == 0
        config.gradient_accumulation_steps //= ddp_world_size
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = torch.device(config.device)
        master_process = True
        seed_offset = 0
    
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, seed_offset


def create_modality_registry(config: TrainingConfig) -> ModalityRegistry:
    """Create modality registry for images and spectra."""
    modalities = [
        ModalityConfig(
            name="images",
            input_size=config.image_patch_size * config.image_patch_size * config.n_chan,
            patch_size=config.image_patch_size,
            loss_weight=779/196,  # Mathematically balanced: 779/196 ≈ 3.97
            embed_pos=True,
            pos_input_size=1,
        ),
        ModalityConfig(
            name="spectra",
            input_size=config.spectrum_patch_size,
            patch_size=config.spectrum_patch_size,
            pos_input_size=1,
            loss_weight=196/779,  # Mathematically balanced: 196/779 ≈ 0.25
            embed_pos=True,
        ),
    ]
    return ModalityRegistry(modalities)


def create_datasets_and_loaders(config: TrainingConfig, ddp: bool, ddp_rank: int, ddp_world_size: int, device):
    """Create datasets and data loaders."""
    
    # Create datasets
    train_dataset = EuclidDESIMultimodalDataset(
        split=config.train_split,
        image_size=config.image_size,
        spectrum_length=config.spectrum_length,
        cache_dir=config.cache_dir,
        verbose=False,
    )
    
    val_dataset = EuclidDESIMultimodalDataset(
        split=config.val_split,
        image_size=config.image_size,
        spectrum_length=config.spectrum_length,
        cache_dir=config.cache_dir,
        verbose=False,
    )
    
    # Create samplers for DDP
    train_sampler = None
    val_sampler = None
    if ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=False,
            drop_last=False
        )
    
    # Use pin_memory only for CUDA
    use_pin_memory = device.type == 'cuda'
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(not ddp),
        num_workers=config.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=use_pin_memory,
        drop_last=True,
        sampler=train_sampler,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=use_pin_memory,
        drop_last=False,
        sampler=val_sampler,
    )
    
    return train_dataset, val_dataset, train_loader, val_loader


def get_lr(it: int, config: TrainingConfig) -> float:
    """Learning rate schedule with linear warmup and cosine decay."""
    # Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # If beyond decay iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, config, modality_registry, device, ctx):
    """Estimate loss on train and validation sets."""
    model.eval()
    losses = {}
    
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        split_losses = {modality: [] for modality in ["images", "spectra"]}
        
        for i, batch in enumerate(loader):
            if i >= config.eval_iters:
                break
            
            # Prepare batch
            inputs = prepare_multimodal_batch(
                batch, config.image_patch_size, config.spectrum_patch_size, 
                device, modality_registry
            )
            
            if not inputs:  # Skip if no valid inputs
                print(f"Warning: Empty inputs for {split} batch {i}")
                continue
            
            # Debug: print inputs info
            if i == 0:  # Only for first batch
                print(f"Debug {split} inputs keys: {inputs.keys()}")
                for key, val in inputs.items():
                    if isinstance(val, torch.Tensor):
                        print(f"  {key}: {val.shape}, dtype: {val.dtype}")
            
            with ctx:
                # Proper target preparation for autoregressive training
                # Model outputs seq_len-1 for autoregressive modalities, full seq_len for others
                targets = {}
                for modality in inputs.keys():
                    if modality.endswith('_positions'):
                        continue  # Skip position tensors
                    
                    if modality == 'images':
                        # For autoregressive modality: target = input[1:] (remove first token)
                        targets[modality] = inputs[modality][:, 1:, :]
                    else:
                        # For non-autoregressive modality: target = input (full sequence)
                        targets[modality] = inputs[modality]
                
                logits, loss = model(inputs, targets=targets)
                
            # Debug: check loss
            if loss is None:
                print(f"Warning: model returned None loss for {split} batch {i}")
                print(f"  Model inputs: {inputs.keys()}")
                continue
                
            # Collect losses per modality - use total loss for all modalities
            # since the model returns aggregated loss
            split_losses["images"].append(loss.item())
            split_losses["spectra"].append(loss.item())
        
        # Average losses
        avg_losses = {}
        for modality, losses_list in split_losses.items():
            if losses_list:
                avg_losses[modality] = sum(losses_list) / len(losses_list)
            else:
                avg_losses[modality] = float('inf')
        
        losses[split] = avg_losses
    
    model.train()
    return losses


def save_checkpoint(model, optimizer, iter_num, best_val_loss, config, ddp, filename="ckpt.pt"):
    """Save model checkpoint."""
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Get the raw model (unwrap DDP if needed)
    raw_model = model.module if ddp else model
    
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "bias": config.bias,
            "dropout": config.dropout,
        },
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config.__dict__,
    }
    
    checkpoint_path = os.path.join(config.out_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


@torch.no_grad()
def visualize_reconstructions(model, val_loader, config, modality_registry, device, ctx, iter_num):
    """Generate and save reconstruction visualizations."""
    model.eval()
    
    # Get one batch for visualization
    batch = next(iter(val_loader))
    inputs = prepare_multimodal_batch(
        batch, config.image_patch_size, config.spectrum_patch_size,
        device, modality_registry
    )
    
    if not inputs:
        print("Warning: No valid inputs for visualization")
        return
    
    # Forward pass to get reconstructions
    with ctx:
        logits, _ = model(inputs)
    
    # Convert to float32 for matplotlib compatibility
    if logits:
        for key in logits:
            if isinstance(logits[key], torch.Tensor):
                logits[key] = logits[key].float()
    
    # Create visualization
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'Reconstructions at Iteration {iter_num}', fontsize=16)
    
    # Process each modality
    for i in range(min(5, len(batch['all_object_ids']))):  # Show up to 5 examples
        
        # Row 0: Original images
        if 'images' in batch and len(batch['images']) > i:
            orig_img = batch['images'][i].cpu().numpy()
            if orig_img.shape[0] == 3:  # RGB
                orig_img = np.transpose(orig_img, (1, 2, 0))
                # Normalize to [0,1] for display
                orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')
        else:
            axes[0, i].axis('off')
        
        # Row 1: Reconstructed images
        if 'images' in inputs and 'images' in logits and len(logits['images']) > i:
            recon_patches = logits['images'][i].cpu().numpy()  # Shape: [195, 768]
            
            # Reshape patches back to image format
            # Each patch is 16x16x3 = 768 values
            patch_size = config.image_patch_size
            n_channels = config.n_chan
            
            # Calculate grid dimensions (14x14 patches for 224x224 image)
            patches_per_side = config.image_size // patch_size
            
            # Only use the first 196 patches to reconstruct 14x14 grid
            if recon_patches.shape[0] >= patches_per_side * patches_per_side - 1:
                # We have 195 patches but need 196 for 14x14 grid
                # Pad with zeros for the missing patch
                if recon_patches.shape[0] == 195:
                    # Add one zero patch at the end
                    zero_patch = np.zeros((1, recon_patches.shape[1]))
                    recon_patches = np.vstack([recon_patches, zero_patch])
                
                # Take first 196 patches (14x14)
                patches_to_use = recon_patches[:patches_per_side*patches_per_side]
                
                # Reshape to [14, 14, 768]
                patch_grid = patches_to_use.reshape(patches_per_side, patches_per_side, -1)
                
                # Reshape each patch from 768 -> 16x16x3
                recon_img = np.zeros((patches_per_side*patch_size, patches_per_side*patch_size, n_channels))
                
                for py in range(patches_per_side):
                    for px in range(patches_per_side):
                        patch_data = patch_grid[py, px].reshape(patch_size, patch_size, n_channels)
                        y_start, y_end = py*patch_size, (py+1)*patch_size
                        x_start, x_end = px*patch_size, (px+1)*patch_size
                        recon_img[y_start:y_end, x_start:x_end] = patch_data
                
                # Normalize for display
                recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
                recon_img = np.clip(recon_img, 0, 1)
                
                axes[1, i].imshow(recon_img)
                axes[1, i].set_title(f'Reconstructed Image {i+1}')
                axes[1, i].axis('off')
            else:
                # Not enough patches - show black image
                axes[1, i].imshow(np.zeros((224, 224, 3)))
                axes[1, i].set_title(f'Insufficient patches ({recon_patches.shape[0]})')
                axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
        
        # Row 2: Original and reconstructed spectra on same plot
        if 'spectra' in batch and len(batch['spectra']) > i:
            orig_spec = batch['spectra'][i].cpu().numpy()
            axes[2, i].plot(orig_spec, label='Original', alpha=0.8, linewidth=1, color='blue')
            
            # Add reconstructed spectrum if available
            if 'spectra' in inputs and 'spectra' in logits and len(logits['spectra']) > i:
                recon_patches = logits['spectra'][i].cpu().numpy()  # Shape: [31, 256]
                
                # Flatten patches back to spectrum
                recon_spec = recon_patches.flatten()  # Shape: [31*256] = [7936]
                
                # Truncate to original spectrum length
                if len(orig_spec) <= len(recon_spec):
                    recon_spec = recon_spec[:len(orig_spec)]
                else:
                    # Pad if needed
                    recon_spec = np.pad(recon_spec, (0, len(orig_spec) - len(recon_spec)), 'constant')
                
                axes[2, i].plot(recon_spec, label='Reconstructed', alpha=0.8, linewidth=1, color='orange')
            
            axes[2, i].set_ylim([orig_spec.min()-1, orig_spec.max()+1])
            axes[2, i].set_title(f'Spectrum {i+1}')
            axes[2, i].legend(fontsize=8)
            axes[2, i].grid(True, alpha=0.3)
            axes[2, i].set_xlabel('Wavelength Index')
            axes[2, i].set_ylabel('Flux')
        else:
            axes[2, i].axis('off')
    
    # Hide unused subplots
    for i in range(5):
        if i >= len(batch['all_object_ids']):
            for row in range(3):
                axes[row, i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    vis_dir = os.path.join(config.out_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, f'reconstructions_iter_{iter_num:06d}.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reconstruction visualization: {vis_path}")
    model.train()


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load model and optimizer state from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model"])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Get training state
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    
    print(f"Resumed from iteration {iter_num}, best val loss: {best_val_loss:.6f}")
    return iter_num, best_val_loss


def print_training_summary(iter_num, max_iters, best_val_loss, current_loss):
    """Print a training progress summary."""
    progress = (iter_num / max_iters) * 100
    print(f"\n{'='*60}")
    print(f"TRAINING PROGRESS SUMMARY")
    print(f"{'='*60}")
    print(f"Progress: {progress:.1f}% ({iter_num}/{max_iters} iterations)")
    print(f"Current loss: {current_loss:.6f}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Loss improvement: {((0.19 - best_val_loss) / 0.19 * 100):.1f}% from start")
    print(f"{'='*60}\n")


def main():
    """Main training loop."""
    config, resume_path = parse_args()
    
    # Setup DDP
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, seed_offset = setup_ddp(config)
    
    # Seed for reproducibility
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create output directory
    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print(f"Output directory: {config.out_dir}")
        print(f"Training on device: {device}")
        print(f"DDP: {ddp}, World size: {ddp_world_size}")
    
    # Create modality registry
    modality_registry = create_modality_registry(config)
    
    # Create datasets and loaders
    train_dataset, val_dataset, train_loader, val_loader = create_datasets_and_loaders(
        config, ddp, ddp_rank, ddp_world_size, device
    )
    
    if master_process:
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        print(f"Modalities: {modality_registry.names()}")
        
        # Print optimization settings
        print(f"\nOptimization settings:")
        for modality_name in modality_registry.names():
            modality_config = modality_registry.get_config(modality_name)
            print(f"  {modality_config.name}: loss_weight={modality_config.loss_weight}, patch_size={modality_config.patch_size}")
        print(f"  batch_size: {config.batch_size}")
        print(f"  spectrum_patch_size: {config.spectrum_patch_size}")
    
    # Calculate tokens per iteration
    tokens_per_iter = (
        config.gradient_accumulation_steps * ddp_world_size * 
        config.batch_size * config.block_size * len(modality_registry.names())
    )
    if master_process:
        print(f"Tokens per iteration: {tokens_per_iter:,}")
    
    # Create model
    gpt_config = GPTConfig(
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        bias=config.bias,
        dropout=config.dropout,
        attn_type="causal",
    )
    
    model = GPT(gpt_config, modality_registry)
    model.to(device)
    
    if master_process:
        print(f"Model parameters: {model.get_num_params() / 1e6:.1f}M")
    
    # Compile model if requested
    if config.compile:
        if master_process:
            print("Compiling model...")
        model = torch.compile(model)
    
    # Wrap with DDP if needed
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Create optimizer
    base_model = model.module if ddp else model
    optimizer = base_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device.type,
    )
    
    # Setup mixed precision
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    target_dtype = dtype_map.get(config.dtype, torch.float32)
    use_amp = target_dtype in {torch.bfloat16, torch.float16}
    scaler = torch.amp.GradScaler(device.type, enabled=(target_dtype == torch.float16))
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=target_dtype)
        if use_amp else nullcontext()
    )
    
    # Initialize wandb
    if config.log_via_wandb and master_process:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__,
        )
    
    # Initialize loss logging
    loss_log_file = None
    if master_process:
        # Create timestamped log file to avoid overwrites
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        loss_log_file = os.path.join(config.out_dir, f'training_log_{timestamp}.txt')
        
        # Always create new log file with header
        with open(loss_log_file, 'w') as f:
            f.write("iter,train_loss,val_loss,lr,time_ms\n")
        print(f"Training log will be saved to: {loss_log_file}")
    
    # Load checkpoint if resuming
    start_iter = 0
    if resume_path and master_process:
        try:
            start_iter, best_val_loss = load_checkpoint(resume_path, model, optimizer, device)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch...")
            start_iter = 0
            best_val_loss = float('inf')
    else:
        best_val_loss = float('inf')
    
    # Training loop
    if master_process:
        if resume_path and start_iter > 0:
            print(f"Resuming training from iteration {start_iter}...")
        else:
            print("Starting training from scratch...")
    
    model.train()
    iter_num = start_iter
    t0 = time.time()
    
    # Convert loaders to iterators
    train_iter = iter(train_loader)
    
    while iter_num < config.max_iters:
        # Set learning rate
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation and checkpointing
        if iter_num % config.eval_interval == 0 and master_process:
            metrics = estimate_loss(model, train_loader, val_loader, config, modality_registry, device, ctx)
            train_loss = metrics["train"]["spectra"]  # Using spectra loss as representative
            val_loss = metrics["val"]["spectra"]
            
            print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, lr {lr:.2e}")
            
            # Print summary every now and then
            if iter_num % (config.eval_interval * 5) == 0:
                 print_training_summary(iter_num, config.max_iters, best_val_loss, train_loss)
            
            # Log to wandb
            if config.log_via_wandb:
                wandb.log({
                    "iter": iter_num,
                    "train/loss_spectra": metrics["train"]["spectra"],
                    "train/loss_images": metrics["train"]["images"],
                    "val/loss_spectra": metrics["val"]["spectra"],
                    "val/loss_images": metrics["val"]["images"],
                    "lr": lr,
                })
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iter_num, best_val_loss, config, ddp, "ckpt_best.pt")
                print(f"New best validation loss: {best_val_loss:.4f}")
            
            # Regular checkpoint
            if config.always_save_checkpoint or (iter_num > 0 and iter_num % config.checkpoint_interval == 0):
                save_checkpoint(model, optimizer, iter_num, best_val_loss, config, ddp, f"ckpt_{iter_num}.pt")
            
            # Visualize reconstructions
            if iter_num % (config.eval_interval * 2) == 0:
                visualize_reconstructions(model, val_loader, config, modality_registry, device, ctx, iter_num)
        
        # Training step
        # Handle data loading with potential end of epoch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Prepare inputs
        inputs = prepare_multimodal_batch(
            batch, config.image_patch_size, config.spectrum_patch_size,
            device, modality_registry
        )
        
        if not inputs:
            print(f"Warning: Empty batch at iteration {iter_num}")
            continue
            
        with ctx:
             # Proper target preparation (same as in estimate_loss)
            targets = {}
            for modality in inputs.keys():
                if modality.endswith('_positions'):
                    continue
                if modality == 'images':
                    targets[modality] = inputs[modality][:, 1:, :]
                else:
                    targets[modality] = inputs[modality]
                    
            logits, loss = model(inputs, targets=targets)
            loss = loss / config.gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Step optimizer
        if (iter_num + 1) % config.gradient_accumulation_steps == 0:
            if config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iter_num += 1
            
            # Log training step
            if iter_num % config.log_interval == 0 and master_process:
                t1 = time.time()
                dt = (t1 - t0) * 1000
                loss_val = loss.item() * config.gradient_accumulation_steps
                print(f"iter {iter_num}: loss {loss_val:.4f}, time {dt:.2f}ms")
                
                # Write to log file
                if loss_log_file:
                    with open(loss_log_file, 'a') as f:
                        f.write(f"{iter_num},{loss_val:.6f},,{lr:.2e},{dt:.2f}\n")
                
                t0 = t1
                
    if master_process:
        print("Training finished!")
        save_checkpoint(model, optimizer, iter_num, best_val_loss, config, ddp, "ckpt_final.pt")
        if config.log_via_wandb:
            wandb.finish()
    
    destroy_process_group()


if __name__ == "__main__":
    main()
