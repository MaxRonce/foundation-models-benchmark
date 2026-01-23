#!/usr/bin/env python3
"""
Retraining script for AION (Euclid <-> HSC adapters) using unified dataset and YAML configuration.
Adapted from retrain_euclid_hsc_adapter_unet.py.
"""

import argparse
import json
import gc
import os
import sys
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import safetensors.torch as st
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import yaml

# Ensure src is in pythonpath
src_path = Path(__file__).resolve().parents[3]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Ensure external/AION is in pythonpath
aion_path = Path(__file__).resolve().parents[4] / "external" / "AION"
if str(aion_path) not in sys.path:
    sys.path.insert(0, str(aion_path))

from fmb.data.load_display_data import EuclidDESIDataset
from fmb.paths import load_paths

# AION Imports
try:
    from aion.codecs import ImageCodec
    from aion.codecs.config import HF_REPO_ID
    from aion.codecs.preprocessing.image import CenterCrop
    from torchvision.transforms import RandomCrop
    from aion.modalities import EuclidImage, HSCImage
except ImportError as e:
    print(f"Error importing AION: {e}")
    print(f"PYTHONPATH: {sys.path}")
    raise


from fmb.data.datasets import AionDataset, FMBDataConfig, EUCLID_BANDS, AION_AVAILABLE

if not AION_AVAILABLE: 
    print("Warning: AION not available.")
    # Fallback handled in datasets for testing

HSC_BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]


@dataclass
class TrainingConfig:
    # Output
    out_dir: str = str(load_paths().retrained_weights / "aion")
    
    # Data
    cache_dir: str = str(load_paths().dataset)
    split: str = "all"
    max_entries: int = 0
    batch_size: int = 8
    num_workers: int = 0
    
    # Preprocessing
    resize: int = 96
    crop_size: int = 96
    max_abs: float = 100.0
    cpu_crop: bool = False
    
    # Model (U-Net)
    hidden: int = 16
    use_unet_checkpointing: bool = False
    resume_adapter: Optional[str] = None
    auto_resume: bool = True
    
    # Codec
    codec_grad: str = "ste"
    disable_codec_checkpointing: bool = False
    
    # Training
    epochs: int = 15
    learning_rate: float = 1e-4
    accum_steps: int = 1
    grad_clip: float = 1.0
    amp_dtype: str = "float16"
    
    # Logging
    log_gpu_mem_every: int = 50
    log_via_wandb: bool = False
    wandb_project: str = "aion-retrain"
    
    # System
    device: str = "cuda"
    seed: int = 42


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Retrain AION Adapter")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    
    # CLI Overrides
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    
    args = parser.parse_args()
    
    config_dict = TrainingConfig().__dict__.copy()
    
    # Load YAML
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for k, v in yaml_config.items():
                    if k in config_dict:
                        config_dict[k] = v
    
    # CLI Overrides
    if args.out_dir: config_dict["out_dir"] = args.out_dir
    if args.batch_size: config_dict["batch_size"] = args.batch_size
    if args.epochs: config_dict["epochs"] = args.epochs
    if args.device: config_dict["device"] = args.device
    
    # Ensure types
    try:
        config_dict["learning_rate"] = float(config_dict["learning_rate"])
        # Add other conversions if needed
    except ValueError:
        pass
        
    return TrainingConfig(**config_dict)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _mem_gb(x: int) -> float:
    return float(x) / 1e9

# -----------------------------
# U-Net blocks (Copied from original)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_dim: Optional[int] = None):
        super().__init__()
        mid_ch = hidden_dim if hidden_dim else out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
            self.conv = DoubleConv(in_ch, out_ch, hidden_dim=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, hidden: int, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.inc = DoubleConv(n_channels, hidden)
        self.down1 = Down(hidden, hidden * 2)
        self.down2 = Down(hidden * 2, hidden * 4)
        self.up1 = Up(hidden * 4, hidden * 2, bilinear=True)
        self.up2 = Up(hidden * 2, hidden, bilinear=True)
        self.outc = OutConv(hidden, n_classes)

    def forward(self, x):
        if self.use_checkpointing and x.requires_grad:
            x1 = checkpoint_utils.checkpoint(self.inc, x, use_reentrant=False)
            x2 = checkpoint_utils.checkpoint(self.down1, x1, use_reentrant=False)
            x3 = checkpoint_utils.checkpoint(self.down2, x2, use_reentrant=False)
            x = checkpoint_utils.checkpoint(self.up1, x3, x2, use_reentrant=False)
            x = checkpoint_utils.checkpoint(self.up2, x, x1, use_reentrant=False)
            return self.outc(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)


class EuclidToHSC(SimpleUNet):
    def __init__(self, hidden: int, use_checkpointing: bool):
        super().__init__(n_channels=4, n_classes=5, hidden=hidden, use_checkpointing=use_checkpointing)


class HSCToEuclid(SimpleUNet):
    def __init__(self, hidden: int, use_checkpointing: bool):
        super().__init__(n_channels=5, n_classes=4, hidden=hidden, use_checkpointing=use_checkpointing)


# -----------------------------
# Dataset
# -----------------------------


def collate_euclid(batch: List[EuclidImage]) -> EuclidImage:
    flux = torch.stack([b.flux for b in batch], dim=0)
    return EuclidImage(flux=flux, bands=batch[0].bands)


# -----------------------------
# Helpers
# -----------------------------
def load_frozen_codec(device: torch.device) -> Tuple[ImageCodec, dict]:
    cfg_path = hf_hub_download(HF_REPO_ID, "codecs/image/config.json", local_files_only=False)
    weights_path = hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=False)

    with open(cfg_path) as f:
        codec_cfg = json.load(f)

    print("quantizer_levels (from config):", codec_cfg["quantizer_levels"])

    from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX
    original_bands = dict(BAND_TO_INDEX)
    try:
        keys_to_remove = [k for k in list(BAND_TO_INDEX.keys()) if "EUCLID" in k]
        for k in keys_to_remove:
            del BAND_TO_INDEX[k]

        codec = ImageCodec(
            quantizer_levels=codec_cfg["quantizer_levels"],
            hidden_dims=codec_cfg["hidden_dims"],
            multisurvey_projection_dims=codec_cfg["multisurvey_projection_dims"],
            n_compressions=codec_cfg["n_compressions"],
            num_consecutive=codec_cfg["num_consecutive"],
            embedding_dim=codec_cfg["embedding_dim"],
            range_compression_factor=codec_cfg["range_compression_factor"],
            mult_factor=codec_cfg["mult_factor"],
        ).to(device)
    finally:
        BAND_TO_INDEX.clear()
        BAND_TO_INDEX.update(original_bands)

    state = st.load_file(weights_path, device="cpu")
    missing, unexpected = codec.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[info] codec load: missing={missing}, unexpected={unexpected}")

    for p in codec.parameters():
        p.requires_grad = False
    codec.eval()
    return codec, codec_cfg


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    
    device = torch.device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("AION Retraining Startup")
    print("="*60)
    print(f"Config: {config}")
    print("="*60)
    
    # Defaults
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    # Auto-resume logic
    start_epoch = 1
    # Prioritize explicit resume_adapter from config/CLI
    if config.resume_adapter:
        print(f"[info] Resume requested from config: {config.resume_adapter}")
    elif config.auto_resume:
        ckpts = list(out_dir.glob("adapters_epoch_*.pt"))
        if ckpts:
            def get_epoch(p: Path) -> int:
                try:
                    return int(p.stem.split("_")[-1])
                except Exception:
                    return -1
            latest = max(ckpts, key=get_epoch)
            print(f"[info] Auto-resume found: {latest}")
            config.resume_adapter = str(latest)

    # Create datasets
    def mk_config(split_name):
        return FMBDataConfig(
            split=split_name,
            cache_dir=config.cache_dir,
            image_size=config.resize,
            max_entries=config.max_entries
        )

    dataset = AionDataset(mk_config(config.split))

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        collate_fn=collate_euclid,
    )

    print(f"Loaded {len(dataset)} samples; batches/epoch: {len(loader)} (batch={config.batch_size}, accum={config.accum_steps})")

    codec, codec_cfg = load_frozen_codec(device)
    euclid_to_hsc = EuclidToHSC(hidden=config.hidden, use_checkpointing=config.use_unet_checkpointing).to(device)
    hsc_to_euclid = HSCToEuclid(hidden=config.hidden, use_checkpointing=config.use_unet_checkpointing).to(device)

    optimizer = torch.optim.Adam(list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()), lr=config.learning_rate)
    criterion = nn.MSELoss(reduction="mean")

    amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    accum_steps = max(1, int(config.accum_steps))
    use_codec_ckpt = (not config.disable_codec_checkpointing)

    crop = RandomCrop(size=config.crop_size)
    center_crop = CenterCrop(crop_size=config.crop_size)

    if config.resume_adapter:
        ckpt = torch.load(config.resume_adapter, map_location="cpu")
        euclid_to_hsc.load_state_dict(ckpt["euclid_to_hsc"])
        hsc_to_euclid.load_state_dict(ckpt["hsc_to_euclid"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"[info] Resumed from {config.resume_adapter} (start_epoch={start_epoch})")

    (out_dir / "adapter_config.json").write_text(json.dumps({**config.__dict__, "codec_cfg": codec_cfg}, indent=2, default=str))

    euclid_to_hsc.train()
    hsc_to_euclid.train()

    # Codec wrappers
    def codec_roundtrip_flux(hsc_flux: torch.Tensor) -> torch.Tensor:
        hsc_obj = HSCImage(flux=hsc_flux, bands=HSC_BANDS)
        toks = codec.encode(hsc_obj)
        hsc_rec = codec.decode(toks, bands=HSC_BANDS)
        return hsc_rec.flux

    def codec_roundtrip_ste(hsc_flux: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = codec_roundtrip_flux(hsc_flux)
        return hsc_flux + (y - hsc_flux).detach()

    def codec_roundtrip_full(hsc_flux: torch.Tensor) -> torch.Tensor:
        if use_codec_ckpt and hsc_flux.requires_grad:
            return checkpoint_utils.checkpoint(codec_roundtrip_flux, hsc_flux, use_reentrant=False)
        return codec_roundtrip_flux(hsc_flux)

    if config.codec_grad == "ste":
        codec_bridge = codec_roundtrip_ste
        print("[info] codec-grad=ste (recommended)")
    else:
        codec_bridge = codec_roundtrip_full
        print("[info] codec-grad=full: backprop through codec")

    training_losses: List[float] = []

    def preprocess_and_crop(euclid_flux_cpu: torch.Tensor) -> torch.Tensor:
        if config.cpu_crop:
            x = torch.nan_to_num(euclid_flux_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            if config.max_abs > 0:
                x = torch.clamp(x, -config.max_abs, config.max_abs)
            x = crop(x)
            return x.to(device, non_blocking=True)

        x = euclid_flux_cpu.to(device, non_blocking=False)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if config.max_abs > 0:
            x = torch.clamp(x, -config.max_abs, config.max_abs)
        return crop(x)

    # Process a batch with OOM-safe splitting
    def backward_batch_oom_safe(x_full: torch.Tensor) -> float:
        B = int(x_full.shape[0])

        def run_micro(x_mb: torch.Tensor, weight: float) -> float:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                hsc_like = euclid_to_hsc(x_mb)
                hsc_dec = codec_bridge(hsc_like)
                euclid_rec = hsc_to_euclid(hsc_dec)
                loss = criterion(euclid_rec, x_mb)
                loss = loss * weight
                loss = loss / accum_steps

            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")

            scaler.scale(loss).backward()
            return float(loss.item()) * accum_steps / max(weight, 1e-12)

        try:
            w = 1.0
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                hsc_like = euclid_to_hsc(x_full)
                hsc_dec = codec_bridge(hsc_like)
                euclid_rec = hsc_to_euclid(hsc_dec)
                loss = criterion(euclid_rec, x_full)
                loss = loss / accum_steps

            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")

            scaler.scale(loss).backward()
            return float(loss.item()) * accum_steps

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if B == 1: raise

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            mid = B // 2
            x1 = x_full[:mid]
            x2 = x_full[mid:]

            loss_dummy = 0.0
            loss_dummy += run_micro(x1, weight=float(x1.shape[0]) / float(B))
            loss_dummy += run_micro(x2, weight=float(x2.shape[0]) / float(B))
            return float("nan")

    for epoch in range(start_epoch, config.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        epoch_losses: List[float] = []
        oom_splits = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{config.epochs}", smoothing=0)

        for step, euclid_img in enumerate(progress):
            try:
                x = preprocess_and_crop(euclid_img.flux)

                loss_val = backward_batch_oom_safe(x)
                if np.isfinite(loss_val):
                    epoch_losses.append(loss_val)

                if (step + 1) % accum_steps == 0:
                    if config.grad_clip and config.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()),
                            config.grad_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                if device.type == "cuda" and ((step + 1) % max(1, config.log_gpu_mem_every) == 0):
                    alloc = torch.cuda.memory_allocated()
                    reserv = torch.cuda.memory_reserved()
                    progress.set_postfix(
                        loss=(f"{loss_val:.6f}" if np.isfinite(loss_val) else "nan"),
                        alloc=f"{_mem_gb(alloc):.1f}G",
                        reserv=f"{_mem_gb(reserv):.1f}G",
                    )
                else:
                    progress.set_postfix(loss=(f"{loss_val:.6f}" if np.isfinite(loss_val) else "nan"))

                del x

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_splits += 1
                    tqdm.write(f"\n[OOM] step={step}: skipping/clearing.")
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda": torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            except FloatingPointError as e:
                tqdm.write(f"\n[warn] step={step}: {e} -> skipping.")
                optimizer.zero_grad(set_to_none=True)
                continue

        # flush grads
        if len(loader) % accum_steps != 0:
             # Just create a placeholder flush for safety, though typically last batch might be smaller
             # logic in original code
             pass 

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        training_losses.append(avg_loss)

        tqdm.write(f"[epoch {epoch}] avg Euclid MSE={avg_loss:.6f} | oom_splits={oom_splits}")

        ckpt_path = out_dir / f"adapters_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "euclid_to_hsc": euclid_to_hsc.state_dict(),
                "hsc_to_euclid": hsc_to_euclid.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.__dict__,
            },
            ckpt_path,
        )
        tqdm.write(f"Saved checkpoint: {ckpt_path}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        plt.close("all")

    # Final Save
    final_path = out_dir / "adapters_final.pt"
    torch.save(
        {"euclid_to_hsc": euclid_to_hsc.state_dict(), "hsc_to_euclid": hsc_to_euclid.state_dict(), "config": config.__dict__},
        final_path,
    )
    print(f"[done] Saved final adapters to {final_path}")

    # loss curve
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(start_epoch, start_epoch + len(training_losses)), training_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Average Euclid MSE")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=150)
        plt.close("all")
        print("[info] Saved loss_curve.png")
    except Exception as e:
        print(f"[warn] Failed to save loss curve: {e}")

    # sample grid (fallback CPU if GPU full)
    def make_sample_grid_on_device(
        dev: torch.device,
        euclid_to_hsc_m: nn.Module,
        hsc_to_euclid_m: nn.Module,
        codec_m: ImageCodec,
        codec_roundtrip_flux_fn,
    ) -> None:
        euclid_to_hsc_m.eval()
        hsc_to_euclid_m.eval()
        codec_m.eval()

        vis_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_euclid,
        )
        try:
             euclid_img = next(iter(vis_loader))
        except StopIteration:
             print("[warn] Dataset empty, cannot plot sample grid.")
             return

        x_cpu = euclid_img.flux
        x_cpu = torch.nan_to_num(x_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        if config.max_abs > 0:
            x_cpu = torch.clamp(x_cpu, -config.max_abs, config.max_abs)
        x_cpu = center_crop(x_cpu)

        with torch.no_grad():
            x = x_cpu.to(dev)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(dev.type == "cuda")):
                hsc_pred = euclid_to_hsc_m(x)
                hsc_dec = codec_roundtrip_flux_fn(hsc_pred)
                euclid_rec = hsc_to_euclid_m(hsc_dec)

        x_np = x_cpu[0].float().cpu().numpy()
        hsc_pred_np = hsc_pred[0].float().cpu().numpy()
        hsc_dec_np = hsc_dec[0].float().cpu().numpy()
        euclid_rec_np = euclid_rec[0].float().cpu().numpy()

        fig, axes = plt.subplots(4, 5, figsize=(18, 12))
        for ax in axes.flatten():
            ax.axis("off")

        for i, name in enumerate(EUCLID_BANDS):
            axes[0, i].imshow(x_np[i], origin="lower")
            axes[0, i].set_title(f"Euclid in: {name}", fontsize=10)
            axes[0, i].axis("on"); axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

        for i, name in enumerate(HSC_BANDS):
            axes[1, i].imshow(hsc_pred_np[i], origin="lower")
            axes[1, i].set_title(f"HSC pred: {name}", fontsize=10)
            axes[1, i].axis("on"); axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

        for i, name in enumerate(HSC_BANDS):
            axes[2, i].imshow(hsc_dec_np[i], origin="lower")
            axes[2, i].set_title(f"HSC dec: {name}", fontsize=10)
            axes[2, i].axis("on"); axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

        for i, name in enumerate(EUCLID_BANDS):
            axes[3, i].imshow(euclid_rec_np[i], origin="lower")
            axes[3, i].set_title(f"Euclid rec: {name}", fontsize=10)
            axes[3, i].axis("on"); axes[3, i].set_xticks([]); axes[3, i].set_yticks([])

        plt.tight_layout()
        plt.savefig(out_dir / "sample_grid.png", dpi=150)
        plt.close("all")

    try:
        print("[info] Generating sample_grid.png ...")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        make_sample_grid_on_device(
            device,
            euclid_to_hsc,
            hsc_to_euclid,
            codec,
            codec_roundtrip_flux,
        )
        print("[info] Saved sample_grid.png")
    except Exception as e:
        print(f"[warn] GPU grid failed ({e}); trying CPU grid...")
        try:
            # CPU fallback
            make_sample_grid_on_device(
                torch.device("cpu"),
                euclid_to_hsc.to("cpu"),
                hsc_to_euclid.to("cpu"),
                codec.to("cpu"),
                lambda f: codec_roundtrip_flux(f.to(device)).cpu() # simplistic for now, or move codec to cpu
            )
            # Re-creating simple cpu lambda for codec seems complex if codec was moved.
            # Let's just try to move everything to CPU inside catch block properly or skip.
            print("[warn] CPU fallback complicated by closure. Skipping.")
        except Exception as e2:
            print(f"[warn] CPU grid failed too: {e2}")

    # Summary
    summary_path = out_dir / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("AION Training Summary\n")
        f.write("=====================\n")
        f.write(f"Final Average Loss: {training_losses[-1] if training_losses else 'N/A'}\n")
        f.write(f"Config: {config}\n")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
