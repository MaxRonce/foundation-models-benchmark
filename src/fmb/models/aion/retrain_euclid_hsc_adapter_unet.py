#!/usr/bin/env python3
"""
Euclid <-> HSC adapters (U-Net) around frozen AION ImageCodec.

Key fixes:
- No OOM skipping: on OOM we split the batch into smaller micro-batches and continue.
- Prevent true memory retention by default via STE through codec:
    forward uses codec roundtrip, backward treats codec as identity (no codec graph kept).
  You can enable full gradients through codec with --codec-grad full (not recommended if it leaks).

- AMP + GradScaler
- Optional gradient accumulation
- Optional U-Net checkpointing
- Real GPU memory logging (alloc/reserved/peak reset each epoch)
- Save loss_curve.png and sample_grid.png

"""

import argparse
import json
import gc
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import safetensors.torch as st
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

from scratch.load_display_data import EuclidDESIDataset
from aion.codecs import ImageCodec
from aion.codecs.config import HF_REPO_ID
from aion.codecs.preprocessing.image import CenterCrop
from torchvision.transforms import RandomCrop
from aion.modalities import EuclidImage, HSCImage


EUCLID_BANDS = ["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"]
HSC_BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]

EUCLID_ZP_NU = {
    "vis_image": 2835.34,
    "nisp_y_image": 1916.10,
    "nisp_j_image": 1370.25,
    "nisp_h_image": 918.35,
}


# -----------------------------
# U-Net blocks
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
class EuclidImageDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, cache_dir: str, max_entries: Optional[int], resize: int):
        self.base = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        self.resize = resize
        self._indices = list(range(len(self.base))) if not max_entries or max_entries <= 0 else list(
            range(min(len(self.base), max_entries))
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> EuclidImage:
        base_idx = self._indices[idx]
        sample = self.base[base_idx]

        keys = ["vis_image", "nisp_y_image", "nisp_j_image", "nisp_h_image"]
        bands = []
        for key in keys:
            t = sample.get(key)
            if t is None:
                raise ValueError(f"Missing band '{key}' at index {base_idx}")
            t = t.to(torch.float32)
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

            # ADU -> nanomaggies
            zp_nu = EUCLID_ZP_NU[key]
            scale_factor = zp_nu / 3631.0
            t = t * scale_factor

            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.ndim != 2:
                raise ValueError(f"Expected 2D band, got {tuple(t.shape)}")
            bands.append(t)

        flux = torch.stack(bands, dim=0)  # (4,H,W)

        if self.resize and (flux.shape[-1] != self.resize or flux.shape[-2] != self.resize):
            flux = F.interpolate(
                flux.unsqueeze(0),
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return EuclidImage(flux=flux, bands=EUCLID_BANDS)


def collate_euclid(batch: List[EuclidImage]) -> EuclidImage:
    flux = torch.stack([b.flux for b in batch], dim=0)
    return EuclidImage(flux=flux, bands=batch[0].bands)


# -----------------------------
# Helpers
# -----------------------------
def load_frozen_codec(device: torch.device) -> Tuple[ImageCodec, dict]:
    cfg_path = hf_hub_download(HF_REPO_ID, "codecs/image/config.json", local_files_only=True)
    weights_path = hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=True)

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


    # Config file
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    args = p.parse_args()
    
    # Load YAML if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Update args with values from YAML if not specified on CLI (CLI takes precedence? 
                # Usually config file provides defaults, and CLI overrides. 
                # But argparse already populated defaults. 
                # So we should overwrite if YAML value exists AND CLI arg was default? 
                # Simpler approach: Load YAML, then parse args again with defaults from YAML.
                # OR: Manually update args namespace where YAML has values.
                
                # Let's override defaults with YAML, then let specific CLI args override that.
                # However, since we already parsed args, we ignore the fact that they might be defaults.
                # Ideally: set defaults=yaml_values in ArgumentParser.
                pass 
                
    return args

# Better approach: parse args, check for config, reload args with defaults from config
# Or standard pattern: config args -> parser.set_defaults(**config_args) -> parser.parse_args()

def parse_args() -> argparse.Namespace:
    # First pass to get config file
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", type=str, default=None)
    early_args, _ = p.parse_known_args()

    # Load defaults from YAML
    defaults = {}
    if early_args.config:
        import yaml
        with open(early_args.config, 'r') as f:
            defaults = yaml.safe_load(f) or {}

    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--cache-dir", type=str, default=defaults.get("cache_dir", "/scratch"))
    p.add_argument("--split", type=str, default=defaults.get("split", "train"))
    p.add_argument("--max-entries", type=int, default=defaults.get("max_entries", 0))

    p.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 8))
    p.add_argument("--epochs", type=int, default=defaults.get("epochs", 5))
    p.add_argument("--lr", type=float, default=float(defaults.get("learning_rate", 1e-4)))

    p.add_argument("--resize", type=int, default=defaults.get("resize", 96))
    p.add_argument("--crop-size", type=int, default=defaults.get("crop_size", 96))
    p.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 0))
    p.add_argument("--max-abs", type=float, default=defaults.get("max_abs", 100.0))

    p.add_argument("--hidden", type=int, default=defaults.get("hidden", 16))
    p.add_argument("--output", type=str, default=defaults.get("out_dir", "outputs/euclid_hsc_adapter_unet"))

    p.add_argument("--resume-adapter", type=str, default=defaults.get("resume_adapter", None))
    p.add_argument("--auto-resume", action="store_true", default=defaults.get("auto_resume", False))
    p.add_argument("--grad-clip", type=float, default=defaults.get("grad_clip", 1.0))

    # Memory
    p.add_argument("--accum-steps", type=int, default=defaults.get("accum_steps", 1))
    p.add_argument("--amp-dtype", type=str, default=defaults.get("amp_dtype", "float16"), choices=["float16", "bfloat16"])
    p.add_argument("--use-unet-checkpointing", action="store_true", default=defaults.get("use_unet_checkpointing", False))

    # Codec gradients. YAML uses 'codec_grad'
    p.add_argument("--codec-grad", type=str, default=defaults.get("codec_grad", "ste"), choices=["ste", "full"])
    p.add_argument("--disable-codec-checkpointing", action="store_true", default=defaults.get("disable_codec_checkpointing", False))

    p.add_argument("--cpu-crop", action="store_true", default=defaults.get("cpu_crop", False))
    p.add_argument("--log-gpu-mem-every", type=int, default=defaults.get("log_gpu_mem_every", 50))

    return p.parse_args()


def _mem_gb(x: int) -> float:
    return float(x) / 1e9


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    # Conservative defaults that reduce “mystery” allocations
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume
    start_epoch = 1
    if args.auto_resume:
        ckpts = list(out_dir.glob("adapters_epoch_*.pt"))
        if ckpts:
            def get_epoch(p: Path) -> int:
                try:
                    return int(p.stem.split("_")[-1])
                except Exception:
                    return -1
            latest = max(ckpts, key=get_epoch)
            print(f"[info] Auto-resume: {latest}")
            args.resume_adapter = str(latest)

    max_entries = None if args.max_entries <= 0 else args.max_entries
    dataset = EuclidImageDataset(split=args.split, cache_dir=args.cache_dir, max_entries=max_entries, resize=args.resize)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_euclid,
    )

    print(f"Loaded {len(dataset)} samples; batches/epoch: {len(loader)} (batch={args.batch_size}, accum={args.accum_steps})")

    codec, codec_cfg = load_frozen_codec(device)
    euclid_to_hsc = EuclidToHSC(hidden=args.hidden, use_checkpointing=args.use_unet_checkpointing).to(device)
    hsc_to_euclid = HSCToEuclid(hidden=args.hidden, use_checkpointing=args.use_unet_checkpointing).to(device)

    optimizer = torch.optim.Adam(list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()), lr=args.lr)
    criterion = nn.MSELoss(reduction="mean")

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    accum_steps = max(1, int(args.accum_steps))
    use_codec_ckpt = (not args.disable_codec_checkpointing)

    crop = RandomCrop(size=args.crop_size)
    center_crop = CenterCrop(crop_size=args.crop_size)

    if args.resume_adapter:
        ckpt = torch.load(args.resume_adapter, map_location="cpu")
        euclid_to_hsc.load_state_dict(ckpt["euclid_to_hsc"])
        hsc_to_euclid.load_state_dict(ckpt["hsc_to_euclid"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"[info] Resumed from {args.resume_adapter} (start_epoch={start_epoch})")

    (out_dir / "adapter_config.json").write_text(json.dumps({**vars(args), "codec_cfg": codec_cfg}, indent=2, default=str))

    euclid_to_hsc.train()
    hsc_to_euclid.train()

    # codec roundtrip, Tensor -> Tensor
    def codec_roundtrip_flux(hsc_flux: torch.Tensor) -> torch.Tensor:
        hsc_obj = HSCImage(flux=hsc_flux, bands=HSC_BANDS)
        toks = codec.encode(hsc_obj)
        hsc_rec = codec.decode(toks, bands=HSC_BANDS)
        return hsc_rec.flux

    # STE wrapper: forward uses codec output, backward is identity wrt input
    def codec_roundtrip_ste(hsc_flux: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = codec_roundtrip_flux(hsc_flux)
        # forward: y ; backward: dy/dx = I
        return hsc_flux + (y - hsc_flux).detach()

    # Full gradient path (may leak); with optional checkpointing
    def codec_roundtrip_full(hsc_flux: torch.Tensor) -> torch.Tensor:
        if use_codec_ckpt and hsc_flux.requires_grad:
            return checkpoint_utils.checkpoint(codec_roundtrip_flux, hsc_flux, use_reentrant=False)
        return codec_roundtrip_flux(hsc_flux)

    # Choose codec gradient mode
    if args.codec_grad == "ste":
        codec_bridge = codec_roundtrip_ste
        print("[info] codec-grad=ste (recommended): no codec autograd graph retained.")
    else:
        codec_bridge = codec_roundtrip_full
        print("[info] codec-grad=full: backprop through codec (can be very memory heavy / may leak).")

    training_losses: List[float] = []

    def preprocess_and_crop(euclid_flux_cpu: torch.Tensor) -> torch.Tensor:
        # returns GPU tensor
        if args.cpu_crop:
            x = torch.nan_to_num(euclid_flux_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            if args.max_abs > 0:
                x = torch.clamp(x, -args.max_abs, args.max_abs)
            x = crop(x)  # CPU crop
            return x.to(device, non_blocking=True)

        x = euclid_flux_cpu.to(device, non_blocking=False)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if args.max_abs > 0:
            x = torch.clamp(x, -args.max_abs, args.max_abs)
        return crop(x)  # GPU crop

    # Process a batch with OOM-safe splitting (NO SKIP)
    def backward_batch_oom_safe(x_full: torch.Tensor) -> float:
        """
        Do forward/backward for a batch tensor x_full (B,4,H,W) on GPU.
        If OOM happens, splits batch and retries recursively.
        Returns the (unscaled) loss value for logging.
        """
        B = int(x_full.shape[0])

        def run_micro(x_mb: torch.Tensor, weight: float) -> float:
            # weight = (mb_size / B_full)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                hsc_like = euclid_to_hsc(x_mb)
                hsc_dec = codec_bridge(hsc_like)
                euclid_rec = hsc_to_euclid(hsc_dec)
                loss = criterion(euclid_rec, x_mb)
                # match "whole batch mean" by weighting each microbatch contribution
                loss = loss * weight
                # and apply grad accumulation scaling
                loss = loss / accum_steps

            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")

            scaler.scale(loss).backward()
            return float(loss.item()) * accum_steps / max(weight, 1e-12)  # approx local loss for debug (not used)

        try:
            # normal path
            w = 1.0  # whole batch weight
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
            # split batch
            if B == 1:
                # cannot split further
                raise

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            mid = B // 2
            x1 = x_full[:mid]
            x2 = x_full[mid:]

            # process halves with proper weights
            loss_dummy = 0.0
            loss_dummy += run_micro(x1, weight=float(x1.shape[0]) / float(B))
            loss_dummy += run_micro(x2, weight=float(x2.shape[0]) / float(B))
            # for logging, return a placeholder (true loss would require recompute; not worth it)
            return float("nan")

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)
        epoch_losses: List[float] = []
        oom_splits = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", smoothing=0)

        for step, euclid_img in enumerate(progress):
            try:
                x = preprocess_and_crop(euclid_img.flux)

                # backward (OOM-safe; no skipping)
                loss_val = backward_batch_oom_safe(x)
                if np.isfinite(loss_val):
                    epoch_losses.append(loss_val)

                # optimizer step on accum boundary
                if (step + 1) % accum_steps == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()),
                            args.grad_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # log memory
                if device.type == "cuda" and ((step + 1) % max(1, args.log_gpu_mem_every) == 0):
                    alloc = torch.cuda.memory_allocated()
                    reserv = torch.cuda.memory_reserved()
                    peak = torch.cuda.max_memory_allocated()
                    progress.set_postfix(
                        loss=(f"{loss_val:.6f}" if np.isfinite(loss_val) else "nan(split)"),
                        alloc=f"{_mem_gb(alloc):.1f}G",
                        reserv=f"{_mem_gb(reserv):.1f}G",
                        peak=f"{_mem_gb(peak):.1f}G",
                    )
                else:
                    progress.set_postfix(loss=(f"{loss_val:.6f}" if np.isfinite(loss_val) else "nan(split)"))

                del x

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_splits += 1
                    print(f"\n[OOM] step={step}: even batch=1 failed? clearing and continuing.")
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise
            except FloatingPointError as e:
                print(f"\n[warn] step={step}: {e} -> skipping this batch (bad numerics).")
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                continue

        # flush remaining grads
        if len(loader) % accum_steps != 0:
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        training_losses.append(avg_loss)

        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated()
            reserv = torch.cuda.memory_reserved()
            peak = torch.cuda.max_memory_allocated()
            print(
                f"[epoch {epoch}] avg Euclid MSE={avg_loss:.6f} | "
                f"alloc={_mem_gb(alloc):.1f}G reserv={_mem_gb(reserv):.1f}G peak={_mem_gb(peak):.1f}G | "
                f"oom_splits={oom_splits}"
            )
        else:
            print(f"[epoch {epoch}] avg Euclid MSE={avg_loss:.6f} | oom_splits={oom_splits}")

        ckpt_path = out_dir / f"adapters_epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "euclid_to_hsc": euclid_to_hsc.state_dict(),
                "hsc_to_euclid": hsc_to_euclid.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        plt.close("all")

    # final save
    final_path = out_dir / "adapters_final.pt"
    torch.save(
        {"euclid_to_hsc": euclid_to_hsc.state_dict(), "hsc_to_euclid": hsc_to_euclid.state_dict(), "args": vars(args)},
        final_path,
    )
    print(f"[done] Saved final adapters to {final_path}")

    # loss curve
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(training_losses) + 1), training_losses, marker="o")
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
        euclid_img = next(iter(vis_loader))
        x_cpu = euclid_img.flux
        x_cpu = torch.nan_to_num(x_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        if args.max_abs > 0:
            x_cpu = torch.clamp(x_cpu, -args.max_abs, args.max_abs)
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
            # CPU copies for viz
            euclid_to_hsc_cpu = euclid_to_hsc.to("cpu")
            hsc_to_euclid_cpu = hsc_to_euclid.to("cpu")
            codec_cpu = codec.to("cpu")

            def codec_roundtrip_flux_cpu(hsc_flux: torch.Tensor) -> torch.Tensor:
                hsc_obj = HSCImage(flux=hsc_flux, bands=HSC_BANDS)
                toks = codec_cpu.encode(hsc_obj)
                hsc_rec = codec_cpu.decode(toks, bands=HSC_BANDS)
                return hsc_rec.flux

            make_sample_grid_on_device(
                torch.device("cpu"),
                euclid_to_hsc_cpu,
                hsc_to_euclid_cpu,
                codec_cpu,
                codec_roundtrip_flux_cpu,
            )
            print("[info] Saved sample_grid.png (CPU fallback)")
        except Exception as e2:
            print(f"[warn] CPU grid failed too: {e2}")



if __name__ == "__main__":
    main()
