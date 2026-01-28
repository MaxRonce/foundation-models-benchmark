#!/usr/bin/env python3
"""
Foundation Models Benchmark (FMB)

Module: fmb.models.aion.retrain_euclid_hsc_adapter_unet
Description: AION fine-tuning with U-Net adapter
"""

"""
Euclid <-> HSC adapters (U-Net) around frozen AION ImageCodec.

Key features:
- OOM-safe: automatically splits batches into micro-batches if GPU memory is full.
- STE (Straight-Through Estimator): allows training adapters through the frozen codec.
- Dynamic Configuration: supports YAML and CLI arguments.
- Path Management: integrates with fmb.paths for datasets and outputs.
- Logging: real-time GPU memory tracking and training metrics.
"""

import argparse
import math
from pathlib import Path

# CRITICAL: Monkey-patch aion.modalities to use our local classes
import aion.modalities
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop
from tqdm.auto import tqdm

from fmb.models.aion.modalities import EuclidImage, HSCImage, Image
from fmb.models.aion.model import (
    EUCLID_BANDS,
    HSC_BANDS,
    EuclidImageDataset,
    EuclidToHSC,
    HSCToEuclid,
    collate_euclid,
    load_frozen_codec,
)
from fmb.paths import load_paths

aion.modalities.Image = Image
aion.modalities.HSCImage = HSCImage
aion.modalities.EuclidImage = EuclidImage

# Fix typo/legacy method in aion library
from aion.codecs.preprocessing.image import CenterCrop, RescaleToLegacySurvey


def _fixed_reverse_zeropoint(self, scale):
    return 22.5 - 2.5 * math.log10(scale)


if not hasattr(RescaleToLegacySurvey, "_reverse_zeropoint"):
    RescaleToLegacySurvey._reverse_zeropoint = _fixed_reverse_zeropoint
RescaleToLegacySurvey.reverse_zeropoint = _fixed_reverse_zeropoint


def parse_args() -> argparse.Namespace:
    """Parse arguments from CLI and YAML config."""
    paths = load_paths()

    # First pass: Get config file
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    early_args, _ = parser.parse_known_args()

    # Load defaults from YAML
    defaults = {}
    if early_args.config:
        import yaml

        with open(early_args.config, "r") as f:
            defaults = yaml.safe_load(f) or {}

    # Second pass: Full arguments
    parser = argparse.ArgumentParser(description="Retrain AION adapters.")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Data
    parser.add_argument(
        "--cache-dir", type=str, default=defaults.get("cache_dir", str(paths.dataset))
    )
    parser.add_argument("--split", type=str, default=defaults.get("split", "train"))
    parser.add_argument(
        "--max-entries", type=int, default=defaults.get("max_entries", 0)
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 8))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 5))
    parser.add_argument(
        "--lr", type=float, default=float(defaults.get("learning_rate", 1e-4))
    )
    parser.add_argument(
        "--grad-clip", type=float, default=defaults.get("grad_clip", 1.0)
    )
    parser.add_argument(
        "--accum-steps", type=int, default=defaults.get("accum_steps", 1)
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default=defaults.get("amp_dtype", "float16"),
        choices=["float16", "bfloat16"],
    )

    # Preprocessing
    parser.add_argument("--resize", type=int, default=defaults.get("resize", 96))
    parser.add_argument("--crop-size", type=int, default=defaults.get("crop_size", 96))
    parser.add_argument("--max-abs", type=float, default=defaults.get("max_abs", 100.0))
    parser.add_argument(
        "--cpu-crop", action="store_true", default=defaults.get("cpu_crop", False)
    )

    # Model
    parser.add_argument("--hidden", type=int, default=defaults.get("hidden", 16))
    parser.add_argument(
        "--use-unet-checkpointing",
        action="store_true",
        default=defaults.get("use_unet_checkpointing", False),
    )
    parser.add_argument(
        "--codec-grad",
        type=str,
        default=defaults.get("codec_grad", "ste"),
        choices=["ste", "full"],
    )
    parser.add_argument(
        "--disable-codec-checkpointing",
        action="store_true",
        default=defaults.get("disable_codec_checkpointing", False),
    )

    # Output
    default_out = paths.retrained_weights / "aion"
    parser.add_argument(
        "--output", type=str, default=defaults.get("out_dir", str(default_out))
    )
    parser.add_argument(
        "--resume-adapter", type=str, default=defaults.get("resume_adapter", None)
    )
    parser.add_argument(
        "--auto-resume", action="store_true", default=defaults.get("auto_resume", True)
    )

    # Logging
    parser.add_argument(
        "--num-workers", type=int, default=defaults.get("num_workers", 0)
    )
    parser.add_argument(
        "--log-gpu-mem-every", type=int, default=defaults.get("log_gpu_mem_every", 50)
    )

    return parser.parse_args()


def _mem_gb(x: int) -> float:
    return float(x) / 1e9


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume Logic
    start_epoch = 1
    if args.auto_resume and not args.resume_adapter:
        ckpts = list(out_dir.glob("adapters_epoch_*.pt"))
        if ckpts:

            def get_epoch(p: Path) -> int:
                try:
                    return int(p.stem.split("_")[-1])
                except:
                    return -1

            latest = max(ckpts, key=get_epoch)
            print(f"[info] Auto-resume from latest: {latest}")
            args.resume_adapter = str(latest)

    # Data
    max_entries = None if args.max_entries <= 0 else args.max_entries
    dataset = EuclidImageDataset(
        split=args.split,
        cache_dir=args.cache_dir,
        max_entries=max_entries,
        resize=args.resize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_euclid,
    )

    print(f"Loaded {len(dataset)} samples; {len(loader)} batches/epoch")

    # Models
    codec, codec_cfg = load_frozen_codec(device)
    euclid_to_hsc = EuclidToHSC(
        hidden=args.hidden, use_checkpointing=args.use_unet_checkpointing
    ).to(device)
    hsc_to_euclid = HSCToEuclid(
        hidden=args.hidden, use_checkpointing=args.use_unet_checkpointing
    ).to(device)

    optimizer = torch.optim.Adam(
        list(euclid_to_hsc.parameters()) + list(hsc_to_euclid.parameters()), lr=args.lr
    )
    criterion = nn.MSELoss(reduction="mean")

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16)
    )

    if args.resume_adapter:
        ckpt = torch.load(args.resume_adapter, map_location="cpu")
        euclid_to_hsc.load_state_dict(ckpt["euclid_to_hsc"])
        hsc_to_euclid.load_state_dict(ckpt["hsc_to_euclid"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        print(f"[info] Resumed from {args.resume_adapter} (next epoch={start_epoch})")

    # Helpers for codec roundtrip
    def run_codec_roundtrip(hsc_flux: torch.Tensor) -> torch.Tensor:
        hsc_obj = HSCImage(flux=hsc_flux, bands=HSC_BANDS)
        toks = codec.encode(hsc_obj)
        hsc_rec = codec.decode(toks, bands=HSC_BANDS)
        return hsc_rec.flux

    def codec_bridge(hsc_flux: torch.Tensor) -> torch.Tensor:
        if args.codec_grad == "ste":
            with torch.no_grad():
                y = run_codec_roundtrip(hsc_flux)
            return hsc_flux + (y - hsc_flux).detach()
        else:
            # Full gradients through codec
            from torch.utils.checkpoint import checkpoint

            if not args.disable_codec_checkpointing and hsc_flux.requires_grad:
                return checkpoint(run_codec_roundtrip, hsc_flux, use_reentrant=False)
            return run_codec_roundtrip(hsc_flux)

    crop = RandomCrop(size=args.crop_size)
    center_crop = CenterCrop(crop_size=args.crop_size)

    # OOM-safe Batch Processor
    def process_batch(x_full: torch.Tensor) -> float:
        B = x_full.shape[0]
        try:
            with torch.amp.autocast(
                "cuda", dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                hsc_like = euclid_to_hsc(x_full)
                hsc_dec = codec_bridge(hsc_like)
                euclid_rec = hsc_to_euclid(hsc_dec)
                loss = criterion(euclid_rec, x_full) / args.accum_steps

            if not torch.isfinite(loss):
                raise FloatingPointError("Loss is non-finite")

            scaler.scale(loss).backward()
            return loss.item() * args.accum_steps
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or B == 1:
                raise
            # Split and retry
            torch.cuda.empty_cache()
            mid = B // 2
            loss1 = process_batch(x_full[:mid])
            loss2 = process_batch(x_full[mid:])
            return (loss1 + loss2) / 2.0

    # Training Loop
    training_losses = []
    print(f"Starting training (Epoch {start_epoch} to {args.epochs})...")

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        euclid_to_hsc.train()
        hsc_to_euclid.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []

        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", smoothing=0.1)
        for step, euclid_img in enumerate(progress):
            # Preprocess and move to GPU
            x = euclid_img.flux.to(device)
            if args.max_abs > 0:
                x = torch.clamp(x, -args.max_abs, args.max_abs)
            x = crop(x)

            loss_val = process_batch(x)
            epoch_losses.append(loss_val)

            if (step + 1) % args.accum_steps == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        list(euclid_to_hsc.parameters())
                        + list(hsc_to_euclid.parameters()),
                        args.grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Update progress bar
            if device.type == "cuda" and (step + 1) % args.log_gpu_mem_every == 0:
                alloc = _mem_gb(torch.cuda.memory_allocated())
                peak = _mem_gb(torch.cuda.max_memory_allocated())
                progress.set_postfix(
                    loss=f"{loss_val:.4f}", mem=f"{alloc:.1f}/{peak:.1f}G"
                )
            else:
                progress.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = np.mean(epoch_losses)
        training_losses.append(avg_loss)
        print(f"[Epoch {epoch}] Avg MSE: {avg_loss:.6f}")

        # Save Checkpoint
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

    # Final Save and Plots
    torch.save(
        {
            "euclid_to_hsc": euclid_to_hsc.state_dict(),
            "hsc_to_euclid": hsc_to_euclid.state_dict(),
            "args": vars(args),
        },
        out_dir / "adapters_final.pt",
    )

    # Plot sample results
    def save_sample_grid():
        euclid_to_hsc.eval()
        hsc_to_euclid.eval()
        with torch.no_grad():
            sample = next(iter(loader))
            x = sample.flux[:1].to(device)
            x = center_crop(x)
            hsc_pred = euclid_to_hsc(x)
            hsc_dec = run_codec_roundtrip(hsc_pred)
            euclid_rec = hsc_to_euclid(hsc_dec)

        fig, axes = plt.subplots(4, 5, figsize=(15, 10))
        for i in range(4):  # Euclid bands
            axes[0, i].imshow(x[0, i].cpu(), origin="lower")
            axes[0, i].set_title(f"Input {EUCLID_BANDS[i]}")
            axes[3, i].imshow(euclid_rec[0, i].cpu(), origin="lower")
            axes[3, i].set_title(f"Rec {EUCLID_BANDS[i]}")
        for i in range(5):  # HSC bands
            axes[1, i].imshow(hsc_pred[0, i].cpu(), origin="lower")
            axes[1, i].set_title(f"Pred {HSC_BANDS[i]}")
            axes[2, i].imshow(hsc_dec[0, i].cpu(), origin="lower")
            axes[2, i].set_title(f"Codec {HSC_BANDS[i]}")

        plt.tight_layout()
        plt.savefig(out_dir / "sample_grid.png")
        plt.close()

    save_sample_grid()
    print(f"Finished. Results saved in {out_dir}")


if __name__ == "__main__":
    main()
