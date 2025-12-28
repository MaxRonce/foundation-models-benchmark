#!/usr/bin/env python3
"""
Retrain ImageCodec on raw Euclid VIS/Y/J/H bands (with flux renormalization to nanomaggies).

Example:
  python -m scratch.retrain_euclid_codec \\
    --cache-dir /scratch/ronceray/datasets \\
    --split train \\
    --max-entries 5000 \\
    --batch-size 8 \\
    --epochs 5 \\
    --lr 1e-4 \\
    --resize 160 \\
    --crop-size 96 \\
    --output outputs/retrained_euclid_codec
"""
import argparse
from encodings.cp932 import codec
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import safetensors.torch as st
from huggingface_hub import hf_hub_download
from torch.optim.lr_scheduler import CosineAnnealingLR

from fmb.data.load_display_data_hsc import EuclidDESIDataset
# Paths management
from fmb.paths import load_paths

# AION might need sys.path for Phase 1 if not in python path
try:
    from aion.codecs import ImageCodec
    from aion.codecs.config import HF_REPO_ID
    from aion.codecs.preprocessing.image import CenterCrop
    from aion.modalities import EuclidImage
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[4] / "external" / "AION"))
    from aion.codecs import ImageCodec
    from aion.codecs.config import HF_REPO_ID
    from aion.codecs.preprocessing.image import CenterCrop
    from aion.modalities import EuclidImage
from typing import List, Optional

BANDS = ["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"]

# Zero points in nJy/ADU (derived from user-provided ZP_nu table)
# Target is nanomaggies (ZP=22.5 mag, 1 nmgy = 3631 nJy)
# Scale factor = ZP_nu / 3631.0
EUCLID_ZP_NU = {
    "vis_image": 2835.34,
    "nisp_y_image": 1916.10,
    "nisp_j_image": 1370.25,
    "nisp_h_image": 918.35,
}


class EuclidImageDataset(torch.utils.data.Dataset):
    """Wrap EuclidDESIDataset to emit EuclidImage objects."""

    def __init__(self, split: str, cache_dir: Optional[str], max_entries: Optional[int], resize: int):
        if cache_dir is None:
             cache_dir = str(load_paths().data)
        self.base = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        self.max_entries = max_entries if max_entries is not None and max_entries > 0 else None
        self.resize = resize
        self._indices = list(range(len(self.base))) if self.max_entries is None else list(
            range(min(len(self.base), self.max_entries)),
        )

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return len(self._indices)

    def __getitem__(self, idx: int) -> EuclidImage:
        base_idx = self._indices[idx]
        sample = self.base[base_idx]

        bands = []
        # Map dataset keys to ZP keys
        keys = ["vis_image", "nisp_y_image", "nisp_j_image", "nisp_h_image"]
        
        for key in keys:
            tensor = sample.get(key)
            if tensor is None:
                raise ValueError(f"Missing band '{key}' at index {base_idx}")
            tensor = tensor.to(torch.float32)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply zero-point scaling to convert to nanomaggies (Legacy Survey scale)
            zp_nu = EUCLID_ZP_NU[key]
            scale_factor = zp_nu / 3631.0
            tensor = tensor * scale_factor

            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 2:
                raise ValueError(f"Expected band '{key}' to be 2D, got shape {tuple(tensor.shape)}")
            bands.append(tensor)

        flux = torch.stack(bands, dim=0)  # (4, H, W)
        if self.resize and (flux.shape[-1] != self.resize or flux.shape[-2] != self.resize):
            flux = F.interpolate(
                flux.unsqueeze(0),
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return EuclidImage(flux=flux, bands=BANDS)


def collate_euclid(batch: List[EuclidImage]) -> EuclidImage:
    if not batch:
        raise ValueError("Empty batch received by DataLoader")
    flux = torch.stack([item.flux for item in batch], dim=0)
    return EuclidImage(flux=flux, bands=batch[0].bands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetune ImageCodec on Euclid VIS/Y/J/H bands without flux renormalization.",
    )
    parser.add_argument("--cache-dir", type=str, default="/scratch", help="Local HF cache/dataset root.")
    parser.add_argument("--split", type=str, default="train", help="Splits: 'train', 'test', 'train,test', 'all'.")
    parser.add_argument("--max-entries", type=int, default=5000, help="Limit samples (<=0 means all).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resize", type=int, default=160, help="Resize Euclid bands to NxN before cropping.")
    parser.add_argument("--crop-size", type=int, default=96, help="Center-crop size for reconstruction loss.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a previously finetuned codec directory (containing a 'codecs/' subfolder) to resume from.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers (use 0 or 1 on memory-constrained nodes).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max grad norm (<=0 disables clipping).",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine"],
        default="none",
        help="Optional LR scheduler; cosine anneals over total steps.",
    )
    parser.add_argument("--output", type=str, default=None, help="Save directory.")
    parser.add_argument(
        "--save-viz",
        type=str,
        default=None,
        help="Optional path to save an original/cropped/reconstructed grid from the first batch.",
    )
    parser.add_argument(
        "--max-abs",
        type=float,
        default=1e4,
        help="Clamp absolute flux before range compression to avoid NaNs/infs (set <=0 to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_entries = None if args.max_entries is None or args.max_entries <= 0 else args.max_entries

    dataset = EuclidImageDataset(split=args.split, cache_dir=args.cache_dir, max_entries=max_entries, resize=args.resize)

    def make_loader(num_workers: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_euclid,
        )

    loader = make_loader(args.num_workers)
    # If worker spawning fails (common with fork+OOM), fall back to num_workers=0.
    if args.num_workers > 0:
        try:
            next(iter(loader))
        except OSError as exc:
            print(f"[warn] DataLoader failed with num_workers={args.num_workers} ({exc}); retrying with 0 workers.")
            loader = make_loader(0)
    print(f"Loaded {len(dataset)} samples; batches/epoch: {len(loader)} using num_workers={loader.num_workers}")

    # Load codec: either resume from a previously finetuned checkpoint, or start from the base HF weights.
    if args.resume_from:
        print(f"[info] Resuming codec from {args.resume_from}")
        resume_dir = Path(args.resume_from)
        cfg_path = resume_dir / "codecs" / "image" / "config.json"
        weights_path = resume_dir / "codecs" / "image" / "model.safetensors"
        if not cfg_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Expected config and weights under {resume_dir}/codecs/image/, "
                f"but found cfg={cfg_path.is_file()} weights={weights_path.is_file()}",
            )
        with cfg_path.open() as f:
            codec_cfg = json.load(f)
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
        state = st.load_file(weights_path, device="cpu")
        missing, unexpected = codec.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[info] resume state_dict load: missing={missing}, unexpected={unexpected}")
    else:
        # Manually load codec weights and pad band-projection layers to include Euclid bands.
        cfg_path = hf_hub_download(HF_REPO_ID, "codecs/image/config.json", local_files_only=True)
        weights_path = hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=True)
        with open(cfg_path) as f:
            codec_cfg = json.load(f)
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
        state = st.load_file(weights_path, device="cpu")
        print("quantizer_levels (from config):", codec_cfg["quantizer_levels"])

        # Pad subsample_in/out to accommodate extra Euclid channels (13 vs pretrained 9).
        def _pad_param(name: str, target_shape):
            tensor = state.get(name)
            if tensor is None:
                return

            old_shape = tensor.shape
            if old_shape == target_shape:
                return

            new_tensor = torch.zeros(target_shape, dtype=tensor.dtype)

            # Copy overlapping region
            common_slices_old = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, target_shape))
            common_slices_new = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, target_shape))
            new_tensor[common_slices_new] = tensor[common_slices_old]

            # If this is one of the band-related params, and there is a 9 -> 13 change,
            # then we want to COPY the last original band (index 8) into the new Euclid slots.
            if name in ("subsample_in.weight", "subsample_out.weight", "subsample_out.bias"):
                # Find axis where 9 -> 13 (old bands -> new bands)
                band_axis = None
                for i, (o, n) in enumerate(zip(old_shape, target_shape)):
                    if o == 9 and n == 13:
                        band_axis = i
                        break

                if band_axis is not None:
                    # number of extra bands (13 - 9 = 4 for your case)
                    num_extra = target_shape[band_axis] - old_shape[band_axis]

                    # slice for the source band: index 8 along band_axis
                    src_slice = [slice(None)] * len(target_shape)
                    src_slice[band_axis] = slice(8, 9)  # last original band

                    # slice for the destination bands: indices [9:13] along band_axis
                    dst_slice = [slice(None)] * len(target_shape)
                    dst_slice[band_axis] = slice(old_shape[band_axis], target_shape[band_axis])

                    # Take the last band and repeat it num_extra times along band_axis
                    patch = new_tensor[tuple(src_slice)]  # shape with size 1 along band_axis

                    repeat_factors = []
                    for i in range(len(target_shape)):
                        if i == band_axis:
                            repeat_factors.append(num_extra)
                        else:
                            repeat_factors.append(1)

                    patch_expanded = patch.repeat(*repeat_factors)
                    new_tensor[tuple(dst_slice)] = patch_expanded

            state[name] = new_tensor

        _pad_param("subsample_in.weight", codec.subsample_in.weight.shape)
        _pad_param("subsample_out.weight", codec.subsample_out.weight.shape)
        _pad_param("subsample_out.bias", codec.subsample_out.bias.shape)
        missing, unexpected = codec.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[info] state_dict load: missing={missing}, unexpected={unexpected}")

    # Freeze everything
    for name, p in codec.named_parameters():
        p.requires_grad = False

    # Unfreeze the projection layers we just padded
    for name, p in codec.named_parameters():
        if "subsample_in" in name or "subsample_out" in name or "pre_quant_proj" in name or "post_quant_proj" in name:
            p.requires_grad = True

    # Log unfrozen parameters
    trainable_params = [n for n, p in codec.named_parameters() if p.requires_grad]
    print(f"[info] Unfrozen parameters: {trainable_params}")
    print(f"[info] Total trainable parameters: {sum(p.numel() for p in codec.parameters() if p.requires_grad)}")

    # Explicitly freeze biases to prevent degradation of other modalities
    # This is critical: updating the shared bias shifts activations for ALL modalities,
    # causing catastrophic forgetting/degradation of existing ones (e.g. HSC).
    if codec.subsample_in.bias is not None:
        codec.subsample_in.bias.requires_grad = False
    if codec.subsample_out.bias is not None:
        codec.subsample_out.bias.requires_grad = False

    codec.train()

    optimizer = torch.optim.Adam(codec.parameters(), lr=args.lr)
    scheduler = None
    if args.scheduler == "cosine":
        total_steps = max(len(dataset) // max(args.batch_size, 1), 1) * max(args.epochs, 1)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = torch.nn.MSELoss()
    crop = CenterCrop(crop_size=args.crop_size)

    training_losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        skipped = 0
        for euclid_img in progress:
            flux = euclid_img.flux.to(device)
            flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
            if args.max_abs and args.max_abs > 0:
                flux = torch.clamp(flux, min=-args.max_abs, max=args.max_abs)
            cropped = crop(flux)
            euclid_cropped = EuclidImage(flux=cropped, bands=euclid_img.bands)

            optimizer.zero_grad(set_to_none=True)
            tokens = codec.encode(euclid_cropped)

            with torch.no_grad():
                t_min = float(tokens.min())
                t_max = float(tokens.max())
                n_unique = len(torch.unique(tokens))
            global_step = (epoch - 1) * len(loader) + progress.n
            print(f"[debug] batch {global_step} tokens range: {t_min} {t_max}, unique: {n_unique}")



            if not torch.isfinite(tokens).all():
                print("[debug] non-finite tokens from encode()")
                print("  flux min/max:", float(cropped.min()), float(cropped.max()))
                break

            recon = codec.decode(tokens, bands=euclid_cropped.bands)

            if not torch.isfinite(recon.flux).all():
                print("[debug] non-finite recon from decode()")
                print("  tokens min/max:", float(tokens.min()), float(tokens.max()))
                print("  input flux min/max:", float(cropped.min()), float(cropped.max()))
                # Optionally inspect a few recon values:
                bad = recon.flux[~torch.isfinite(recon.flux)]
                print("  example bad values:", bad.view(-1)[:10].tolist())
                break
            if not torch.isfinite(recon.flux).all():
                skipped += 1
                progress.set_postfix_str("non-finite recon")
                continue
            loss = criterion(recon.flux, cropped)
            if not torch.isfinite(loss):
                skipped += 1
                progress.set_postfix_str("non-finite loss")
                continue
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        training_losses.append(avg_loss)
        print(f"[epoch {epoch}] avg MSE={avg_loss:.6f} (skipped {skipped} batches)")
        if skipped == len(loader):
            print("[error] all batches skipped this epoch (non-finite). Try lowering --lr, lowering --max-abs, or increasing --grad-clip.")
            break

    codec.eval()
    viz_payload = None
    with torch.no_grad():
        try:
            batch = next(iter(loader))
            flux = batch.flux.to(device)
            flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
            if args.max_abs and args.max_abs > 0:
                flux = torch.clamp(flux, min=-args.max_abs, max=args.max_abs)
            cropped = crop(flux)
            euclid_cropped = EuclidImage(flux=cropped, bands=batch.bands)
            tokens = codec.encode(euclid_cropped)
            recon = codec.decode(tokens, bands=euclid_cropped.bands)
            val_loss = criterion(recon.flux, cropped).item()
            print(f"[val] first-batch MSE={val_loss:.6f}")
            viz_payload = (flux.cpu(), cropped.cpu(), recon.flux.cpu(), batch.bands)
        except StopIteration:
            print("[val] loader is empty; skipping quick validation.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    codec.save_pretrained(out_dir, modality=EuclidImage)
    print(f"Saved finetuned codec to {out_dir}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(training_losses) + 1), training_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Average MSE")
        plt.title("Euclid codec training loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        curve_path = out_dir / "training_loss.png"
        plt.savefig(curve_path, dpi=150)
        print(f"Saved training curve to {curve_path}")

        if viz_payload is not None:
            orig, cropped, recon_flux, bands = viz_payload
            sample_idx = 0
            num_bands = orig.shape[1]
            fig, axes = plt.subplots(3, num_bands, figsize=(4 * num_bands, 9))
            titles = ["Original (resized)", "Cropped input", "Reconstructed"]
            for b in range(num_bands):
                images = [orig[sample_idx, b].numpy(), cropped[sample_idx, b].numpy(), recon_flux[sample_idx, b].numpy()]
                for row_idx, img in enumerate(images):
                    ax = axes[row_idx, b]
                    ax.imshow(img, cmap="viridis")
                    if b == 0:
                        ax.set_ylabel(titles[row_idx])
                    ax.set_title(bands[b])
                    ax.axis("off")
            plt.tight_layout()
            viz_path = Path(args.save_viz) if args.save_viz else out_dir / "recon_grid.png"
            viz_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(viz_path, dpi=150)
            print(f"Saved reconstruction grid to {viz_path}")

    except Exception as exc:  # pragma: no cover
        print(f"Could not save training curve: {exc}")


if __name__ == "__main__":
    main()
