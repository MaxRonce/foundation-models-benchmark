#!/usr/bin/env python3
"""
Recovery script to generate sample_grid.png from a trained adapter checkpoint.
Standalone version (does not import from broken training script).
"""

import argparse
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
import matplotlib.pyplot as plt
import safetensors.torch as st
from pathlib import Path
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from scratch.load_display_data import EuclidDESIDataset
from aion.codecs import ImageCodec
from aion.codecs.config import HF_REPO_ID
from aion.codecs.preprocessing.image import CenterCrop
from aion.modalities import EuclidImage, HSCImage

# -----------------------------
# Constants & Config
# -----------------------------
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
    """(conv => BN => ReLU) * 2"""
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
    """Small U-Net with optional checkpointing on blocks."""
    def __init__(self, n_channels: int, n_classes: int, hidden: int = 32, bilinear: bool = True, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.inc = DoubleConv(n_channels, hidden)
        self.down1 = Down(hidden, hidden * 2)
        self.down2 = Down(hidden * 2, hidden * 4)
        self.up1 = Up(hidden * 4, hidden * 2, bilinear)
        self.up2 = Up(hidden * 2, hidden, bilinear)
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
    """Wrap EuclidDESIDataset to emit EuclidImage objects (flux in nanomaggies)."""
    def __init__(self, split: str, cache_dir: str, max_entries: Optional[int], resize: int):
        self.base = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        self.resize = resize
        self._indices = list(range(len(self.base))) if not max_entries or max_entries <= 0 else list(range(min(len(self.base), max_entries)))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> EuclidImage:
        base_idx = self._indices[idx]
        sample = self.base[base_idx]

        keys = ["vis_image", "nisp_y_image", "nisp_j_image", "nisp_h_image"]
        bands = []
        for key in keys:
            tensor = sample.get(key)
            if tensor is None:
                raise ValueError(f"Missing band '{key}' at index {base_idx}")
            tensor = tensor.to(torch.float32)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

            # ADU -> nanomaggies (Legacy Survey scale)
            zp_nu = EUCLID_ZP_NU[key]
            scale_factor = zp_nu / 3631.0
            tensor = tensor * scale_factor

            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 2:
                raise ValueError(f"Expected band '{key}' to be 2D, got {tuple(tensor.shape)}")
            bands.append(tensor)

        flux = torch.stack(bands, dim=0)  # (4, H, W)

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
    """Load pretrained ImageCodec from HF cache and freeze it."""
    cfg_path = hf_hub_download(HF_REPO_ID, "codecs/image/config.json", local_files_only=True)
    weights_path = hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=True)

    with open(cfg_path) as f:
        codec_cfg = json.load(f)

    from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX

    original_bands = dict(BAND_TO_INDEX)
    try:
        # Remove Euclid bands so codec matches checkpoint input
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


# -----------------------------
# Main Recovery Logic
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate sample_grid.png from checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to adapters_final.pt or epoch checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save png. Defaults to ckpt parent dir.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to visualize from")
    args_cli = parser.parse_args()

    ckpt_path = Path(args_cli.ckpt)
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        sys.exit(1)

    out_dir = Path(args_cli.output_dir) if args_cli.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args_cli.device)

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Extract training args to rebuild model with correct hyperparameters
    train_args = ckpt.get("args", {})
    if not train_args:
        print("Warning: 'args' not found in checkpoint. Assuming defaults (hidden=16, checkpointing=False).")
        hidden = 16
        use_ckpt = False
        resize = 96
        crop_size = 96
        amp_dtype_str = "float16"
        max_abs = 100.0
    else:
        # train_args is likely a dict if saved via vars(args), or Namespace
        if hasattr(train_args, "hidden"): # Namespace
            hidden = train_args.hidden
            use_ckpt = getattr(train_args, "use_unet_checkpointing", False)
            resize = getattr(train_args, "resize", 96)
            crop_size = getattr(train_args, "crop_size", 96)
            amp_dtype_str = getattr(train_args, "amp_dtype", "float16")
            max_abs = getattr(train_args, "max_abs", 100.0)
            cache_dir = getattr(train_args, "cache_dir", "/scratch")
        else: # dict
            hidden = train_args.get("hidden", 16)
            use_ckpt = train_args.get("use_unet_checkpointing", False)
            resize = train_args.get("resize", 96)
            crop_size = train_args.get("crop_size", 96)
            amp_dtype_str = train_args.get("amp_dtype", "float16")
            max_abs = train_args.get("max_abs", 100.0)
            cache_dir = train_args.get("cache_dir", "/scratch")

    print(f"Model config: hidden={hidden}, use_checkpointing={use_ckpt}")
    print(f"Data config: resize={resize}, crop_size={crop_size}, max_abs={max_abs}")

    # Load Codec
    print("Loading frozen codec...")
    codec, _ = load_frozen_codec(device)
    codec.eval()

    # Load Adapter Models
    euclid_to_hsc = EuclidToHSC(hidden=hidden, use_checkpointing=use_ckpt).to(device)
    hsc_to_euclid = HSCToEuclid(hidden=hidden, use_checkpointing=use_ckpt).to(device)

    euclid_to_hsc.load_state_dict(ckpt["euclid_to_hsc"])
    hsc_to_euclid.load_state_dict(ckpt["hsc_to_euclid"])

    euclid_to_hsc.eval()
    hsc_to_euclid.eval()

    # Setup Dataset
    print(f"Loading dataset (split={args_cli.split})...")
    dataset = EuclidImageDataset(split=args_cli.split, cache_dir=cache_dir, max_entries=100, resize=resize)
    
    # We want a random sample, but reproducibility might be nice. 
    # For now, just shuffle=True.
    vis_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True, 
        num_workers=0,
        collate_fn=collate_euclid,
    )
    
    center_crop = CenterCrop(crop_size=crop_size)
    amp_dtype = torch.float16 if amp_dtype_str == "float16" else torch.bfloat16

    # Helper for decoding
    def codec_roundtrip_flux(hsc_flux: torch.Tensor) -> torch.Tensor:
        hsc_obj = HSCImage(flux=hsc_flux, bands=HSC_BANDS)
        toks = codec.encode(hsc_obj)
        hsc_rec = codec.decode(toks, bands=HSC_BANDS)
        return hsc_rec.flux

    print("Generating sample...")
    try:
        euclid_img = next(iter(vis_loader))
        x_cpu = euclid_img.flux  # (1,4,H,W)
        x_cpu = torch.nan_to_num(x_cpu, nan=0.0, posinf=0.0, neginf=0.0)
        
        if max_abs > 0:
            x_cpu = torch.clamp(x_cpu, -max_abs, max_abs)
        
        x_cpu = center_crop(x_cpu)

        # Main generation loop
        with torch.no_grad():
            x = x_cpu.to(device)
            # Use autocast for consistency with training
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                hsc_pred = euclid_to_hsc(x)
                hsc_dec = codec_roundtrip_flux(hsc_pred)
                euclid_rec = hsc_to_euclid(hsc_dec)

        # Move to CPU for plotting
        x_np = x_cpu[0].float().cpu().numpy()
        hsc_pred_np = hsc_pred[0].float().cpu().numpy()
        hsc_dec_np = hsc_dec[0].float().cpu().numpy()
        euclid_rec_np = euclid_rec[0].float().cpu().numpy()

        # Plotting
        fig, axes = plt.subplots(4, 5, figsize=(18, 12))
        for ax in axes.flatten():
            ax.axis("off")

        # Row 1: Euclid In (4 bands)
        for i, name in enumerate(EUCLID_BANDS):
            if i < 4:
                axes[0, i].imshow(x_np[i], origin="lower")
                axes[0, i].set_title(f"Euclid in: {name}", fontsize=10)
                axes[0, i].axis("on"); axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

        # Row 2: HSC Pred (5 bands)
        for i, name in enumerate(HSC_BANDS):
            axes[1, i].imshow(hsc_pred_np[i], origin="lower")
            axes[1, i].set_title(f"HSC pred: {name}", fontsize=10)
            axes[1, i].axis("on"); axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

        # Row 3: HSC Decoded (5 bands)
        for i, name in enumerate(HSC_BANDS):
            axes[2, i].imshow(hsc_dec_np[i], origin="lower")
            axes[2, i].set_title(f"HSC dec: {name}", fontsize=10)
            axes[2, i].axis("on"); axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

        # Row 4: Euclid Recon (4 bands)
        for i, name in enumerate(EUCLID_BANDS):
            if i < 4:
                axes[3, i].imshow(euclid_rec_np[i], origin="lower")
                axes[3, i].set_title(f"Euclid rec: {name}", fontsize=10)
                axes[3, i].axis("on"); axes[3, i].set_xticks([]); axes[3, i].set_yticks([])

        plt.tight_layout()
        save_path = out_dir / "sample_grid.png"
        plt.savefig(save_path, dpi=150)
        plt.close("all")
        print(f"[Success] Saved sample_grid.png to {save_path}")

    except Exception as e:
        print(f"[Error] Failed to generate sample grid: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
