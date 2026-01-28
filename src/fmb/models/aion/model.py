"""
Foundation Models Benchmark (FMB)

Module: fmb.models.aion.model
Description: AION multimodal foundation model
"""

import json
from typing import List, Optional, Tuple

import safetensors.torch as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from fmb.data.load_display_data import EuclidDESIDataset
from fmb.models.aion.modalities import EuclidImage

# Constants
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
    """(convolution => [BN] => ReLU) * 2"""

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
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscaling then double conv"""

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
        # input is CHW
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    """Small U-Net for domain adaptation."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden: int,
        use_checkpointing: bool = False,
    ):
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
    """Adapter to translate Euclid (4 bands) into HSC-like (5 bands)."""

    def __init__(self, hidden: int, use_checkpointing: bool):
        super().__init__(
            n_channels=4,
            n_classes=5,
            hidden=hidden,
            use_checkpointing=use_checkpointing,
        )


class HSCToEuclid(SimpleUNet):
    """Adapter to translate HSC (5 bands) back to Euclid (4 bands)."""

    def __init__(self, hidden: int, use_checkpointing: bool):
        super().__init__(
            n_channels=5,
            n_classes=4,
            hidden=hidden,
            use_checkpointing=use_checkpointing,
        )


# -----------------------------
# Dataset
# -----------------------------


class EuclidImageDataset(Dataset):
    """Dataset wrapper for Euclid images from DESI."""

    def __init__(
        self, split: str, cache_dir: str, max_entries: Optional[int], resize: int
    ):
        self.base = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        self.resize = resize
        self._indices = (
            list(range(len(self.base)))
            if not max_entries or max_entries <= 0
            else list(range(min(len(self.base), max_entries)))
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

        if self.resize and (
            flux.shape[-1] != self.resize or flux.shape[-2] != self.resize
        ):
            flux = F.interpolate(
                flux.unsqueeze(0),
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return EuclidImage(flux=flux, bands=EUCLID_BANDS)


def collate_euclid(batch: List[EuclidImage]) -> EuclidImage:
    """Collate function for EuclidImage batches."""
    flux = torch.stack([b.flux for b in batch], dim=0)
    return EuclidImage(flux=flux, bands=batch[0].bands)


# -----------------------------
# Helpers
# -----------------------------


def load_frozen_codec(device: torch.device) -> Tuple[nn.Module, dict]:
    """Load frozen AION ImageCodec from HF Hub."""
    # Import here to avoid circular dependency
    from aion.codecs import ImageCodec
    from aion.codecs.config import HF_REPO_ID

    cfg_path = hf_hub_download(
        HF_REPO_ID, "codecs/image/config.json", local_files_only=True
    )
    weights_path = hf_hub_download(
        HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=True
    )

    with open(cfg_path) as f:
        codec_cfg = json.load(f)

    print(f"Loading codec with quantizer_levels={codec_cfg['quantizer_levels']}")

    # Patch band registry to avoid collisions with Euclid bands during codec init
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
        print(
            f"[info] Codec load: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    for p in codec.parameters():
        p.requires_grad = False
    codec.eval()
    return codec, codec_cfg


def load_aion_components(
    device: torch.device,
    hidden: int = 16,
    use_checkpointing: bool = False,
) -> Tuple[EuclidToHSC, HSCToEuclid, nn.Module]:
    """Load adapters and codec."""
    codec, _ = load_frozen_codec(device)
    euclid_to_hsc = EuclidToHSC(hidden=hidden, use_checkpointing=use_checkpointing).to(
        device
    )
    hsc_to_euclid = HSCToEuclid(hidden=hidden, use_checkpointing=use_checkpointing).to(
        device
    )
    return euclid_to_hsc, hsc_to_euclid, codec
