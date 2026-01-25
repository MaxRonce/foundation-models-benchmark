"""
2026-01-23
aion/model.py


Description
-----------
AION model components: U-Net adapters and codec loading utilities.

Usage
-----
from fmb.models.aion.model import load_aion_components, EuclidToHSC, HSCToEuclid

euclid_to_hsc, hsc_to_euclid, codec = load_aion_components(device='cuda')
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
import safetensors.torch as st
from huggingface_hub import hf_hub_download


# U-Net Building Blocks
class DoubleConv(nn.Module):
    """
    Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU.
    
    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    hidden_dim : Optional[int]
        Hidden dimension (defaults to out_ch).
    """
    
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
    """Downsampling block: MaxPool -> DoubleConv."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )
    
    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling block with skip connections.
    
    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    bilinear : bool
        Use bilinear upsampling instead of transposed convolution.
    """
    
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
        
        # Pad x1 to match x2 size
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution: 1x1 conv to desired number of channels."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    """
    Simple U-Net architecture for image-to-image translation.
    
    Parameters
    ----------
    n_channels : int
        Number of input channels.
    n_classes : int
        Number of output channels.
    hidden : int
        Base hidden dimension (multiplied at each level).
    use_checkpointing : bool
        Enable gradient checkpointing to save memory.
    """
    
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
    """Euclid (4 bands) → HSC (5 bands) adapter."""
    
    def __init__(self, hidden: int, use_checkpointing: bool):
        super().__init__(
            n_channels=4,
            n_classes=5,
            hidden=hidden,
            use_checkpointing=use_checkpointing,
        )


class HSCToEuclid(SimpleUNet):
    """HSC (5 bands) → Euclid (4 bands) adapter."""
    
    def __init__(self, hidden: int, use_checkpointing: bool):
        super().__init__(
            n_channels=5,
            n_classes=4,
            hidden=hidden,
            use_checkpointing=use_checkpointing,
        )


def load_frozen_codec(device: torch.device):
    """
    Load frozen AION ImageCodec from HuggingFace Hub.
    
    Parameters
    ----------
    device : torch.device
        Device to load the codec on.
    
    Returns
    -------
    codec : ImageCodec
        Frozen codec model.
    codec_cfg : dict
        Codec configuration dictionary.
    """
    # Import here to avoid circular dependencies
    try:
        from aion.codecs import ImageCodec
        from aion.codecs.config import HF_REPO_ID
        from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX
    except ImportError as e:
        raise ImportError(
            "AION not found. Make sure external/AION is initialized:\n"
            "  git submodule update --init --recursive"
        ) from e
    
    # Download config and weights
    cfg_path = hf_hub_download(
        HF_REPO_ID,
        "codecs/image/config.json",
        local_files_only=False,
    )
    weights_path = hf_hub_download(
        HF_REPO_ID,
        "codecs/image/model.safetensors",
        local_files_only=False,
    )
    
    with open(cfg_path) as f:
        codec_cfg = json.load(f)
    
    print(f" Loading codec with quantizer_levels={codec_cfg['quantizer_levels']}")
    
    # Temporarily remove Euclid bands from registry
    original_bands = dict(BAND_TO_INDEX)
    try:
        keys_to_remove = [k for k in list(BAND_TO_INDEX.keys()) if "EUCLID" in k]
        for k in keys_to_remove:
            del BAND_TO_INDEX[k]
        
        # Create codec
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
        # Restore original bands
        BAND_TO_INDEX.clear()
        BAND_TO_INDEX.update(original_bands)
    
    # Load weights
    state = st.load_file(weights_path, device="cpu")
    missing, unexpected = codec.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  Codec load: missing={len(missing)}, unexpected={len(unexpected)}")
    
    # Freeze codec
    for p in codec.parameters():
        p.requires_grad = False
    codec.eval()
    
    print(" Codec loaded and frozen")
    return codec, codec_cfg


def load_aion_components(
    device: torch.device,
    hidden: int = 16,
    use_checkpointing: bool = False,
) -> Tuple[EuclidToHSC, HSCToEuclid, nn.Module]:
    """
    Load all AION components: adapters and frozen codec.
    
    Parameters
    ----------
    device : torch.device
        Device to load models on.
    hidden : int
        Hidden dimension for U-Net adapters.
    use_checkpointing : bool
        Enable gradient checkpointing in adapters.
    
    Returns
    -------
    euclid_to_hsc : EuclidToHSC
        Euclid → HSC adapter.
    hsc_to_euclid : HSCToEuclid
        HSC → Euclid adapter.
    codec : ImageCodec
        Frozen AION codec.
    """
    codec, _ = load_frozen_codec(device)
    
    euclid_to_hsc = EuclidToHSC(
        hidden=hidden,
        use_checkpointing=use_checkpointing,
    ).to(device)
    
    hsc_to_euclid = HSCToEuclid(
        hidden=hidden,
        use_checkpointing=use_checkpointing,
    ).to(device)
    
    return euclid_to_hsc, hsc_to_euclid, codec
