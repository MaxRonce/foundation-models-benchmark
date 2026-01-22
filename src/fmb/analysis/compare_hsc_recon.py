#!/usr/bin/env python3
"""
Compare reconstructions from a retrained Euclid codec vs the base (unretrained) codec using HSC data.

Features:
- Processes multiple HSC images.
- Uses native HSC support for both Base (multi-survey) and Retrained codecs.
- Visualization rows:
  1. Original HSC
  2. Base Recon (HSC)
  3. Retrained Recon (HSC)
  4. (Optional) Euclid Mapped Recon (HSC->Euclid->Retrained->Euclid)
  5. |Base - Original|
  6. |Retrained - Original|

Example:
  python3 -m scratch.compare_hsc_recon \
    --retrained-codec outputs/retrained_euclid_codec_418738 \
    --hsc-cache-dir ./hsc_cache \
    --num-samples 4 \
    --output outputs/hsc_recon_comparison.png \
    --as-euclid
"""


import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from scratch.load_display_data_hsc import HSCDataset
from aion.codecs import ImageCodec
from aion.codecs.config import HF_REPO_ID
from aion.codecs.preprocessing.image import CenterCrop, RescaleToLegacySurvey
from aion.modalities import HSCImage, EuclidImage

# Monkey-patch to fix typo and TypeError in aion library
def _fixed_reverse_zeropoint(self, scale):
    return 22.5 - 2.5 * math.log10(scale)

if not hasattr(RescaleToLegacySurvey, "_reverse_zeropoint"):
    RescaleToLegacySurvey._reverse_zeropoint = _fixed_reverse_zeropoint
RescaleToLegacySurvey.reverse_zeropoint = _fixed_reverse_zeropoint

# HSC Bands: G, R, I, Z, Y
HSC_BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
# Euclid Bands: VIS, Y, J, H
EUCLID_BANDS = ["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"]

# Mapping indices for HSC -> Euclid
# HSC: 0=G, 1=R, 2=I, 3=Z, 4=Y
# Euclid: 0=VIS, 1=Y, 2=J, 3=H
HSC_TO_EUCLID_INDICES = [0, 1, 2, 3]


class HSCImageDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, cache_dir: str, max_entries: Optional[int], resize: int):
        self.base = HSCDataset(split=split, cache_dir=cache_dir, streaming=True, max_items=max_entries or 500)
        self.max_entries = max_entries
        self.resize = resize
        self._indices = range(len(self.base))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        target_bands = ["hsc_g", "hsc_r", "hsc_i", "hsc_z", "hsc_y"]
        bands_data = []
        
        for key in target_bands:
            tensor = sample.get(key)
            if tensor is None:
                tensor = torch.zeros((1, 1))
            if not isinstance(tensor, torch.Tensor):
                 tensor = torch.as_tensor(tensor, dtype=torch.float32)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            bands_data.append(tensor)

        resized_bands = []
        for b in bands_data:
            if b.ndim < 2:
                b = torch.zeros((self.resize, self.resize), dtype=torch.float32)
            if b.shape[-1] != self.resize or b.shape[-2] != self.resize:
                b = F.interpolate(
                    b.unsqueeze(0).unsqueeze(0),
                    size=(self.resize, self.resize),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            resized_bands.append(b)
            
        flux = torch.stack(resized_bands, dim=0)
        return HSCImage(flux=flux, bands=HSC_BANDS)


def collate_fn(batch):
    flux = torch.stack([item.flux for item in batch], dim=0)
    return HSCImage(flux=flux, bands=batch[0].bands)


def _load_config(path: Path) -> dict:
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    with open(path) as f:
        return json.load(f)


def _pad_state(state: dict, codec: ImageCodec) -> dict:
    """Pad subsample layers to accept extra channels if needed."""
    def _pad_param(name: str, target):
        tensor = state.get(name)
        if tensor is None:
            return
        if tuple(tensor.shape) == tuple(target.shape):
            return
        new_tensor = torch.zeros_like(target)
        common_slices = tuple(slice(0, min(a, b)) for a, b in zip(new_tensor.shape, tensor.shape))
        new_tensor[common_slices] = tensor[common_slices]
        state[name] = new_tensor

    _pad_param("subsample_in.weight", codec.subsample_in.weight)
    _pad_param("subsample_out.weight", codec.subsample_out.weight)
    _pad_param("subsample_out.bias", codec.subsample_out.bias)
    return state


def load_codec(codec_dir: Optional[Path], device: torch.device) -> ImageCodec:
    if codec_dir is None:
        # Base codec (from HF)
        cfg_path = Path(hf_hub_download(HF_REPO_ID, "codecs/image/config.json", local_files_only=True))
        weights_path = Path(hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=True))
    else:
        # Retrained codec
        # Try HSCImage first, then EuclidImage, then root
        cfg_path = codec_dir / "codecs" / HSCImage.name / "config.json"
        weights_path = codec_dir / "codecs" / HSCImage.name / "model.safetensors"
        
        if not cfg_path.exists():
            cfg_path = codec_dir / "codecs" / EuclidImage.name / "config.json"
            weights_path = codec_dir / "codecs" / EuclidImage.name / "model.safetensors"
        
        if not cfg_path.exists():
             cfg_path = codec_dir / "config.json"
             weights_path = codec_dir / "model.safetensors"
             
        if not cfg_path.exists():
             raise FileNotFoundError(f"Could not find config.json in {codec_dir}")

    cfg = _load_config(cfg_path)

    codec = ImageCodec(
        quantizer_levels=cfg["quantizer_levels"],
        hidden_dims=cfg["hidden_dims"],
        multisurvey_projection_dims=cfg["multisurvey_projection_dims"],
        n_compressions=cfg["n_compressions"],
        num_consecutive=cfg["num_consecutive"],
        embedding_dim=cfg["embedding_dim"],
        range_compression_factor=cfg["range_compression_factor"],
        mult_factor=cfg["mult_factor"],
    ).to(device)
    
    state = st.load_file(weights_path, device="cpu")
    # Pad weights if necessary (e.g. loading 9-band checkpoint into 13-band model)
    state = _pad_state(state, codec)
    codec.load_state_dict(state, strict=False)
    codec.eval()
    
    return codec


def plot_grid(samples: List[Dict[str, np.ndarray]], bands: List[str], out_path: Path) -> None:
    """
    samples: List of dicts, each dict contains images for a row type.
             Keys: 'Original', 'Base', 'Retrained', 'EuclidMapped', 'DiffBase', 'DiffRetrained'
    """
    num_samples = len(samples)
    num_bands = len(bands)
    
    # Determine rows per sample
    sample_keys = list(samples[0].keys())
    rows_per_sample = len(sample_keys)
    total_rows = num_samples * rows_per_sample
    
    fig, axes = plt.subplots(total_rows, num_bands, figsize=(3 * num_bands, 3 * total_rows))
    if total_rows == 1 and num_bands == 1: axes = np.array([[axes]])
    elif total_rows == 1: axes = axes[None, :]
    elif num_bands == 1: axes = axes[:, None]
    
    for s in range(num_samples):
        sample_data = samples[s]
        for r, key in enumerate(sample_keys):
            row_idx = s * rows_per_sample + r
            img_row = sample_data[key] # Shape (bands, H, W)
            
            for b in range(num_bands):
                ax = axes[row_idx, b]
                
                # Handle missing bands (e.g. EuclidMapped might have 4 bands, HSC has 5)
                if b < img_row.shape[0]:
                    img = img_row[b]
                    im = ax.imshow(img, cmap="viridis")
                else:
                    ax.axis("off")
                    continue
                
                if s == 0 and r == 0:
                    ax.set_title(bands[b])
                if b == 0:
                    # Row label
                    label = key
                    if s > 0 and r == 0:
                        label = f"Sample {s+1}\n{label}"
                    elif r == 0:
                        label = f"Sample {s+1}\n{label}"
                    ax.set_ylabel(label, rotation=90, size='large')
                
                ax.set_xticks([])
                ax.set_yticks([])
                
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare retrained vs base Euclid ImageCodec reconstructions on HSC data.")
    p.add_argument("--retrained-codec", required=True, help="Path to retrained codec directory.")
    p.add_argument("--hsc-cache-dir", type=str, default="./hsc_cache", help="HSC Dataset cache dir.")
    p.add_argument("--split", type=str, default="train", help="Dataset split.")
    p.add_argument("--max-entries", type=int, default=500, help="Limit dataset size.")
    p.add_argument("--num-samples", type=int, default=4, help="Number of samples to visualize.")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    p.add_argument("--resize", type=int, default=160, help="Resize before encode.")
    p.add_argument("--crop-size", type=int, default=96, help="Center-crop size inside codec.")
    p.add_argument("--output", type=str, default="outputs/hsc_recon_comparison.png", help="Output image path.")
    p.add_argument("--as-euclid", action="store_true", help="Add a row with HSC mapped to Euclid bands.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("Loading HSC dataset...")
    dataset = HSCImageDataset(args.split, args.hsc_cache_dir, args.max_entries, args.resize)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    # 2. Load Codecs
    print("Loading codecs...")
    retrained_codec = load_codec(Path(args.retrained_codec), device)
    base_codec = load_codec(None, device)

    crop = CenterCrop(crop_size=args.crop_size)

    collected_samples = []
    needed = args.num_samples

    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            flux = batch.flux.to(device)
            flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
            cropped_flux = crop(flux)
            input_hsc = HSCImage(flux=cropped_flux, bands=batch.bands)

            # 1. Base Recon (HSC)
            tok_base = base_codec.encode(input_hsc)
            recon_base = base_codec.decode(tok_base, bands=batch.bands).flux

            # 2. Retrained Recon (HSC)
            tok_ret = retrained_codec.encode(input_hsc)
            recon_ret = retrained_codec.decode(tok_ret, bands=batch.bands).flux

            # 3. Euclid Mapped (Optional)
            recon_euclid_mapped = None
            if args.as_euclid:
                # Map HSC -> Euclid
                input_euclid_flux = cropped_flux[:, HSC_TO_EUCLID_INDICES, :, :]
                input_euclid = EuclidImage(flux=input_euclid_flux, bands=EUCLID_BANDS)
                
                # Encode/Decode with Retrained (as debug)
                tok_euclid = retrained_codec.encode(input_euclid)
                recon_euclid = retrained_codec.decode(tok_euclid, bands=EUCLID_BANDS).flux
                
                # We have 4 bands. To align with 5 HSC bands for plotting:
                # Map VIS->G, Y->R, J->I, H->Z, Zero->Y
                # Create 5-band tensor
                B, _, H, W = recon_euclid.shape
                recon_euclid_mapped = torch.zeros((B, 5, H, W), device=device)
                recon_euclid_mapped[:, 0:4, :, :] = recon_euclid
                # 5th band remains zero

            # Collect samples
            B = cropped_flux.shape[0]
            for i in range(B):
                if len(collected_samples) >= needed:
                    break
                
                sample_dict = {}
                # Original
                sample_dict["Original"] = cropped_flux[i].cpu().numpy()
                
                # Base
                sample_dict["Base"] = recon_base[i].cpu().numpy()
                
                # Retrained
                sample_dict["Retrained"] = recon_ret[i].cpu().numpy()
                
                # Euclid Mapped
                if recon_euclid_mapped is not None:
                    sample_dict["EuclidMapped"] = recon_euclid_mapped[i].cpu().numpy()
                
                # Diffs
                sample_dict["|Base-Orig|"] = np.abs(sample_dict["Base"] - sample_dict["Original"])
                sample_dict["|Ret-Orig|"] = np.abs(sample_dict["Retrained"] - sample_dict["Original"])
                
                collected_samples.append(sample_dict)

            if len(collected_samples) >= needed:
                break

    print(f"Collected {len(collected_samples)} samples.")
    
    # Plot
    plot_grid(collected_samples, HSC_BANDS, Path(args.output))
    print(f"Saved comparison grid to {args.output}")


if __name__ == "__main__":
    main()
