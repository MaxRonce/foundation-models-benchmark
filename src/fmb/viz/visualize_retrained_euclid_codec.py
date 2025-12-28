#!/usr/bin/env python3
"""
Load a retrained Euclid ImageCodec, run encode→decode on a batch of samples, and
save grids of original vs reconstructed Euclid bands.

Example:
  python -m scratch.visualize_retrained_euclid_codec \
    --codec-dir outputs/retrained_euclid_codec_418738 \
    --cache-dir /n03data/ronceray/datasets \
    --split train \
    --max-entries 100 \
    --num-samples 10 \
    --resize 160 \
    --crop-size 96 \
    --output outputs/retrained_euclid_codec_418738/viz_grid.png
"""
import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import safetensors.torch as st

from scratch.load_display_data import EuclidDESIDataset
from aion.codecs import ImageCodec
from aion.codecs.config import HF_REPO_ID
from aion.codecs.preprocessing.image import CenterCrop
from aion.modalities import EuclidImage
from huggingface_hub import hf_hub_download

BANDS = ["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"]


class EuclidImageDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, cache_dir: str, max_entries: Optional[int], resize: int):
        self.base = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        self.max_entries = max_entries if max_entries and max_entries > 0 else None
        self.resize = resize
        self._indices = list(range(len(self.base))) if self.max_entries is None else list(
            range(min(len(self.base), self.max_entries)),
        )

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx: int) -> EuclidImage:
        base_idx = self._indices[idx]
        sample = self.base[base_idx]
        bands: List[torch.Tensor] = []
        for key in ("vis_image", "nisp_y_image", "nisp_j_image", "nisp_h_image"):
            tensor = sample.get(key)
            if tensor is None:
                raise ValueError(f"Missing band '{key}' at index {base_idx}")
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 2:
                raise ValueError(f"Expected band '{key}' to be 2D, got shape {tuple(tensor.shape)}")
            bands.append(tensor)
        flux = torch.stack(bands, dim=0)
        if self.resize and (flux.shape[-1] != self.resize or flux.shape[-2] != self.resize):
            flux = F.interpolate(
                flux.unsqueeze(0),
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return EuclidImage(flux=flux, bands=BANDS)


def collate_euclid(batch: List[EuclidImage]) -> EuclidImage:
    flux = torch.stack([item.flux for item in batch], dim=0)
    return EuclidImage(flux=flux, bands=batch[0].bands)


def load_codec(codec_dir: Path, device: torch.device) -> ImageCodec:
    cfg_path = codec_dir / "codecs" / EuclidImage.name / "config.json"
    weights_path = codec_dir / "codecs" / EuclidImage.name / "model.safetensors"
    if not cfg_path.exists() or not weights_path.exists():
        # fallback to base HF config if local modality folder missing
        cfg_path = Path(hf_hub_download(HF_REPO_ID, "codecs/image/config.json", local_files_only=True))
        weights_path = Path(hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=True))
    cfg = torch.load(cfg_path, map_location="cpu") if cfg_path.suffix == ".pt" else None
    if cfg is None:
        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
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
    codec.load_state_dict(state, strict=False)
    codec.eval()
    return codec


def plot_grid(orig: torch.Tensor, recon: torch.Tensor, bands: List[str], out_path: Path) -> None:
    num_samples = orig.shape[0]
    num_bands = orig.shape[1]
    rows = 2 * num_samples  # original and reconstructed rows per sample
    fig, axes = plt.subplots(rows, num_bands, figsize=(4 * num_bands, 3 * rows))
    for s in range(num_samples):
        for b in range(num_bands):
            for r, data, title_prefix in [
                (0, orig[s, b].cpu().numpy(), "Original"),
                (1, recon[s, b].cpu().numpy(), "Reconstructed"),
            ]:
                ax = axes[2 * s + r, b]
                ax.imshow(data, cmap="viridis")
                if s == 0:
                    ax.set_title(bands[b])
                if b == 0:
                    ax.set_ylabel(title_prefix)
                ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize reconstructions from a retrained Euclid ImageCodec.")
    p.add_argument("--codec-dir", required=True, help="Path to retrained codec directory (e.g., outputs/retrained_euclid_codec_XXXX).")
    p.add_argument("--cache-dir", type=str, default="/n03data/ronceray/datasets", help="Dataset cache dir.")
    p.add_argument("--split", type=str, default="train", help="Dataset split.")
    p.add_argument("--max-entries", type=int, default=200, help="Limit dataset size for speed.")
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize.")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    p.add_argument("--resize", type=int, default=160, help="Resize before encode.")
    p.add_argument("--crop-size", type=int, default=96, help="Center-crop size inside codec.")
    p.add_argument("--output", type=str, default="outputs/recon_grid.png", help="Where to save the grid image.")
    p.add_argument("--max-abs", type=float, default=1e3, help="Clamp flux magnitude before encode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EuclidImageDataset(args.split, args.cache_dir, args.max_entries, args.resize)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_euclid,
    )
    codec = load_codec(Path(args.codec_dir), device)
    crop = CenterCrop(crop_size=args.crop_size)

    collected_orig = []
    collected_recon = []
    needed = args.num_samples

    with torch.no_grad():
        for batch in loader:
            flux = batch.flux.to(device)
            flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
            if args.max_abs and args.max_abs > 0:
                flux = torch.clamp(flux, min=-args.max_abs, max=args.max_abs)
            cropped = crop(flux)
            euclid_cropped = EuclidImage(flux=cropped, bands=batch.bands)
            tokens = codec.encode(euclid_cropped)
            recon = codec.decode(tokens, bands=batch.bands)
            collected_orig.append(flux.cpu())
            collected_recon.append(recon.flux.cpu())
            if sum(x.shape[0] for x in collected_orig) >= needed:
                break

    orig = torch.cat(collected_orig, dim=0)[:needed]
    recon_flux = torch.cat(collected_recon, dim=0)[:needed]
    plot_grid(orig, recon_flux, batch.bands, Path(args.output))
    print(f"Saved reconstruction grid to {args.output}")


if __name__ == "__main__":
    main()
