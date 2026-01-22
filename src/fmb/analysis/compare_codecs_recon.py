#!/usr/bin/env python3
"""
Compare reconstructions from a retrained Euclid codec vs the base (unretrained) codec.

Outputs a grid with rows: original, retrained recon, base recon, |retrained - base|,
and prints summary metrics (MSE to original, L1 between reconstructions).

Example:
  python -m scratch.compare_codecs_recon \
    --retrained-codec outputs/retrained_euclid_codec_418738 \
    --cache-dir /n03data/ronceray/datasets \
    --split train \
    --max-entries 200 \
    --num-samples 10 \
    --batch-size 8 \
    --output outputs/retrained_euclid_codec_418738/compare_grid.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from scratch.load_display_data import EuclidDESIDataset
from aion.codecs import ImageCodec
from aion.codecs.config import HF_REPO_ID
from aion.codecs.preprocessing.image import CenterCrop
from aion.modalities import EuclidImage

BANDS = ["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"]


class EuclidImageDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, cache_dir: str, max_entries: Optional[int], resize: int):
        self.base = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        self.max_entries = max_entries if max_entries and max_entries > 0 else None
        self.resize = resize
        self._indices = list(range(len(self.base))) if self.max_entries is None else list(
            range(min(len(self.base), self.max_entries)),
        )

    def __len__(self) -> int:
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


def _load_config(path: Path) -> dict:
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    with open(path) as f:
        return json.load(f)


def _pad_state_for_euclid(state: dict, codec: ImageCodec) -> dict:
    """Pad subsample layers to accept extra Euclid channels if needed."""
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
        cfg_path = Path(hf_hub_download(HF_REPO_ID, "codecs/image/config.json", local_files_only=True))
        weights_path = Path(hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=True))
    else:
        cfg_path = codec_dir / "codecs" / EuclidImage.name / "config.json"
        weights_path = codec_dir / "codecs" / EuclidImage.name / "model.safetensors"
        if not cfg_path.exists() or not weights_path.exists():
            cfg_path = Path(hf_hub_download(HF_REPO_ID, "codecs/image/config.json", local_files_only=True))
            weights_path = Path(hf_hub_download(HF_REPO_ID, "codecs/image/model.safetensors", local_files_only=True))

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
    state = _pad_state_for_euclid(state, codec)
    codec.load_state_dict(state, strict=False)
    codec.eval()
    return codec


def plot_grid(orig: torch.Tensor, recon_ret: torch.Tensor, recon_base: torch.Tensor, bands: List[str], out_path: Path) -> None:
    num_samples = orig.shape[0]
    num_bands = orig.shape[1]
    rows = 4 * num_samples  # original, retrained, base, diff
    fig, axes = plt.subplots(rows, num_bands, figsize=(4 * num_bands, 3 * rows))
    for s in range(num_samples):
        for b in range(num_bands):
            cells: List[Tuple[np.ndarray, str]] = [
                (orig[s, b].cpu().numpy(), "Original"),
                (recon_ret[s, b].cpu().numpy(), "Retrained"),
                (recon_base[s, b].cpu().numpy(), "Base"),
                (np.abs(recon_ret[s, b].cpu().numpy() - recon_base[s, b].cpu().numpy()), "|Ret-Base|"),
            ]
            for r, (img, label) in enumerate(cells):
                ax = axes[4 * s + r, b]
                ax.imshow(img, cmap="viridis")
                if s == 0:
                    ax.set_title(bands[b])
                if b == 0:
                    ax.set_ylabel(label)
                ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare retrained vs base Euclid ImageCodec reconstructions.")
    p.add_argument("--retrained-codec", required=True, help="Path to retrained codec directory.")
    p.add_argument("--cache-dir", type=str, default="/n03data/ronceray/datasets", help="Dataset cache dir.")
    p.add_argument("--split", type=str, default="train", help="Dataset split.")
    p.add_argument("--max-entries", type=int, default=200, help="Limit dataset size for speed.")
    p.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize.")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    p.add_argument("--resize", type=int, default=160, help="Resize before encode.")
    p.add_argument("--crop-size", type=int, default=96, help="Center-crop size inside codec.")
    p.add_argument("--output", type=str, default="outputs/compare_grid.png", help="Where to save the grid image.")
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

    retrained_codec = load_codec(Path(args.retrained_codec), device)
    base_codec = load_codec(None, device)
    crop = CenterCrop(crop_size=args.crop_size)

    orig_batches = []
    ret_batches = []
    base_batches = []
    needed = args.num_samples

    with torch.no_grad():
        for batch in loader:
            flux = batch.flux.to(device)
            flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
            if args.max_abs and args.max_abs > 0:
                flux = torch.clamp(flux, min=-args.max_abs, max=args.max_abs)
            cropped = crop(flux)
            euclid_cropped = EuclidImage(flux=cropped, bands=batch.bands)

            tok_ret = retrained_codec.encode(euclid_cropped)
            recon_ret = retrained_codec.decode(tok_ret, bands=batch.bands).flux

            tok_base = base_codec.encode(euclid_cropped)
            recon_base = base_codec.decode(tok_base, bands=batch.bands).flux

            orig_batches.append(flux.cpu())
            ret_batches.append(recon_ret.cpu())
            base_batches.append(recon_base.cpu())

            if sum(x.shape[0] for x in orig_batches) >= needed:
                break

    orig = torch.cat(orig_batches, dim=0)[:needed]
    recon_ret = torch.cat(ret_batches, dim=0)[:needed]
    recon_base = torch.cat(base_batches, dim=0)[:needed]

    # Metrics
    mse_ret = torch.mean((recon_ret - crop(orig.to(device)).cpu()) ** 2).item()
    mse_base = torch.mean((recon_base - crop(orig.to(device)).cpu()) ** 2).item()
    l1_between = torch.mean(torch.abs(recon_ret - recon_base)).item()
    print(f"[metrics] MSE(retrained vs cropped orig): {mse_ret:.6f}")
    print(f"[metrics] MSE(base vs cropped orig):      {mse_base:.6f}")
    print(f"[metrics] L1(retrained vs base):           {l1_between:.6f}")

    plot_grid(orig, recon_ret, recon_base, batch.bands, Path(args.output))
    print(f"Saved comparison grid to {args.output}")


if __name__ == "__main__":
    main()
