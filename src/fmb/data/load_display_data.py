# load_display_data.py
"""
Utility script to load and display samples from the Euclid+DESI dataset.
It provides a PyTorch Dataset class `EuclidDESIDataset` and a function `display_one_sample`
to visualize images, spectra, and SEDs.

Usage:
    python -m scratch.load_display_data --index 5 --show-bands --save outputs/img_5.png
"""

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from random import sample
import sys
import warnings

# Matplotlib defaults to interactive; switch to "Agg" if --no-gui is passed
import matplotlib

from typing import Optional, Sequence

def _maybe_switch_to_agg(no_gui: bool):
    if no_gui:
        matplotlib.use("Agg")
    else:
        # If no display available (clusters/headless), automatically fallback to Agg
        try:
            import tkinter  # noqa: F401
        except Exception:
            matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from datasets import load_dataset, concatenate_datasets, load_from_disk
except ImportError as e:
    raise SystemExit(
        "The 'datasets' package is required. Install it with: pip install datasets"
    ) from e

from torch.utils.data import DataLoader
from pathlib import Path

# Add src to pythonpath
src_path = Path(__file__).resolve().parents[2]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fmb.paths import load_paths

HF_DATASET_ID = "msiudek/astroPT_euclid_Q1_desi_dr1_dataset" # Fallback if not in paths, but paths has default

class EuclidDESIDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for the Euclid+DESI HuggingFace dataset."""
    def __init__(
        self,
        split="train",
        transform=None,
        cache_dir=None,
        verbose: bool = False,
    ):
        import os
        paths = load_paths()
        if cache_dir is None:
            cache_dir = str(paths.dataset)
            
        os.makedirs(cache_dir, exist_ok=True)
        self.verbose = verbose
        self.transform = transform
        
        self.hf_dataset_id = getattr(paths, "dataset_hf_id", HF_DATASET_ID)

        requested_splits: list[str]
        datasets_to_concat: list = []

        local_split_paths = {}
        if paths.dataset_train and paths.dataset_train.exists():
             local_split_paths["train"] = paths.dataset_train
        if paths.dataset_test and paths.dataset_test.exists():
             local_split_paths["test"] = paths.dataset_test

        def _load_split(split_name: str):
            """Load a split from local disk if available, otherwise from HF."""
            if split_name in local_split_paths:
                if self.verbose:
                    print(f"Loading split '{split_name}' from {local_split_paths[split_name]}")
                return load_from_disk(str(local_split_paths[split_name]))
            if self.verbose:
                print(f"Loading split '{split_name}' from HF dataset {self.hf_dataset_id}")
            return load_dataset(
                self.hf_dataset_id,
                split=split_name,
                cache_dir=cache_dir,
            )

        if isinstance(split, str):
            normalized = split.strip()
            if normalized.lower() in {"all", "*"}:
                requested_splits = list(local_split_paths) or ["train", "test"]
            else:
                requested_splits = [part.strip() for part in normalized.split(",") if part.strip()]
                if not requested_splits:
                    raise ValueError("No valid split names provided")
            for split_name in requested_splits:
                try:
                    datasets_to_concat.append(_load_split(split_name))
                except Exception as e:
                    raise RuntimeError(
                        f"Unable to load split '{split_name}' (local or Hub): {e}"
                    ) from e
        elif isinstance(split, Sequence):
            requested_splits = [str(part) for part in split]
            for split_name in requested_splits:
                try:
                    datasets_to_concat.append(_load_split(split_name))
                except Exception as e:
                    raise RuntimeError(
                        f"Unable to load split '{split_name}' (local or Hub): {e}"
                    ) from e
        else:
            raise TypeError("split must be a string, list or tuple of split names")

        if len(datasets_to_concat) == 1:
            self.dataset = datasets_to_concat[0]
        else:
            self.dataset = concatenate_datasets(datasets_to_concat)

        self.splits = requested_splits
        if self.verbose:
            per_split_sizes = {
                name: len(ds)
                for name, ds in zip(self.splits, datasets_to_concat)
            }
            print(
                f"Loaded EuclidDESIDataset with splits={self.splits} total_samples={len(self.dataset)}"
            )
            print(f"Per-split sizes: {per_split_sizes}")
            preview = [
                (self.dataset[i].get("object_id") or self.dataset[i].get("targetid")) 
                for i in range(min(3, len(self.dataset)))
            ]
            print(f"Object ID preview: {preview}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample = self.dataset[idx]

        # Convert PIL image to tensor
        rgb_image = sample['RGB_image']
        if Image is not None and isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)

        # Convert to tensor format (C, H, W)
        if isinstance(rgb_image, np.ndarray):
            if rgb_image.ndim == 3:
                rgb_image_t = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
            else:
                rgb_image_t = torch.from_numpy(rgb_image).unsqueeze(0).float() / 255.0
        else:
            # Fallback: if already tensor-like
            rgb_image_t = torch.as_tensor(rgb_image).float()
            if rgb_image_t.ndim == 3 and rgb_image_t.shape[0] in (1, 3):
                pass
            else:
                # Try to put back in (C,H,W)
                rgb_image_t = rgb_image_t.permute(2, 0, 1).contiguous()

        # Process spectrum data
        spectrum_data = None
        if sample.get('spectrum') is not None:
            if self.verbose:
                print(f"Sample spectrum keys: {sample['spectrum'].keys()}")
            flux = sample['spectrum'].get('flux')
            wavelength = sample['spectrum'].get('wavelength')
            error = sample['spectrum'].get('error')

            flux = np.array(flux) if flux is not None else None
            wavelength = np.array(wavelength) if wavelength is not None else None
            error = np.array(error) if error is not None else None

            ivar = 1.0 / (error ** 2) if error is not None else None

            # mask not provided → make a "valid empty" boolean mask
            mask = np.zeros_like(flux, dtype=bool) if flux is not None else None

            if flux is not None:
                spectrum_data = {
                    'flux': torch.from_numpy(flux).float(),
                    'wavelength': torch.from_numpy(wavelength).float() if wavelength is not None else None,
                    'error': torch.from_numpy(error).float() if error is not None else None,
                    'ivar': torch.from_numpy(ivar).float() if ivar is not None else None,
                    'mask': torch.from_numpy(mask).bool() if mask is not None else None,
                }

        # Process SED data
        sed_fluxes = None
        if sample.get('sed_data') is not None:
            flux_keys = [k for k in sample['sed_data'].keys() if k.startswith('flux_')]
            if flux_keys:
                sed_fluxes = torch.tensor([sample['sed_data'][k] for k in flux_keys]).float()

        # Individual band images (optional)
        def _to_tensor_img(x):
            return torch.from_numpy(np.array(x)).float() if x is not None else None

        vis_image    = _to_tensor_img(sample.get('VIS_image'))
        nisp_y_image = _to_tensor_img(sample.get('NISP_Y_image'))
        nisp_j_image = _to_tensor_img(sample.get('NISP_J_image'))
        nisp_h_image = _to_tensor_img(sample.get('NISP_H_image'))

        return {
            'object_id': sample.get('object_id') or sample.get('targetid'),
            'targetid': sample.get('targetid'),
            'redshift': sample.get('redshift'),
            'rgb_image': rgb_image_t,
            'vis_image': vis_image,
            'nisp_y_image': nisp_y_image,
            'nisp_j_image': nisp_j_image,
            'nisp_h_image': nisp_h_image,
            'spectrum': spectrum_data,
            'sed_fluxes': sed_fluxes,
        }


def display_one_sample(
    split: str = "train",
    index: int = 0,
    cache_dir: str = "/n03data/ronceray/datasets",
    save_path: Optional[str] = None,
    show_bands: bool = False,
):
    """
    Load a sample from the dataset and display the RGB image (+ optional bands/spectrum/SED).
    """
    print(f"Loading dataset split='{split}'...")
    ds = EuclidDESIDataset(split=split, cache_dir=cache_dir)
    print(f"Dataset loaded. Total samples ({split}): {len(ds)}")

    if not (0 <= index < len(ds)):
        raise IndexError(f"--index {index} out of bounds (0..{len(ds)-1})")

    sample = ds[index]
    title = f"object_id={sample['object_id']} | z={sample['redshift']}"
    print(f"Displaying index {index}: {title}")

    # Prepare figure
    if show_bands:
        fig, axes = plt.subplots(2, 4, figsize=(12, 8))
        ax_rgb, ax_spec, ax_sed, _ = axes[0]
        ax_vis, ax_y, ax_j, ax_h = axes[1]
    else:
        fig, ax_rgb = plt.subplots(figsize=(5, 5))

    # ----- RGB -----
    rgb = sample['rgb_image']
    if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
        rgb_np = rgb.permute(1, 2, 0).numpy()
        if rgb_np.shape[2] == 1:  # grayscale
            ax_rgb.imshow(rgb_np[..., 0], cmap="gray")
        else:
            ax_rgb.imshow(np.clip(rgb_np, 0, 1))
    else:
        # Abnormal case: try to display as 2D
        ax_rgb.imshow(rgb.squeeze().numpy(), cmap="gray")
    ax_rgb.set_title(f"RGB — {title}")
    ax_rgb.axis("off")

    if show_bands:
        # ----- Spectrum (if available) -----
        spec = sample.get('spectrum')
        if spec is not None and spec.get('flux') is not None:
            flux = spec['flux'].numpy()
            wavelength = spec['wavelength'].numpy() if spec.get('wavelength') is not None else np.arange(len(flux))
            ax_spec.plot(wavelength, flux, linewidth=0.8)
            ax_spec.set_title("DESI Spectrum")
            ax_spec.set_xlabel("Wavelength (Å)")
            ax_spec.set_ylabel("Flux")
        else:
            ax_spec.text(0.5, 0.5, "No spectrum", ha="center", va="center")
            ax_spec.set_axis_off()

        # ----- SED (if available) -----
        sed = sample.get('sed_fluxes')
        if sed is not None:
            ax_sed.bar(range(len(sed)), sed.numpy())
            ax_sed.set_title(f"SED ({len(sed)} bands)")
            ax_sed.set_xlabel("Filter")
            ax_sed.set_ylabel("Flux")
        else:
            ax_sed.text(0.5, 0.5, "No SED", ha="center", va="center")
            ax_sed.set_axis_off()

        # ----- Individual bands -----
        for ax, band_tensor, label in [
            (ax_vis, sample.get('vis_image'), "VIS"),
            (ax_y, sample.get('nisp_y_image'), "NIR-Y"),
            (ax_j, sample.get('nisp_j_image'), "NIR-J"),
            (ax_h, sample.get('nisp_h_image'), "NIR-H"),
        ]:
            if band_tensor is not None:
                im = ax.imshow(band_tensor.numpy(), cmap="viridis")
                ax.set_title(label)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, f"{label} unavailable", ha="center", va="center")
                ax.set_axis_off()

        fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Image saved: {save_path}")

    # If non-interactive backend, plt.show() will do nothing (OK)
    plt.show()
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Load and display a sample from the Euclid+DESI dataset."
    )
    p.add_argument("--index", type=int, default=0, help="Index of the sample to display (default: 0)")
    p.add_argument("--split", type=str, default="train", help="HF split to use")
    
    # Default to configured path
    try:
        default_cache = str(load_paths().dataset)
    except Exception:
        default_cache = "./data"

    p.add_argument("--cache-dir", type=str,
                   default=default_cache,
                   help="HuggingFace cache directory")
    p.add_argument("--save", type=str, default=None, help="Save path for the figure (png/jpg, optional)")
    p.add_argument("--no-gui", action="store_true", help="Do not open a window (save only if --save is provided)")
    p.add_argument("--show-bands", action="store_true", help="Display spectrum/SED + individual bands if available")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _maybe_switch_to_agg(args.no_gui)

    try:
        display_one_sample(
            split=args.split,
            index=args.index,
            cache_dir=args.cache_dir,
            save_path=args.save,
            show_bands=args.show_bands,
        )
    except Exception as e:
        warnings.warn(f"Error during display: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
