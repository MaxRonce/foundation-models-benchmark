"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.outliers.single_object
Description: FMB module: fmb.viz.outliers.single_object
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from fmb.paths import load_paths
from fmb.data.load_display_data import EuclidDESIDataset
from fmb.viz.utils import load_index, prepare_rgb_image
from fmb.viz.spectrum import extract_spectrum, LATEX_REST_LINES
from fmb.viz.style import apply_style

def get_sample_by_id(object_id: str, index_path: Optional[Path], cache_dir: str, verbose: bool = True) -> dict:
    if index_path and index_path.exists():
        index_map = load_index(index_path)
        if object_id in index_map:
            split, idx = index_map[object_id]
            # Use offset logic if needed? load_display_data handles dataset loading.
            # We just load the specific sample.
            # Ideally we reuse valid offsets, but simply loading dataset and using generic access is safest.
            dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=verbose)
            
            # Since index might be global/unsafe, checking ID is better, 
            # but EuclidDESIDataset doesn't support ID lookup.
            # We blindly trust index for now, but handle IndexError
            try:
                # Correct index if strictly sequential? 
                # See my previous logic in utils.py.
                # Here we will assume the index in index_map is usable or relative.
                # If it fails, we fail.
                sample = dataset[int(idx)]
                return sample
            except IndexError:
                # Fallback: scan whole split
                if verbose: print(f"Index {idx} invalid, scanning split '{split}'...")
                for s in dataset:
                    if str(s.get("object_id") or "") == object_id:
                        return s
                raise ValueError(f"Object {object_id} not found in split {split}")
    
    # Fallback if no index: scan both splits
    if verbose: print("No index provided or object not found. Scanning 'train' and 'test'...")
    for split in ["train", "test"]:
        try:
            ds = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
            for s in ds:
                if str(s.get("object_id") or "") == object_id:
                    return s
        except Exception:
            pass
            
    raise ValueError(f"Object {object_id} not found in dataset.")

def plot_spectrum(ax: plt.Axes, sample: dict, smooth_sigma: float = 2.0):
    wavelength, flux = extract_spectrum(sample)
    redshift = sample.get("redshift", 0.0)
    
    if hasattr(redshift, "item"): redshift = redshift.item()
    if redshift is None or np.isnan(redshift): redshift = 0.0
    
    if flux is None or wavelength is None:
        ax.text(0.5, 0.5, "No Spectrum", ha="center", va="center", transform=ax.transAxes)
        return

    # Sort and Smooth
    order = np.argsort(wavelength)
    w = wavelength[order]
    f = flux[order]
    
    if smooth_sigma > 0:
        f = gaussian_filter1d(f, sigma=smooth_sigma)
    
    # Restframe conversion
    w_rest = w / (1.0 + redshift)
    
    ax.plot(w_rest, f, color="black", linewidth=0.8)
    
    # Overlay emission lines
    y_min, y_max = ax.get_ylim()
    for name, line_rest in LATEX_REST_LINES.items():
        if w_rest.min() <= line_rest <= w_rest.max():
            ax.axvline(line_rest, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            # Label at top
            ax.text(line_rest, y_max * 0.95, name, rotation=90, fontsize=6, 
                    ha="center", va="top", alpha=0.7)

    ax.set_xlabel(r"Rest-frame Wavelength [\AA]")
    ax.set_ylabel("Flux [arb.]")
    ax.set_xlim(w_rest.min(), w_rest.max())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_band(ax: plt.Axes, image_data, title: str):
    if image_data is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
        
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.detach().cpu().numpy()
    
    im = np.nan_to_num(image_data)
    v_min = np.percentile(im, 1)
    v_max = np.percentile(im, 99.5)
    
    ax.imshow(im, cmap="magma", origin="lower", vmin=v_min, vmax=v_max)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    # Simple frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

def run_single_object_plot(
    object_id: str,
    index_path: Union[str, Path, None] = None,
    cache_dir: Union[str, Path, None] = None,
    save_path: Union[str, Path, None] = None,
    smooth: float = 2.0,
    dpi: int = 300
):
    apply_style()
    paths = load_paths()
    if not cache_dir:
        cache_dir = paths.dataset
    
    if index_path is None:
        index_path = paths.dataset_index
        
    if save_path is None:
        out_dir = paths.analysis / "objects"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"object_{object_id}.png"
    
    print(f"Loading object {object_id}...")
    # Ensure index path is resolved to Path if it exists
    idx_p = Path(index_path) if index_path else None
    sample = get_sample_by_id(object_id, idx_p, str(cache_dir))
    
    # 5 panels: Spectrum, VIS, Y, J, H
    fig = plt.figure(figsize=(15, 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 5, width_ratios=[2.5, 1, 1, 1, 1])

    # 1. Spectrum
    ax_spec = fig.add_subplot(gs[0])
    plot_spectrum(ax_spec, sample, smooth)
    
    # 2. VIS
    ax_vis = fig.add_subplot(gs[1])
    plot_band(ax_vis, sample.get("vis_image"), "VIS")
    
    # 3. Y
    ax_y = fig.add_subplot(gs[2])
    plot_band(ax_y, sample.get("nisp_y_image"), "Y")
    
    # 4. J
    ax_j = fig.add_subplot(gs[3])
    plot_band(ax_j, sample.get("nisp_j_image"), "J")
    
    # 5. H
    ax_h = fig.add_subplot(gs[4])
    plot_band(ax_h, sample.get("nisp_h_image"), "H")
    
    z = sample.get("redshift", 0.0)
    if hasattr(z, "item"): z = z.item()
    if z is None or np.isnan(z): z = 0.0
    
    fig.suptitle(f"Object {object_id} ($z = {z:.3f}$)", fontsize=14, y=1.05)
    
    plt.tight_layout()
    sp = Path(save_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(sp, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {sp}")

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paper-ready single object visualization")
    parser.add_argument("--object-id", required=True, help="ID of the object to plot")
    parser.add_argument("--index", default=None, help="Path to index CSV")
    parser.add_argument("--cache-dir", default=None, help="Data cache directory")
    parser.add_argument("--save", default="object_viz.pdf", help="Output filename")
    parser.add_argument("--smooth", type=float, default=2.0, help="Smoothing for spectrum")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saving")
    args = parser.parse_args(argv)

    run_single_object_plot(
        object_id=args.object_id,
        index_path=args.index,
        cache_dir=args.cache_dir,
        save_path=args.save,
        smooth=args.smooth,
        dpi=args.dpi
    )

if __name__ == "__main__":
    main()
