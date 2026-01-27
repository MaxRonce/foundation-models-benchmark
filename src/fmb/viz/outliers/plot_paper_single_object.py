"""
Script to generate a publication-ready single object visualization.
Displays a single row with:
[Spectrum (Restframe)] [VIS band] [NISP-Y band] [NISP-J band] [NISP-H band]

Usage:
    python -m scratch.plot_paper_single_object --object-id <ID> --index euclid_index.csv --save object_viz.pdf
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from scratch.load_display_data import EuclidDESIDataset
from scratch.display_outlier_images_spectrum import extract_spectrum, REST_LINES
from scratch.show_object_detail import load_index

# --- Publication Style Settings ---
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
except Exception:
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    })

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

LATEX_REST_LINES = {
    r"Ly$\alpha$": 1216.0,
    "C IV": 1549.0,
    "C III]": 1909.0,
    "Mg II": 2798.0,
    "[O II]": 3727.0,
    "[Ne III]": 3869.0,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0,
    "[O III]": 4959.0,
    "[O III]": 5007.0,
    "[N II]": 6548.0,
    r"H$\alpha$": 6563.0,
    "[N II]": 6584.0,
    "[S II]": 6717.0,
    "[S II]": 6731.0,
}

def get_sample(object_id: str, index_path: str, cache_dir: str) -> dict:
    index_map = load_index(Path(index_path))
    if object_id not in index_map:
        raise ValueError(f"Object ID {object_id} not found in index {index_path}")
    
    split, idx = index_map[object_id]
    dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=True)
    return dataset[int(idx)]

def plot_spectrum(ax: plt.Axes, sample: dict, smooth_sigma: float = 2.0):
    wavelength, flux = extract_spectrum(sample)
    redshift = sample.get("redshift", 0.0)
    
    if hasattr(redshift, "item"): redshift = redshift.item()
    
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
    # ax.set_title(f"Redshift $z={redshift:.3f}$", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_band(ax: plt.Axes, image_data, title: str):
    if image_data is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
        
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.detach().cpu().numpy()
    
    # Normalize for display
    # Use arcsinh or similar for better contrast? 
    # Let's use simple percentile clipping for paper look
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

def main():
    parser = argparse.ArgumentParser(description="Paper-ready single object visualization")
    parser.add_argument("--object-id", required=True, help="ID of the object to plot")
    parser.add_argument("--index", required=True, help="Path to index CSV")
    parser.add_argument("--cache-dir", default="/n03data/ronceray/datasets", help="Data cache directory")
    parser.add_argument("--save", default="object_viz.pdf", help="Output filename")
    parser.add_argument("--smooth", type=float, default=2.0, help="Smoothing for spectrum")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saving")
    args = parser.parse_args()

    print(f"Loading object {args.object_id}...")
    sample = get_sample(args.object_id, args.index, args.cache_dir)
    
    # 5 panels: Spectrum, VIS, Y, J, H
    fig = plt.figure(figsize=(15, 3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 5, width_ratios=[2.5, 1, 1, 1, 1])

    # 1. Spectrum
    ax_spec = fig.add_subplot(gs[0])
    plot_spectrum(ax_spec, sample, args.smooth)
    
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
    
    # Global title? User asked for "object visualization"
    z = sample.get("redshift", 0.0)
    if hasattr(z, "item"): z = z.item()
    fig.suptitle(f"Object {args.object_id} ($z = {z:.3f}$)", fontsize=14, y=1.05)
    
    plt.tight_layout()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to {args.save}")

if __name__ == "__main__":
    main()
