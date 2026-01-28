"""
Similarity Visualization Utilities.
Functions to display grids of query and neighbor objects (Images + Spectra).
"""
from pathlib import Path
from typing import Sequence, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch

from fmb.data.utils import prepare_rgb_image

REST_LINES = {
    "Lyα": 1216.0,
    "C IV": 1549.0,
    "C III]": 1909.0,
    "Mg II": 2798.0,
    "[O II]": 3727.0,
    "[Ne III]": 3869.0,
    "Hδ": 4102.0,
    "Hγ": 4341.0,
    "Hβ": 4861.0,
    "[O III]": 4959.0,
    "[O III]": 5007.0,
    "[N II]": 6548.0,
    "Hα": 6563.0,
    "[N II]": 6584.0,
    "[S II]": 6717.0,
    "[S II]": 6731.0,
}

def extract_spectrum(sample: Dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Helper to extract flux and wavelength from sample dict."""
    spec = sample.get("spectrum")
    if spec is None:
        return None, None
    flux = spec.get("flux")
    if flux is None:
        return None, None
        
    if isinstance(flux, torch.Tensor):
        flux_np = flux.detach().cpu().numpy()
    else:
        flux_np = np.asarray(flux)
    flux_np = np.squeeze(flux_np)
    
    wavelength = spec.get("wavelength")
    if wavelength is None:
        wavelength_np = np.arange(len(flux_np))
    else:
        if isinstance(wavelength, torch.Tensor):
            wavelength_np = wavelength.detach().cpu().numpy()
        else:
            wavelength_np = np.asarray(wavelength)
        wavelength_np = np.squeeze(wavelength_np)
        
    return wavelength_np, flux_np

def plot_vertical_panels(
    samples: Sequence[Dict],
    cols: int,
    save_path: Optional[Path],
    show: bool,
    row_labels: Optional[Sequence[str]] = None,
    smooth_sigma: float = 3.0
) -> None:
    """
    Plots a grid where each cell contains a vertical stack of (Image, Spectrum).
    Designed for showing Query + Neighbors side-by-side.
    """
    count = len(samples)
    if count == 0:
        print("No samples to display.")
        return
        
    rows = int(np.ceil(count / cols))
    
    # Figure size estimation
    # Width: ~3 inches per col
    # Height: ~4 inches per row (2 inches img, 2 inches spec)
    fig_w = 3.5 * cols
    fig_h = 4.5 * rows
    
    fig, axes = plt.subplots(rows * 2, cols, figsize=(fig_w, fig_h))
    
    # Ensure axes is 2D array [2*rows, cols]
    if rows * 2 == 1 and cols == 1:
        axes = np.array([[axes]])
    elif (rows * 2 == 1) or (cols == 1):
        axes = axes.reshape(rows * 2, cols)
        
    # Add row labels if provided (e.g. Model Name)
    # We place them to the left of the First column of each Row-Block
    if row_labels:
        for r_idx in range(rows):
            if r_idx < len(row_labels):
                lbl = row_labels[r_idx]
                # Attach to the top-left axis of this row-group
                ax_ref = axes[2 * r_idx, 0]
                # Coordinates in axes fraction? Or Figure?
                # Using text with negative x
                ax_ref.text(
                    -0.2, 0.5, 
                    lbl, 
                    transform=ax_ref.transAxes, 
                    rotation=90, 
                    va='center', 
                    ha='right', 
                    fontsize=12, 
                    fontweight='bold'
                )

    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        
        img_ax = axes[2 * row, col]
        spec_ax = axes[2 * row + 1, col]

        # 1. Image
        image = prepare_rgb_image(sample)
        if image.ndim == 3 and image.shape[2] == 1:
            img_ax.imshow(image[..., 0], cmap="gray", origin="lower")
        else:
            img_ax.imshow(image, origin="lower")
            
        img_ax.axis("off")
        
        # Title/ID
        obj_id = str(sample.get("object_id", "N/A"))
        # Clean up ID if it has prefixes like [QUERY]
        # Maybe split lines if too long?
        
        redshift = sample.get("redshift")
        title_str = f"{obj_id}"
        if redshift is not None:
             try:
                 title_str += f"\nz={float(redshift):.3f}"
             except:
                 pass
                 
        img_ax.set_title(title_str, fontsize=8)

        # 2. Spectrum
        wavelength, flux = extract_spectrum(sample)
        spec_ax.clear()
        
        if flux is not None and wavelength is not None:
            sort_idx = np.argsort(wavelength)
            wave_sorted = wavelength[sort_idx]
            flux_sorted = flux[sort_idx]
            
            if smooth_sigma > 0:
                smoothed_flux = scipy.ndimage.gaussian_filter1d(flux_sorted, sigma=smooth_sigma)
            else:
                smoothed_flux = flux_sorted
                
            redshift = sample.get("redshift")
            rest_wave = wave_sorted
            z_val = None
            
            if redshift is not None:
                try:
                    z_val = float(redshift)
                    rest_wave = wave_sorted / (1.0 + z_val)
                except (TypeError, ValueError):
                    pass
            
            spec_ax.plot(rest_wave, smoothed_flux, linewidth=0.8, color="black")
            
            # Lines
            if z_val is not None:
                for name, line_rest in REST_LINES.items():
                    if rest_wave.min() <= line_rest <= rest_wave.max():
                        spec_ax.axvline(line_rest, color="red", linestyle=":", alpha=0.5, linewidth=0.8)
                        # Label? Maybe too crowded for small plots.
                        
            spec_ax.set_xlim(rest_wave.min(), rest_wave.max())
            spec_ax.set_yticks([]) # Clean look
            
            # Only label X axis on bottom row? Or all?
            # All is safer for now.
            # spec_ax.set_xlabel("Wavelength [Å]", fontsize=7)
            
        else:
            spec_ax.text(0.5, 0.5, "No Spectrum", ha="center", va="center", fontsize=8)
            spec_ax.axis("off")
            
    # clear unused
    total_cells = rows * cols
    for idx in range(count, total_cells):
        row = idx // cols
        col = idx % cols
        axes[2 * row, col].axis("off")
        axes[2 * row + 1, col].axis("off")

    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        
    if show:
        plt.show()
        
    plt.close(fig)
