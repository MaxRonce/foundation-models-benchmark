"""
Script to display a publication-ready grid of Euclid RGB images AND DESI spectra for a list of object IDs.
Designed for paper figures with LaTeX styling.

Usage:
    python -m scratch.plot_paper_outlier_grid \
        --csv outliers.csv \
        --save outliers_grid_paper.png
"""
import argparse
from pathlib import Path
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm
import matplotlib

from scratch.load_display_data import EuclidDESIDataset
from scratch.display_outlier_images import (
    read_object_ids,
    collect_samples,
    collect_samples_with_index,
    load_index,
    prepare_rgb_image,
)

# --- Publication Style Settings ---
# Try to use LaTeX, fallback if not available
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
except Exception:
    print("Warning: LaTeX not available, falling back to STIX fonts.")
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    })

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

REST_LINES = {
    r"Ly$\alpha$": 1216.0,
    r"C IV": 1549.0,
    r"C III]": 1909.0,
    r"Mg II": 2798.0,
    r"[O II]": 3727.0,
    r"[Ne III]": 3869.0,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0,
    r"[O III]": 4959.0,
    r"[O III]": 5007.0,
    r"[N II]": 6548.0,
    r"H$\alpha$": 6563.0,
    r"[N II]": 6584.0,
    r"[S II]": 6717.0,
    r"[S II]": 6731.0,
}


def extract_spectrum(sample: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def plot_publication_grid(
    samples: Sequence[dict],
    cols: int,
    save_path: Path | None,
    show: bool,
) -> None:
    count = len(samples)
    if count == 0:
        print("No samples to display.")
        return
    
    rows = int(np.ceil(count / cols))
    
    # Grid: 1 row per object (Image | Spectrum)
    #       2 cols per object
    total_grid_rows = rows
    total_grid_cols = cols * 2
    
    # Figure size
    # Width: ~5 inches per object (1.5" img + 3.5" spec) -> ~10 inches for 2 cols
    # Height: ~1.5 inches per object row
    fig_width = 5.0 * cols
    fig_height = 1.3 * rows 
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # GridSpec
    # Alternating widths: [1, 2.5] for [Image, Spectrum]
    gs = fig.add_gridspec(
        total_grid_rows, 
        total_grid_cols, 
        width_ratios=[1, 2.5] * cols,
        hspace=0.03, 
        wspace=0.03,
        top=0.98,
        bottom=0.05,
        left=0.02,
        right=0.98
    )

    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        
        # Grid indices
        # Row is just 'row'
        # Cols are 2*col (Image) and 2*col+1 (Spectrum)
        c_img = 2 * col
        c_spec = 2 * col + 1
        
        # 1. Image (Left)
        ax_img = fig.add_subplot(gs[row, c_img])
        
        image = prepare_rgb_image(sample)
        if image.ndim == 3 and image.shape[2] == 1:
            ax_img.imshow(image[..., 0], cmap="gray", origin="lower")
        else:
            ax_img.imshow(image, origin="lower")
        
        ax_img.axis("off")
        
        # Overlay Text
        obj_id = str(sample.get("object_id", "N/A"))
        redshift = sample.get("redshift")
        
        text_content = f"{obj_id}"
        if redshift is not None:
            try:
                text_content += f"\n$z={float(redshift):.3f}$"
            except (TypeError, ValueError):
                text_content += f"\n$z={redshift}$"
                
        # Top-left of image
        ax_img.text(
            0.05, 0.95, 
            text_content, 
            transform=ax_img.transAxes, 
            fontsize=7, 
            va="top", 
            ha="left",
            color="white",
            fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1.0)
        )

        # 2. Spectrum (Right)
        ax_spec = fig.add_subplot(gs[row, c_spec])

        wavelength, flux = extract_spectrum(sample)
        
        if flux is not None and wavelength is not None:
            sort_idx = np.argsort(wavelength)
            wave_sorted = wavelength[sort_idx]
            flux_sorted = flux[sort_idx]
            
            # Use sigma=3 as in the original script
            smoothed_flux = scipy.ndimage.gaussian_filter1d(flux_sorted, sigma=3)
            
            redshift = sample.get("redshift")
            rest_wave = wave_sorted
            z_val = None
            
            if redshift is not None:
                try:
                    z_val = float(redshift)
                    rest_wave = wave_sorted / (1.0 + z_val)
                    x_label = r"Rest-frame Wavelength [\AA]"
                except (TypeError, ValueError):
                    x_label = r"Observed Wavelength [\AA]"
            else:
                x_label = r"Wavelength [\AA]"

            ax_spec.plot(rest_wave, smoothed_flux, linewidth=0.6, color="black")
            
            # Emission lines
            if z_val is not None:
                ymin, ymax = ax_spec.get_ylim()
                for name, line_rest in REST_LINES.items():
                    if rest_wave.min() <= line_rest <= rest_wave.max():
                        ax_spec.axvline(line_rest, color="red", linestyle=":", alpha=0.6, linewidth=0.8)
                        # Label inside graph, vertical, small
                        ax_spec.text(
                            line_rest,
                            ymax * 0.98, # Near top
                            name,
                            rotation=90,
                            va="top",
                            ha="center",
                            fontsize=5,
                            color="#cc0000",
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5)
                        )

            ax_spec.set_xlim(rest_wave.min(), rest_wave.max())
            ax_spec.spines['top'].set_visible(False)
            ax_spec.spines['right'].set_visible(False)
            
            # X-label only for bottom row of OBJECTS
            if row == rows - 1:
                ax_spec.set_xlabel(x_label, fontsize=7, labelpad=2)
            else:
                ax_spec.set_xticklabels([])
            
            # Y-label only for first column of OBJECTS (if we care about flux label)
            # Or just remove it to save space as "arb."
            if col == 0 and row == rows // 2: # Middle left
                 # ax_spec.set_ylabel(r"Flux [arb.]", fontsize=7) 
                 pass # User asked for compact, Flux label often skippable or put on one axis
            
            ax_spec.set_yticks([]) # Hide Y ticks mostly for cleanliness or minimal ticks
            # Maybe keep ticks but no labels?
            # ax_spec.tick_params(axis='y', left=False, labelleft=False)
            
            ax_spec.minorticks_on()
            ax_spec.tick_params(axis='both', which='major', labelsize=6, length=3)
            ax_spec.tick_params(axis='both', which='minor', length=1.5)
            
        else:
            ax_spec.text(0.5, 0.5, "No spectrum", ha="center", va="center", transform=ax_spec.transAxes)
            ax_spec.axis("off")

    # Hide unused axes if any
    total_cells = rows * cols
    for idx in range(count, total_cells):
        row = idx // cols
        col = idx % cols
        r_start = 2 * row
        c_start = 2 * col
        # Remove the placeholders
        # Note: we haven't created them, so nothing to delete, 
        # but if we were iterating grid cells we would need to hide them.
        # Since we iterate by samples, we just stop. 
        # But we might want to turn off the "ticks" of the empty slots if they were shared?
        # Here axes are independent, so it's fine.
        pass

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved publication grid to {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Display publication-ready grid of outliers with spectra",
    )
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s) with object_id column")
    parser.add_argument("--split", type=str, default="all", help="Dataset split(s) for EuclidDESIDataset")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/n03data/ronceray/datasets",
    )
    parser.add_argument("--max", type=int, default=12, help="Maximum number of images to display")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid")
    parser.add_argument("--save", type=str, default="outliers_paper_grid.png", help="Path to save the figure")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive display")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--index", type=str, default=None, help="Optional CSV mapping object_id -> split/index")
    args = parser.parse_args(argv)

    csv_paths = [Path(p) for p in args.csv]
    object_ids = read_object_ids(csv_paths, limit=args.max, verbose=args.verbose)
    if not object_ids:
        raise SystemExit("No object IDs found in provided CSV files")

    if args.index:
        index_map = load_index(Path(args.index))
        samples = collect_samples_with_index(
            cache_dir=args.cache_dir,
            object_ids=object_ids,
            index_map=index_map,
            verbose=args.verbose,
        )
    else:
        dataset = EuclidDESIDataset(split=args.split, cache_dir=args.cache_dir, verbose=args.verbose)
        samples = collect_samples(dataset, object_ids, verbose=args.verbose)

    if not samples:
        raise SystemExit("None of the requested object IDs were found in the dataset")

    plot_publication_grid(
        samples,
        cols=max(1, args.cols),
        save_path=Path(args.save) if args.save else None,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
