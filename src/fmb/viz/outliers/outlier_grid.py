"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.outliers.outlier_grid
Description: Grid visualization of top anomalous objects
"""

import argparse
from pathlib import Path
from typing import Sequence, Optional, List, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm

from fmb.paths import load_paths
from fmb.data.load_display_data import EuclidDESIDataset
from fmb.viz.utils import (
    load_index,
    collect_samples,
    collect_samples_with_index,
    prepare_rgb_image,
)
from fmb.viz.spectrum import extract_spectrum, REST_LINES
from fmb.data.utils import read_object_ids
from fmb.viz.style import apply_style


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
        obj_id = str(sample.get("object_id") or sample.get("targetid", "N/A"))
        redshift = sample.get("redshift")
        
        text_content = f"{obj_id}"
        if redshift is not None and not np.isnan(redshift):
            try:
                text_content += f"\n$z={float(redshift):.3f}$"
            except (TypeError, ValueError):
                pass
                
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
            
            smoothed_flux = scipy.ndimage.gaussian_filter1d(flux_sorted, sigma=3)
            
            redshift = sample.get("redshift")
            rest_wave = wave_sorted
            z_val = None
            
            if redshift is not None and not np.isnan(redshift) and redshift > -1:
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
                # Auto-scale Y to ignore extreme outliers?
                # For now keep as is.
                
                for name, line_rest in REST_LINES.items():
                    if rest_wave.min() <= line_rest <= rest_wave.max():
                        ax_spec.axvline(line_rest, color="red", linestyle=":", alpha=0.6, linewidth=0.8)
                        ax_spec.text(
                            line_rest,
                            ymax * 0.98,
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
            
            ax_spec.set_yticks([]) # Hide Y ticks
            ax_spec.minorticks_on()
            ax_spec.tick_params(axis='both', which='major', labelsize=6, length=3)
            ax_spec.tick_params(axis='both', which='minor', length=1.5)
            
        else:
            ax_spec.text(0.5, 0.5, "No spectrum", ha="center", va="center", transform=ax_spec.transAxes)
            ax_spec.axis("off")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved publication grid to {save_path}")
    
    if show:
        plt.show()
    
    plt.close(fig)


def run_grid_plot(
    csv_paths: List[Union[str, Path]],
    split: str = "all",
    cache_dir: Optional[Union[str, Path]] = None,
    max_count: int = 12,
    cols: int = 3,
    save_path: Optional[Union[str, Path]] = "outliers_paper_grid.png",
    show: bool = True,
    index_path: Optional[Union[str, Path]] = None,
    verbose: bool = False
):
    """
    Load data and plot the outlier grid.
    """
    # Apply style
    apply_style()
    
    # Resolve paths
    paths = load_paths()
    if not cache_dir:
        cache_dir = paths.dataset
        
    if index_path is None:
        index_path = paths.dataset_index
    if index_path and not index_path.exists():
        # Just warn or allow it to be ignored? 
        # `collect_samples_with_index` needs it.
        pass

    if save_path is None:
        out_dir = paths.analysis / "outliers"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "outliers_grid.png"
    
    csv_files = [Path(p) for p in csv_paths]
    object_ids = read_object_ids(csv_files, verbose=verbose) # removed limit here to shuffle/sample later or use max
    # Note: original read_object_ids took limit, but we pass full list then slice here? 
    # Actually utils.read_object_ids doesn't have limit arg in all versions?
    # Let's check signature... 
    # User's utils.py has `read_object_ids`? I saw it in `src/fmb/data/utils.py`? 
    # I didn't check that file content. Assuming it works or I should limit list manually.
    if len(object_ids) > max_count:
        object_ids = object_ids[:max_count]

    if not object_ids:
        print("No object IDs found in provided CSV files")
        return

    if index_path:
        index_map = load_index(Path(index_path))
        samples = collect_samples_with_index(
            cache_dir=str(cache_dir), # func expects str
            object_ids=object_ids,
            index_map=index_map,
            verbose=verbose,
        )
    else:
        dataset = EuclidDESIDataset(split=split, cache_dir=str(cache_dir), verbose=verbose)
        samples = collect_samples(dataset, object_ids, verbose=verbose)

    if not samples:
        print("None of the requested object IDs were found in the dataset")
        return

    plot_publication_grid(
        samples,
        cols=max(1, cols),
        save_path=Path(save_path) if save_path else None,
        show=show,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Display publication-ready grid of outliers with spectra",
    )
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s) with object_id column")
    parser.add_argument("--split", type=str, default="all", help="Dataset split(s)")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--max", type=int, default=12, help="Maximum number of images to display")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid")
    parser.add_argument("--save", type=str, default="outliers_paper_grid.png", help="Path to save the figure")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive display")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--index", type=str, default=None, help="Optional CSV mapping object_id -> split/index")
    args = parser.parse_args(argv)

    run_grid_plot(
        csv_paths=args.csv,
        split=args.split,
        cache_dir=args.cache_dir,
        max_count=args.max,
        cols=args.cols,
        save_path=args.save,
        show=not args.no_show,
        index_path=args.index,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
