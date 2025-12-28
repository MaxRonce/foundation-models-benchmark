"""
Script to display a grid of Euclid RGB images AND DESI spectra for a list of object IDs.
Similar to display_outlier_images.py but includes spectral data.
Adapted for AstroPT (mostly a copy, but kept separate for workflow consistency).

Usage:
    python -m scratch.display_outlier_images_spectrum_astropt \
        --csv outliers.csv \
        --save outliers_grid_with_spectra.png
"""
import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm

from scratch.load_display_data import EuclidDESIDataset
from scratch.display_outlier_images import (
    read_object_ids,
    collect_samples,
    collect_samples_with_index,
    load_index,
    prepare_rgb_image,
)

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


def plot_vertical_panels(
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
    fig, axes = plt.subplots(rows * 2, cols, figsize=(4 * cols, 6 * rows))
    axes = np.atleast_2d(axes)

    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        img_ax = axes[2 * row, col]
        spec_ax = axes[2 * row + 1, col]

        image = prepare_rgb_image(sample)
        if image.ndim == 3 and image.shape[2] == 1:
            img_ax.imshow(image[..., 0], cmap="gray")
        else:
            img_ax.imshow(image, cmap="gray")
        img_ax.axis("off")
        redshift = sample.get("redshift")
        obj_label = str(sample.get("object_id", "N/A"))
        if redshift is not None:
            try:
                obj_label += f" (z={float(redshift):.3f})"
            except (TypeError, ValueError):
                obj_label += f" (z={redshift})"
        img_ax.set_title(obj_label)

        wavelength, flux = extract_spectrum(sample)
        spec_ax.clear()
        if flux is not None and wavelength is not None:
            sort_idx = np.argsort(wavelength)
            wave_sorted = wavelength[sort_idx]
            flux_sorted = flux[sort_idx]
            smoothed_flux = scipy.ndimage.gaussian_filter1d(flux_sorted, sigma=3)
            redshift = sample.get("redshift")
            rest_wave = wave_sorted
            if redshift is not None:
                try:
                    z = float(redshift)
                    rest_wave = wave_sorted / (1.0 + z)
                except (TypeError, ValueError):
                    z = None
            else:
                z = None
            spec_ax.plot(rest_wave, smoothed_flux, linewidth=0.8, color="black")
            if redshift is not None:
                if z is not None:
                    for name, line_rest in REST_LINES.items():
                        if rest_wave.min() <= line_rest <= rest_wave.max():
                            spec_ax.axvline(line_rest, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
                            ymax = spec_ax.get_ylim()[1]
                            spec_ax.text(
                                line_rest,
                                ymax * 0.9,
                                name,
                                rotation=90,
                                va="top",
                                ha="center",
                                fontsize=7,
                                color="black",
                                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
                            )
            spec_ax.set_xlim(rest_wave.min(), rest_wave.max())
            spec_ax.set_xlabel("Rest-frame Wavelength [Å]")
            spec_ax.set_ylabel("Flux")
            spec_ax.grid(True, alpha=0.2)
        else:
            spec_ax.text(0.5, 0.5, "No spectrum", ha="center", va="center")
            spec_ax.axis("off")

    total_cells = rows * cols
    for idx in range(count, total_cells):
        row = idx // cols
        col = idx % cols
        axes[2 * row, col].axis("off")
        axes[2 * row + 1, col].axis("off")

    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved grid to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Display Euclid RGB images and DESI spectra with emission lines (AstroPT version)",
    )
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s) with object_id column")
    parser.add_argument("--split", type=str, default="all", help="Dataset split(s) for EuclidDESIDataset")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/n03data/ronceray/datasets",
    )
    parser.add_argument("--max", type=int, default=12, help="Maximum number of images to display")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure")
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

    plot_vertical_panels(
        samples,
        cols=max(1, args.cols),
        save_path=Path(args.save) if args.save else None,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
