"""
Script to display a grid of Euclid RGB images (and optionally DESI spectra) for a list of object IDs.
Useful for visual inspection of outliers or specific samples.

Usage:
    python -m fmb.viz.display_anomalies \\
        --csv outliers.csv \\
        --mode combined \\
        --save outliers_grid.png
"""
import argparse
import csv
import math
from pathlib import Path
from typing import Sequence, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm

from fmb.data.load_display_data import EuclidDESIDataset

try:
    import seaborn as sns
    sns.set_context("paper")
    sns.set_style("white")
except ImportError:
    pass

# Force serif fonts for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

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

def read_object_ids(
    csv_paths: Sequence[Path],
    limit: int | None = None,
    verbose: bool = False,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for path in csv_paths:
        if verbose:
            print(f"Reading object IDs from {path}")
        with path.open() as handle:
            reader = csv.DictReader(handle)
            if "object_id" not in reader.fieldnames:
                raise ValueError(f"CSV {path} is missing 'object_id' column")
            for row in reader:
                oid = str(row["object_id"]).strip()
                if not oid or oid in seen:
                    continue
                ids.append(oid)
                seen.add(oid)
                if limit is not None and len(ids) >= limit:
                    return ids
    return ids


def load_index(index_path: Path) -> Dict[str, Tuple[str, int]]:
    mapping: Dict[str, Tuple[str, int]] = {}
    with index_path.open() as handle:
        reader = csv.DictReader(handle)
        required = {"object_id", "split", "index"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Index file must contain columns {required}")
        for row in reader:
            oid = str(row["object_id"]).strip()
            split = row["split"].strip()
            try:
                idx = int(row["index"])
            except ValueError as exc:
                raise ValueError(f"Invalid index for object_id={oid}: {row['index']}") from exc
            if oid:
                mapping[oid] = (split, idx)
    return mapping


def collect_samples(
    dataset: EuclidDESIDataset,
    target_ids: Sequence[str],
    verbose: bool = False,
) -> list[dict]:
    wanted = {str(oid): None for oid in target_ids}
    collected: list[dict] = []
    remaining = set(wanted.keys())
    iterator = dataset
    if verbose:
        iterator = tqdm(dataset, desc="Scanning dataset", unit="sample")
    for sample in iterator:
        oid = str(sample.get("object_id"))
        if oid in remaining:
            collected.append(sample)
            remaining.remove(oid)
            if not remaining:
                break
    missing = set(wanted.keys()) - {str(s.get("object_id")) for s in collected}
    if missing:
        print(f"Warning: {len(missing)} object IDs not found: {sorted(list(missing))[:5]}...")
    return collected


def collect_samples_with_index(
    cache_dir: str,
    object_ids: Sequence[str],
    index_map: Dict[str, Tuple[str, int]],
    verbose: bool = False,
) -> list[dict]:
    from collections import defaultdict

    grouped: Dict[str, list[Tuple[str, int]]] = defaultdict(list)
    for oid in object_ids:
        if oid not in index_map:
            continue
        grouped[index_map[oid][0]].append((oid, index_map[oid][1]))

    samples_by_id: Dict[str, dict] = {}
    for split, entries in grouped.items():
        if verbose:
            print(f"Loading split '{split}' to fetch {len(entries)} samples")
        dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        for oid, idx in entries:
            if idx < 0 or idx >= len(dataset):
                print(f"Warning: index {idx} out of range for split '{split}' (object {oid})")
                continue
            sample = dataset[idx]
            samples_by_id[oid] = sample

    missing = [oid for oid in object_ids if oid not in samples_by_id]
    if missing:
        print(f"Warning: {len(missing)} object IDs missing in index/dataset: {missing[:5]}...")

    return [samples_by_id[oid] for oid in object_ids if oid in samples_by_id]


def prepare_rgb_image(sample: dict) -> np.ndarray:
    rgb = sample.get("rgb_image")
    if rgb is None:
        raise ValueError("Sample missing 'rgb_image'")
    if isinstance(rgb, torch.Tensor):
        tensor = rgb.detach().cpu()
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
            # Tensor is usually (C, H, W) -> we want (H, W, C) for plotting
            tensor = tensor.permute(1, 2, 0)
        elif tensor.dim() == 2:
            pass # (H, W) is fine
        else:
             # Just in case (1, H, W) was squeezed incorrectly before, or (B, C, H, W)
             if tensor.dim() == 4: tensor = tensor.squeeze(0).permute(1, 2, 0)
        
        array = tensor.numpy()
    else:  # assume numpy array
        array = np.asarray(rgb)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
            
    array = np.clip(array, 0.0, 1.0)
    return array


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


def plot_grid(samples: Sequence[dict], cols: int, save_path: Path | None, show: bool) -> None:
    count = len(samples)
    if count == 0:
        print("No samples to display.")
        return
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < count:
                sample = samples[idx]
                image = prepare_rgb_image(sample)
                if image.ndim == 3 and image.shape[2] == 1:
                    ax.imshow(image[..., 0], cmap="gray")
                else:
                    ax.imshow(image, cmap="gray")
                title = f"{sample.get('object_id', 'N/A')}"
                ax.set_title(title, fontsize=10)
            ax.axis("off")
            idx += 1
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"Saved grid to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_vertical_panels(
    samples: Sequence[dict],
    cols: int,
    save_path: Path | None,
    show: bool,
    smooth_sigma: float = 3.0,
) -> None:
    count = len(samples)
    if count == 0:
        print("No samples to display.")
        return
    rows = int(np.ceil(count / cols))
    
    fig = plt.figure(figsize=(4.5 * cols, 5.5 * rows))
    gs = fig.add_gridspec(rows, cols, wspace=0.1, hspace=0.2, left=0.02, right=0.98, top=0.95, bottom=0.05)

    for idx, sample in enumerate(samples):
        row = idx // cols
        col = idx % cols
        
        gs_cell = gs[row, col].subgridspec(2, 1, height_ratios=[1, 0.8], hspace=0.0)
        img_ax = fig.add_subplot(gs_cell[0])
        spec_ax = fig.add_subplot(gs_cell[1])

        image = prepare_rgb_image(sample)
        if image.ndim == 3 and image.shape[2] == 1:
            img_ax.imshow(image[..., 0], cmap="gray")
        else:
            img_ax.imshow(image, cmap="gray")
        img_ax.axis("off")
        
        redshift = sample.get("redshift")
        obj_label = str(sample.get("object_id", "N/A"))
        title_text = obj_label
        
        if redshift is not None:
             try:
                z_val = float(redshift)
                title_text += f"\\n$z={z_val:.3f}$"
             except:
                title_text += f"\\n$z={redshift}$"

        img_ax.set_title(title_text, fontsize=12, fontfamily='serif', y=1.02)

        wavelength, flux = extract_spectrum(sample)
        spec_ax.clear()
        if flux is not None and wavelength is not None:
            sort_idx = np.argsort(wavelength)
            wave_sorted = wavelength[sort_idx]
            flux_sorted = flux[sort_idx]
            smoothed_flux = scipy.ndimage.gaussian_filter1d(flux_sorted, sigma=smooth_sigma)
            
            # Rest frame
            rest_wave = wave_sorted
            z = None
            if redshift is not None:
                try:
                    z = float(redshift)
                    rest_wave = wave_sorted / (1.0 + z)
                except (TypeError, ValueError):
                    pass
            
            spec_ax.plot(rest_wave, smoothed_flux, linewidth=1.0, color="#222222")
            
            if z is not None:
                for name, line_rest in REST_LINES.items():
                    if rest_wave.min() <= line_rest <= rest_wave.max():
                        spec_ax.axvline(line_rest, color="darkred", linestyle=":", alpha=0.4, linewidth=0.8)
                        ymax = spec_ax.get_ylim()[1]
                        spec_ax.text(
                            line_rest,
                            ymax * 0.95,
                            name,
                            rotation=90,
                            va="top",
                            ha="center",
                            fontsize=8,
                            color="#444444",
                            fontfamily='serif',
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.2),
                        )
            
            spec_ax.set_xlim(rest_wave.min(), rest_wave.max())
            spec_ax.set_xlabel(r"Rest-frame Wavelength [$\\AA$]", fontsize=10, fontfamily='serif')
            spec_ax.set_ylabel("Flux", fontsize=10, fontfamily='serif')
            spec_ax.tick_params(labelsize=9, direction='in')
            spec_ax.grid(True, alpha=0.1, linestyle="-", linewidth=0.5)
            
            for spine in spec_ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('#555555')
        else:
            spec_ax.text(0.5, 0.5, "No spectrum", ha="center", va="center", fontsize=10, fontfamily='serif')
            spec_ax.axis("off")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Display Euclid RGB images and optionally DESI spectra from outliers CSV.",
    )
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s) with object_id column")
    parser.add_argument("--mode", choices=["images", "combined"], default="combined", help="Display mode: 'images' (grid only) or 'combined' (vertical panels with spectra)")
    parser.add_argument("--split", type=str, default="all", help="Dataset split(s) for EuclidDESIDataset")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/n03data/ronceray/datasets",
    )
    parser.add_argument("--max", type=int, default=12, help="Maximum number of images to display")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure")
    parser.add_argument("--smooth", type=float, default=3.0, help="Sigma for Gaussian smoothing of spectrum (combined mode)")
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

    if args.mode == "images":
        plot_grid(
            samples,
            cols=max(1, args.cols),
            save_path=Path(args.save) if args.save else None,
            show=not args.no_show,
        )
    else:
        plot_vertical_panels(
            samples,
            cols=max(1, args.cols),
            save_path=Path(args.save) if args.save else None,
            show=not args.no_show,
            smooth_sigma=args.smooth,
        )


if __name__ == "__main__":
    main()
