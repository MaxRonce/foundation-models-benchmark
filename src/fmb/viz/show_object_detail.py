"""
Script to display detailed information for a specific object or list of objects.
It shows the RGB image, NISP bands, composite cutouts, and spectrum.

Usage:
    python -m scratch.show_object_detail --object-id 12345
    python -m scratch.show_object_detail --csv list_of_ids.csv --output-dir details/
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from scratch.display_outlier_images import prepare_rgb_image
from scratch.display_outlier_images_spectrum import extract_spectrum, REST_LINES
from scratch.load_display_data import EuclidDESIDataset

try:
    from datasets import get_dataset_split_names
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'datasets' package is required. Install it with 'pip install datasets'.") from exc

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required. Install it with 'pip install scipy'.") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _to_numpy(image: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()
    return image


def _normalize_id(value) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.cpu().numpy().tolist()
    return str(value)


def adjust_dynamic_range(flux: np.ndarray, q: float = 100, clip: float = 99.85) -> np.ndarray:
    im = np.arcsinh(flux * q)
    if clip < 100:
        im = np.clip(im, 0, np.percentile(im, clip))
    return im


def is_valid_image(im: np.ndarray | None, min_range: float = 1e-6) -> bool:
    if im is None:
        return False
    if not np.isfinite(im).any():
        return False
    im_min = np.nanmin(im)
    im_max = np.nanmax(im)
    if not np.isfinite(im_min) or not np.isfinite(im_max):
        return False
    return (im_max - im_min) > min_range


def to_uint8(im: np.ndarray | None) -> np.ndarray | None:
    if im is None:
        return None
    if not np.isfinite(im).any():
        return np.full(im.shape, 128, dtype=np.uint8)
    im_min = np.nanmin(im)
    im_max = np.nanmax(im)
    if not np.isfinite(im_min) or not np.isfinite(im_max) or im_max <= im_min:
        return np.full(im.shape, 128, dtype=np.uint8)
    im = (im - im_min) / (im_max - im_min)
    return (255 * im).astype(np.uint8)


def make_nisp_rgb_composite(sample: dict) -> np.ndarray | None:
    nisp_bands = {
        "Y": _to_numpy(sample.get("nisp_y_image")),
        "J": _to_numpy(sample.get("nisp_j_image")),
        "H": _to_numpy(sample.get("nisp_h_image")),
    }
    order = ("H", "J", "Y")
    valid = [nisp_bands[b] for b in order if is_valid_image(nisp_bands[b])]
    if not valid:
        return None
    ref_shape = valid[0].shape
    if not all(arr.shape == ref_shape for arr in valid):
        return None

    channels = []
    for band in order:
        arr = nisp_bands.get(band)
        if is_valid_image(arr):
            adj = adjust_dynamic_range(arr, q=1, clip=99.85)
            ch = to_uint8(adj)
        else:
            ch = np.full(ref_shape, 128, dtype=np.uint8)
        channels.append(ch)
    return np.stack(channels, axis=2)


def make_composite_cutout(sample: dict) -> np.ndarray | None:
    vis = _to_numpy(sample.get("vis_image"))
    if not is_valid_image(vis):
        return None
    nisp_arrays = [
        _to_numpy(sample.get("nisp_y_image")),
        _to_numpy(sample.get("nisp_j_image")),
        _to_numpy(sample.get("nisp_h_image")),
    ]
    nisp_valid = [arr for arr in nisp_arrays if is_valid_image(arr)]
    if not nisp_valid:
        return None
    if not all(arr.shape == vis.shape for arr in nisp_valid):
        return None

    nisp_mean = np.mean(np.stack(nisp_valid, axis=0), axis=0)
    vis_adj = adjust_dynamic_range(vis, q=100, clip=99.85)
    nisp_adj = adjust_dynamic_range(nisp_mean, q=1, clip=99.85)
    mean_adj = np.mean([vis_adj, nisp_adj], axis=0)

    vis_uint8 = to_uint8(vis_adj)
    nisp_uint8 = to_uint8(nisp_adj)
    mean_uint8 = to_uint8(mean_adj)
    return np.stack([nisp_uint8, mean_uint8, vis_uint8], axis=2)


def render_spectrum(ax: plt.Axes, sample: dict, smooth_sigma: float | None) -> None:
    wavelength, flux = extract_spectrum(sample)
    redshift = sample.get("redshift")
    if flux is None or wavelength is None or redshift is None:
        ax.text(0.5, 0.5, "No spectrum", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if isinstance(redshift, torch.Tensor):
        redshift = redshift.item() if redshift.numel() == 1 else redshift.cpu().numpy()

    order = np.argsort(wavelength)
    wave_sorted = wavelength[order]
    flux_sorted = flux[order]
    if smooth_sigma is not None and smooth_sigma > 0:
        flux_sorted = gaussian_filter1d(flux_sorted, sigma=smooth_sigma)

    try:
        z = float(redshift)
    except (TypeError, ValueError):
        z = None

    rest_wave = wave_sorted / (1.0 + z) if z is not None else wave_sorted
    ax.plot(rest_wave, flux_sorted, linewidth=1.0, color="black")
    if z is not None:
        for name, line_rest in REST_LINES.items():
            if rest_wave.min() <= line_rest <= rest_wave.max():
                ax.axvline(line_rest, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
                ymax = ax.get_ylim()[1]
                ax.text(
                    line_rest,
                    ymax * 0.9,
                    name,
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=8,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.0),
                )
    ax.set_xlim(rest_wave.min(), rest_wave.max())
    ax.set_xlabel("Rest-frame Wavelength [Å]")
    ax.set_ylabel("Flux")
    ax.grid(True, alpha=0.2)


def render_band(ax: plt.Axes, image: torch.Tensor | np.ndarray | None, title: str) -> None:
    image = _to_numpy(image)
    if image is None:
        ax.text(0.5, 0.5, f"No {title}", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    ax.imshow(image, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")


def load_index(index_path: Path | None) -> dict[str, tuple[str, int]]:
    if index_path is None:
        return {}
    if not index_path.exists():
        raise SystemExit(f"Index file {index_path} not found")
    with index_path.open() as handle:
        reader = csv.DictReader(handle)
        return {row["object_id"]: (row["split"], int(row["index"])) for row in reader}


def parse_object_ids(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        raise SystemExit(f"CSV file {csv_path} not found")
    ids: list[str] = []
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        if "object_id" in reader.fieldnames:
            for row in reader:
                oid = row.get("object_id")
                if oid:
                    ids.append(str(oid).strip())
        else:
            handle.seek(0)
            for line in handle:
                line = line.strip()
                if line and not line.lower().startswith("object"):
                    ids.append(line)
    if not ids:
        raise SystemExit(f"No object IDs found in {csv_path}")
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Display detailed views for one or multiple objects (spectrum + VIS/NISP cutouts)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--object-id", help="Single object ID to display")
    group.add_argument("--csv", type=str, help="CSV file with column 'object_id'")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Comma-separated splits to search, or 'all' to scan every split",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets",
    )
    parser.add_argument("--index", type=str, default=None, help="Optional CSV index mapping object_id -> split/index")
    parser.add_argument("--smooth", type=float, default=0.0, help="Gaussian smoothing sigma for spectrum (0 disables)")
    parser.add_argument("--save", type=str, default=None, help="Path to save figure (single-object mode)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save figures when using --csv")
    parser.add_argument("--dpi", type=int, default=250, help="Output figure DPI")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive display")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    if args.object_id:
        object_ids = [args.object_id]
    else:
        object_ids = parse_object_ids(Path(args.csv))

    smooth_sigma = args.smooth if args.smooth and args.smooth > 0 else None
    index_map = load_index(Path(args.index) if args.index else None)

    if args.split.lower() == "all":
        splits = get_dataset_split_names("msiudek/astroPT_euclid_desi_dataset")
    else:
        splits = [s.strip() for s in args.split.split(",") if s.strip()]
        if not splits:
            raise SystemExit("No valid splits provided")

    dataset_cache: dict[str, EuclidDESIDataset] = {}

    def get_dataset(split: str) -> EuclidDESIDataset:
        if split not in dataset_cache:
            dataset_cache[split] = EuclidDESIDataset(split=split, cache_dir=args.cache_dir, verbose=args.verbose)
        return dataset_cache[split]

    def locate_sample(object_id: str) -> dict | None:
        if index_map and object_id in index_map:
            split, idx = index_map[object_id]
            ds = get_dataset(split)
            return ds[int(idx)]
        for split in splits:
            ds = get_dataset(split)
            iterator = range(len(ds))
            if args.verbose and tqdm is not None:
                iterator = tqdm(iterator, desc=f"Scanning {split} ({object_id})", unit="obj", leave=False)
            for i in iterator:
                sample = ds[i]
                if _normalize_id(sample.get("object_id")) == object_id:
                    return sample
        return None

    output_dir = None
    if len(object_ids) > 1:
        output_dir = Path(args.output_dir or "object_details")
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.save:
            print("Warning: --save is ignored when processing multiple objects; using --output-dir instead.")

    object_iter = tqdm(object_ids, desc="Objects", unit="obj") if len(object_ids) > 1 and tqdm else object_ids

    for obj_id in object_iter:
        sample = locate_sample(obj_id)
        if sample is None:
            print(f"Warning: object {obj_id} not found; skipping")
            continue

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(15, 7))
        gs = GridSpec(2, 5, figure=fig)

        ax_spec = fig.add_subplot(gs[:, :2])
        render_spectrum(ax_spec, sample, smooth_sigma)

        rgb = prepare_rgb_image(sample)
        rgb = np.clip(rgb, 0.0, 1.0)
        if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
            rgb_plot = np.moveaxis(rgb, 0, -1)
        elif rgb.ndim == 2:
            rgb_plot = rgb
        else:
            rgb_plot = rgb
        ax_vis = fig.add_subplot(gs[0, 2])
        ax_vis.imshow(rgb_plot)
        ax_vis.axis("off")
        ax_vis.set_title(f"RGB (object {obj_id})")

        ax_nisp_rgb = fig.add_subplot(gs[0, 3])
        nisp_rgb = make_nisp_rgb_composite(sample)
        if nisp_rgb is not None:
            ax_nisp_rgb.imshow(nisp_rgb, origin="lower")
            ax_nisp_rgb.set_title("NISP RGB (R=H,G=J,B=Y)")
            ax_nisp_rgb.axis("off")
        else:
            ax_nisp_rgb.text(0.5, 0.5, "No NISP RGB", ha="center", va="center", transform=ax_nisp_rgb.transAxes)
            ax_nisp_rgb.axis("off")

        ax_comp = fig.add_subplot(gs[0, 4])
        composite = make_composite_cutout(sample)
        if composite is not None:
            ax_comp.imshow(composite, origin="lower")
            ax_comp.set_title("Composite (NISP, mean, VIS)")
            ax_comp.axis("off")
        else:
            ax_comp.text(0.5, 0.5, "No composite", ha="center", va="center", transform=ax_comp.transAxes)
            ax_comp.axis("off")

        band_titles = ["NIR-Y", "NIR-J", "NIR-H"]
        band_images = [sample.get("nisp_y_image"), sample.get("nisp_j_image"), sample.get("nisp_h_image")]
        for idx, (title, image) in enumerate(zip(band_titles, band_images)):
            ax_band = fig.add_subplot(gs[1, 2 + idx])
            render_band(ax_band, image, title)

        fig.subplots_adjust(top=0.92, hspace=0.75, wspace=0.35)
        fig.tight_layout()

        if output_dir is not None:
            save_path = output_dir / f"object_{obj_id}.png"
        else:
            save_path = Path(args.save) if args.save else None

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=args.dpi)
            print(f"Saved figure to {save_path}")

        if not args.no_show and len(object_ids) == 1:
            plt.show()
        plt.close(fig)

    if len(object_ids) > 1 and not args.no_show:
        print("Multiple objects processed; figures saved to output directory. Enable --no-show to suppress GUI windows.")


if __name__ == "__main__":
    main()
