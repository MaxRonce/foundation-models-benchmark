"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.visualize_embedding_umap
Description: FMB module: fmb.viz.visualize_embedding_umap
"""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from scratch.display_outlier_images import (
    collect_samples,
    collect_samples_with_index,
    load_index,
    prepare_rgb_image,
)
from scratch.display_outlier_images_spectrum import REST_LINES, extract_spectrum
from scratch.load_display_data import EuclidDESIDataset
from tqdm import tqdm

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'umap-learn' package is required. Install it with 'pip install umap-learn'."
    ) from exc


def load_records(path: Path) -> list[dict]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def _to_str_id(value) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.cpu().numpy().tolist()
    return str(value)


def stack_embeddings(records: Sequence[dict], key: str) -> np.ndarray:
    vectors = []
    for rec in records:
        tensor = rec.get(key)
        if tensor is None:
            continue
        if isinstance(tensor, torch.Tensor):
            vectors.append(tensor.detach().cpu().numpy())
        else:
            vectors.append(np.asarray(tensor))
    if not vectors:
        raise ValueError(f"No embeddings found for key '{key}'")
    return np.stack(vectors, axis=0)


def compute_umap(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    reducer = umap.UMAP(random_state=random_state)
    return reducer.fit_transform(embeddings)


def assign_to_grid(
    object_ids: Sequence[str],
    coords: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    random_state: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    if grid_rows <= 0 or grid_cols <= 0 or coords.size == 0:
        return [], []

    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0

    cell_w = (max_x - min_x) / grid_cols
    cell_h = (max_y - min_y) / grid_rows

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(coords):
        gx = int((x - min_x) / cell_w)
        gy = int((y - min_y) / cell_h)
        gx = min(max(gx, 0), grid_cols - 1)
        gy = min(max(gy, 0), grid_rows - 1)
        buckets.setdefault((gx, gy), []).append(idx)

    rng = np.random.default_rng(random_state)
    selected_ids: list[str] = []
    cell_positions: list[tuple[int, int]] = []
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            cell = (gx, gy)
            indices = buckets.get(cell)
            if not indices:
                continue
            idx = rng.choice(indices)
            selected_ids.append(object_ids[idx])
            cell_positions.append((gx, gy))

    return selected_ids, cell_positions


def add_thumbnails(
    ax: plt.Axes,
    cell_positions: list[tuple[int, int]],
    samples: Sequence[dict],
) -> None:
    for (gx, gy), sample in zip(cell_positions, samples):
        try:
            image = prepare_rgb_image(sample)
            image = np.clip(image, 0.0, 1.0)
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=2)
            elif image.ndim == 3 and image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)

            xmin, xmax = gx, gx + 1
            ymin, ymax = gy, gy + 1
            ax.imshow(
                image,
                extent=(xmin, xmax, ymin, ymax),
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                zorder=3,
            )
        except Exception as exc:  # pragma: no cover
            print(
                f"Failed to attach thumbnail for object {sample.get('object_id')}: {exc}"
            )


def render_spectrum(ax: plt.Axes, sample: dict) -> None:
    wavelength, flux = extract_spectrum(sample)
    redshift = sample.get("redshift")
    if flux is not None and wavelength is not None and redshift is not None:
        sort_idx = np.argsort(wavelength)
        wave_sorted = wavelength[sort_idx]
        flux_sorted = flux[sort_idx]
        try:
            z = float(redshift)
        except (TypeError, ValueError):
            z = None
        rest_wave = wave_sorted / (1.0 + z) if z is not None else wave_sorted
        ax.plot(rest_wave, flux_sorted, linewidth=0.6, color="black")
        if z is not None:
            for name, line_rest in REST_LINES.items():
                if rest_wave.min() <= line_rest <= rest_wave.max():
                    ax.axvline(
                        line_rest, color="red", linestyle="--", alpha=0.6, linewidth=0.6
                    )
                    ymax = ax.get_ylim()[1]
                    ax.text(
                        line_rest,
                        ymax * 0.9,
                        name,
                        rotation=90,
                        va="top",
                        ha="center",
                        fontsize=6,
                        color="black",
                        bbox=dict(
                            facecolor="white", alpha=0.8, edgecolor="none", pad=1.0
                        ),
                    )
        ax.set_xlim(rest_wave.min(), rest_wave.max())
    else:
        ax.text(
            0.5,
            0.5,
            "No spectrum",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
    ax.set_xticks([])
    ax.set_yticks([])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize embedding UMAP with thumbnail images organized on a grid",
    )
    parser.add_argument("--input", required=True, help="Path to embeddings .pt file")
    parser.add_argument(
        "--embedding-key",
        default="embedding_hsc_desi",
        choices=["embedding_hsc_desi", "embedding_hsc", "embedding_spectrum"],
        help="Embedding field to visualize",
    )
    parser.add_argument(
        "--figure", required=True, help="Output image path for thumbnails"
    )
    parser.add_argument(
        "--figure-spectrum", default=None, help="Optional path for spectrum-grid figure"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Dataset split(s) if no index is provided (comma-separated or 'all')",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/n03data/ronceray/datasets",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Optional CSV mapping object_id -> split/index",
    )
    parser.add_argument(
        "--grid-rows", type=int, default=12, help="Number of rows in the thumbnail grid"
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=12,
        help="Number of columns in the thumbnail grid",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for UMAP and sampling",
    )
    parser.add_argument("--dpi", type=int, default=450, help="Output resolution in DPI")
    parser.add_argument(
        "--point-size", type=float, default=6.0, help="Scatter point size"
    )
    parser.add_argument("--alpha", type=float, default=0.35, help="Scatter alpha")
    args = parser.parse_args(argv)

    records = load_records(Path(args.input))
    coords = compute_umap(
        stack_embeddings(records, args.embedding_key), args.random_state
    )

    object_ids = [_to_str_id(rec.get("object_id", "")) for rec in records]

    redshifts = []
    for rec in records:
        z = rec.get("redshift", np.nan)
        if isinstance(z, torch.Tensor):
            if z.numel() == 1:
                z = z.item()
            else:
                z = z.cpu().numpy()
        redshifts.append(z)
    redshifts = np.array(redshifts, dtype=float)

    # Normalize coordinates to [0, 1] range
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]
    coords_x = (coords_x - coords_x.min()) / (coords_x.max() - coords_x.min() + 1e-9)
    coords_y = (coords_y - coords_y.min()) / (coords_y.max() - coords_y.min() + 1e-9)
    coords_norm = np.column_stack((coords_x, coords_y))

    thumb_ids, cell_positions = assign_to_grid(
        object_ids,
        coords_norm,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        random_state=args.random_state,
    )

    fig, ax = plt.subplots(figsize=(args.grid_cols * 1.5, args.grid_rows * 1.5))
    if np.isnan(redshifts).all():
        ax.scatter(
            coords_norm[:, 0] * args.grid_cols,
            coords_norm[:, 1] * args.grid_rows,
            s=args.point_size,
            alpha=args.alpha,
            color="slateblue",
            zorder=1,
        )
    else:
        mask = ~np.isnan(redshifts)
        scatter = ax.scatter(
            coords_norm[mask, 0] * args.grid_cols,
            coords_norm[mask, 1] * args.grid_rows,
            s=args.point_size,
            alpha=args.alpha,
            c=redshifts[mask],
            cmap="viridis",
            zorder=1,
        )
        fig.colorbar(scatter, ax=ax, label="Redshift")
        if (~mask).any():
            ax.scatter(
                coords_norm[~mask, 0] * args.grid_cols,
                coords_norm[~mask, 1] * args.grid_rows,
                s=args.point_size,
                alpha=args.alpha,
                color="lightgray",
                label="redshift NA",
                zorder=1,
            )
            ax.legend(loc="best")

    ax.set_title(f"UMAP of {args.embedding_key.replace('embedding_', '').upper()}")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_xlim(0, args.grid_cols)
    ax.set_ylim(0, args.grid_rows)
    ax.set_xticks(np.arange(0, args.grid_cols + 1, 1))
    ax.set_yticks(np.arange(0, args.grid_rows + 1, 1))
    ax.grid(True, which="both", alpha=0.3)

    ordered_samples: list[dict] = []
    ordered_cells: list[tuple[int, int]] = []
    if thumb_ids:
        if args.index:
            index_map = load_index(Path(args.index))
            samples = collect_samples_with_index(
                cache_dir=args.cache_dir,
                object_ids=thumb_ids,
                index_map=index_map,
                verbose=True,
            )
            retrieved_ids = {_to_str_id(sample.get("object_id")) for sample in samples}
            missing = [oid for oid in thumb_ids if oid not in retrieved_ids]
            if missing:
                dataset = EuclidDESIDataset(
                    split=args.split, cache_dir=args.cache_dir, verbose=True
                )
                samples.extend(collect_samples(dataset, missing, verbose=True))
        else:
            dataset = EuclidDESIDataset(
                split=args.split, cache_dir=args.cache_dir, verbose=True
            )
            samples = collect_samples(dataset, thumb_ids, verbose=True)
        id_to_sample = {
            _to_str_id(sample.get("object_id")): sample for sample in samples
        }
        for oid, cell in zip(thumb_ids, cell_positions):
            sample = id_to_sample.get(oid)
            if sample is not None:
                ordered_samples.append(sample)
                ordered_cells.append(cell)
        if args.embedding_key != "embedding_spectrum":
            add_thumbnails(ax, ordered_cells, ordered_samples)

    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.figure, dpi=args.dpi)
    print(f"Saved UMAP visualization to {args.figure}")
    plt.close(fig)

    if args.figure_spectrum:
        if ordered_samples:
            grid_pairs = [
                (gx, gy, sample)
                for (gx, gy), sample in zip(cell_positions, ordered_samples)
            ]
            fig_spec, axes = plt.subplots(
                args.grid_rows,
                args.grid_cols,
                figsize=(args.grid_cols * 2.5, args.grid_rows * 1.8),
                sharex=True,
                sharey=True,
            )
            axes = np.atleast_2d(axes)
            for gx, gy, sample in tqdm(
                grid_pairs, desc="Rendering spectra", unit="spec"
            ):
                ax_spec = axes[gy, gx]
                render_spectrum(ax_spec, sample)
            for gy in range(args.grid_rows):
                for gx in range(args.grid_cols):
                    if (gx, gy) not in ordered_cells:
                        axes[gy, gx].axis("off")
            fig_spec.tight_layout()
            Path(args.figure_spectrum).parent.mkdir(parents=True, exist_ok=True)
            fig_spec.savefig(args.figure_spectrum, dpi=args.dpi)
            print(f"Saved UMAP spectrum visualization to {args.figure_spectrum}")
            plt.close(fig_spec)
        else:
            print("No samples available for spectrum visualization")


if __name__ == "__main__":
    main()
