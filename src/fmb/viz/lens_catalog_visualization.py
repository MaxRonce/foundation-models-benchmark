"""
Script to visualize a lens catalog in the context of AION embeddings.
It matches a lens catalog (with object_ids) to the embeddings, highlights them on a UMAP,
and generates a grid of images/spectra for the matched lenses.

Usage:
    python -m scratch.lens_catalog_visualization \
        --lens-csv /path/to/lenses.csv \
        --embeddings /path/to/embeddings.pt \
        --output-umap scratch/outputs/lens_umap.png \
        --output-grid scratch/outputs/lens_grid.png
"""
import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # Optional dependency; only required when loading .pt embeddings.
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

try:  # Optional dependency; only required when computing fresh UMAP coordinates.
    import umap  # type: ignore[import]
except ModuleNotFoundError:
    umap = None  # type: ignore[assignment]

try:  # Reuse the display helpers when available (requires torch+dataset).
    from scratch.display_outlier_images import (
        collect_samples,
        collect_samples_with_index,
        load_index,
    )
    from scratch.display_outlier_images_spectrum import plot_vertical_panels
    from scratch.load_display_data import EuclidDESIDataset

    _DISPLAY_HELPERS_AVAILABLE = True
except ModuleNotFoundError:
    collect_samples = None  # type: ignore[assignment]
    collect_samples_with_index = None  # type: ignore[assignment]
    load_index = None  # type: ignore[assignment]
    plot_vertical_panels = None  # type: ignore[assignment]
    EuclidDESIDataset = None  # type: ignore[assignment]
    _DISPLAY_HELPERS_AVAILABLE = False


EMBEDDING_COLUMN_MAP = {
    "embedding_hsc_desi": ("embedding_hsc_desi_x", "embedding_hsc_desi_y"),
    "embedding_hsc": ("embedding_hsc_x", "embedding_hsc_y"),
    "embedding_spectrum": ("embedding_spectrum_x", "embedding_spectrum_y"),
}


def _require_torch(context: str) -> None:
    if torch is None:  # pragma: no cover - defensive branch
        raise SystemExit(
            f"PyTorch is required to {context}. Install the optional 'torch' extra for this project."
        )


def _require_umap(context: str) -> None:
    if umap is None:  # pragma: no cover - defensive branch
        raise SystemExit(
            f"The 'umap-learn' package is required to {context}. "
            "Install it or provide pre-computed UMAP coordinates via --umap-coords."
        )


def load_records(path: Path) -> list[dict]:
    _require_torch("load embeddings saved with torch.save")
    data = torch.load(path, map_location="cpu")  # type: ignore[union-attr]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def _to_numpy_array(value) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _object_id_to_str(value) -> str:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.cpu().numpy().tolist()
    return str(value)


def stack_embeddings(records: Iterable[dict], key: str) -> np.ndarray:
    vectors = []
    for entry in records:
        tensor = entry.get(key)
        if tensor is None:
            continue
        vectors.append(_to_numpy_array(tensor))
    if not vectors:
        raise ValueError(f"No embeddings found for key '{key}'")
    return np.stack(vectors, axis=0)


def compute_umap_coords(records: list[dict], embedding_key: str, random_state: int) -> pd.DataFrame:
    embeddings = stack_embeddings(records, embedding_key)
    _require_umap("compute UMAP coordinates from embeddings")
    reducer = umap.UMAP(random_state=random_state)  # type: ignore[call-arg]
    coords = reducer.fit_transform(embeddings)
    object_ids = [_object_id_to_str(rec.get("object_id", "")) for rec in records]
    return pd.DataFrame({"object_id": object_ids, "umap_x": coords[:, 0], "umap_y": coords[:, 1]})


def load_umap_from_csv(path: Path, embedding_key: str) -> pd.DataFrame:
    if embedding_key not in EMBEDDING_COLUMN_MAP:
        valid = "', '".join(sorted(EMBEDDING_COLUMN_MAP))
        raise ValueError(f"Unsupported embedding key '{embedding_key}'. Expected one of '{valid}'.")
    columns = EMBEDDING_COLUMN_MAP[embedding_key]
    df = pd.read_csv(path, dtype={"object_id": str})
    missing = [col for col in ("object_id", *columns) if col not in df.columns]
    if missing:
        raise ValueError(f"UMAP CSV {path} is missing required columns: {missing}")
    coords = df.loc[:, ["object_id", *columns]].copy()
    coords = coords.rename(columns={columns[0]: "umap_x", columns[1]: "umap_y"})
    return coords


def normalize_object_id_series(series: pd.Series) -> pd.Series:
    def _convert(value) -> str | None:
        if pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return None
        try:
            # Decimal preserves large integers expressed in scientific notation or with trailing ".0".
            quantized = Decimal(text)
        except (InvalidOperation, ValueError):
            # Fall back to raw string (e.g., unexpected alphanumeric IDs).
            return text
        if quantized == quantized.to_integral_value():
            return f"{quantized.to_integral_value()}"
        return text

    return series.map(_convert)


def plot_highlighted_umap(
    all_coords: pd.DataFrame,
    lens_coords: pd.DataFrame,
    output_path: Path,
    dpi: int,
    title: str,
) -> None:
    if all_coords.empty:
        raise ValueError("No embedding coordinates available to plot.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        all_coords["umap_x"],
        all_coords["umap_y"],
        c="lightgray",
        s=10,
        alpha=0.35,
        label=f"All embeddings (n={len(all_coords)})",
    )

    if not lens_coords.empty:
        if "expert_score" in lens_coords.columns:
            sc = ax.scatter(
                lens_coords["umap_x"],
                lens_coords["umap_y"],
                c=lens_coords["expert_score"],
                cmap="plasma",
                s=60,
                edgecolor="black",
                linewidths=0.5,
                label=f"Lenses matched (n={len(lens_coords)})",
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.01)
            cbar.set_label("Expert Score")
        else:
            ax.scatter(
                lens_coords["umap_x"],
                lens_coords["umap_y"],
                c="crimson",
                s=60,
                edgecolor="black",
                linewidths=0.5,
                label=f"Lenses matched (n={len(lens_coords)})",
            )
    else:
        ax.text(
            0.5,
            0.95,
            "No lenses matched to embeddings",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved highlighted UMAP to {output_path}")


def fetch_samples_for_grid(
    object_ids: list[str],
    cache_dir: str,
    split: str,
    index_csv: Path | None,
    verbose: bool,
) -> list[dict]:
    if not _DISPLAY_HELPERS_AVAILABLE:  # pragma: no cover - defensive branch
        raise SystemExit(
            "Displaying the image/spectrum grid requires the dataset display helpers, "
            "which in turn depend on PyTorch. Install the 'torch' optional dependency."
        )

    unique_ids = list(dict.fromkeys(object_ids))

    if index_csv is not None:
        index_map = load_index(index_csv)  # type: ignore[operator]
        samples = collect_samples_with_index(  # type: ignore[operator]
            cache_dir=cache_dir,
            object_ids=unique_ids,
            index_map=index_map,
            verbose=verbose,
        )
    else:
        dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=verbose)  # type: ignore[operator]
        samples = collect_samples(dataset, unique_ids, verbose=verbose)  # type: ignore[arg-type,operator]

    samples_by_id = {}
    for sample in samples:
        oid = str(sample.get("object_id"))
        samples_by_id[oid] = sample

    ordered = []
    missing = []
    for oid in object_ids:
        sample = samples_by_id.get(oid)
        if sample is None:
            missing.append(oid)
        else:
            ordered.append(sample)

    if missing:
        print(f"Warning: {len(missing)} lens IDs missing from dataset: {missing[:5]}...")

    return ordered


def prepare_lens_grid(
    lenses: pd.DataFrame,
    output_path: Path | None,
    cache_dir: str,
    split: str,
    index_csv: Path | None,
    max_items: int,
    cols: int,
    verbose: bool,
) -> None:
    if output_path is None:
        return

    if lenses.empty:
        print("No matched lenses available for the spectrum grid.")
        return

    selected = lenses.head(max_items).copy()
    object_ids = [str(oid) for oid in selected["object_id"]]

    samples = fetch_samples_for_grid(
        object_ids=object_ids,
        cache_dir=cache_dir,
        split=split,
        index_csv=index_csv,
        verbose=verbose,
    )

    if not samples:
        print("Unable to retrieve samples for the requested lenses; skipping grid generation.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_vertical_panels(  # type: ignore[operator]
        samples=samples,
        cols=max(1, cols),
        save_path=output_path,
        show=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join the Q1 discovery engine lens catalog with AION embeddings, "
            "highlight the lenses on a UMAP projection, and save an image+spectrum grid."
        )
    )
    parser.add_argument(
        "--lens-csv",
        type=Path,
        default=Path("q1_discovery_engine_lens_catalog.csv"),
        help="Path to the discovery engine lens catalog (with an 'object_id' column).",
    )
    parser.add_argument(
        "--umap-coords",
        type=Path,
        default=Path("umap_coords.csv"),
        help="Optional CSV with pre-computed UMAP coordinates for each embedding key.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Optional .pt file with raw embeddings. Used when --umap-coords is not provided.",
    )
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="embedding_hsc_desi",
        choices=sorted(EMBEDDING_COLUMN_MAP),
        help="Embedding field to highlight on the UMAP.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when computing new UMAP coordinates.",
    )
    parser.add_argument(
        "--output-umap",
        type=Path,
        default=Path("scratch/outputs/lens_umap.png"),
        help="Output path for the highlighted UMAP image.",
    )
    parser.add_argument(
        "--output-grid",
        type=Path,
        default=Path("scratch/outputs/lens_spectrum_grid.png"),
        help="Output path for the lens image+spectrum grid.",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Number of columns in the lens grid.",
    )
    parser.add_argument(
        "--max-grid-items",
        type=int,
        default=12,
        help="Maximum number of lens samples to include in the grid.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Dataset split(s) when fetching samples (used if --index is not provided).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets",
        help="Cache directory for the Euclid+DESI dataset.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Optional CSV mapping object_id -> split/index for faster retrieval.",
    )
    parser.add_argument(
        "--sort-column",
        type=str,
        default="expert_score",
        help="Lens catalog column used to rank items for the grid.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort the lens catalog in ascending order (default is descending).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=350,
        help="Resolution for saved figures.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose dataset loading logs.",
    )
    args = parser.parse_args()

    if not args.lens_csv.exists():
        raise SystemExit(f"Lens catalog not found at {args.lens_csv}")

    lens_catalog = pd.read_csv(args.lens_csv, dtype={"object_id": str})
    if "object_id" not in lens_catalog.columns:
        raise SystemExit("Lens catalog must contain an 'object_id' column.")

    lens_catalog["object_id"] = normalize_object_id_series(lens_catalog["object_id"])
    lens_catalog = lens_catalog.dropna(subset=["object_id"]).copy()

    coords: pd.DataFrame
    if args.embeddings is not None:
        records = load_records(args.embeddings)
        coords = compute_umap_coords(records, args.embedding_key, args.random_state)
    elif args.umap_coords is not None and args.umap_coords.exists():
        coords = load_umap_from_csv(args.umap_coords, args.embedding_key)
    else:
        raise SystemExit(
            "Either provide --embeddings to compute UMAP coordinates or supply --umap-coords "
            "with the desired embedding columns."
        )

    coords["object_id"] = normalize_object_id_series(coords["object_id"])
    coords = coords.dropna(subset=["object_id"]).copy()

    merged = lens_catalog.merge(coords, on="object_id", how="inner", suffixes=("", "_embedding"))
    missing_count = len(lens_catalog) - len(merged)
    if missing_count:
        print(f"Warning: {missing_count} lenses did not match any embeddings.")

    plot_highlighted_umap(
        all_coords=coords,
        lens_coords=merged,
        output_path=args.output_umap,
        dpi=args.dpi,
        title=f"UMAP of {args.embedding_key.replace('embedding_', '').upper()} embeddings with lenses highlighted",
    )

    sort_col = args.sort_column if args.sort_column in merged.columns else "umap_x"
    if sort_col not in merged.columns:
        print(
            f"Sort column '{args.sort_column}' not found after merging; "
            "falling back to UMAP X coordinate ordering."
        )
    merged_sorted = merged.sort_values(by=sort_col, ascending=args.ascending)

    prepare_lens_grid(
        lenses=merged_sorted,
        output_path=args.output_grid,
        cache_dir=args.cache_dir,
        split=args.split,
        index_csv=args.index,
        max_items=args.max_grid_items,
        cols=args.grid_cols,
        verbose=args.verbose,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
