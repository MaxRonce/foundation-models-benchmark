"""
Script to highlight known Dual AGN candidates on UMAP projections.
It reads a list of Dual AGN object IDs and overlays them on the UMAP of embeddings.

Usage:
    python -m scratch.highlight_dual_agn_umap \
        --embeddings /path/to/embeddings.pt \
        --dual-csv /path/to/dual_agn_catalog.csv \
        --output scratch/outputs/dual_agn_umap.png
"""
import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'umap-learn' package is required. Install it with 'pip install umap-learn'.") from exc

from scratch.detect_outliers import EMBEDDING_KEYS, load_records, stack_embeddings  # type: ignore[import]


def read_dual_ids(path: Path) -> set[str]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "object_id" not in (reader.fieldnames or []):
            raise SystemExit(f"CSV '{path}' must contain an 'object_id' column")
        dual_ids = {row["object_id"].strip() for row in reader if row.get("object_id")}
    return dual_ids


def prepare_coords(
    records: list[dict],
    embedding_key: str,
    dual_ids: set[str],
    random_state: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    embeddings = stack_embeddings(records, embedding_key)
    object_ids = [str(rec.get("object_id", "")) for rec in records]
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    coords = reducer.fit_transform(embeddings)
    mask = np.array([oid in dual_ids for oid in object_ids], dtype=bool)
    return coords, mask, object_ids


def plot_dual_umap(
    coords: np.ndarray,
    mask: np.ndarray,
    embedding_key: str,
    ax: plt.Axes,
) -> None:
    ax.scatter(coords[~mask, 0], coords[~mask, 1], s=6, color="lightgray", alpha=0.4, label="Others")
    if mask.any():
        ax.scatter(coords[mask, 0], coords[mask, 1], s=35, color="crimson", edgecolor="black", linewidth=0.4, label="Dual AGN")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No Dual AGN in embedding", ha="center", va="center", transform=ax.transAxes)
    title = embedding_key.replace("embedding_", "").replace("_", " ").upper()
    ax.set_title(f"{title} UMAP")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.2)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Highlight dual AGN candidates on UMAP projections of each embedding set.",
    )
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .pt file")
    parser.add_argument("--dual-csv", required=True, help="CSV containing dual AGN object_ids")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--embedding-key", choices=EMBEDDING_KEYS, nargs="+", default=EMBEDDING_KEYS)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    embeddings_path = Path(args.embeddings)
    dual_path = Path(args.dual_csv)
    output_path = Path(args.output)

    records = load_records(embeddings_path)
    dual_ids = read_dual_ids(dual_path)

    selected_keys = args.embedding_key
    fig, axes = plt.subplots(1, len(selected_keys), figsize=(6 * len(selected_keys), 5))
    if len(selected_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, selected_keys):
        try:
            coords, mask, _ = prepare_coords(
                records,
                embedding_key=key,
                dual_ids=dual_ids,
                random_state=args.random_state,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
            )
        except ValueError as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        plot_dual_umap(coords, mask, key, ax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved dual AGN UMAP figure to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
