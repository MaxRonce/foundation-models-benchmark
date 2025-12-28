"""
Script to visualize Dual AGN classifier scores on UMAP projections.
It colors the UMAP points based on the score assigned by the Dual AGN regressor.

Usage:
    python -m scratch.plot_dual_agn_scores_umap \
        --embeddings /path/to/embeddings.pt \
        --scores-csv scratch/outputs/dual_agn_scores.csv \
        --output scratch/outputs/dual_agn_scores_umap.png
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


def read_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "object_id" not in (reader.fieldnames or []) or "score" not in (reader.fieldnames or []):
            raise SystemExit(f"CSV '{path}' must contain 'object_id' and 'score' columns")
        for row in reader:
            oid = row.get("object_id")
            score = row.get("score")
            if oid is None or score is None:
                continue
            oid = oid.strip()
            try:
                value = float(score)
            except ValueError:
                continue
            scores[oid] = value
    return scores


def compute_umap_coords(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(embeddings)


def standardize(array: np.ndarray) -> np.ndarray:
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (array - mean) / std


def plot_scores(
    coords: np.ndarray,
    scores: np.ndarray,
    embedding_key: str,
    ax: plt.Axes,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=scores,
        s=8,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        alpha=0.9,
    )
    title = embedding_key.replace("embedding_", "").replace("_", " ").upper()
    ax.set_title(f"{title} UMAP (Dual AGN score)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.2)
    return sc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colour UMAP projections by Dual AGN regression scores.",
    )
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .pt file")
    parser.add_argument("--scores-csv", required=True, help="CSV produced by scratch.train_dual_agn_regressor")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--embedding-key",
        choices=EMBEDDING_KEYS,
        nargs="+",
        default=EMBEDDING_KEYS,
        help="Embedding key(s) to visualise",
    )
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--standardize", action="store_true", help="Standardize features before UMAP")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    embeddings_path = Path(args.embeddings)
    scores_path = Path(args.scores_csv)
    output_path = Path(args.output)

    score_dict = read_scores(scores_path)
    records = load_records(embeddings_path)

    selected_keys = args.embedding_key
    fig, axes = plt.subplots(1, len(selected_keys), figsize=(6 * len(selected_keys), 5))
    if len(selected_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, selected_keys):
        try:
            embeddings = stack_embeddings(records, key)
        except ValueError as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        if args.standardize:
            embeddings = standardize(embeddings)
        object_ids = [str(rec.get("object_id", "")) for rec in records]
        scores = np.array([score_dict.get(oid, 0.0) for oid in object_ids], dtype=float)
        coords = compute_umap_coords(
            embeddings,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
        )
        sc = plot_scores(coords, scores, key, ax, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label("Dual AGN score")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved Dual AGN score UMAP to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
