"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.plot_anomaly_scores
Description: FMB module: fmb.viz.plot_anomaly_scores
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
    raise SystemExit(
        "The 'umap-learn' package is required. Install it with 'pip install umap-learn'."
    ) from exc

from .detect_outliers import EMBEDDING_KEYS  # type: ignore[import]
from .detect_outliers import load_records, stack_embeddings

METRICS: list[tuple[str, str, str]] = [
    ("log_prob", "Log Probability", "viridis"),
    ("anomaly_sigma", "Anomaly Sigma (z-score)", "coolwarm"),
    ("rank", "Anomaly Rank", "viridis_r"),
]


def parse_scores_csv(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    by_key: dict[str, dict[str, dict[str, float]]] = {}
    with path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        required = {
            "object_id",
            "embedding_key",
            "log_prob",
            "neg_log_prob",
            "anomaly_sigma",
            "rank",
        }
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise SystemExit(
                f"Scores CSV is missing required columns: {sorted(missing_cols)}"
            )
        for row in reader:
            key = row["embedding_key"]
            object_id = row["object_id"]
            if key not in by_key:
                by_key[key] = {}
            by_key[key][object_id] = {
                "log_prob": float(row["log_prob"]),
                "anomaly_sigma": float(row["anomaly_sigma"]),
                "rank": float(row["rank"]),
            }
    return by_key


def compute_umap_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def standardize(array: np.ndarray) -> np.ndarray:
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (array - mean) / std


def sanitize_filename(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_").replace("embedding_", "")


def plot_metric_umap(
    coords: np.ndarray,
    values: np.ndarray,
    embedding_key: str,
    metric_key: str,
    metric_label: str,
    cmap: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=12, linewidths=0, alpha=0.9
    )
    ax.set_title(f"{embedding_key.replace('embedding_', '').upper()} – {metric_label}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.1)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(metric_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot UMAP projections of embeddings coloured by flow-based anomaly scores.",
    )
    parser.add_argument(
        "--embeddings", required=True, help="Path to embeddings .pt file"
    )
    parser.add_argument(
        "--scores-csv",
        required=True,
        help="CSV produced by scratch.detect_outliers_NFs",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to store generated figures"
    )
    parser.add_argument(
        "--embedding-key",
        choices=EMBEDDING_KEYS,
        nargs="+",
        help="Subset of embedding keys to plot. Defaults to keys present in the CSV.",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=30, help="UMAP number of neighbours"
    )
    parser.add_argument(
        "--min-dist", type=float, default=0.05, help="UMAP minimum distance"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for UMAP reproducibility",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable feature standardization before UMAP.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    embeddings_path = Path(args.embeddings)
    scores_path = Path(args.scores_csv)
    output_dir = Path(args.output_dir)

    score_map = parse_scores_csv(scores_path)
    requested_keys = args.embedding_key or list(score_map.keys())

    records = load_records(embeddings_path)

    for key in requested_keys:
        if key not in score_map:
            print(f"[skip] No scores found for embedding key '{key}'")
            continue
        try:
            embeddings = stack_embeddings(records, key)
        except ValueError as err:
            print(f"[skip] {err}")
            continue
        object_ids = [rec.get("object_id", "") for rec in records]
        rows = []
        vectors = []
        for idx, oid in enumerate(object_ids):
            scores_for_key = score_map[key].get(str(oid))
            if scores_for_key is None:
                continue
            vectors.append(embeddings[idx])
            rows.append(scores_for_key)

        if not vectors:
            print(f"[skip] Found no overlapping objects for key '{key}'")
            continue

        array = np.stack(vectors, axis=0)
        if not args.no_standardize:
            array = standardize(array)
        coords = compute_umap_embeddings(
            array,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
        )

        print(f"[{key}] plotting {len(rows)} objects")
        for metric_key, metric_label, cmap in METRICS:
            values = np.array([row[metric_key] for row in rows], dtype=float)
            filename = f"{sanitize_filename(key)}_{metric_key}_umap.png"
            plot_metric_umap(
                coords,
                values,
                embedding_key=key,
                metric_key=metric_key,
                metric_label=metric_label,
                cmap=cmap,
                output_path=output_dir / filename,
            )


if __name__ == "__main__":
    main()
