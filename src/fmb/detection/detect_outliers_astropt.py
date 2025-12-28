"""
Script to detect outliers in the embedding space using Isolation Forest.
It identifies anomalous objects based on their embeddings and visualizes them on UMAP plots.
It also saves the list of outlier IDs for further analysis.

Adapted for AstroPT embeddings (images + spectra).

Usage:
    python -m scratch.detect_outliers_astropt \
        --input /path/to/astropt_embeddings.pt \
        --figure-joint umap_joint_outliers.png \
        --outliers-joint outliers_joint.csv \
        --contamination 0.02
"""
import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import umap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'umap-learn' package is required. Install it with 'pip install umap-learn'.") from exc

try:
    from sklearn.ensemble import IsolationForest
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scikit-learn is required. Install it with 'pip install scikit-learn'.") from exc


EMBEDDING_KEYS = [
    "embedding_joint",
    "embedding_images",
    "embedding_spectra",
]


def load_records(path: Path) -> list[dict]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def stack_embeddings(records: Sequence[dict], key: str) -> np.ndarray:
    vectors = []
    for rec in records:
        # Special handling for joint embedding
        if key == "embedding_joint":
            img = rec.get("embedding_images")
            spec = rec.get("embedding_spectra")
            if img is None or spec is None:
                continue
            
            # Ensure numpy arrays
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            else:
                img = np.asarray(img)
                
            if isinstance(spec, torch.Tensor):
                spec = spec.detach().cpu().numpy()
            else:
                spec = np.asarray(spec)
                
            # Concatenate
            vectors.append(np.concatenate([img, spec]))
        else:
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


def run_isolation_forest(embeddings: np.ndarray, contamination: float, random_state: int) -> np.ndarray:
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(embeddings)
    return labels == -1


def compute_umap(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    reducer = umap.UMAP(random_state=random_state)
    return reducer.fit_transform(embeddings)


def plot_umap_highlights(
    base_key: str,
    coords_map: dict[str, np.ndarray],
    outlier_masks: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    ordered = [base_key] + [k for k in coords_map.keys() if k != base_key]
    coords = coords_map[base_key]
    fig, axes = plt.subplots(1, len(ordered), figsize=(6 * len(ordered), 5))
    if len(ordered) == 1:
        axes = [axes]
    for ax, highlight_key in zip(axes, ordered):
        mask = outlier_masks[highlight_key]
        ax.scatter(
            coords[~mask, 0],
            coords[~mask, 1],
            s=8,
            color="lightgray",
            alpha=0.5,
            label="Inliers",
        )
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=24,
                color="crimson",
                edgecolors="black",
                linewidths=0.5,
                label=highlight_key.replace("embedding_", "").replace("_", " ").upper(),
            )
        title = (
            f"{base_key.replace('embedding_', '').replace('_', ' ').upper()} UMAP\n"
            f"Highlight: {highlight_key.replace('embedding_', '').replace('_', ' ').upper()}"
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def save_umap_csv(path: Path, object_ids: Sequence[str], coords_map: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["object_id"]
    for key in coords_map:
        headers.extend([f"{key}_x", f"{key}_y"])
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for idx, oid in enumerate(object_ids):
            row = [oid]
            for key in coords_map:
                coord = coords_map[key][idx]
                row.extend([f"{coord[0]:.6f}", f"{coord[1]:.6f}"])
            writer.writerow(row)


def save_outlier_ids(path: Path, object_ids: Sequence[str], mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["object_id"])
        for oid in np.array(object_ids)[mask]:
            writer.writerow([oid])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Detect embedding outliers with Isolation Forest and visualize them (AstroPT version)",
    )
    parser.add_argument("--input", required=True, help="Path to embeddings .pt file")
    parser.add_argument("--figure-joint", required=True, help="Output path for Joint highlight figure")
    parser.add_argument("--figure-images", required=True, help="Output path for Images-only highlight figure")
    parser.add_argument("--figure-spectra", required=True, help="Output path for Spectra-only highlight figure")
    parser.add_argument("--umap-csv", required=True, help="CSV path to store UMAP coordinates for all embeddings")
    parser.add_argument("--outliers-joint", required=True, help="CSV path for Joint outlier IDs")
    parser.add_argument("--outliers-images", required=True, help="CSV path for Images-only outlier IDs")
    parser.add_argument("--outliers-spectra", required=True, help="CSV path for Spectra-only outlier IDs")
    parser.add_argument("--contamination", type=float, default=0.02, help="Isolation Forest contamination fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args(argv)

    records = load_records(Path(args.input))
    object_ids = [rec.get("object_id", "") for rec in records]

    embeddings = {}
    for key in EMBEDDING_KEYS:
        try:
            embeddings[key] = stack_embeddings(records, key)
        except ValueError:
            continue
    
    if len(embeddings) < 2:
        # We might accept just 2 if one is joint and one is single, but ideally we want all 3
        print(f"Warning: Found only {len(embeddings)} embedding types: {list(embeddings.keys())}")

    outlier_masks = {
        key: run_isolation_forest(array, args.contamination, args.random_state)
        for key, array in embeddings.items()
    }
    coords_map = {
        key: compute_umap(array, random_state=args.random_state)
        for key, array in embeddings.items()
    }

    figure_paths = {
        "embedding_joint": Path(args.figure_joint),
        "embedding_images": Path(args.figure_images),
        "embedding_spectra": Path(args.figure_spectra),
    }

    for key, path in figure_paths.items():
        if key in embeddings:
            plot_umap_highlights(key, coords_map, outlier_masks, path)

    save_umap_csv(Path(args.umap_csv), object_ids, coords_map)

    outlier_outputs = {
        "embedding_joint": Path(args.outliers_joint),
        "embedding_images": Path(args.outliers_images),
        "embedding_spectra": Path(args.outliers_spectra),
    }
    for key, path in outlier_outputs.items():
        if key in outlier_masks:
            save_outlier_ids(path, object_ids, outlier_masks[key])

    for key, mask in outlier_masks.items():
        print(f"Detected {mask.sum()} outliers for {key} (contamination={args.contamination})")
    print(f"UMAP coordinates saved to {args.umap_csv}")


if __name__ == "__main__":
    main()
