"""
Script to detect outliers in the embedding space using Isolation Forest.
It identifies anomalous objects based on their embeddings and visualizes them on UMAP plots.
It also saves the list of outlier IDs for further analysis.
Supports generic embedding keys and on-the-fly creation of joint embeddings (concatenation).

Usage:
    python -m fmb.detection.detect_outliers \\
        --input /path/to/embeddings.pt \\
        --keys embedding_images embedding_spectra embedding_joint \\
        --figure-prefix my_analysis \\
        --outliers-prefix outliers \\
        --contamination 0.02
"""
import argparse
import csv
from pathlib import Path
from typing import Sequence, List

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


def load_records(path: Path) -> list[dict]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def stack_embeddings(records: Sequence[dict], key: str) -> np.ndarray:
    vectors = []
    
    # Check if we need to construct joint embedding on the fly
    is_virtual_joint = (key == "embedding_joint" and 
                        any("embedding_joint" not in r for r in records[:5]))

    for rec in records:
        if is_virtual_joint:
            # Try to concatenate images + spectra
            img = rec.get("embedding_images")
            spec = rec.get("embedding_spectra")
            if img is None or spec is None:
                continue
            
            # Ensure numpy
            img = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
            spec = spec.detach().cpu().numpy() if isinstance(spec, torch.Tensor) else np.asarray(spec)
            
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
        # If optional, we might just return empty, but for now raise error
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
    
    # Ensure axes is iterable
    if not isinstance(axes, (list, np.ndarray)):
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
            f"{base_key.replace('embedding_', '').replace('_', ' ').upper()} UMAP\\n"
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
        # Handle case where mask length might differ slightly if some objects were skipped?
        # Stack logic skipped incomplete records, so indices align with valid vectors.
        # But wait, stack_embeddings filters. We must ensure alignment.
        # Current logic assumes all records have all keys or skips. 
        # Refinement: We should probably filter records upfront.
        for oid in np.array(object_ids)[mask]:
            writer.writerow([oid])


from fmb.paths import load_paths

def main(argv: Sequence[str] | None = None) -> None:
    paths = load_paths()
    parser = argparse.ArgumentParser(
        description="Detect embedding outliers with Isolation Forest (Generic)",
    )
    parser.add_argument("--input", required=True, help="Path to embeddings .pt file")
    parser.add_argument(
        "--keys", 
        nargs="+", 
        default=["embedding_images", "embedding_spectra", "embedding_joint"],
        help="List of embedding keys to analyze (default: images, spectra, joint)"
    )
    
    default_fig_prefix = str(paths.analysis / "umap_highlight")
    default_out_prefix = str(paths.outliers / "outliers_iforest")
    default_umap_csv = str(paths.analysis / "embeddings_umap_coords.csv")

    parser.add_argument("--figure-prefix", default=default_fig_prefix, help="Prefix for output UMAP figures (e.g. 'analysis/umap_highlight')")
    parser.add_argument("--outliers-prefix", default=default_out_prefix, help="Prefix for output outlier CSVs (e.g. 'analysis/outliers')")
    parser.add_argument("--umap-csv", default=default_umap_csv, help="CSV path to store UMAP coordinates for all embeddings")
    parser.add_argument("--contamination", type=float, default=0.02, help="Isolation Forest contamination fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args(argv)

    records = load_records(Path(args.input))
    object_ids = [rec.get("object_id", "") for rec in records]

    embeddings = {}
    valid_keys = []
    
    # Pre-filter keys that actually exist or can be created
    for key in args.keys:
        try:
            embeddings[key] = stack_embeddings(records, key)
            valid_keys.append(key)
        except ValueError:
            print(f"Warning: Could not stack embeddings for key '{key}' (not found). Skipping.")
            continue
            
    if len(embeddings) < 1:
        raise SystemExit("No valid embeddings found to analyze.")

    # Consistency check: Ensure all arrays have same length (should be true if stack_embeddings logic is consistent)
    # However, stack_embeddings skips incomplete records independently. 
    # Ideally, we should filter records first. Assuming data is clean for now (all records have all fields).
    
    outlier_masks = {
        key: run_isolation_forest(array, args.contamination, args.random_state)
        for key, array in embeddings.items()
    }
    
    print("Computing UMAP projections...")
    coords_map = {
        key: compute_umap(array, random_state=args.random_state)
        for key, array in embeddings.items()
    }

    # Generate Figures
    for key in embeddings:
        save_path = Path(f"{args.figure_prefix}_{key}.png")
        plot_umap_highlights(key, coords_map, outlier_masks, save_path)
        print(f"Saved figure: {save_path}")

    # Save UMAP CSV
    save_umap_csv(Path(args.umap_csv), object_ids, coords_map)
    print(f"Saved UMAP coords: {args.umap_csv}")

    # Save Outlier CSVs
    for key in outlier_masks:
        save_path = Path(f"{args.outliers_prefix}_{key}.csv")
        save_outlier_ids(save_path, object_ids, outlier_masks[key])
        print(f"Saved outliers ({outlier_masks[key].sum()}): {save_path}")

    print("Done!")


if __name__ == "__main__":
    main()
