#!/usr/bin/env python3
"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.visualize_multimodel_embeddings
Description: FMB module: fmb.viz.visualize_multimodel_embeddings
"""

"""
Script to visualize UMAP projections of embeddings from AION, AstroPT, and AstroCLIP models,
colored by physical parameters from a FITS catalog.

This script:
- Loads embeddings from AION, AstroPT, and AstroCLIP
- Key prefixing is applied to avoid collisions (e.g. 'embedding_joint')
- Matches them with a FITS catalog to extract physical parameters
- Computes UMAP projections for all embedding types
- Generates a grid visualization with color-coded physical parameters

Usage:
    python -m scratch.visualize_multimodel_embeddings \
        --aion-embeddings /path/to/aion.pt \
        --astropt-embeddings /path/to/astropt.pt \
        --astroclip-embeddings /path/to/astroclip.pt \
        --catalog /path/to/catalog.fits \
        --output-dir /path/to/output \
        --physical-param redshift \
        --random-state 42
"""
import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from astropy.io import fits
except ImportError as exc:
    raise SystemExit(
        "The 'astropy' package is required. Install it with 'pip install astropy'."
    ) from exc

try:
    import umap
except ImportError as exc:
    raise SystemExit(
        "The 'umap-learn' package is required. Install it with 'pip install umap-learn'."
    ) from exc


# Embedding keys for models
AION_EMBEDDING_KEYS = [
    "embedding_hsc_desi",
    "embedding_hsc",
    "embedding_spectrum",
]

ASTROPT_EMBEDDING_KEYS = [
    "embedding_images",
    "embedding_spectra",
    "embedding_joint",
]

ASTROCLIP_EMBEDDING_KEYS = [
    "embedding_images",
    "embedding_spectra",
    "embedding_joint",
]

# UMAP preset configurations
UMAP_PRESETS = {
    "balanced": {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "metric": "cosine",
        "densmap": True,
        "description": "Équilibré - Structure globale + densité (recommandé)",
    },
    "local": {
        "n_neighbors": 15,
        "min_dist": 0.05,
        "metric": "cosine",
        "densmap": True,
        "description": "Local - Structure locale fine + densité",
    },
    "global": {
        "n_neighbors": 50,
        "min_dist": 0.1,
        "metric": "cosine",
        "densmap": False,
        "description": "Global - Vue d'ensemble (plus rapide)",
    },
}


def load_embeddings(path: Path) -> list[dict]:
    """Load embedding records from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def load_fits_catalog(path: Path) -> tuple[dict, list[str], str]:
    """Load FITS catalog."""
    with fits.open(path) as hdul:
        data = hdul[1].data
        columns = hdul[1].columns.names
        catalog_dict = {}
        id_column = None
        for priority_col in ["TARGETID", "targetid", "TargetID"]:
            if priority_col in columns:
                id_column = priority_col
                break
        if id_column is None:
            for col in columns:
                if col.lower() in ["object_id", "objid", "id"]:
                    id_column = col
                    break
        if id_column is None:
            raise ValueError(
                f"Could not find object ID column in FITS. Available columns: {columns}"
            )

        print(f"Using '{id_column}' as object ID column")

        for row in data:
            obj_id = str(row[id_column])
            catalog_dict[obj_id] = {col: row[col] for col in columns}

        numeric_columns = []
        for col in columns:
            if col == id_column:
                continue
            try:
                col_format = hdul[1].columns[col].format
                if any(
                    fmt in col_format.upper() for fmt in ["E", "D", "I", "J", "K", "F"]
                ):
                    numeric_columns.append(col)
                    continue
                # sample check can be skipped for brevity or kept if robustness needed
            except:
                pass

        print(f"Found {len(numeric_columns)} numeric physical parameters")
        return catalog_dict, numeric_columns, id_column


def stack_embeddings_with_joint(records: Sequence[dict], key: str) -> np.ndarray:
    """Stack embeddings for a given key, with joint handling."""
    vectors = []
    # Key might be prefixed, e.g. "astropt_embedding_joint".
    # But records have keys "astropt_embedding_joint" directly merged?
    # Yes, we merge with prefixes.

    for rec in records:
        tensor = rec.get(key)

        # Fallback: if key is "astropt_embedding_joint" but it's missing, try to construct from existing fields
        # BUT `merge_embedding_records` handles the merging and prefixing.
        # So we should rely on keys simply being present.

        # Exception: "joint" construction from components.
        # If the merged record has "astropt_embedding_images" and "astropt_embedding_spectra", we can built joint.
        if tensor is None and "embedding_joint" in key:
            prefix = key.replace("embedding_joint", "")  # "astropt_"
            img_key = f"{prefix}embedding_images"
            spec_key = f"{prefix}embedding_spectra"

            img = rec.get(img_key)
            spec = rec.get(spec_key)

            if img is not None and spec is not None:
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu().numpy()
                else:
                    img = np.asarray(img)
                if isinstance(spec, torch.Tensor):
                    spec = spec.detach().cpu().numpy()
                else:
                    spec = np.asarray(spec)
                tensor = np.concatenate([img, spec])

        if tensor is None:
            continue

        if isinstance(tensor, torch.Tensor):
            vectors.append(tensor.detach().cpu().numpy())
        else:
            vectors.append(np.asarray(tensor))

    if not vectors:
        raise ValueError(f"No embeddings found for key '{key}'")
    return np.stack(vectors, axis=0)


def save_umap_coordinates(coords_map: dict[str, np.ndarray], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(coords_map, save_path)
    print(f"  Saved UMAP coordinates to {save_path}")


def load_umap_coordinates(load_path: Path) -> dict[str, np.ndarray]:
    coords_map = torch.load(load_path, map_location="cpu", weights_only=False)
    for key in coords_map:
        if isinstance(coords_map[key], torch.Tensor):
            coords_map[key] = coords_map[key].numpy()
    print(f"  Loaded UMAP coordinates from {load_path}")
    return coords_map


def compute_umap(
    embeddings: np.ndarray, random_state: int, preset: str = "balanced"
) -> np.ndarray:
    config = UMAP_PRESETS[preset]
    reducer = umap.UMAP(
        random_state=random_state,
        n_neighbors=config["n_neighbors"],
        min_dist=config["min_dist"],
        metric=config["metric"],
        densmap=config["densmap"],
    )
    return reducer.fit_transform(embeddings)


def merge_embedding_records(
    aion_records: list[dict],
    astropt_records: list[dict],
    astroclip_records: list[dict],
    catalog: dict,
    physical_param: str,
) -> tuple[list[dict], np.ndarray, list[str]]:
    """
    Merge AION, AstroPT, and AstroCLIP embeddings with PROPER PREFIXING.
    """
    # Build dictionaries
    aion_dict = {str(r.get("object_id", "")): r for r in aion_records}
    astropt_dict = {str(r.get("object_id", "")): r for r in astropt_records}
    astroclip_dict = {str(r.get("object_id", "")): r for r in astroclip_records}

    all_ids = (
        set(aion_dict.keys()) | set(astropt_dict.keys()) | set(astroclip_dict.keys())
    )
    all_ids.discard("")

    merged_records = []
    physical_values = []
    valid_ids = []

    for obj_id in sorted(all_ids):
        # We start with empty dict, NOT reusing aion_dict content directly to strictly control keys.
        # But for AION we can keep original keys or prefix them?
        # AION keys (hsc_desi) are generally unique.
        # But let's standardize: "aion_" vs "astropt_" vs "astroclip_"
        # To maintain compatibility with existing cache consumer `plot_paper_combined_umap.py` lines:
        # KEY_AION_CACHE = "aion_embedding_hsc_desi"
        # KEY_ASTROPT_CACHE = "astropt_embedding_joint"
        # KEY_ASTROCLIP_CACHE = "astroclip_embedding_joint"

        merged_rec = {"object_id": obj_id}

        # Merge AION
        if obj_id in aion_dict:
            for k in AION_EMBEDDING_KEYS:
                if k in aion_dict[obj_id]:
                    # Standardize prefix "aion_" if strictly following updated plan
                    # OR check if keys already overlap.
                    merged_rec[f"aion_{k}"] = aion_dict[obj_id][k]

        # Merge AstroPT
        if obj_id in astropt_dict:
            for k in ASTROPT_EMBEDDING_KEYS:
                if k in astropt_dict[obj_id]:
                    merged_rec[f"astropt_{k}"] = astropt_dict[obj_id][k]

        # Merge AstroCLIP
        if obj_id in astroclip_dict:
            for k in ASTROCLIP_EMBEDDING_KEYS:
                if k in astroclip_dict[obj_id]:
                    merged_rec[f"astroclip_{k}"] = astroclip_dict[obj_id][k]

        # Get physical parameter
        phys_val = np.nan
        if obj_id in catalog:
            try:
                raw_val = catalog[obj_id][physical_param]
                if hasattr(raw_val, "item"):
                    phys_val = float(raw_val.item())
                else:
                    phys_val = float(raw_val)
                if np.isnan(phys_val) or np.isinf(phys_val):
                    phys_val = np.nan
            except:
                pass

        merged_records.append(merged_rec)
        physical_values.append(phys_val)
        valid_ids.append(obj_id)

    return merged_records, np.array(physical_values), valid_ids


def plot_umap_grid(
    coords_map: dict[str, np.ndarray],
    colors: np.ndarray,
    param_name: str,
    save_path: Path,
) -> None:
    # Sort keys for consistent display
    # Group by model
    # Prefer order: AION -> AstroPT -> AstroCLIP and within that: Image -> Spec -> Joint

    # helper for sorting
    def sort_key(k):
        score = 0
        if "aion" in k:
            score += 100
        elif "astropt" in k:
            score += 200
        elif "astroclip" in k:
            score += 300

        if "image" in k or "hsc" in k:
            score += 1
        elif "spectr" in k:
            score += 2  # spectrum or spectra
        elif "joint" in k or "hsc_desi" in k:
            score += 3
        return score

    names = sorted(list(coords_map.keys()), key=sort_key)
    n_plots = len(names)

    # Flexible Grid
    cols = 3
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Colors
    valid_mask = ~np.isnan(colors)
    vmin, vmax = 0, 1
    if valid_mask.any():
        valid_colors = colors[valid_mask]
        vmin = np.percentile(valid_colors, 2)
        vmax = np.percentile(valid_colors, 98)
        if vmax - vmin < 1e-6:
            vmin, vmax = valid_colors.min(), valid_colors.max()
        print(f"    🎨 Color scale: [{vmin:.3f}, {vmax:.3f}]")

    for i, name in enumerate(names):
        ax = axes[i]
        coords = coords_map[name]

        if valid_mask.any():
            scatter = ax.scatter(
                coords[valid_mask, 0],
                coords[valid_mask, 1],
                c=colors[valid_mask],
                cmap="viridis",
                s=10,
                alpha=0.7,
                edgecolors="none",
                vmin=vmin,
                vmax=vmax,
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(param_name, fontsize=8)

        if (~valid_mask).any():
            ax.scatter(
                coords[~valid_mask, 0],
                coords[~valid_mask, 1],
                s=10,
                color="lightgray",
                alpha=0.3,
                label="N/A",
            )

        pretty = name.replace("embedding_", "").replace("_", " ").title()
        ax.set_title(pretty, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)

    # Hide unused
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Multi-Model UMAP Projections: {param_name}", fontsize=16, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # space for tile
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved UMAP grid to {save_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Model UMAP Visualization (AION/AstroPT/AstroCLIP)"
    )
    parser.add_argument(
        "--aion-embeddings", required=True, help="Path to AION embeddings .pt"
    )
    parser.add_argument(
        "--astropt-embeddings", required=True, help="Path to AstroPT embeddings .pt"
    )
    parser.add_argument(
        "--astroclip-embeddings", required=True, help="Path to AstroCLIP embeddings .pt"
    )
    parser.add_argument("--catalog", required=True, help="Path to FITS catalog")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--physical-param", help="Physical parameter to color by")
    parser.add_argument(
        "--all-params", action="store_true", help="Visualize all numeric params"
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--umap-cache", help="Path to cache file")
    parser.add_argument(
        "--umap-preset", default="balanced", choices=list(UMAP_PRESETS.keys())
    )

    args = parser.parse_args(argv)

    print("Loading data...")
    aion_recs = load_embeddings(Path(args.aion_embeddings))
    astropt_recs = load_embeddings(Path(args.astropt_embeddings))
    astroclip_recs = load_embeddings(Path(args.astroclip_embeddings))
    catalog, numeric_cols, _ = load_fits_catalog(Path(args.catalog))

    params_to_viz = (
        numeric_cols
        if args.all_params
        else ([args.physical_param] if args.physical_param else [])
    )
    if not params_to_viz:
        print(
            "No parameters selected to visualize (use --physical-param or --all-params)."
        )
        return

    # Compute UMAPs (Merged dummy first)
    print("Merging records and computing UMAPs...")
    all_recs, _, _ = merge_embedding_records(
        aion_recs, astropt_recs, astroclip_recs, catalog, params_to_viz[0]
    )

    coords_map = {}
    if args.umap_cache and Path(args.umap_cache).exists():
        coords_map = load_umap_coordinates(Path(args.umap_cache))
    else:
        # Need to find all valid keys in merged records (that are embeddings)
        sample_keys = [
            k for k in all_recs[0].keys() if k not in ["object_id", "redshift"]
        ]
        for key in sample_keys:
            try:
                embeddings = stack_embeddings_with_joint(all_recs, key)
                print(f"  Computing UMAP for {key}...")
                coords_map[key] = compute_umap(
                    embeddings, args.random_state, args.umap_preset
                )
            except ValueError:
                print(f"  Skipping {key} (no data)")

        if args.umap_cache:
            save_umap_coordinates(coords_map, Path(args.umap_cache))

    # Plotting
    for param in params_to_viz:
        print(f"Plotting {param}...")
        _, values, _ = merge_embedding_records(
            aion_recs, astropt_recs, astroclip_recs, catalog, param
        )
        save_path = Path(args.output_dir) / f"umap_grid_{param}.png"
        plot_umap_grid(coords_map, values, param, save_path)

    print("Done.")


if __name__ == "__main__":
    main()
