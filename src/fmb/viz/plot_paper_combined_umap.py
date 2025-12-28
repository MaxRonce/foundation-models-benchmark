"""
Script to generate a publication-ready combined UMAP figure.
Displays four panels in a 2x2 grid:
- Top Row: UMAP colored by physical parameter (e.g., Redshift) for AstroPT and AION.
- Bottom Row: UMAP with thumbnails for AstroPT and AION.

Usage:
    python -m scratch.plot_paper_combined_umap \
        --aion-embeddings /path/to/aion.pt \
        --astropt-embeddings /path/to/astropt.pt \
        --catalog /path/to/catalog.fits \
        --index euclid_index.csv \
        --coords-cache paper/umap_coords_cache.pt \
        --physical-param Z \
        --save paper/paper_combined_umap.png \
        --grid-rows 25 --grid-cols 25
"""
import argparse
from pathlib import Path
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from tqdm import tqdm
import umap

from scratch.display_outlier_images import (
    load_index,
    collect_samples,
    collect_samples_with_index,
    prepare_rgb_image,
)
from scratch.load_display_data import EuclidDESIDataset

# --- Publication Style Settings ---
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
except Exception:
    print("Warning: LaTeX not available, falling back to STIX fonts.")
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    })

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 14,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

KEY_ASTROPT = "embedding_joint"
KEY_AION = "embedding_hsc_desi"

def load_embeddings(path: Path) -> list[dict]:
    """Load embedding records from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")

def load_coordinates(load_path: Path) -> dict[str, np.ndarray]:
    """Load previously computed coordinates (t-SNE or UMAP)."""
    if not load_path.exists():
        raise FileNotFoundError(f"Cache file not found: {load_path}")
    
    coords_map = torch.load(load_path, map_location="cpu", weights_only=False)
    # Convert tensors to numpy if needed
    for key in coords_map:
        if isinstance(coords_map[key], torch.Tensor):
            coords_map[key] = coords_map[key].numpy()
    print(f"  Loaded coordinates from {load_path}")
    return coords_map

def load_fits_catalog(path: Path) -> tuple[dict, str]:
    """Load FITS catalog and return dict mapping object_id -> row."""
    with fits.open(path) as hdul:
        data = hdul[1].data
        columns = hdul[1].columns.names
        
        catalog_dict = {}
        id_column = None
        
        # Find ID column
        for priority_col in ['TARGETID', 'targetid', 'TargetID', 'object_id', 'objid', 'id']:
            if priority_col in columns:
                id_column = priority_col
                break
        
        if id_column is None:
            raise ValueError(f"Could not find object ID column in FITS. Available: {columns}")
            
        print(f"Using '{id_column}' as object ID column")
        
        for row in data:
            obj_id = str(row[id_column])
            catalog_dict[obj_id] = {col: row[col] for col in columns}
            
    return catalog_dict, id_column

def assign_to_grid(
    object_ids: Sequence[str],
    coords: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    random_state: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    if grid_rows <= 0 or grid_cols <= 0 or coords.size == 0:
        return [], []

    # Normalize coords to [0, 1]
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    
    # Avoid div by zero
    if max_x == min_x: max_x += 1e-9
    if max_y == min_y: max_y += 1e-9
    
    norm_x = (coords[:, 0] - min_x) / (max_x - min_x)
    norm_y = (coords[:, 1] - min_y) / (max_y - min_y)

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (nx, ny) in enumerate(zip(norm_x, norm_y)):
        gx = int(nx * grid_cols)
        gy = int(ny * grid_rows)
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
    grid_rows: int,
    grid_cols: int,
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
            
            # Add image
            ax.imshow(
                image,
                extent=(xmin, xmax, ymin, ymax),
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                zorder=10,
            )
            
            # Add frame
            rect = plt.Rectangle((xmin, ymin), 1, 1, linewidth=0.5, edgecolor='white', facecolor='none', zorder=11, alpha=0.5)
            ax.add_patch(rect)
            
        except Exception as exc:
            print(f"Failed to attach thumbnail for object {sample.get('object_id')}: {exc}")

def plot_scatter_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "plasma",
    point_size: float = 3.0,
    use_hexbin: bool = False,
) -> None:
    mask = ~np.isnan(values)
    
    # Normalize coordinates to [0, 1] to match the grid visualization shape
    c_min = coords.min(axis=0)
    c_max = coords.max(axis=0)
    denom = c_max - c_min
    denom[denom == 0] = 1e-9
    norm_coords = (coords - c_min) / denom
    
    mappable = None
    
    if use_hexbin:
        # Hexbin plot
        mappable = ax.hexbin(
            norm_coords[mask, 0], 
            norm_coords[mask, 1], 
            C=values[mask], 
            gridsize=75, 
            reduce_C_function=np.mean,
            mincnt=1,
            linewidths=0.2,
            edgecolors='face',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True
        )
    else:
        # Scatter plot
        # Plot NaNs
        if (~mask).any():
            ax.scatter(norm_coords[~mask, 0], norm_coords[~mask, 1], s=point_size, c='lightgray', alpha=0.3, rasterized=True)
            
        # Plot valid
        if mask.any():
            mappable = ax.scatter(
                norm_coords[mask, 0], 
                norm_coords[mask, 1], 
                s=point_size, 
                c=values[mask], 
                cmap=cmap, 
                alpha=0.7, 
                edgecolors="none",
                rasterized=True,
                vmin=vmin, vmax=vmax
            )
    
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Add frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
        
    return mappable

def plot_thumbnail_panel(
    ax: plt.Axes,
    thumb_ids: list[str],
    cell_positions: list[tuple[int, int]],
    samples: list[dict],
    title: str,
    grid_rows: int,
    grid_cols: int,
) -> None:
    # Map samples to cells
    id_to_sample = {str(s.get("object_id")): s for s in samples}
    ordered_samples = []
    ordered_cells = []
    
    for oid, cell in zip(thumb_ids, cell_positions):
        sample = id_to_sample.get(str(oid))
        if sample:
            ordered_samples.append(sample)
            ordered_cells.append(cell)
            
    add_thumbnails(ax, ordered_cells, ordered_samples, grid_rows, grid_cols)
    
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_aspect('equal')
    
    # Add frame but hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate publication combined UMAP figure")
    parser.add_argument("--aion-embeddings", required=True, help="AION .pt file")
    parser.add_argument("--astropt-embeddings", required=True, help="AstroPT .pt file")
    parser.add_argument("--catalog", required=True, help="FITS catalog")
    parser.add_argument("--index", required=True, help="Index CSV mapping object_id -> split/index")
    parser.add_argument("--coords-cache", required=True, help="Path to pre-computed coords .pt file")
    parser.add_argument("--physical-param", required=True, help="Parameter to color by")
    parser.add_argument("--save", default="paper_combined_umap.png", help="Output filename")
    parser.add_argument("--grid-rows", type=int, default=25, help="Grid rows")
    parser.add_argument("--grid-cols", type=int, default=25, help="Grid cols")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--cache-dir", default="/n03data/ronceray/datasets", help="Dataset cache dir")
    parser.add_argument("--hexbin", action="store_true", help="Use hexbin plot instead of scatter")
    
    args = parser.parse_args(argv)
    
    # 1. Load Data
    print("Loading embeddings...")
    aion_recs = load_embeddings(Path(args.aion_embeddings))
    astropt_recs = load_embeddings(Path(args.astropt_embeddings))
    
    print("Loading catalog...")
    catalog, _ = load_fits_catalog(Path(args.catalog))
    
    print("Loading coordinates...")
    coords_map = load_coordinates(Path(args.coords_cache))
    
    if KEY_ASTROPT not in coords_map or KEY_AION not in coords_map:
        raise ValueError("Missing keys in coordinates cache")
        
    # 2. Extract IDs and Values for Coloring (Top Row)
    print(f"Extracting parameter '{args.physical_param}'...")
    
    def get_ids_for_key(records, key):
        ids = []
        for rec in records:
            tensor = rec.get(key)
            if tensor is None and key == "embedding_joint":
                if rec.get("embedding_images") is not None and rec.get("embedding_spectra") is not None:
                    tensor = True
            if tensor is not None:
                oid = rec.get("object_id", "")
                if isinstance(oid, torch.Tensor):
                     oid = oid.item() if oid.numel() == 1 else str(oid.tolist())
                ids.append(str(oid))
        return ids

    aion_ids_full = get_ids_for_key(aion_recs, KEY_AION)
    astropt_ids_full = get_ids_for_key(astropt_recs, KEY_ASTROPT)
    
    coords_aion = coords_map[KEY_AION]
    coords_astro = coords_map[KEY_ASTROPT]
    
    def get_values(ids):
        vals = []
        for oid in ids:
            val = np.nan
            if oid in catalog:
                try:
                    raw = catalog[oid].get(args.physical_param)
                    if raw is not None:
                        val = float(raw) if not hasattr(raw, 'item') else float(raw.item())
                except:
                    pass
            vals.append(val)
        return np.array(vals)
        
    values_aion = get_values(aion_ids_full)
    values_astro = get_values(astropt_ids_full)
    
    # Determine color limits
    all_values = np.concatenate([values_astro[~np.isnan(values_astro)], values_aion[~np.isnan(values_aion)]])
    if len(all_values) == 0:
        vmin, vmax = 0, 1
    else:
        vmin = np.percentile(all_values, 2)
        vmax = np.percentile(all_values, 98)
        if vmax - vmin < 1e-6: vmin, vmax = all_values.min(), all_values.max()
        
    print(f"Color scale: [{vmin:.3f}, {vmax:.3f}]")
    
    # 3. Process Grid Assignments (Bottom Row)
    print("Assigning to grid...")
    thumbs_aion, cells_aion = assign_to_grid(
        aion_ids_full, coords_aion, args.grid_rows, args.grid_cols, args.random_state
    )
    thumbs_astro, cells_astro = assign_to_grid(
        astropt_ids_full, coords_astro, args.grid_rows, args.grid_cols, args.random_state
    )
    
    # 4. Fetch Images
    print("Fetching image samples...")
    all_thumb_ids = list(set(thumbs_aion) | set(thumbs_astro))
    index_map = load_index(Path(args.index))
    samples = collect_samples_with_index(
        cache_dir=args.cache_dir,
        object_ids=all_thumb_ids,
        index_map=index_map,
        verbose=True,
    )
    
    retrieved_ids = {str(s.get("object_id")) for s in samples}
    missing = [oid for oid in all_thumb_ids if oid not in retrieved_ids]
    if missing:
        print(f"Fetching {len(missing)} missing samples...")
        dataset = EuclidDESIDataset(split="all", cache_dir=args.cache_dir, verbose=True)
        samples.extend(collect_samples(dataset, missing, verbose=True))
        
    # 5. Plot
    print("Plotting...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    # Layout:
    # axes[0, 0]: AstroPT Scatter
    # axes[0, 1]: AION Scatter
    # axes[1, 0]: AstroPT Thumbnails
    # axes[1, 1]: AION Thumbnails
    
    # Top Row: Scatter
    sc = plot_scatter_panel(
        axes[0, 0], coords_astro, values_astro, 
        r"\textbf{AstroPT} (Spectra + Images)", vmin, vmax,
        use_hexbin=args.hexbin
    )
    plot_scatter_panel(
        axes[0, 1], coords_aion, values_aion, 
        r"\textbf{AION} (Spectra + Images)", vmin, vmax,
        use_hexbin=args.hexbin
    )
    
    # Bottom Row: Thumbnails
    plot_thumbnail_panel(
        axes[1, 0], thumbs_astro, cells_astro, samples, 
        "", args.grid_rows, args.grid_cols
    )
    plot_thumbnail_panel(
        axes[1, 1], thumbs_aion, cells_aion, samples, 
        "", args.grid_rows, args.grid_cols
    )
    
    # Colorbar for Top Row
    if sc:
        fig.subplots_adjust(right=0.9, wspace=0.1, hspace=0.1)
        # Add colorbar ax spanning the height of the top row
        pos0 = axes[0, 1].get_position()
        cbar_ax = fig.add_axes([0.92, pos0.ymin, 0.015, pos0.height])
        
        label = args.physical_param
        if label == "Z" or label.lower() == "redshift":
            label = r"Redshift $z$"
        
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label(label, fontsize=14)
        cbar.solids.set_edgecolor("face")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=600, bbox_inches="tight")
    print(f"Saved figure to {args.save}")
    plt.close(fig)

if __name__ == "__main__":
    main()
