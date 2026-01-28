"""
Script to generate a publication-ready combined UMAP figure.
Displays six panels in a 2x3 grid:
- Top Row: UMAP colored by physical parameter (e.g., Redshift) for AstroPT, AION, and AstroCLIP.
- Bottom Row: UMAP with thumbnails for AstroPT, AION, and AstroCLIP.

Usage:
    python -m scratch.plot_paper_combined_umap \
        --aion-embeddings /path/to/aion.pt \
        --astropt-embeddings /path/to/astropt.pt \
        --astroclip-embeddings /path/to/astroclip.pt \
        --catalog /path/to/catalog.fits \
        --index euclid_index.csv \
        --coords-cache paper/umap_coords_cache.pt \
        --physical-param Z \
        --save paper/paper_combined_umap.png \
        --grid-rows 25 --grid-cols 25
"""
import argparse
from pathlib import Path
from typing import Sequence, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from tqdm import tqdm
import umap.umap_ as umap

from fmb.viz.utils import (
    load_index,
    collect_samples,
    collect_samples_with_index,
    prepare_rgb_image,
    load_viz_style,
)
from fmb.data.load_display_data import EuclidDESIDataset
from fmb.paths import load_paths

# --- Publication Style Settings ---
load_viz_style()

# Parameters describing WHAT to load from files
KEY_ASTROPT_MSG = "embedding_joint"
KEY_AION_MSG = "embedding_hsc_desi"
KEY_ASTROCLIP_MSG = "embedding_joint"

# Parameters describing WHERE to store in cache (prefixed to avoid collision)
KEY_ASTROPT_CACHE = "astropt_embedding_joint"
KEY_AION_CACHE = "aion_embedding_hsc_desi"
KEY_ASTROCLIP_CACHE = "astroclip_embedding_joint"

def load_embeddings(path: Path) -> list[dict]:
    """Load embedding records from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")

def load_coordinates(load_path: Path) -> dict[str, np.ndarray]:
    """Load previously computed coordinates (t-SNE or UMAP). Returns empty dict if not found."""
    if not load_path.exists():
        print(f"  Cache file {load_path} not found. Will start with empty cache.")
        return {}
    
    try:
        coords_map = torch.load(load_path, map_location="cpu", weights_only=False)
        # Convert tensors to numpy if needed
        for key in coords_map:
            if isinstance(coords_map[key], torch.Tensor):
                coords_map[key] = coords_map[key].numpy()
        print(f"  Loaded coordinates from {load_path}")
        return coords_map
    except Exception as e:
        print(f"Warning: Failed to load cache {load_path}: {e}. Starting with empty cache.")
        return {}

def stack_embeddings(records: Sequence[dict], key: str) -> Tuple[np.ndarray, list[str]]:
    """Stack embeddings. If 'embedding_joint' missing, construct from images/spectra."""
    vectors = []
    ids = []
    
    for rec in records:
        tensor = rec.get(key)
        
        # Special handling for Joint embedding (concatenation)
        # Check if we should try to construct it
        if "embedding_joint" in key and tensor is None:
             # Try un-prefixed components (common case in raw files)
             img = rec.get("embedding_images")
             spec = rec.get("embedding_spectra")
             
             if img is not None and spec is not None:
                 img = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
                 spec = spec.detach().cpu().numpy() if isinstance(spec, torch.Tensor) else np.asarray(spec)
                 tensor = np.concatenate([img, spec])
        
        if tensor is None:
            continue
            
        if isinstance(tensor, torch.Tensor):
            vectors.append(tensor.detach().cpu().numpy())
        else:
            vectors.append(np.asarray(tensor))
            
        # ID
        oid = rec.get("object_id", "")
        if isinstance(oid, torch.Tensor):
             oid = oid.item() if oid.numel() == 1 else str(oid.tolist())
        ids.append(str(oid))
            
    if not vectors:
        raise ValueError(f"No embeddings found for key '{key}' and construction failed.")
        
    return np.stack(vectors, axis=0), ids

def compute_umap(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    print(f"  Computing UMAP for {len(embeddings)} samples (dim={embeddings.shape[1]})...")
    # Using presets similar to visualize script (balanced)
    # Using n_neighbors=30 to maintain structure, metric=cosine standard
    reducer = umap.UMAP(random_state=random_state, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(embeddings)

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

def normalize_simple(coords: np.ndarray) -> np.ndarray:
    c_min = coords.min(axis=0)
    c_max = coords.max(axis=0)
    denom = c_max - c_min
    denom[denom == 0] = 1e-9
    return (coords - c_min) / denom


def robust_normalize(coords: np.ndarray) -> np.ndarray:
    """Normalize coords to [0, 1] using robust limits (percentiles) to avoid outlier compression."""
    # Use 1st and 99th percentile for robust range
    c_min = np.percentile(coords, 1, axis=0)
    c_max = np.percentile(coords, 99, axis=0)
    
    # Clip extreme outliers for the purpose of grid assignment/visualization boundaries
    clipped = np.clip(coords, c_min, c_max)
    
    denom = c_max - c_min
    denom[denom == 0] = 1e-9
    
    return (clipped - c_min) / denom

def assign_to_grid(
    object_ids: Sequence[str],
    coords: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    random_state: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    if grid_rows <= 0 or grid_cols <= 0 or coords.size == 0:
        return [], []

    # Robust normalization to fill the grid effectively (matching scatter plot view)
    norm_coords = robust_normalize(coords)
    norm_x = norm_coords[:, 0]
    norm_y = norm_coords[:, 1]

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
            pass

def plot_scatter_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "plasma",
    point_size: float = 3.0,
    use_hexbin: bool = True, # Default to Hexbin
) -> None:
    mask = ~np.isnan(values)
    
    # Use robust normalization for plotting to ensure main structure fills frame
    norm_coords = robust_normalize(coords)
    
    mappable = None
    
    if use_hexbin:
        # Hexbin plot
        mappable = ax.hexbin(
            norm_coords[mask, 0], 
            norm_coords[mask, 1], 
            C=values[mask], 
            gridsize=100, # Increased resolution 
            reduce_C_function=np.mean,
            mincnt=1,
            linewidths=0.0, # Smoother look
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
    
    ax.set_title(title, fontsize=24, pad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    # Set explicit limits since we normalized to 0-1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
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
    
    ax.set_title(title, fontsize=24, pad=15)
    ax.set_xlim(0, grid_cols)
    ax.set_ylim(0, grid_rows)
    ax.set_aspect('auto')
    
    # Add frame but hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

def plot_similarity_histogram(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    color: str = "gray",
    bins: int = 50,
) -> None:
    # Remove NaNs
    valid_values = values[~np.isnan(values)]
    
    if len(valid_values) == 0:
        return

    # Plot histogram
    ax.hist(
        np.abs(valid_values), 
        bins=bins, 
        range=(0, 1), 
        density=True, 
        color=color, 
        alpha=0.7, 
        edgecolor='none',
        rasterized=True
    )
    
    # Add statistics lines
    mean_val = np.mean(np.abs(valid_values))
    median_val = np.median(np.abs(valid_values))
    
    ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='red', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.2f}')
    
    ax.set_title(title, fontsize=24, pad=15)
    ax.set_xlim(0, 1)
    
    # Hide y-axis ticks/labels as density is relative
    ax.set_yticks([])
    ax.set_ylabel("Density", fontsize=32)
    ax.set_xlabel("|Cosine Similarity|", fontsize=32)
    
    ax.legend(loc='upper left', fontsize=26, frameon=False)
    
    # Add frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

def main(argv: Sequence[str] | None = None) -> None:
    paths = load_paths()
    
    parser = argparse.ArgumentParser(description="Generate publication combined UMAP figure (AstroPT, AION, AstroCLIP)")
    parser.add_argument("--aion-embeddings", default=str(paths.embeddings / "aions_embeddings.pt"), help="AION .pt file")
    parser.add_argument("--astropt-embeddings", default=str(paths.embeddings / "astropt_embeddings.pt"), help="AstroPT .pt file")
    parser.add_argument("--astroclip-embeddings", default=str(paths.embeddings / "embeddings_astroclip.pt"), help="AstroCLIP .pt file")
    parser.add_argument("--catalog", default=str(paths.dataset / "DESI_DR1_Euclid_Q1_dataset_catalog_EM.fits"), help="FITS catalog")
    parser.add_argument("--index", default=str(paths.dataset_index), help="Index CSV mapping object_id -> split/index")
    parser.add_argument("--coords-cache", default=str(paths.analysis / "umap" / "coords_cache.pt"), help="Path to pre-computed coords .pt file")
    parser.add_argument("--physical-param", default="Z", help="Parameter to color by")
    parser.add_argument("--save", default=str(paths.analysis / "umap" / "combined_umap.png"), help="Output filename")
    parser.add_argument("--grid-rows", type=int, default=25, help="Grid rows")
    parser.add_argument("--grid-cols", type=int, default=25, help="Grid cols")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--cache-dir", default=str(paths.dataset), help="Dataset cache dir")
    parser.add_argument("--hexbin", action="store_true", default=True, help="Use hexbin plot instead of scatter (Default: True)")
    parser.add_argument("--show-similarity", action="store_true", default=True, help="Add 3rd row with cosine similarity (Images vs Spectra)")  # Default True now
    parser.add_argument("--no-hexbin", action="store_false", dest="hexbin", help="Disable hexbin plot")
    
    parser.set_defaults(hexbin=True)
    
    args = parser.parse_args(argv)

    # Helper for similarity
    def compute_cosine_similarity(records):
        sims = []
        for rec in records:
            # Try specific keys often found in these files
            img = rec.get("embedding_images")
            spec = rec.get("embedding_spectra")

            # AION specific keys
            if img is None: img = rec.get("embedding_hsc")
            if spec is None: spec = rec.get("embedding_spectrum")
            
            # If not found, try to look for keys ending in _images/_spectra/_spectrum
            if img is None:
                for k in rec.keys():
                    if k.endswith("_images"):
                        img = rec[k]
                        break
            if spec is None:
                for k in rec.keys():
                    if k.endswith("_spectra") or k.endswith("_spectrum"):
                        spec = rec[k]
                        break
            
            if img is None or spec is None:
                sims.append(np.nan)
                continue
                
            if hasattr(img, "detach"): img = img.detach().cpu().numpy()
            else: img = np.array(img)
                
            if hasattr(spec, "detach"): spec = spec.detach().cpu().numpy()
            else: spec = np.array(spec)
            
            img = img.flatten()
            spec = spec.flatten()
            
            ni = np.linalg.norm(img)
            ns = np.linalg.norm(spec)
            if ni == 0 or ns == 0:
                sims.append(np.nan)
            else:
                sims.append(np.dot(img, spec) / (ni * ns))
        return np.array(sims)
    
    # 1. Load Data
    print("Loading embeddings...")
    aion_recs = load_embeddings(Path(args.aion_embeddings))
    astropt_recs = load_embeddings(Path(args.astropt_embeddings))
    astroclip_recs = load_embeddings(Path(args.astroclip_embeddings))
    
    print("Loading catalog...")
    catalog, _ = load_fits_catalog(Path(args.catalog))
    
    print("Loading coordinates cache...")
    coords_cache_path = Path(args.coords_cache)
    coords_map = load_coordinates(coords_cache_path)
    
    # helper to get or compute
    cache_dirty = False
    
    def get_or_compute(records, native_key, cache_key):
        nonlocal cache_dirty
        if cache_key in coords_map:
            print(f"  Using cached coords for '{cache_key}'")
            return coords_map[cache_key]
        else:
            print(f"  Missing coords for '{cache_key}'. Computing...")
            vecs, _ = stack_embeddings(records, native_key)
            coords = compute_umap(vecs, args.random_state)
            coords_map[cache_key] = coords
            cache_dirty = True
            return coords

    print("Checking AION coords...")
    coords_aion = get_or_compute(aion_recs, KEY_AION_MSG, KEY_AION_CACHE)
    
    print("Checking AstroPT coords...")
    coords_astropt = get_or_compute(astropt_recs, KEY_ASTROPT_MSG, KEY_ASTROPT_CACHE)
    
    print("Checking AstroCLIP coords...")
    coords_astroclip = get_or_compute(astroclip_recs, KEY_ASTROCLIP_MSG, KEY_ASTROCLIP_CACHE)
    
    # Save cache if needed
    if cache_dirty:
        print(f"Saving updated cache to {coords_cache_path}...")
        coords_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(coords_map, coords_cache_path)
        
    # Resolve physical parameter name (handle aliases like 'redshift' -> 'Z')
    param = args.physical_param
    if param not in catalog:
        # Common aliases
        aliases = {
            "redshift": "Z",
            "z": "Z",
            "Redshift": "Z",
            "mass": "LOGM",
            "sfr": "LOGSFR"
        }
        if param in aliases and aliases[param] in catalog:
            print(f"Parameter '{param}' not found in catalog, using alias '{aliases[param]}'")
            param = aliases[param]
    
    # 2. Extract IDs and Values for Coloring (Top Row)
    print(f"Extracting parameter '{param}'...")
    
    def get_ids(records, native_key):
        _, ids = stack_embeddings(records, native_key)
        return ids

    aion_ids_full = get_ids(aion_recs, KEY_AION_MSG)
    astropt_ids_full = get_ids(astropt_recs, KEY_ASTROPT_MSG)
    astroclip_ids_full = get_ids(astroclip_recs, KEY_ASTROCLIP_MSG)
    
    def get_values(ids):
        vals = []
        matches = 0
        samples = 0
        print(f"DEBUG: Processing {len(ids)} IDs...")
        if len(ids) > 0:
            print(f"DEBUG: Sample ID (embedding): '{ids[0]}' (type: {type(ids[0])})")
            cat_sample = next(iter(catalog.keys()))
            print(f"DEBUG: Sample ID (catalog): '{cat_sample}' (type: {type(cat_sample)})")
            
        for oid in ids:
            # Ensure strict string matching
            oid_str = str(oid).strip()
            val = np.nan
            if oid_str in catalog:
                matches += 1
                try:
                    raw = catalog[oid_str].get(param)
                    if raw is not None:
                        val = float(raw) if not hasattr(raw, 'item') else float(raw.item())
                except:
                    pass
            vals.append(val)
            samples += 1
        print(f"DEBUG: Found {matches}/{samples} matches in catalog for param '{param}'")
        return np.array(vals)
        
    values_aion = get_values(aion_ids_full)
    values_astro = get_values(astropt_ids_full)
    values_clip = get_values(astroclip_ids_full)
    
    # Determine color limits
    all_values = np.concatenate([
        values_astro[~np.isnan(values_astro)], 
        values_aion[~np.isnan(values_aion)],
        values_clip[~np.isnan(values_clip)]
    ])
    
    if len(all_values) == 0:
        vmin, vmax = 0, 1
    else:
        vmin = np.percentile(all_values, 2)
        vmax = np.percentile(all_values, 98)
        if vmax - vmin < 1e-6: vmin, vmax = all_values.min(), all_values.max()
        
    print(f"Color scale: [{vmin:.3f}, {vmax:.3f}]")
    
    # 2b. Compute Similarity if requested
    val_sim_aion = None
    val_sim_astro = None
    val_sim_clip = None
    vmin_sim, vmax_sim = 0, 1
    
    if args.show_similarity:
        print("Computing cosine similarities...")
        val_sim_aion = compute_cosine_similarity(aion_recs)
        val_sim_astro = compute_cosine_similarity(astropt_recs)
        val_sim_clip = compute_cosine_similarity(astroclip_recs)
        
        all_sims = np.concatenate([
            val_sim_aion[~np.isnan(val_sim_aion)],
            val_sim_astro[~np.isnan(val_sim_astro)],
            val_sim_clip[~np.isnan(val_sim_clip)]
        ])
        
        if len(all_sims) > 0:
            vmin_sim = np.percentile(all_sims, 2)
            vmax_sim = np.percentile(all_sims, 98)
            print(f"Similarity Color scale: [{vmin_sim:.3f}, {vmax_sim:.3f}]")
        else:
            print("Warning: No valid similarities found.")
    
    # 3. Process Grid Assignments (Bottom Row)
    print("Assigning to grid...")
    thumbs_aion, cells_aion = assign_to_grid(
        aion_ids_full, coords_aion, args.grid_rows, args.grid_cols, args.random_state
    )
    thumbs_astro, cells_astro = assign_to_grid(
        astropt_ids_full, coords_astropt, args.grid_rows, args.grid_cols, args.random_state
    )
    thumbs_clip, cells_clip = assign_to_grid(
        astroclip_ids_full, coords_astroclip, args.grid_rows, args.grid_cols, args.random_state
    )
    
    # 4. Fetch Images
    print("Fetching image samples...")
    all_thumb_ids = list(set(thumbs_aion) | set(thumbs_astro) | set(thumbs_clip))
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
    rows = 3 if args.show_similarity else 2
    fig_height = 18 if args.show_similarity else 12
    fig, axes = plt.subplots(rows, 3, figsize=(24, fig_height))
    
    # Top Row: Scatter (Physical Param)
    sc = plot_scatter_panel(
        axes[0, 0], coords_astropt, values_astro, 
        r"\textbf{AstroPT} (Spectra + Images)", vmin, vmax,
        use_hexbin=args.hexbin
    )
    plot_scatter_panel(
        axes[0, 1], coords_aion, values_aion, 
        r"\textbf{AION} (Spectra + Images)", vmin, vmax,
        use_hexbin=args.hexbin
    )
    plot_scatter_panel(
        axes[0, 2], coords_astroclip, values_clip, 
        r"\textbf{AstroCLIP} (Spectra + Images)", vmin, vmax,
        use_hexbin=args.hexbin
    )
    
    # Switch rows if similarity is added? 
    # Standard: Row 1=Param, Row 2=Thumbnails. 
    # Requested: Add 3rd line.
    # Let's put Thumbnails in Row 2 (index 1), Similarity in Row 3 (index 2).
    # Or Similarity in Row 2, Thumbnails in Row 3?
    # Keeping Thumbnails at bottom (last row) is often cleanest.
    # But user asked to "add a 3rd line". 
    # I will put Similarity in the middle (Row 2), Thumbnails at bottom (Row 3).
    # Wait, existing code puts Thumbnails in Row 2 (axes[1, ...]).
    # If I add similarity, I will put it as axes[2, ...] (Row 3).
    # This matches "Add a 3rd line".
    
    thumb_row_idx = 1
    sim_row_idx = 2
    
    # ADD ROW TITLES
    # We place text relative to first axes of each row
    # Row 1: Latent Space
    axes[0, 0].text(-0.15, 0.5, "Latent Space\n(Colored by Param)", transform=axes[0, 0].transAxes, 
                    rotation=90, va='center', ha='right', fontsize=28, fontweight='bold')
    # Row 2: Thumbnails
    axes[1, 0].text(-0.15, 0.5, "Representative\nThumbnails", transform=axes[1, 0].transAxes, 
                    rotation=90, va='center', ha='right', fontsize=28, fontweight='bold')
                    
    # Middle Row: Thumbnails (Original Row 2)
    # If we want Similarity in the middle, we'd change indices.
    # Let's append Similarity at the bottom to follow "Add 3rd line" literally.
    
    plot_thumbnail_panel(
        axes[1, 0], thumbs_astro, cells_astro, samples, 
        "", args.grid_rows, args.grid_cols
    )
    plot_thumbnail_panel(
        axes[1, 1], thumbs_aion, cells_aion, samples, 
        "", args.grid_rows, args.grid_cols
    )
    plot_thumbnail_panel(
        axes[1, 2], thumbs_clip, cells_clip, samples, 
        "", args.grid_rows, args.grid_cols
    )
    
    sc_sim = None
    if args.show_similarity:
        # Row 3: Similarity
        axes[2, 0].text(-0.15, 0.5, "Cosine Similarity\n(Images vs Spectra)", transform=axes[2, 0].transAxes, 
                        rotation=90, va='center', ha='right', fontsize=28, fontweight='bold')
        
        # Bottom Row: Similarity Histograms
        # Use different colors for each model if desired, or uniform
        plot_similarity_histogram(
            axes[2, 0], val_sim_astro, 
            "", color="C0" # Removed title to avoid overlap
        )
        plot_similarity_histogram(
            axes[2, 1], val_sim_aion, 
            "", color="C1"
        )
        plot_similarity_histogram(
            axes[2, 2], val_sim_clip, 
            "", color="C2"
        )

    # Colorbars
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.1, hspace=0.2) # Increased hspace for titles, added left margin
    
    # 1. Colorbar for Param
    if sc:
        pos_top_right = axes[0, 2].get_position()
        cbar_ax = fig.add_axes([0.92, pos_top_right.ymin, 0.015, pos_top_right.height])
        
        label = param
        if label == "Z" or label.lower() == "redshift":
            label = r"Redshift $z$"
        
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(label, fontsize=24)
    cbar.ax.tick_params(labelsize=18)
    cbar.solids.set_edgecolor("face")



    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=600, bbox_inches="tight")
    print(f"Saved figure to {args.save}")
    plt.close(fig)

if __name__ == "__main__":
    main()
