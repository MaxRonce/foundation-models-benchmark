"""
Script to generate a publication-ready UMAP grid figure.
Displays three panels:
1. AstroPT Joint Embeddings (Left)
2. AION Joint Embeddings (Center)
3. AstroCLIP Joint Embeddings (Right)

Both with thumbnails overlaid on a grid.

Usage:
    python -m scratch.plot_paper_umap_grid \
        --aion-embeddings /path/to/aion.pt \
        --astropt-embeddings /path/to/astropt.pt \
        --astroclip-embeddings /path/to/astroclip.pt \
        --index euclid_index.csv \
        --save paper_umap_grid.png \
        --grid-rows 20 --grid-cols 20
"""
import argparse
from pathlib import Path
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
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
KEY_ASTROCLIP = "embedding_joint" # Using prefix logic in main/cache 

def load_embeddings(path: Path) -> list[dict]:
    """Load embedding records from a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")

def stack_embeddings(records: Sequence[dict], key: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    vectors = []
    ids = []
    redshifts = []
    
    for rec in records:
        tensor = rec.get(key)
        
        # Special handling for AstroPT/AstroCLIP Joint if key is missing but components exist
        if tensor is None and key == "embedding_joint":
            img_emb = rec.get("embedding_images")
            spec_emb = rec.get("embedding_spectra")
            if img_emb is not None and spec_emb is not None:
                if isinstance(img_emb, torch.Tensor): img_emb = img_emb.detach().cpu().numpy()
                else: img_emb = np.asarray(img_emb)
                
                if isinstance(spec_emb, torch.Tensor): spec_emb = spec_emb.detach().cpu().numpy()
                else: spec_emb = np.asarray(spec_emb)
                
                tensor = np.concatenate([img_emb, spec_emb])
        
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
        
        # Redshift
        z = rec.get("redshift", np.nan)
        if isinstance(z, torch.Tensor):
            z = z.item() if z.numel() == 1 else np.nan
        redshifts.append(float(z))
            
    if not vectors:
        raise ValueError(f"No embeddings found for key '{key}'")
        
    return np.stack(vectors, axis=0), ids, np.array(redshifts)

def compute_coords(embeddings: np.ndarray, random_state: int, use_tsne: bool = False) -> np.ndarray:
    if use_tsne:
        print(f"Computing t-SNE for {len(embeddings)} samples...")
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto')
        return reducer.fit_transform(embeddings)
    else:
        print(f"Computing UMAP for {len(embeddings)} samples...")
        reducer = umap.UMAP(random_state=random_state, n_neighbors=15, min_dist=0.1)
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

def plot_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    redshifts: np.ndarray,
    thumb_ids: list[str],
    cell_positions: list[tuple[int, int]],
    samples: list[dict],
    title: str,
    grid_rows: int,
    grid_cols: int,
    point_size: float = 2.0,
    alpha: float = 0.5,
) -> None:
    # Normalize coords to grid size for plotting
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    
    if max_x == min_x: max_x += 1e-9
    if max_y == min_y: max_y += 1e-9
    
    norm_x = (coords[:, 0] - min_x) / (max_x - min_x) * grid_cols
    norm_y = (coords[:, 1] - min_y) / (max_y - min_y) * grid_rows
    
    # Scatter plot
    mask = ~np.isnan(redshifts)
    
    # Plot NaNs
    if (~mask).any():
        ax.scatter(norm_x[~mask], norm_y[~mask], s=point_size, c='lightgray', alpha=0.3, rasterized=True)
        
    # Plot valid
    if mask.any():
        sc = ax.scatter(
            norm_x[mask], 
            norm_y[mask], 
            s=point_size, 
            c=redshifts[mask], 
            cmap="viridis", 
            alpha=alpha, 
            rasterized=True,
            vmin=0, vmax=4 # Reasonable redshift limits
        )
    
    # Add thumbnails
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
    
    # Add frame (border) but hide ticks/labels
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    return sc if mask.any() else None


def get_coords_cached(
    records: list[dict], 
    key: str, 
    random_state: int, 
    cache_path: Optional[Path],
    cache_key_override: Optional[str] = None,
    use_tsne: bool = False
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Get coordinates, either from cache or by computing them.
    Returns (coords, ids, redshifts).
    """
    # 1. Stack embeddings first to get IDs and Redshifts (needed for alignment)
    vecs, ids, redshifts = stack_embeddings(records, key)
    
    # Use override key for cache lookup if provided (e.g. "astroclip_embedding_joint")
    lookup_key = cache_key_override if cache_key_override else key
    
    # 2. Check cache
    coords = None
    if cache_path and cache_path.exists():
        print(f"Loading cached coordinates from {cache_path} (key: {lookup_key})...")
        try:
            cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
            
            if isinstance(cache_data, dict) and lookup_key in cache_data:
                cached_coords_map = cache_data[lookup_key] 
                
                if isinstance(cached_coords_map, (np.ndarray, torch.Tensor)):
                    # Array format
                    if len(cached_coords_map) == len(vecs):
                        coords = cached_coords_map
                        if isinstance(coords, torch.Tensor): coords = coords.numpy()
                    else:
                        print(f"Cache length mismatch ({len(cached_coords_map)} vs {len(vecs)}). Recomputing.")
                elif isinstance(cached_coords_map, dict):
                    # Dict format {id: coord}
                    temp_coords = []
                    valid_mask = []
                    for i, oid in enumerate(ids):
                        if oid in cached_coords_map:
                            temp_coords.append(cached_coords_map[oid])
                            valid_mask.append(True)
                        else:
                            valid_mask.append(False)
                    
                    if all(valid_mask):
                        coords = np.array(temp_coords)
                    else:
                        print(f"Cache missing {len(ids) - sum(valid_mask)} IDs. Recomputing.")
            else:
                 print(f"Key '{lookup_key}' not found in cache. Recomputing.")
                 
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing.")
    
    # 3. Compute if needed
    if coords is None:
        coords = compute_coords(vecs, random_state, use_tsne)
        
        # Save to cache if path provided
        if cache_path:
            print(f"Saving coordinates to {cache_path}...")
            full_cache = {}
            if cache_path.exists():
                try:
                    full_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                    if not isinstance(full_cache, dict): full_cache = {}
                except:
                    pass
            
            full_cache[lookup_key] = coords
            torch.save(full_cache, cache_path)
            
    return coords, ids, redshifts

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate publication UMAP/t-SNE grid comparison")
    parser.add_argument("--aion-embeddings", required=True, help="AION .pt file")
    parser.add_argument("--astropt-embeddings", required=True, help="AstroPT .pt file")
    parser.add_argument("--astroclip-embeddings", required=True, help="AstroCLIP .pt file")
    parser.add_argument("--index", required=True, help="Index CSV mapping object_id -> split/index")
    parser.add_argument("--save", default="paper_umap_grid.png", help="Output filename")
    parser.add_argument("--grid-rows", type=int, default=20, help="Grid rows")
    parser.add_argument("--grid-cols", type=int, default=20, help="Grid cols")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--cache-dir", default="/n03data/ronceray/datasets", help="Dataset cache dir")
    parser.add_argument("--coords-cache", help="Path to .pt file to save/load coordinates")
    parser.add_argument("--use-tsne", action="store_true", help="Use t-SNE instead of UMAP")
    
    args = parser.parse_args(argv)
    
    # 1. Load Data
    print("Loading embeddings...")
    aion_recs = load_embeddings(Path(args.aion_embeddings))
    astropt_recs = load_embeddings(Path(args.astropt_embeddings))
    astroclip_recs = load_embeddings(Path(args.astroclip_embeddings))
    
    cache_path = Path(args.coords_cache) if args.coords_cache else None
    
    # 2. Process AION
    print("Processing AION...")
    # Use specific cache key "aion_embedding_hsc_desi" if standardized, or rely on file key.
    coords_aion, ids_aion, z_aion = get_coords_cached(
        aion_recs, KEY_AION, args.random_state, cache_path, "aion_embedding_hsc_desi", args.use_tsne
    )
    
    thumbs_aion, cells_aion = assign_to_grid(
        ids_aion, coords_aion, args.grid_rows, args.grid_cols, args.random_state
    )
    
    # 3. Process AstroPT
    print("Processing AstroPT...")
    coords_astro, ids_astro, z_astro = get_coords_cached(
        astropt_recs, KEY_ASTROPT, args.random_state, cache_path, "astropt_embedding_joint", args.use_tsne
    )
    
    thumbs_astro, cells_astro = assign_to_grid(
        ids_astro, coords_astro, args.grid_rows, args.grid_cols, args.random_state
    )
    
    # 3.5 Process AstroCLIP
    print("Processing AstroCLIP...")
    coords_clip, ids_clip, z_clip = get_coords_cached(
        astroclip_recs, KEY_ASTROCLIP, args.random_state, cache_path, "astroclip_embedding_joint", args.use_tsne
    )
    
    thumbs_clip, cells_clip = assign_to_grid(
        ids_clip, coords_clip, args.grid_rows, args.grid_cols, args.random_state
    )
    
    # 4. Fetch Samples (Images)
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
        print(f"Fetching {len(missing)} missing samples from dataset...")
        dataset = EuclidDESIDataset(split="all", cache_dir=args.cache_dir, verbose=True)
        samples.extend(collect_samples(dataset, missing, verbose=True))
        
    # 5. Plot
    print("Plotting...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # AstroPT (Left)
    plot_panel(
        axes[0], coords_astro, z_astro, thumbs_astro, cells_astro, samples,
        r"\textbf{AstroPT} (Spectra + Images)", args.grid_rows, args.grid_cols
    )
    
    # AION (Center)
    plot_panel(
        axes[1], coords_aion, z_aion, thumbs_aion, cells_aion, samples,
        r"\textbf{AION} (Spectra + Images)", args.grid_rows, args.grid_cols
    )
    
    # AstroCLIP (Right)
    plot_panel(
        axes[2], coords_clip, z_clip, thumbs_clip, cells_clip, samples,
        r"\textbf{AstroCLIP} (Spectra + Images)", args.grid_rows, args.grid_cols
    )
    
    # No Colorbar
        
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=500, bbox_inches="tight")
    print(f"Saved figure to {args.save}")
    plt.close(fig)

if __name__ == "__main__":
    main()
