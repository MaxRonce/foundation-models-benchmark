#!/usr/bin/env python3
"""
Script to visualize t-SNE projections of embeddings from both AION and AstroPT models,
colored by physical parameters from a FITS catalog.

This script:
- Loads embeddings from AION (embedding_hsc_desi, embedding_hsc, embedding_spectrum)
- Loads embeddings from AstroPT (embedding_images, embedding_spectra, embedding_joint)
- Matches them with a FITS catalog to extract physical parameters
- Computes t-SNE projections for all 6 embedding types
- Generates a grid visualization with color-coded physical parameters

Usage:
    python -m scratch.visualize_multimodel_embeddings_tsne \
        --aion-embeddings /n03data/ronceray/embeddings/euclid_desi_all_embeddings.pt \
        --astropt-embeddings /n03data/ronceray/embeddings/astropt_embeddings.pt \
        --catalog /home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_EM.fits \
        --output-dir /path/to/output \
        --physical-param redshift \
        --random-state 42
"""
import argparse
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
    from sklearn.manifold import TSNE
except ImportError as exc:
    raise SystemExit(
        "The 'scikit-learn' package is required. Install it with 'pip install scikit-learn'."
    ) from exc


# Embedding keys for AION and AstroPT models
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

# t-SNE preset configurations
TSNE_PRESETS = {
    "balanced": {
        "perplexity": 30,
        "metric": "cosine",
        "description": "Équilibré - Structure globale + locale (recommandé)",
    },
    "local": {
        "perplexity": 15,
        "metric": "cosine",
        "description": "Local - Structure locale fine",
    },
    "global": {
        "perplexity": 50,
        "metric": "cosine",
        "description": "Global - Vue d'ensemble plus large",
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
    """
    Load FITS catalog and return a dictionary mapping object_id to row data,
    along with available numeric columns and the ID column name.
    
    Returns:
        - catalog_dict: Dictionary mapping object_id to row data
        - numeric_columns: List of numeric column names (excluding ID column)
        - id_column: Name of the object ID column
    """
    with fits.open(path) as hdul:
        # Typically the data is in the first extension (HDU 1)
        data = hdul[1].data
        columns = hdul[1].columns.names
        
        # Build a dictionary mapping object_id to all column values
        catalog_dict = {}
        
        # Find the object ID column (TARGETID is the correct one for DESI)
        id_column = None
        # Priority order: TARGETID (for DESI), then fallbacks
        for priority_col in ['TARGETID', 'targetid', 'TargetID']:
            if priority_col in columns:
                id_column = priority_col
                break
        
        # Fallback to other common names if TARGETID not found
        if id_column is None:
            for col in columns:
                if col.lower() in ['object_id', 'objid', 'id']:
                    id_column = col
                    break
        
        if id_column is None:
            raise ValueError(f"Could not find object ID column in FITS. Available columns: {columns}")
        
        print(f"Using '{id_column}' as object ID column")
        
        for row in data:
            obj_id = str(row[id_column])
            catalog_dict[obj_id] = {col: row[col] for col in columns}
        
        # Identify numeric columns (excluding ID column)
        numeric_columns = []
        for col in columns:
            if col == id_column:
                continue
            # Check if column is numeric by checking the dtype or trying to convert
            try:
                # Get the column type from the FITS table
                col_format = hdul[1].columns[col].format
                # Common numeric formats in FITS: E (float32), D (float64), I (int16), J (int32), K (int64)
                if any(fmt in col_format.upper() for fmt in ['E', 'D', 'I', 'J', 'K', 'F']):
                    numeric_columns.append(col)
                    continue
                # Fallback: try to convert a sample of values
                for row in data[:min(100, len(data))]:
                    try:
                        float(row[col])
                        # If we can convert to float, it's numeric
                        numeric_columns.append(col)
                        break
                    except (ValueError, TypeError):
                        continue
            except Exception:
                # Not numeric, skip
                continue
        
        print(f"Found {len(numeric_columns)} numeric physical parameters")
        print(f"  Sample: {numeric_columns[:5]}")
        
        return catalog_dict, numeric_columns, id_column


def stack_embeddings_with_joint(records: Sequence[dict], key: str) -> np.ndarray:
    """
    Stack embeddings for a given key.
    Special handling for 'embedding_joint' which concatenates images+spectra.
    """
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


def save_tsne_coordinates(coords_map: dict[str, np.ndarray], save_path: Path) -> None:
    """Save t-SNE coordinates to a file for reuse."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(coords_map, save_path)
    print(f"  Saved t-SNE coordinates to {save_path}")


def load_tsne_coordinates(load_path: Path) -> dict[str, np.ndarray]:
    """Load previously computed t-SNE coordinates."""
    coords_map = torch.load(load_path, map_location="cpu", weights_only=False)
    # Convert tensors to numpy if needed
    for key in coords_map:
        if isinstance(coords_map[key], torch.Tensor):
            coords_map[key] = coords_map[key].numpy()
    print(f"  Loaded t-SNE coordinates from {load_path}")
    return coords_map


def compute_tsne(embeddings: np.ndarray, random_state: int, preset: str = "balanced") -> np.ndarray:
    """Compute t-SNE projection for the given embeddings using a preset configuration."""
    config = TSNE_PRESETS[preset]
    
    print(f"    t-SNE params: perplexity={config['perplexity']}, " 
          f"metric={config['metric']}")
    
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=config["perplexity"],
        metric=config["metric"],
        init="pca",  # Initialize with PCA for better results
        verbose=0,
    )
    return tsne.fit_transform(embeddings)


def merge_embeddings_only(
    aion_records: list[dict],
    astropt_records: list[dict],
) -> list[dict]:
    """
    Merge AION and AstroPT embeddings without extracting parameters.
    Used for initial merge before computing t-SNE.
    
    Returns:
        - merged_records: List of merged records with all embedding types
    """
    # Build dictionaries keyed by object_id
    aion_dict = {str(rec.get("object_id", "")): rec for rec in aion_records}
    astropt_dict = {str(rec.get("object_id", "")): rec for rec in astropt_records}
    
    # Get all unique object IDs
    all_ids = set(aion_dict.keys()) | set(astropt_dict.keys())
    all_ids.discard("")  # Remove empty IDs
    
    merged_records = []
    
    for obj_id in sorted(all_ids):
        merged_rec = {"object_id": obj_id}
        
        # Merge AION embeddings
        if obj_id in aion_dict:
            for key in AION_EMBEDDING_KEYS:
                if key in aion_dict[obj_id]:
                    merged_rec[key] = aion_dict[obj_id][key]
        
        # Merge AstroPT embeddings
        if obj_id in astropt_dict:
            for key in ASTROPT_EMBEDDING_KEYS:
                if key in astropt_dict[obj_id]:
                    merged_rec[key] = astropt_dict[obj_id][key]
        
        merged_records.append(merged_rec)
    
    return merged_records


def merge_embedding_records(
    aion_records: list[dict],
    astropt_records: list[dict],
    catalog: dict,
    physical_param: str,
) -> tuple[list[dict], np.ndarray, list[str]]:
    """
    Merge AION and AstroPT embeddings, match with catalog, and extract physical parameter.
    
    Returns:
        - merged_records: List of merged records with all embedding types
        - physical_values: Array of physical parameter values (NaN for missing)
        - valid_object_ids: List of object IDs that have at least one embedding
    """
    # Build dictionaries keyed by object_id
    aion_dict = {str(rec.get("object_id", "")): rec for rec in aion_records}
    astropt_dict = {str(rec.get("object_id", "")): rec for rec in astropt_records}
    
    # Get all unique object IDs
    all_ids = set(aion_dict.keys()) | set(astropt_dict.keys())
    all_ids.discard("")  # Remove empty IDs
    
    merged_records = []
    physical_values = []
    valid_object_ids = []
    
    # Debug: track matching statistics
    catalog_matches = 0
    param_found = 0
    param_valid = 0
    
    for obj_id in sorted(all_ids):
        merged_rec = {"object_id": obj_id}
        
        # Merge AION embeddings
        if obj_id in aion_dict:
            for key in AION_EMBEDDING_KEYS:
                if key in aion_dict[obj_id]:
                    merged_rec[key] = aion_dict[obj_id][key]
        
        # Merge AstroPT embeddings
        if obj_id in astropt_dict:
            for key in ASTROPT_EMBEDDING_KEYS:
                if key in astropt_dict[obj_id]:
                    merged_rec[key] = astropt_dict[obj_id][key]
        
        # Get physical parameter from catalog
        phys_val = np.nan
        if obj_id in catalog:
            catalog_matches += 1
            try:
                raw_val = catalog[obj_id][physical_param]
                param_found += 1
                
                # Handle different types (numpy scalar, python float, etc.)
                if hasattr(raw_val, 'item'):  # numpy scalar
                    phys_val = float(raw_val.item())
                else:
                    phys_val = float(raw_val)
                
                # Check if valid (not NaN, not inf)
                if not (np.isnan(phys_val) or np.isinf(phys_val)):
                    param_valid += 1
                else:
                    phys_val = np.nan
                    
            except (KeyError, ValueError, TypeError) as e:
                # Silently set to NaN - parameter doesn't exist or can't be converted
                pass
        
        merged_records.append(merged_rec)
        physical_values.append(phys_val)
        valid_object_ids.append(obj_id)
    
    # Print debug statistics
    if len(all_ids) > 0:
        match_rate = catalog_matches / len(all_ids) * 100
        print(f"    📊 ID matching: {catalog_matches}/{len(all_ids)} ({match_rate:.1f}%) embeddings found in catalog")
        if catalog_matches > 0:
            param_rate = param_found / catalog_matches * 100
            valid_rate = param_valid / catalog_matches * 100
            print(f"    📈 Parameter '{physical_param}': {param_found}/{catalog_matches} exist ({param_rate:.1f}%), {param_valid} valid ({valid_rate:.1f}%)")
        
        # Show sample IDs for debug
        if catalog_matches == 0:
            sample_embed_ids = list(all_ids)[:3]
            sample_cat_ids = list(catalog.keys())[:3]
            print(f"    ⚠️  Sample embedding IDs: {sample_embed_ids}")
            print(f"    ⚠️  Sample catalog IDs: {sample_cat_ids}")
    
    return merged_records, np.array(physical_values), valid_object_ids


def plot_tsne_grid(
    coords_map: dict[str, np.ndarray],
    colors: np.ndarray,
    param_name: str,
    save_path: Path,
) -> None:
    """
    Create a 2x3 grid of t-SNE plots, one for each embedding type.
    
    Args:
        coords_map: Dictionary mapping embedding key to t-SNE coordinates
        colors: Array of physical parameter values for coloring
        param_name: Name of the physical parameter for labeling
        save_path: Path to save the figure
    """
    # Order: AION embeddings first, then AstroPT
    ordered_keys = AION_EMBEDDING_KEYS + ASTROPT_EMBEDDING_KEYS
    available_keys = [k for k in ordered_keys if k in coords_map]
    
    if not available_keys:
        raise ValueError("No t-SNE coordinates available to plot")
    
    # Create grid layout (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Compute robust color limits using percentiles to exclude outliers
    valid_mask = ~np.isnan(colors)
    if valid_mask.any():
        valid_colors = colors[valid_mask]
        # Use 2nd and 98th percentiles to clip outliers
        vmin = np.percentile(valid_colors, 2)
        vmax = np.percentile(valid_colors, 98)
        
        # Add some margin if vmin and vmax are too close
        if vmax - vmin < 1e-6:
            vmin = valid_colors.min()
            vmax = valid_colors.max()
        
        print(f"    🎨 Color scale for '{param_name}': [{vmin:.3f}, {vmax:.3f}] (clipped at 2-98 percentiles)")
    else:
        vmin, vmax = 0, 1
    
    for idx, key in enumerate(available_keys):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        coords = coords_map[key]
        
        # Separate valid and NaN values
        valid_mask = ~np.isnan(colors)
        
        # Plot points with valid physical parameter values
        if valid_mask.any():
            scatter = ax.scatter(
                coords[valid_mask, 0],
                coords[valid_mask, 1],
                c=colors[valid_mask],
                cmap="viridis",
                s=10,
                alpha=0.7,
                edgecolors="none",
                vmin=vmin,  # Clip to robust range
                vmax=vmax,  # Clip to robust range
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(param_name, fontsize=10)
        
        # Plot points with NaN values in gray
        if (~valid_mask).any():
            ax.scatter(
                coords[~valid_mask, 0],
                coords[~valid_mask, 1],
                s=10,
                color="lightgray",
                alpha=0.3,
                edgecolors="none",
                label=f"{param_name} N/A",
            )
            ax.legend(loc="upper right", fontsize=8)
        
        # Format title
        model = "AION" if key in AION_EMBEDDING_KEYS else "AstroPT"
        pretty_key = key.replace("embedding_", "").replace("_", " ").title()
        ax.set_title(f"{model}: {pretty_key}", fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.grid(True, alpha=0.2)
    
    # Hide unused subplots
    for idx in range(len(available_keys), len(axes)):
        axes[idx].axis("off")
    
    fig.suptitle(
        f"t-SNE Projections of AION & AstroPT Embeddings\nColored by {param_name}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE grid to {save_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize t-SNE projections of AION and AstroPT embeddings colored by physical parameters",
    )
    parser.add_argument(
        "--aion-embeddings",
        required=True,
        help="Path to AION embeddings .pt file",
    )
    parser.add_argument(
        "--astropt-embeddings",
        required=True,
        help="Path to AstroPT embeddings .pt file",
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Path to FITS catalog file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--physical-param",
        default=None,
        help="Physical parameter from FITS catalog to use for coloring. If not specified with --all-params, uses all available numeric columns.",
    )
    parser.add_argument(
        "--all-params",
        action="store_true",
        help="Generate a separate t-SNE grid for each numeric physical parameter in the catalog",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for t-SNE reproducibility (default: 42)",
    )
    parser.add_argument(
        "--tsne-cache",
        type=str,
        default=None,
        help="Path to save/load precomputed t-SNE coordinates. If file exists, will load; otherwise will compute and save.",
    )
    parser.add_argument(
        "--tsne-preset",
        type=str,
        choices=list(TSNE_PRESETS.keys()),
        default="balanced",
        help="t-SNE preset configuration: balanced (default), local, or global",
    )
    parser.add_argument(
        "--test-all-presets",
        action="store_true",
        help="Generate visualizations for all 3 t-SNE presets (balanced, local, global) in one run",
    )
    
    args = parser.parse_args(argv)
    
    print("=" * 70)
    print("Multi-Model Embedding t-SNE Visualization")
    print("=" * 70)
    
    # Determine which presets to run
    if args.test_all_presets:
        presets_to_run = list(TSNE_PRESETS.keys())
        print(f"\n🔬 Testing ALL {len(presets_to_run)} t-SNE presets: {', '.join(presets_to_run)}")
    else:
        presets_to_run = [args.tsne_preset]
        print(f"\n📊 Using t-SNE preset: {args.tsne_preset}")
        print(f"   {TSNE_PRESETS[args.tsne_preset]['description']}")
    
    # Load data (once for all presets)
    print("\n[1/6] Loading AION embeddings...")
    aion_records = load_embeddings(Path(args.aion_embeddings))
    print(f"  Loaded {len(aion_records)} AION records")
    
    print("\n[2/6] Loading AstroPT embeddings...")
    astropt_records = load_embeddings(Path(args.astropt_embeddings))
    print(f"  Loaded {len(astropt_records)} AstroPT records")
    
    print("\n[3/6] Loading FITS catalog...")
    catalog, numeric_columns, id_column = load_fits_catalog(Path(args.catalog))
    print(f"  Loaded catalog with {len(catalog)} entries")
    print(f"  Available numeric parameters: {', '.join(numeric_columns[:10])}{'...' if len(numeric_columns) > 10 else ''}")
    
    # VALIDATION: Check if embedding object IDs match catalog IDs
    print("\n[3.5/6] Validating object ID mapping...")
    # Collect sample object IDs from both models
    aion_ids = [str(rec.get("object_id", "")) for rec in aion_records[:100]]  # Sample first 100
    astropt_ids = [str(rec.get("object_id", "")) for rec in astropt_records[:100]]
    all_sample_ids = set(aion_ids + astropt_ids)
    all_sample_ids.discard("")  # Remove empty
    
    # Check how many match the catalog
    matched = sum(1 for obj_id in all_sample_ids if obj_id in catalog)
    match_rate = (matched / len(all_sample_ids) * 100) if all_sample_ids else 0
    
    print(f"  Sample ID matching: {matched}/{len(all_sample_ids)} ({match_rate:.1f}%)")
    
    if match_rate < 10:
        print("\n" + "="*70)
        print("⚠️  WARNING: Very low ID match rate detected!")
        print("="*70)
        print(f"  Only {match_rate:.1f}% of embedding IDs were found in the catalog.")
        print(f"  This suggests a mismatch in object ID formats.")
        print(f"\n  Sample embedding IDs: {list(all_sample_ids)[:5]}")
        print(f"  Sample catalog IDs: {list(catalog.keys())[:5]}")
        print(f"\n  Catalog uses column: '{id_column}'")
        print("\n  Please verify:")
        print("    1. Embeddings and catalog are for the same dataset")
        print("    2. Object ID column is correctly identified")
        print("    3. ID formats match (e.g., integers vs strings)")
        print("="*70)
        
        if match_rate == 0:
            raise SystemExit("\n❌ No matching IDs found. Cannot proceed with visualization.")
        else:
            user_input = input("\n⚠️  Continue anyway? [y/N]: ")
            if user_input.lower() != 'y':
                raise SystemExit("Aborted by user.")
    else:
        print(f"  ✅ ID mapping looks good!")
    
    # Determine which physical parameters to visualize
    if args.all_params:
        params_to_visualize = numeric_columns
        print(f"\n  Will generate {len(params_to_visualize)} t-SNE grids (one per parameter)")
    elif args.physical_param:
        params_to_visualize = [args.physical_param]
        if args.physical_param not in numeric_columns:
            print(f"\n  Warning: '{args.physical_param}' not found in numeric columns, will try anyway...")
    else:
        params_to_visualize = numeric_columns
        print(f"\n  No --physical-param specified, will generate grids for all {len(params_to_visualize)} numeric parameters")
    
    # Process each preset
    all_generated_files = []
    for preset_idx, preset_name in enumerate(presets_to_run, 1):
        if len(presets_to_run) > 1:
            print("\n" + "=" * 70)
            print(f"📊 PRESET {preset_idx}/{len(presets_to_run)}: {preset_name.upper()}")
            print(f"   {TSNE_PRESETS[preset_name]['description']}")
            print("=" * 70)
    
        # Merge records and extract physical parameter
        # Compute or load t-SNE for all embedding types (only once per preset)
        print("\n[4/6] Computing/Loading t-SNE projections...")
        all_embedding_keys = AION_EMBEDDING_KEYS + ASTROPT_EMBEDDING_KEYS
        coords_map = {}
        
        # Build cache path with preset name if testing multiple presets
        cache_path = None
        if args.tsne_cache:
            if len(presets_to_run) > 1:
                # Add preset name to cache path
                cache_base = Path(args.tsne_cache)
                cache_path = cache_base.parent / f"{cache_base.stem}_{preset_name}{cache_base.suffix}"
            else:
                cache_path = Path(args.tsne_cache)
        
        # Check if we can load pre-computed t-SNE coordinates
        if cache_path and cache_path.exists():
            print(f"  Loading pre-computed t-SNE coordinates from cache...")
            coords_map = load_tsne_coordinates(cache_path)
            print(f"  Loaded {len(coords_map)} t-SNE projections")
            # Still need to merge records for matching object IDs
            print("  Merging embedding records...")
            merged_records = merge_embeddings_only(aion_records, astropt_records)
            print(f"  Merged {len(merged_records)} records")
        else:
            # Compute t-SNE projections from scratch
            print(f"  Computing t-SNE projections from scratch with preset '{preset_name}'...")
            # Merge embeddings without extracting parameters (faster and cleaner output)
            print("  Merging embedding records...")
            merged_records = merge_embeddings_only(aion_records, astropt_records)
            print(f"  Merged {len(merged_records)} records")
            
            for key in all_embedding_keys:
                try:
                    embeddings = stack_embeddings_with_joint(merged_records, key)
                    print(f"  Computing t-SNE for '{key}' ({embeddings.shape[0]} samples, {embeddings.shape[1]} dims)...")
                    coords = compute_tsne(embeddings, random_state=args.random_state, preset=preset_name)
                    coords_map[key] = coords
                except ValueError as e:
                    print(f"  Skipping '{key}': {e}")
            
            # Save t-SNE coordinates if cache path is provided
            if cache_path:
                save_tsne_coordinates(coords_map, cache_path)
        
        if not coords_map:
            raise SystemExit("No valid embeddings found. Cannot generate visualization.")
    
        # Generate visualizations for each parameter
        print(f"\n[5/6] Extracting physical parameters and generating visualizations...")
        generated_files = []
        
        for param_idx, param_name in enumerate(params_to_visualize, 1):
            print(f"\n  [{param_idx}/{len(params_to_visualize)}] Processing '{param_name}'...")
            
            # Extract physical values for this parameter
            _, physical_values, _ = merge_embedding_records(
                aion_records, astropt_records, catalog, param_name
            )
            valid_param_count = (~np.isnan(physical_values)).sum()
            print(f"    Found {valid_param_count}/{len(physical_values)} valid values")
            
            # Generate visualization with preset name in filename if testing multiple
            if len(presets_to_run) > 1:
                output_path = Path(args.output_dir) / f"tsne_grid_{param_name}_{preset_name}.png"
            else:
                output_path = Path(args.output_dir) / f"tsne_grid_{param_name}.png"
            plot_tsne_grid(coords_map, physical_values, param_name, output_path)
            generated_files.append(output_path)
        
        print("\n[6/6] Summary for this preset")
        print(f"  Generated {len(generated_files)} t-SNE grid visualizations")
        print(f"  Output directory: {args.output_dir}")
        all_generated_files.extend(generated_files)
    
    print("\n" + "=" * 70)
    print("✅ Done!")
    if len(presets_to_run) > 1:
        print(f"\n📊 Generated visualizations for {len(presets_to_run)} t-SNE presets:")
        for preset in presets_to_run:
            print(f"   - {preset}: {TSNE_PRESETS[preset]['description']}")
    print(f"\n📁 Total files generated: {len(all_generated_files)}")
    print(f"📂 Output directory: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
