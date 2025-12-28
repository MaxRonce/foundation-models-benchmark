"""
Script to generate a publication-ready t-SNE comparison figure.
Displays two panels:
1. AstroPT Joint Embeddings (Left)
2. AION Joint Embeddings (HSC+DESI) (Right)

Colored by a specific physical parameter.
Uses pre-computed t-SNE coordinates.

Usage:
    python -m scratch.plot_paper_tsne_comparison \
        --aion-embeddings /path/to/aion.pt \
        --astropt-embeddings /path/to/astropt.pt \
        --catalog /path/to/catalog.fits \
        --tsne-cache /path/to/tsne_coords.pt \
        --physical-param redshift \
        --save paper_tsne_comparison.png
"""
import argparse
from pathlib import Path
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits

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

# Keys mapping
# AstroPT Joint -> "embedding_joint"
# AION Joint (HSC+DESI) -> "embedding_hsc_desi"
KEY_ASTROPT = "embedding_joint"
KEY_AION = "embedding_hsc_desi"

REQUIRED_KEYS = [KEY_ASTROPT, KEY_AION]


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

def plot_comparison(
    coords_astropt: np.ndarray,
    coords_aion: np.ndarray,
    values_astropt: np.ndarray,
    values_aion: np.ndarray,
    param_name: str,
    save_path: Path,
    use_hexbin: bool = False,
) -> None:
    """Create the 2-panel comparison plot."""
    
    # Setup figure
    # Remove sharex/sharey to allow independent aspect ratio control without issues
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Combine values to find robust color limits
    all_values = np.concatenate([
        values_astropt[~np.isnan(values_astropt)],
        values_aion[~np.isnan(values_aion)]
    ])
    
    if len(all_values) == 0:
        print("Warning: No valid physical parameter values found.")
        vmin, vmax = 0, 1
    else:
        vmin = np.percentile(all_values, 2)
        vmax = np.percentile(all_values, 98)
        if vmax - vmin < 1e-6:
            vmin, vmax = all_values.min(), all_values.max()
        
    print(f"Color scale for {param_name}: [{vmin:.3f}, {vmax:.3f}]")
    
    # Common kwargs
    # Suggested colormaps for Redshift:
    # "RdYlBu_r": Blue -> Yellow -> Red (Standard for temperature/velocity/z)
    # "Spectral_r": Blue -> Yellow -> Red (Similar to above but more vibrant)
    # "coolwarm": Blue -> Red (Diverging, good if centered)
    # "magma": Black -> Red -> Yellow (High contrast)
    plot_kwargs = dict(
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        rasterized=True 
    )
    
    # Helper to plot one panel
    def plot_panel(ax, coords, values, title):
        mask = ~np.isnan(values)
        
        # Normalize coordinates to [0, 1] to match the grid script's "stretch to square" behavior
        # This ensures the shape looks the same as the grid visualization
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        
        # Avoid division by zero
        denom = c_max - c_min
        denom[denom == 0] = 1e-9
        
        norm_coords = (coords - c_min) / denom
        
        if use_hexbin:
            # Hexbin plot
            hb = ax.hexbin(
                norm_coords[mask, 0], 
                norm_coords[mask, 1], 
                C=values[mask], 
                gridsize=75, 
                reduce_C_function=np.mean,
                mincnt=1,
                linewidths=0.2,
                edgecolors='face',
                **plot_kwargs
            )
            return hb
        else:
            # Scatter plot
            # Plot NaNs first (gray)
            if (~mask).any():
                ax.scatter(norm_coords[~mask, 0], norm_coords[~mask, 1], s=3, c='lightgray', alpha=0.3, rasterized=True)
            
            # Plot valid
            sc = None
            if mask.any():
                sc = ax.scatter(norm_coords[mask, 0], norm_coords[mask, 1], s=3, alpha=0.7, edgecolors="none", c=values[mask], **plot_kwargs)
            return sc

    # 1. AstroPT (Left)
    ax_astro = axes[0]
    plot_panel(ax_astro, coords_astropt, values_astropt, "")
    
    ax_astro.set_title(r"\textbf{AstroPT} (Spectra + Images)", fontsize=14)
    # Hide ticks for cleaner look since axes are normalized/arbitrary
    ax_astro.set_xticks([])
    ax_astro.set_yticks([])
    ax_astro.set_aspect('equal')
    
    # 2. AION (Right)
    ax_aion = axes[1]
    mappable = plot_panel(ax_aion, coords_aion, values_aion, "")
    
    ax_aion.set_title(r"\textbf{AION} (Spectra + Images)", fontsize=14)
    ax_aion.set_xticks([])
    ax_aion.set_yticks([])
    ax_aion.set_aspect('equal')
    
    # Colorbar
    if mappable is not None:
        # Adjust layout to make room for colorbar
        fig.subplots_adjust(right=0.85, wspace=0.1)
        cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5]) # Shorter, centered colorbar
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        
        # Format label nicely
        label = param_name
        if param_name == "Z" or param_name.lower() == "redshift":
            label = r"Redshift $z$"
        elif "mass" in param_name.lower():
            label = r"$\log(M_*/M_\odot)$"
        else:
            label = param_name.replace("_", " ").title()
            
        cbar.set_label(label, fontsize=12)
        cbar.solids.set_edgecolor("face") # Remove lines in PDF
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
        
    plt.close(fig)
def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate publication t-SNE/UMAP comparison")
    parser.add_argument("--aion-embeddings", required=True, help="AION .pt file")
    parser.add_argument("--astropt-embeddings", required=True, help="AstroPT .pt file")
    parser.add_argument("--catalog", required=True, help="FITS catalog")
    parser.add_argument("--coords-cache", required=True, help="Path to pre-computed coords .pt file")
    parser.add_argument("--physical-param", required=True, help="Parameter to color by")
    parser.add_argument("--save", default="paper_comparison.png", help="Output filename")
    parser.add_argument("--hexbin", action="store_true", help="Use hexbin plot instead of scatter")
    parser.add_argument("--title-suffix", default="", help="Suffix for plot titles (e.g. 'UMAP')")
    
    args = parser.parse_args(argv)
    
    # 1. Load Data
    print("Loading embeddings...")
    aion_recs = load_embeddings(Path(args.aion_embeddings))
    astropt_recs = load_embeddings(Path(args.astropt_embeddings))
    
    print("Loading catalog...")
    catalog, _ = load_fits_catalog(Path(args.catalog))
    
    print("Loading coordinates...")
    coords_map = load_coordinates(Path(args.coords_cache))
    
    # Check keys
    if KEY_ASTROPT not in coords_map:
        raise ValueError(f"Missing '{KEY_ASTROPT}' in cache")
    if KEY_AION not in coords_map:
        raise ValueError(f"Missing '{KEY_AION}' in cache")
        
    # 2. Match and Extract
    print(f"Extracting parameter '{args.physical_param}'...")
    
    # We need to reconstruct the list of IDs that correspond to the coordinates.
    # The `plot_paper_umap_grid.py` script saves coordinates computed on `stack_embeddings`.
    # `stack_embeddings` iterates through records and keeps those with valid embeddings.
    # So we need to replicate that logic to get the IDs in the correct order.
    
    def get_ids_for_key(records, key):
        ids = []
        for rec in records:
            tensor = rec.get(key)
            # Special handling for AstroPT Joint
            if tensor is None and key == "embedding_joint":
                if rec.get("embedding_images") is not None and rec.get("embedding_spectra") is not None:
                    tensor = True # Just need to know it exists
            
            if tensor is not None:
                oid = rec.get("object_id", "")
                if isinstance(oid, torch.Tensor):
                     oid = oid.item() if oid.numel() == 1 else str(oid.tolist())
                ids.append(str(oid))
        return ids

    aion_ids = get_ids_for_key(aion_recs, KEY_AION)
    astropt_ids = get_ids_for_key(astropt_recs, KEY_ASTROPT)
    
    # Verify lengths
    coords_aion = coords_map[KEY_AION]
    coords_astro = coords_map[KEY_ASTROPT]
    
    if len(coords_aion) != len(aion_ids):
        print(f"Warning: AION coords length ({len(coords_aion)}) != IDs length ({len(aion_ids)})")
    if len(coords_astro) != len(astropt_ids):
        print(f"Warning: AstroPT coords length ({len(coords_astro)}) != IDs length ({len(astropt_ids)})")
        
    # Extract values
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
        
    values_aion = get_values(aion_ids)
    values_astro = get_values(astropt_ids)
    
    # 3. Plot
    plot_comparison(
        coords_astro,
        coords_aion,
        values_astro, 
        values_aion,
        args.physical_param,
        Path(args.save),
        use_hexbin=args.hexbin
    )

if __name__ == "__main__":
    main()
