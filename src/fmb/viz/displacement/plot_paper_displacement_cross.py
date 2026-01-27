"""
Script to generate a publication-ready CROSS-MODALITY displacement analysis figure.
Displays a 3x3 grid:
Rows: Source Modality (Images, Spectra, Joint)
Columns: Target Modality (Images, Spectra, Joint)
Each cell shows overlaid histograms for AION, AstroPT, and AstroCLIP.

Usage:
    python -m scratch.plot_paper_displacement_cross \
        --aion-scores paper/Final_results/_aion.csv \
        --astropt-scores paper/Final_results/_astropt.csv \
        --astroclip-scores paper/Final_results/_astroclip.csv \
        --save paper/Final_results/paper_displacement_cross.png
"""
import argparse
from pathlib import Path
from typing import Sequence, Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

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
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
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

# Mapping of embedding types to keys for each model
KEYS = {
    "AION": {
        "Images": "embedding_hsc",
        "Spectra": "embedding_spectrum",
        "Joint": "embedding_hsc_desi"
    },
    "AstroPT": {
        "Images": "embedding_images",
        "Spectra": "embedding_spectra",
        "Joint": "embedding_joint"
    },
    "AstroCLIP": {
        "Images": "embedding_images",
        "Spectra": "embedding_spectra",
        "Joint": "embedding_joint"
    }
}

COLORS = {
    "AION": "#1f77b4",     # Blue
    "AstroPT": "#ff7f0e",  # Orange
    "AstroCLIP": "#2ca02c" # Green
}

def load_data(path: Path) -> pd.DataFrame:
    """Load anomaly scores from CSV."""
    df = pd.read_csv(path)
    df['object_id'] = df['object_id'].astype(str)
    return df

def plot_cross_panel(ax, src_mod, tgt_mod, model_data_list, n_total):
    """Plot overlaid displacement histograms for multiple models in one panel."""
    
    top_1_threshold = n_total * 0.01
    stats_lines = []
    
    for model_name, ranks_src, ranks_tgt in model_data_list:
        color = COLORS[model_name]
        subset_mask = ranks_src <= top_1_threshold
        
        if subset_mask.sum() == 0:
            continue

        ranks_pct = (ranks_tgt[subset_mask] / n_total) * 100
        
        # Plot Histogram (step for better overlay)
        ax.hist(ranks_pct, bins=40, range=(0, 100), color=color, histtype='step', alpha=0.9, linewidth=1.2, zorder=3, label=model_name)
        # Optional: filled area with low alpha
        ax.hist(ranks_pct, bins=40, range=(0, 100), color=color, alpha=0.1, zorder=2)
        
        # Calculate stats
        retained_1 = (ranks_tgt[subset_mask] <= top_1_threshold).sum() / subset_mask.sum() * 100
        stats_lines.append(rf"{model_name}: {retained_1:.1f}\%")

    # Add markers for Top 1% and Top 10%
    ax.axvline(x=1, color='red', linestyle='--', linewidth=0.8, alpha=0.5, zorder=4)
    ax.axvspan(0, 1, color='red', alpha=0.05, zorder=1)
    
    # Add text box
    stats_text = "\\textbf{Retained Top 1\\%}:\n" + "\n".join(stats_lines)
    ax.text(0.95, 0.92, stats_text, transform=ax.transAxes, ha='right', va='top', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'), zorder=5, fontsize=6)
    
    ax.set_title(rf"\textbf{{{src_mod}}} $\rightarrow$ \textbf{{{tgt_mod}}}", pad=5)
    ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate cross-modality displacement analysis grid")
    parser.add_argument("--aion-scores", required=True, help="AION scores CSV")
    parser.add_argument("--astropt-scores", required=True, help="AstroPT scores CSV")
    parser.add_argument("--astroclip-scores", required=True, help="AstroCLIP scores CSV")
    parser.add_argument("--save", default="paper_displacement_cross.png", help="Output filename")
    
    args = parser.parse_args(argv)
    
    # Load all datasets
    print("Loading scores...")
    dfs = {
        "AION": load_data(Path(args.aion_scores)),
        "AstroPT": load_data(Path(args.astropt_scores)),
        "AstroCLIP": load_data(Path(args.astroclip_scores))
    }
    
    modalities = ["Images", "Spectra", "Joint"]
    
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey='row')
    
    # We need to compute matched objects for *each* model across all *its* modalities
    # Actually, to keep it consistent across models, lets merge everything into one big DF
    print("Merging data for consistency...")
    big_merged = None
    
    for model_name, df in dfs.items():
        model_subset = None
        for mod in modalities:
            key = KEYS[model_name][mod]
            mod_data = df[df['embedding_key'] == key][['object_id', 'rank']].rename(columns={'rank': f'rank_{model_name}_{mod}'})
            if model_subset is None:
                model_subset = mod_data
            else:
                model_subset = model_subset.merge(mod_data, on='object_id')
        
        if big_merged is None:
            big_merged = model_subset
        else:
            big_merged = big_merged.merge(model_subset, on='object_id')
            
    n_total = len(big_merged)
    print(f"Matched {n_total} objects across all models and modalities.")

    for row_idx, src_mod in enumerate(modalities):
        for col_idx, tgt_mod in enumerate(modalities):
            ax = axes[row_idx, col_idx]
            
            model_data_list = []
            for model_name in ["AION", "AstroPT", "AstroCLIP"]:
                ranks_src = big_merged[f'rank_{model_name}_{src_mod}'].values
                ranks_tgt = big_merged[f'rank_{model_name}_{tgt_mod}'].values
                model_data_list.append((model_name, ranks_src, ranks_tgt))
            
            plot_cross_panel(ax, src_mod, tgt_mod, model_data_list, n_total)
            
            if col_idx == 0:
                ax.set_ylabel(rf"\textbf{{{src_mod}}}" + "\nCount")
            if row_idx == 2:
                ax.set_xlabel(rf"Percentile Rank in \textbf{{{tgt_mod}}}")
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower right', fontsize=6)

    plt.tight_layout()
    
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()
