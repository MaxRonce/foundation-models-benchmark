"""
Script to generate a publication-ready displacement analysis figure.
Displays two panels vertically:
1. Top 1% AstroPT Joint -> Where do they fall in AION Joint?
2. Top 1% AION Joint -> Where do they fall in AstroPT Joint?

Usage:
    python -m scratch.plot_paper_displacement \
        --aion-scores /path/to/aion_scores.csv \
        --astropt-scores /path/to/astropt_scores.csv \
        --save paper_displacement.png
"""
import argparse
from pathlib import Path
from typing import Sequence, Optional

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

def load_data(path: Path) -> pd.DataFrame:
    """Load anomaly scores from CSV."""
    df = pd.read_csv(path)
    df['object_id'] = df['object_id'].astype(str)
    return df

def plot_displacement_vertical(
    merged: pd.DataFrame,
    save_path: Path,
) -> None:
    """Create the 2-panel vertical displacement plot."""
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    
    n_samples = len(merged)
    top_1_threshold = n_samples * 0.01
    
    # --- Panel 1: Top 1% AstroPT -> AION ---
    ax1 = axes[0]
    
    # Select Top 1% in AstroPT (rank_astropt <= threshold)
    subset1 = merged[merged['rank_astropt'] <= top_1_threshold]
    
    # Where do they fall in AION? (rank_aion)
    # Normalize to percentile (0-100)
    ranks_pct1 = (subset1['rank_aion'] / n_samples) * 100
    
    # Plot Histogram
    ax1.hist(ranks_pct1, bins=50, range=(0, 100), color='#1f77b4', edgecolor='black', alpha=0.7, linewidth=0.5, zorder=3)
    
    # Add markers for Top 1% and Top 10%
    ax1.axvline(x=1, color='red', linestyle='--', linewidth=1, label=r'Top 1\%', zorder=4)
    ax1.axvline(x=10, color='orange', linestyle='--', linewidth=1, label=r'Top 10\%', zorder=4)
    
    # Shade regions to highlight retention
    ax1.axvspan(0, 1, color='red', alpha=0.1, zorder=1)
    ax1.axvspan(1, 10, color='orange', alpha=0.1, zorder=1)
    
    # Calculate stats
    retained_1 = (subset1['rank_aion'] <= top_1_threshold).sum() / len(subset1) * 100
    retained_10 = (subset1['rank_aion'] <= (n_samples * 0.1)).sum() / len(subset1) * 100
    
    # Add text box
    stats_text1 = (
        f"\\textbf{{Retained}}:\n"
        f"Top 1\\%: {retained_1:.1f}\\%\n"
        f"Top 10\\%: {retained_10:.1f}\\%"
    )
    ax1.text(0.95, 0.85, stats_text1, transform=ax1.transAxes, ha='right', va='top', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'), zorder=5)
    
    ax1.set_title(r"\textbf{AstroPT} (Top 1\%) $\rightarrow$ \textbf{AION}")
    ax1.set_ylabel("Count")
    ax1.legend(loc='center right', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.4, zorder=0)
    
    # --- Panel 2: Top 1% AION -> AstroPT ---
    ax2 = axes[1]
    
    # Select Top 1% in AION
    subset2 = merged[merged['rank_aion'] <= top_1_threshold]
    
    # Where do they fall in AstroPT?
    ranks_pct2 = (subset2['rank_astropt'] / n_samples) * 100
    
    ax2.hist(ranks_pct2, bins=50, range=(0, 100), color='#ff7f0e', edgecolor='black', alpha=0.7, linewidth=0.5, zorder=3)
    
    ax2.axvline(x=1, color='red', linestyle='--', linewidth=1, zorder=4)
    ax2.axvline(x=10, color='orange', linestyle='--', linewidth=1, zorder=4)
    
    # Shade regions
    ax2.axvspan(0, 1, color='red', alpha=0.1, zorder=1)
    ax2.axvspan(1, 10, color='orange', alpha=0.1, zorder=1)
    
    # Calculate stats
    retained_2_1 = (subset2['rank_astropt'] <= top_1_threshold).sum() / len(subset2) * 100
    retained_2_10 = (subset2['rank_astropt'] <= (n_samples * 0.1)).sum() / len(subset2) * 100
    
    stats_text2 = (
        f"\\textbf{{Retained}}:\n"
        f"Top 1\\%: {retained_2_1:.1f}\\%\n"
        f"Top 10\\%: {retained_2_10:.1f}\\%"
    )
    ax2.text(0.95, 0.85, stats_text2, transform=ax2.transAxes, ha='right', va='top', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'), zorder=5)
    
    ax2.set_title(r"\textbf{AION} (Top 1\%) $\rightarrow$ \textbf{AstroPT}")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Percentile Rank in Target Model")
    ax2.grid(True, linestyle=':', alpha=0.4, zorder=0)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
        
    plt.close(fig)

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate publication displacement analysis")
    parser.add_argument("--aion-scores", required=True, help="AION scores CSV")
    parser.add_argument("--astropt-scores", required=True, help="AstroPT scores CSV")
    parser.add_argument("--save", default="paper_displacement.png", help="Output filename")
    
    args = parser.parse_args(argv)
    
    # Load data
    print("Loading scores...")
    aion_df = load_data(Path(args.aion_scores))
    astropt_df = load_data(Path(args.astropt_scores))
    
    # Filter for Joint embeddings
    print(f"Filtering for keys: {KEY_AION} and {KEY_ASTROPT}")
    aion_joint = aion_df[aion_df['embedding_key'] == KEY_AION].copy()
    astropt_joint = astropt_df[astropt_df['embedding_key'] == KEY_ASTROPT].copy()
    
    if len(aion_joint) == 0:
        raise ValueError(f"No data found for AION key: {KEY_AION}")
    if len(astropt_joint) == 0:
        raise ValueError(f"No data found for AstroPT key: {KEY_ASTROPT}")
        
    # Merge on object_id
    print("Merging datasets...")
    merged = pd.merge(
        aion_joint[['object_id', 'rank']],
        astropt_joint[['object_id', 'rank']],
        on='object_id',
        suffixes=('_aion', '_astropt'),
        how='inner'
    )
    
    print(f"Matched {len(merged)} objects")
    
    # Plot
    plot_displacement_vertical(merged, Path(args.save))

if __name__ == "__main__":
    main()
