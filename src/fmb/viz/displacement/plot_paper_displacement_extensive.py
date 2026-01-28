"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.displacement.plot_paper_displacement_extensive
Description: FMB module: fmb.viz.displacement.plot_paper_displacement_extensive
"""

import argparse
from pathlib import Path
from typing import Sequence, Optional, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Publication Style Settings ---
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
except Exception:
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    })

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "savefig.bbox": "tight",
})

KEYS = {
    "AION": {"Images": "embedding_hsc", "Spectra": "embedding_spectrum", "Joint": "embedding_hsc_desi"},
    "AstroPT": {"Images": "embedding_images", "Spectra": "embedding_spectra", "Joint": "embedding_joint"},
    "AstroCLIP": {"Images": "embedding_images", "Spectra": "embedding_spectra", "Joint": "embedding_joint"}
}

COLORS = {"AION": "#1f77b4", "AstroPT": "#ff7f0e", "AstroCLIP": "#2ca02c"}

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['object_id'] = df['object_id'].astype(str)
    return df

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aion-scores", required=True)
    parser.add_argument("--astropt-scores", required=True)
    parser.add_argument("--astroclip-scores", required=True)
    parser.add_argument("--save-prefix", default="paper/Final_results/paper_displacement_extensive")
    args = parser.parse_args(argv)

    print("Loading scores...")
    dfs = {m: load_data(Path(args.aion_scores if m == "AION" else (args.astropt_scores if m == "AstroPT" else args.astroclip_scores))) for m in KEYS}
    
    modalities = ["Images", "Spectra", "Joint"]
    systems = []
    for model in ["AION", "AstroPT", "AstroCLIP"]:
        for mod in modalities:
            systems.append((model, mod))

    print("Merging data for consistency...")
    big_merged = None
    for model, mod in systems:
        key = KEYS[model][mod]
        df = dfs[model]
        mod_data = df[df['embedding_key'] == key][['object_id', 'rank']].rename(columns={'rank': f'rank_{model}_{mod}'})
        if big_merged is None:
            big_merged = mod_data
        else:
            big_merged = big_merged.merge(mod_data, on='object_id')
    
    n_total = len(big_merged)
    print(f"Matched {n_total} objects.")

    retention_matrix = np.zeros((9, 9))
    top_1_thr = n_total * 0.01

    # --- 1. Generate Heatmap ---
    for i, (src_model, src_mod) in enumerate(systems):
        src_col = f'rank_{src_model}_{src_mod}'
        mask = big_merged[src_col] <= top_1_thr
        for j, (tgt_model, tgt_mod) in enumerate(systems):
            tgt_col = f'rank_{tgt_model}_{tgt_mod}'
            retained = (big_merged.loc[mask, tgt_col] <= top_1_thr).sum() / mask.sum() * 100
            retention_matrix[i, j] = retained

    labels = [f"{m}-{mo}" for m, mo in systems]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(retention_matrix, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels, cmap="YlGnBu", cbar_kws={'label': 'Retention Top 1\% (\%)'})
    plt.title("Extensive Anomaly Retention Matrix (Top 1\% $\\rightarrow$ Top 1\%)")
    plt.xlabel("Target System")
    plt.ylabel("Source System")
    plt.tight_layout()
    plt.savefig(f"{args.save_prefix}_heatmap.png", dpi=300)
    print(f"Saved heatmap to {args.save_prefix}_heatmap.png")

    # --- 2. Generate 9x9 Histogram Grid ---
    fig, axes = plt.subplots(9, 9, figsize=(18, 18), sharex=True, sharey='row')
    for i, (src_model, src_mod) in enumerate(systems):
        src_col = f'rank_{src_model}_{src_mod}'
        mask = big_merged[src_col] <= top_1_thr
        color = COLORS[src_model]
        
        for j, (tgt_model, tgt_mod) in enumerate(systems):
            ax = axes[i, j]
            tgt_col = f'rank_{tgt_model}_{tgt_mod}'
            ranks_pct = (big_merged.loc[mask, tgt_col] / n_total) * 100
            
            ax.hist(ranks_pct, bins=30, range=(0, 100), color=color, alpha=0.7, edgecolor='none')
            ax.axvline(x=1, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
            
            if i == 0: ax.set_title(f"To: {tgt_model}\n{tgt_mod}", fontsize=8)
            if j == 0: ax.set_ylabel(f"From: {src_model}\n{src_mod}\nCount", fontsize=8)
            if i == 8: ax.set_xlabel("Rank \%", fontsize=8)
            
            # Retention text
            ax.text(0.95, 0.95, f"{retention_matrix[i, j]:.1f}\%", transform=ax.transAxes, ha='right', va='top', fontsize=7, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.save_prefix}_grid.png", dpi=200)
    print(f"Saved grid to {args.save_prefix}_grid.png")
    plt.close(fig)

if __name__ == "__main__":
    main()
