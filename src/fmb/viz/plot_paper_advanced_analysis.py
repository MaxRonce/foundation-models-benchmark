"""
Script to generate ADVANCED publication figures:
1. Spearman Correlation Clustermap (Global similarity)
2. Jaccard (IoU) Clustermap for Top 1% (Anomaly overlap)
3. Disagreement Scatter Plot (AION Joint vs AstroPT Joint)
4. Extraction of "Controversial" objects (High in one, Low in other)

Usage:
    python -m scratch.plot_paper_advanced_analysis \
        --aion-scores paper/Final_results/_aion.csv \
        --astropt-scores paper/Final_results/_astropt.csv \
        --astroclip-scores paper/Final_results/_astroclip.csv \
        --save-prefix paper/Final_results/paper_advanced
"""
import argparse
from pathlib import Path
from typing import Sequence, Optional, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

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
    "font.size": 10,
    "axes.labelsize": 10,
    "savefig.bbox": "tight",
})

KEYS = {
    "AION": {"Images": "embedding_hsc", "Spectra": "embedding_spectrum", "Joint": "embedding_hsc_desi"},
    "AstroPT": {"Images": "embedding_images", "Spectra": "embedding_spectra", "Joint": "embedding_joint"},
    "AstroCLIP": {"Images": "embedding_images", "Spectra": "embedding_spectra", "Joint": "embedding_joint"}
}

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['object_id'] = df['object_id'].astype(str)
    return df

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aion-scores", required=True)
    parser.add_argument("--astropt-scores", required=True)
    parser.add_argument("--astroclip-scores", required=True)
    parser.add_argument("--save-prefix", default="paper/Final_results/paper_advanced")
    args = parser.parse_args(argv)

    print("Loading scores...")
    dfs = {m: load_data(Path(args.aion_scores if m == "AION" else (args.astropt_scores if m == "AstroPT" else args.astroclip_scores))) for m in KEYS}
    
    systems = []
    modalities = ["Images", "Spectra", "Joint"]
    for model in ["AION", "AstroPT", "AstroCLIP"]:
        for mod in modalities:
            systems.append((model, mod))

    print("Merging data for consistency...")
    big_merged = None
    for model, mod in systems:
        key = KEYS[model][mod]
        df = dfs[model]
        mod_data = df[df['embedding_key'] == key][['object_id', 'rank']].rename(columns={'rank': f'{model}-{mod}'})
        if big_merged is None:
            big_merged = mod_data
        else:
            big_merged = big_merged.merge(mod_data, on='object_id')
    
    n_total = len(big_merged)
    print(f"Matched {n_total} objects.")

    # --- 1. Spearman Clustermap (Rank Correlation) ---
    print("Computing Spearman Correlations...")
    rank_cols = [f'{m}-{mod}' for m, mod in systems]
    corr_matrix = big_merged[rank_cols].corr(method='spearman')
    
    g = sns.clustermap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, 
                       dendrogram_ratio=(.1, .1), cbar_pos=(.02, .32, .03, .2))
    g.fig.suptitle("Hierarchical Clustering of Systems (Spearman Correlation)", y=1.02)
    g.savefig(f"{args.save_prefix}_spearman_clustermap.png", dpi=300)
    print(f"Saved {args.save_prefix}_spearman_clustermap.png")

    # --- 2. Jaccard Index Clustermap (Top 1% Intersection) ---
    print("Computing Jaccard Indices...")
    top_1_thr = n_total * 0.01
    jaccard_matrix = pd.DataFrame(index=rank_cols, columns=rank_cols, dtype=float)
    
    top_1_sets = {}
    for col in rank_cols:
        top_1_sets[col] = set(big_merged[big_merged[col] <= top_1_thr]['object_id'])

    for col1 in rank_cols:
        for col2 in rank_cols:
            s1 = top_1_sets[col1]
            s2 = top_1_sets[col2]
            iou = len(s1.intersection(s2)) / len(s1.union(s2))
            jaccard_matrix.loc[col1, col2] = iou

    g2 = sns.clustermap(jaccard_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                        dendrogram_ratio=(.1, .1), cbar_pos=(.02, .32, .03, .2))
    g2.fig.suptitle("Hierarchical Clustering of Top 1\% Anomalies (Jaccard Index)", y=1.02)
    g2.savefig(f"{args.save_prefix}_jaccard_clustermap.png", dpi=300)
    print(f"Saved {args.save_prefix}_jaccard_clustermap.png")

    # --- 3. Disagreement Analysis (AION-Joint vs AstroPT-Joint) ---
    print("Analyzing Disagreements...")
    sys1 = "AION-Joint"
    sys2 = "AstroPT-Joint"
    
    x = big_merged[sys1].values
    y = big_merged[sys2].values
    
    # Normalize ranks to percentile (0-100)
    x_pct = (x / n_total) * 100
    y_pct = (y / n_total) * 100
    
    # Identify Controversial Zones
    # Case A: Anomaly in Sys1 (Top 1%), Normal in Sys2 (>50%)
    mask_a = (x_pct <= 1) & (y_pct > 50)
    # Case B: Anomaly in Sys2 (Top 1%), Normal in Sys1 (>50%)
    mask_b = (y_pct <= 1) & (x_pct > 50)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    # Hexbin for density
    hb = ax.hexbin(x_pct, y_pct, gridsize=50, cmap='Greys', mincnt=1, bins='log', alpha=0.6)
    
    # Highlight Controversial
    ax.scatter(x_pct[mask_a], y_pct[mask_a], s=10, c='red', label=f'Anomaly in {sys1}\nNormal in {sys2}', alpha=0.8)
    ax.scatter(x_pct[mask_b], y_pct[mask_b], s=10, c='blue', label=f'Anomaly in {sys2}\nNormal in {sys1}', alpha=0.8)
    
    ax.set_xlabel(f"{sys1} Rank (\%)")
    ax.set_ylabel(f"{sys2} Rank (\%)")
    ax.set_title(f"Disagreement Analysis: {sys1} vs {sys2}")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Annotate regions
    ax.axvline(x=1, color='red', linestyle=':', alpha=0.5)
    ax.axhline(y=1, color='blue', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(f"{args.save_prefix}_disagreement_scatter.png", dpi=300)
    print(f"Saved {args.save_prefix}_disagreement_scatter.png")

    # Export Disagreements
    disagreements = big_merged[mask_a | mask_b][['object_id', sys1, sys2]].copy()
    disagreements['type'] = np.where(mask_a[mask_a | mask_b], f'{sys1}_Anomaly', f'{sys2}_Anomaly')
    disagreements.to_csv(f"{args.save_prefix}_disagreement_objects.csv", index=False)
    print(f"Exported {len(disagreements)} controversial objects to CSV.")

if __name__ == "__main__":
    main()
