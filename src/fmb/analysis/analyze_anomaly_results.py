#!/usr/bin/env python3
"""
analyze_anomaly_results.py

Analyzes the output CSV (all.csv) from detect_multimodal_anomalies.py.
Generates:
- Correlation heatmaps (between models)
- Overlap plots (Venn/Bar)
- Scatter plots of ranks
- Distribution of Multimodal Uplift
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import venn, handle if missing
try:
    from matplotlib_venn import venn2, venn3
    HAS_VENN = True
except ImportError:
    HAS_VENN = False

def parse_args():
    p = argparse.ArgumentParser(description="Analyze anomaly results.")
    p.add_argument("--input-csv", type=str, required=True, help="Path to all.csv from detection script.")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to save plots.")
    p.add_argument("--top-k", type=int, default=200, help="Top-K threshold for overlap analysis.")
    return p.parse_args()

def plot_correlations(df, out_dir):
    """Correlation of ranks between models."""
    if "model" not in df.columns:
        return
    
    # Pivot to get rank_cosine/rank_mm_geo per model
    # Index: object_id, Columns: model
    metrics = ["rank_cosine", "rank_mm_geo", "p_joint"]
    
    for m in metrics:
        if m not in df.columns: continue
        
        pivot = df.pivot(index="object_id", columns="model", values=m)
        if pivot.shape[1] < 2:
            continue
            
        plt.figure(figsize=(8, 6))
        # Spearman because ranks
        corr = pivot.corr(method="spearman")
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0, vmax=1)
        plt.title(f"Spearman Correlation of {m} between Models")
        plt.tight_layout()
        plt.savefig(out_dir / f"corr_{m}.png")
        plt.close()

def plot_scatter_ranks(df, out_dir):
    """Scatter p_img vs p_spec."""
    plt.figure(figsize=(8, 8))
    
    # If multiple models, facet grid? Or just overlap with hue?
    if "model" in df.columns:
        sns.scatterplot(data=df, x="p_img", y="p_spec", hue="model", alpha=0.3, s=10)
    else:
        sns.scatterplot(data=df, x="p_img", y="p_spec", alpha=0.3, s=10)
        
    plt.xlabel("Image Percentile (1.0=Anomaly)")
    plt.ylabel("Spectrum Percentile (1.0=Anomaly)")
    plt.title("Image vs Spectrum Anomaly Percentiles")
    plt.plot([0,1], [0,1], "k--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_p_img_p_spec.png")
    plt.close()

def plot_uplift(df, out_dir):
    """Histogram of Uplift"""
    if "uplift_mm" not in df.columns:
        return
        
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x="uplift_mm", hue="model" if "model" in df.columns else None, 
                 element="step", kde=True, bins=50)
    plt.axvline(0, color="k", linestyle="--")
    plt.title("Distribution of Multimodal Uplift\n(p_joint - max(p_img, p_spec))")
    plt.xlabel("Uplift (>0 means multimodal fusion increases anomaly score)")
    plt.tight_layout()
    plt.savefig(out_dir / "uplift_hist.png")
    plt.close()

def plot_overlaps(df, k, out_dir):
    """Venn diagram of Top-K across models."""
    if "model" not in df.columns:
        return
        
    models = df["model"].unique()
    if len(models) not in [2, 3]:
        print(f"Skipping Venn: requires 2 or 3 models, found {len(models)} ({models})")
        return

    if not HAS_VENN:
        print("Skipping Venn: matplotlib_venn not installed.")
        return

    # Get Top-K sets for Multimodal Score
    sets = {}
    for m in models:
        # Lower rank is better? Wait, rank 1 = anomaly.
        # So we take rank <= k
        sub = df[df["model"] == m]
        # Assuming rank_mm_geo exists, or derive it
        if "rank_mm_geo" in sub.columns:
            topk = set(sub[sub["rank_mm_geo"] <= k]["object_id"])
            sets[m] = topk
            
    plt.figure(figsize=(8, 8))
    if len(models) == 2:
        venn2([sets[models[0]], sets[models[1]]], set_labels=models)
    elif len(models) == 3:
        venn3([sets[models[0]], sets[models[1]], sets[models[2]]], set_labels=models)
        
    plt.title(f"Overlap of Top-{k} Multimodal Anomalies")
    plt.savefig(out_dir / f"venn_top{k}_models.png")
    plt.close()

def plot_robustness_comparison(df, k, out_dir):
    """
    If we have robust scores, let's see how they correlate with 'simple' scores.
    Scatter: score_mm_geo vs score_mm_robust_cross_model
    """
    if "score_mm_robust_cross_model" not in df.columns:
        return

    plt.figure(figsize=(8,8))
    sns.scatterplot(data=df, x="score_mm_geo", y="score_mm_robust_cross_model", hue="model", alpha=0.5)
    plt.title("Standard MM Score vs Robust Cross-Model Score")
    plt.plot([0,1], [0,1], "k--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_robust_vs_standard.png")
    plt.close()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    # 1. Correlations
    print("Generating correlation plots...")
    plot_correlations(df, out_dir)
    
    # 2. Scatter
    print("Generating scatter plots...")
    plot_scatter_ranks(df, out_dir)
    
    # 3. Uplift
    print("Generating uplift plot...")
    plot_uplift(df, out_dir)
    
    # 4. Overlaps
    print("Generating overlap plots...")
    plot_overlaps(df, args.top_k, out_dir)
    
    # 5. Robustness
    print("Generating robustness comparisons...")
    plot_robustness_comparison(df, args.top_k, out_dir)
    
    print(f"Done. Results in {out_dir}")

if __name__ == "__main__":
    main()
