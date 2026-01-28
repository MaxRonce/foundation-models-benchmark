#!/usr/bin/env python3
"""
Foundation Models Benchmark (FMB)

Module: fmb.analysis.displacement
Description: Embedding space displacement analysis
"""

"""
Unified Displacement Analysis Module.
Combines:
1. Multi-Model Analysis (Model A vs B)
2. Cross-Modality Analysis (Modality X vs Y)
3. Extensive Analysis (9x9 Pairwise)
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Sequence, Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

from fmb.paths import load_paths
from fmb.viz.style import set_style

# Apply style
set_style()

KEYS = {
    "AION": {"Images": "embedding_hsc", "Spectra": "embedding_spectrum", "Joint": "embedding_hsc_desi"},
    "AstroPT": {"Images": "embedding_images", "Spectra": "embedding_spectra", "Joint": "embedding_joint"},
    "AstroCLIP": {"Images": "embedding_images", "Spectra": "embedding_spectra", "Joint": "embedding_joint"}
}

COLORS = {"AION": "#1f77b4", "AstroPT": "#ff7f0e", "AstroCLIP": "#2ca02c"}

def load_data(path: Path) -> pd.DataFrame:
    """Load anomaly scores from CSV."""
    df = pd.read_csv(path)
    # Ensure object_id is string
    df['object_id'] = df['object_id'].astype(str)
    return df

def find_score_file(model_name: str, base_dir: Path) -> Optional[Path]:
    """Auto-detect score file for a model."""
    # Pattern: anomaly_scores_{model}.csv or *{model}*.csv
    candidates = list(base_dir.glob(f"anomaly_scores_{model_name.lower()}.csv"))
    if not candidates:
        candidates = list(base_dir.glob(f"*{model_name.lower()}*.csv"))
    
    return candidates[0] if candidates else None

def plot_panel_simple(ax, ranks_src, ranks_tgt, n_total, color, label=None):
    """Plot single histogram panel (for Multi-Model)."""
    top_1_threshold = n_total * 0.01
    subset_mask = ranks_src <= top_1_threshold
    
    if subset_mask.sum() == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center')
        return

    ranks_pct = (ranks_tgt[subset_mask] / n_total) * 100
    
    ax.hist(ranks_pct, bins=40, range=(0, 100), color=color, edgecolor='black', alpha=0.7, linewidth=0.5, zorder=3, label=label)
    
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1, zorder=4)
    ax.axvline(x=10, color='orange', linestyle='--', linewidth=1, zorder=4)
    ax.axvspan(0, 1, color='red', alpha=0.1, zorder=1)
    ax.axvspan(1, 10, color='orange', alpha=0.1, zorder=1)
    
    retained_1 = (ranks_tgt[subset_mask] <= top_1_threshold).sum() / subset_mask.sum() * 100
    retained_10 = (ranks_tgt[subset_mask] <= (n_total * 0.1)).sum() / subset_mask.sum() * 100
    
    stats_text = (
        f"\\textbf{{Retained}}:\n"
        f"Top 1\\%: {retained_1:.1f}\\%\n"
        f"Top 10\\%: {retained_10:.1f}\\%"
    )
    ax.text(0.95, 0.92, stats_text, transform=ax.transAxes, ha='right', va='top', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'), zorder=5, fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

def run_multi_model(dfs: Dict[str, pd.DataFrame], output_dir: Path):
    """3x3 Grid: Rows=Modality, Cols=Model Pair comparisons."""
    print("Running Multi-Model Analysis...")
    embedding_types = ["Images", "Spectra", "Joint"]
    comparisons = [("AION", "AstroPT"), ("AstroPT", "AstroCLIP"), ("AstroCLIP", "AION")]
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharex=True, sharey='row')
    
    for row_idx, emb_type in enumerate(embedding_types):
        # Extract subsets
        subsets = {}
        for m, df in dfs.items():
            key = KEYS[m].get(emb_type)
            if key:
                subsets[m] = df[df['embedding_key'] == key].copy()
                
        # Merge
        if not all(m in subsets for m in ["AION", "AstroPT", "AstroCLIP"]):
            print(f"Skipping {emb_type} (missing data)")
            continue
            
        merged = subsets["AION"][['object_id', 'rank']].rename(columns={'rank': 'rank_aion'})
        merged = merged.merge(subsets["AstroPT"][['object_id', 'rank']].rename(columns={'rank': 'rank_astropt'}), on='object_id')
        merged = merged.merge(subsets["AstroCLIP"][['object_id', 'rank']].rename(columns={'rank': 'rank_astroclip'}), on='object_id')
        
        n_total = len(merged)
        
        for col_idx, (src, tgt) in enumerate(comparisons):
            ax = axes[row_idx, col_idx]
            plot_panel_simple(
                ax, 
                merged[f"rank_{src.lower()}"].values, 
                merged[f"rank_{tgt.lower()}"].values, 
                n_total, 
                COLORS[src]
            )
            ax.set_title(rf"\textbf{{{src}}} $\rightarrow$ \textbf{{{tgt}}}", pad=5)
            
            if col_idx == 0: ax.set_ylabel(rf"\textbf{{{emb_type}}}" + "\nCount")
            if row_idx == 2: ax.set_xlabel("Percentile Rank in Target")

    plt.tight_layout()
    fig.savefig(output_dir / "displacement_multi_model.png", dpi=300)
    print(f"Saved {output_dir / 'displacement_multi_model.png'}")
    plt.close(fig)

def run_cross_modality(dfs: Dict[str, pd.DataFrame], output_dir: Path):
    """3x3 Grid: Rows=Source Mod, Cols=Target Mod."""
    print("Running Cross-Modality Analysis...")
    modalities = ["Images", "Spectra", "Joint"]
    
    # Merge all into one big DF
    big_merged = None
    for m_name, df in dfs.items():
        m_sub = None
        for mod in modalities:
            key = KEYS[m_name].get(mod)
            if not key: continue
            
            d = df[df['embedding_key'] == key][['object_id', 'rank']].rename(columns={'rank': f'rank_{m_name}_{mod}'})
            m_sub = d if m_sub is None else m_sub.merge(d, on='object_id')
        
        big_merged = m_sub if big_merged is None else big_merged.merge(m_sub, on='object_id')

    if big_merged is None: return
    n_total = len(big_merged)
    
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey='row')
    
    top_1_threshold = n_total * 0.01
    
    for r, src_mod in enumerate(modalities):
        for c, tgt_mod in enumerate(modalities):
            ax = axes[r, c]
            stats_lines = []
            
            for m_name in ["AION", "AstroPT", "AstroCLIP"]:
                src_col = f'rank_{m_name}_{src_mod}'
                tgt_col = f'rank_{m_name}_{tgt_mod}'
                if src_col not in big_merged.columns or tgt_col not in big_merged.columns: continue
                
                ranks_src = big_merged[src_col].values
                ranks_tgt = big_merged[tgt_col].values
                
                subset_mask = ranks_src <= top_1_threshold
                if subset_mask.sum() == 0: continue
                
                ranks_pct = (ranks_tgt[subset_mask] / n_total) * 100
                color = COLORS[m_name]
                
                ax.hist(ranks_pct, bins=40, range=(0, 100), color=color, histtype='step', alpha=0.9, linewidth=1.2, zorder=3, label=m_name)
                ax.hist(ranks_pct, bins=40, range=(0, 100), color=color, alpha=0.1, zorder=2)
                
                retained = (ranks_tgt[subset_mask] <= top_1_threshold).sum() / subset_mask.sum() * 100
                stats_lines.append(rf"{m_name}: {retained:.1f}\%")
                
            ax.axvline(x=1, color='red', linestyle='--', linewidth=0.8, alpha=0.5, zorder=4)
            
            stats_txt = "\\textbf{Retained Top 1\\%}:\n" + "\n".join(stats_lines)
            ax.text(0.95, 0.92, stats_txt, transform=ax.transAxes, ha='right', va='top', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'), zorder=5, fontsize=6)
            
            ax.set_title(rf"\textbf{{{src_mod}}} $\rightarrow$ \textbf{{{tgt_mod}}}", pad=5)
            if c == 0: ax.set_ylabel(rf"\textbf{{{src_mod}}}" + "\nCount")
            if r == 2: ax.set_xlabel(rf"Percentile Rank in \textbf{{{tgt_mod}}}")
            if r == 0 and c == 0: ax.legend(loc='lower right', fontsize=6)
            ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_dir / "displacement_cross_modality.png", dpi=300)
    print(f"Saved {output_dir / 'displacement_cross_modality.png'}")
    plt.close(fig)

def run_extensive(dfs: Dict[str, pd.DataFrame], output_dir: Path):
    """9x9 Grid."""
    print("Running Extensive Analysis...")
    systems = []
    modalities = ["Images", "Spectra", "Joint"]
    for m in ["AION", "AstroPT", "AstroCLIP"]:
        for mod in modalities:
            systems.append((m, mod))
            
    # Merge
    big_merged = None
    for m, mod in systems:
        key = KEYS[m].get(mod)
        if not key: continue
        d = dfs[m][dfs[m]['embedding_key'] == key][['object_id', 'rank']].rename(columns={'rank': f'rank_{m}_{mod}'})
        big_merged = d if big_merged is None else big_merged.merge(d, on='object_id')
        
    if big_merged is None: return
    n_total = len(big_merged)
    top_1_thr = n_total * 0.01
    
    # Heatmap
    ret_matrix = np.zeros((9, 9))
    for i, (src_m, src_mod) in enumerate(systems):
        src_col = f'rank_{src_m}_{src_mod}'
        mask = big_merged[src_col] <= top_1_thr
        for j, (tgt_m, tgt_mod) in enumerate(systems):
            tgt_col = f'rank_{tgt_m}_{tgt_mod}'
            retained = (big_merged.loc[mask, tgt_col] <= top_1_thr).sum() / mask.sum() * 100
            ret_matrix[i, j] = retained
            
    plt.figure(figsize=(10, 8))
    labels = [f"{m}-{mo}" for m, mo in systems]
    sns.heatmap(ret_matrix, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels, cmap="YlGnBu")
    plt.title("Extensive Retention Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "displacement_extensive_heatmap.png", dpi=300)
    plt.close()
    
    # Grid
    # (Skipping full 9x9 grid code for brevity unless requested, focusing on heatmap which is dense info)
    # The user asked for "everything", so I should include it.
    
    fig, axes = plt.subplots(9, 9, figsize=(18, 18), sharex=True, sharey='row')
    for i, (src_m, src_mod) in enumerate(systems):
        src_col = f'rank_{src_m}_{src_mod}'
        mask = big_merged[src_col] <= top_1_thr
        color = COLORS[src_m]
        
        for j, (tgt_m, tgt_mod) in enumerate(systems):
            tgt_col = f'rank_{tgt_m}_{tgt_mod}'
            ranks_pct = (big_merged.loc[mask, tgt_col] / n_total) * 100
            
            ax = axes[i, j]
            ax.hist(ranks_pct, bins=30, range=(0, 100), color=color, alpha=0.7, edgecolor='none')
            ax.axvline(x=1, color='red', linestyle='--', linewidth=0.5)
            ax.text(0.9, 0.9, f"{ret_matrix[i, j]:.0f}%", transform=ax.transAxes, ha='right', fontsize=6)
            
            if i == 0: ax.set_title(f"{tgt_m}\n{tgt_mod}", fontsize=7)
            if j == 0: ax.set_ylabel(f"{src_m}\n{src_mod}", fontsize=7)
            if i == 8: ax.set_xlabel("%", fontsize=7)
            ax.grid(alpha=0.2)
            
    plt.tight_layout()
    fig.savefig(output_dir / "displacement_extensive_grid.png", dpi=200)
    plt.close(fig)
    print(f"Saved extensive plots to {output_dir}")

def run_analysis(config_path: Optional[Path] = None, output_dir: Optional[Path] = None):
    paths = load_paths()
    if not config_path:
        config_path = paths.repo_root / "src/fmb/configs/analysis/displacement.yaml"
        
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    out_dir_path = Path(output_dir) if output_dir else paths.repo_root / cfg.get("output_dir", "runs/analysis/displacement")
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load Scores
    dfs = {}
    for m in cfg.get("models", ["AION", "AstroPT", "AstroCLIP"]):
        # Check config for overrides
        p = cfg.get(f"{m.lower()}_scores")
        if p:
            p = Path(p)
        else:
            # Auto-detect in outliers path
            p = find_score_file(m, paths.outliers)
            
        if p and p.exists():
            print(f"Loading {m} scores from {p}...")
            dfs[m] = load_data(p)
        else:
            print(f"Warning: Scores for {m} not found.")
            
    if len(dfs) < 2:
        print("Need at least 2 models for displacement analysis.")
        return

    ptype = cfg.get("plot_type", "all")
    
    if ptype in ["all", "multi_model"]:
        run_multi_model(dfs, out_dir_path)
        
    if ptype in ["all", "cross_modality"]:
        run_cross_modality(dfs, out_dir_path)
        
    if ptype in ["all", "extensive"]:
        run_extensive(dfs, out_dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file")
    args = parser.parse_args()
    
    run_analysis(Path(args.config) if args.config else None)
