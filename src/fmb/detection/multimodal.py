"""
Multimodal/Fusion Anomaly Detection.
Combines Cosine mismatches and NF density outliers to find multimodal anomalies.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from fmb.paths import load_paths

# -------------------------------------------------------------------------
# Loaders
# -------------------------------------------------------------------------

def normalize_modality(key: str) -> Optional[str]:
    k = key.lower()
    if "hsc" in k or "image" in k: return "img"
    if "spec" in k: return "spec"
    return None

def load_cosine_csv(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[warn] Cosine CSV not found: {path} (skipping model {model_name})")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Expected: object_id, cosine_similarity, rank
    df["object_id"] = df["object_id"].astype(str)
    df["model"] = model_name
    # Rename rank to rank_cosine
    df = df.rename(columns={"rank": "rank_cosine"})
    return df

def load_nf_csv(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[warn] NF CSV not found: {path} (skipping model {model_name})")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Expected: object_id, embedding_key, anomaly_sigma, rank
    df["object_id"] = df["object_id"].astype(str)
    
    # Pivot logic
    df["modality"] = df["embedding_key"].apply(normalize_modality)
    df = df.dropna(subset=["modality"])
    
    # Deduplicate keeping min rank (most anomalous)
    df = df.sort_values("rank", ascending=True).drop_duplicates(["object_id", "modality"])
    
    p_sigma = df.pivot(index="object_id", columns="modality", values="anomaly_sigma").add_prefix("nf_").add_suffix("_sigma")
    p_rank = df.pivot(index="object_id", columns="modality", values="rank").add_prefix("rank_")
    
    wide = p_sigma.join(p_rank).reset_index()
    wide["model"] = model_name
    return wide

# -------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------

def rank_to_p(rank_col: pd.Series, N: int) -> pd.Series:
    # rank 1 is top anomaly. p=1.0. rank N is p=0.0
    return (N - rank_col + 1.0) / N

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    N = len(df)
    # We want global percentiles usually, or per-model? 
    # The script used global. Let's do global for simplicity of comparison.
    # But usually ranks are relative to the sub-population.
    # Let's do groupings if we have multiple models.
    
    OUT = []
    for model, sub in df.groupby("model"):
        sub = sub.copy()
        Nm = len(sub)
        # Handle missing cols if some model misses a modality
        if "rank_cosine" in sub: sub["p_mis"] = rank_to_p(sub["rank_cosine"], Nm)
        else: sub["p_mis"] = 0.0
            
        if "rank_img" in sub: sub["p_img"] = rank_to_p(sub["rank_img"], Nm)
        else: sub["p_img"] = 0.0 # Not anomalous
            
        if "rank_spec" in sub: sub["p_spec"] = rank_to_p(sub["rank_spec"], Nm)
        else: sub["p_spec"] = 0.0

        # Fusion
        sub["score_mm_geo"] = sub["p_mis"] * np.sqrt(sub["p_img"] * sub["p_spec"])
        sub["score_mm_min"] = sub["p_mis"] * np.minimum(sub["p_img"], sub["p_spec"])
        
        OUT.append(sub)
        
    return pd.concat(OUT, ignore_index=True) if OUT else pd.DataFrame()

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(argv: List[str] = None):
    parser = argparse.ArgumentParser()
    # If not provided, we infer from paths
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--fusion", default="geo", choices=["geo", "min"])
    # Filters
    parser.add_argument("--t-img", type=float, default=0.0)
    parser.add_argument("--t-spec", type=float, default=0.0)
    parser.add_argument("--t-mis", type=float, default=0.0)
    
    args = parser.parse_args(argv)
    
    paths = load_paths()
    root = paths.outliers
    
    # 1. Load Everything
    models = ["aion", "astropt", "astroclip"]
    
    all_cos = []
    all_nf = []
    
    for m in models:
        # Cosine
        p_cos = root / f"cosine_scores_{m}.csv"
        df_c = load_cosine_csv(p_cos, m)
        if not df_c.empty: all_cos.append(df_c)
            
        # NF
        p_nf = root / f"anomaly_scores_{m}.csv"
        df_n = load_nf_csv(p_nf, m)
        if not df_n.empty: all_nf.append(df_n)
        
    if not all_cos or not all_nf:
        print("[error] Missing input CSVs (run 'detect outliers' and 'detect cosine' first).")
        return

    df_cos = pd.concat(all_cos, ignore_index=True)
    df_nf = pd.concat(all_nf, ignore_index=True)
    
    # Merge
    print(f"Merging {len(df_cos)} cosine rows with {len(df_nf)} NF rows...")
    # Inner join mainly
    df = pd.merge(df_cos, df_nf, on=["object_id", "model"], how="inner")
    print(f"Merged total: {len(df)}")
    
    if df.empty:
        print("[error] Empty merge result.")
        return

    # Compute
    df = compute_scores(df)

    # 3. Filter & Sort
    # Apply thresholds
    mask = (df["p_img"] >= args.t_img) & (df["p_spec"] >= args.t_spec) & (df["p_mis"] >= args.t_mis)
    df_filt = df[mask].copy()
    
    key = f"score_mm_{args.fusion}"
    # Sort descending (score high -> anomalous)
    df_filt = df_filt.sort_values(key, ascending=False)
    
    OUT_DIR = root / "multimodal"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput Directory: {OUT_DIR}")
    
    # Save All
    out_all = OUT_DIR / "all_scores.csv"
    df.to_csv(out_all, index=False)
    print(f"[success] Saved ALL scores ({len(df)} objects) to {out_all}")

    # --- Export Unfiltered Ranked Lists (for inspection/plotting) ---
    # The user requested "properly weighted" lists per model for plotting
    
    RANKED_DIR = OUT_DIR / "ranked"
    RANKED_DIR.mkdir(parents=True, exist_ok=True)
    
    # We sort the full dataframe by fusion score
    # And export top-K per model
    # This ignores t_img/t_spec/t_mis thresholds
    
    print(f"\nExporting ranked (unfiltered) lists to: {RANKED_DIR}")
    for model, sub in df.groupby("model"):
        # Sort desc by fusion score
        sub_sorted = sub.sort_values(key, ascending=False).head(args.top_k)
        fname = RANKED_DIR / f"top{args.top_k}_{model}_{args.fusion}.csv"
        sub_sorted.to_csv(fname, index=False)
        print(f"      Saved top-{len(sub_sorted)} '{model}' candidates to {fname}")

    # --- Filtered Lists ---
    if df_filt.empty:
        print("\n[warn] No anomalies passed the STRICT filtering thresholds!")
        print(f"       Thresholds: P_img>={args.t_img}, P_spec>={args.t_spec}, P_mis>={args.t_mis}")
        print("       (You can use the 'ranked' lists above for plotting regardless of these filters)")
        return

    # Save Filtered Top-K per model
    FILTERED_DIR = OUT_DIR / "filtered"
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    
    for model, sub in df_filt.groupby("model"):
        top = sub.head(args.top_k)
        fname = FILTERED_DIR / f"top{args.top_k}_{model}_{args.fusion}_filtered.csv"
        top.to_csv(fname, index=False)
        print(f"[success] Saved filtered top-{len(top)} '{model}' to {fname}")

if __name__ == "__main__":
    main()
