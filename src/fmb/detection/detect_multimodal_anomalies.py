#!/usr/bin/env python3
"""
detect_multimodal_anomalies.py (rewritten)

Goal
----
Compare unsupervised anomaly lists derived from:
  (1) mismatch-only  : low cosine similarity between image/spectrum embeddings
  (2) density-only   : low-density outliers in each modality (NF on image emb, NF on spectrum emb)
  (3) multimodal     : combined mismatch + per-modality rarity (several fusion scores)

NO MANUAL ANNOTATIONS:
We report proxy rates (heuristics) for:
  - matching problems        ~ high mismatch percentile
  - instrumental/artifact    ~ outlier in exactly one modality
  - astrophysical candidates ~ outlier in both modalities

Inputs
------
- Cosine CSV: object_id, cosine_similarity, rank
  (rank=1 means *most* mismatched = lowest cosine, as produced by detect_cosine_anomalies.py)
- NF CSV: object_id, embedding_key, anomaly_sigma, rank
  (rank=1 means most anomalous = highest anomaly_sigma)

Outputs
-------
- output_all: merged table with percentiles + multimodal scores
- output_dir: saves top-K lists for mismatch-only / density-only / multimodal (+ pareto)
- summary_csv: strategy comparison (proxy rates + overlaps)

Usage example
-------------
python detect_multimodal_anomalies.py \
  --cosine-csv outputs/cosine_aion.csv \
  --nf-csv outputs/nf_scores_aion.csv \
  --output-all outputs/mm_all.csv \
  --output-dir outputs/mm_lists \
  --top-k 200 \
  --mode multimodal \
  --fusion geo \
  --t-img 0.99 --t-spec 0.99 --t-mis 0.99

Notes on definitions
--------------------
We convert ranks into percentiles p in [0,1]:
  p = (N - rank + 1) / N
so p=1 is most anomalous / most mismatched.

- mismatch anomaly: high p_mis (low cosine)
- density anomaly (image): high p_img (NF image outlier)
- density anomaly (spec):  high p_spec (NF spec outlier)
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# -----------------------------
# IO helpers
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect and compare multimodal anomalies (no labels).")

    p.add_argument("--aion-embeddings", type=str, required=True, help="Path to AION embeddings (.pt)")
    p.add_argument("--astropt-embeddings", type=str, required=True, help="Path to AstroPT embeddings (.pt)")
    p.add_argument("--astroclip-embeddings", type=str, required=True, help="Path to AstroCLIP embeddings (.pt)")

    p.add_argument("--nf-csv-aion", type=str, required=True, help="Path to AION NF scores (CSV)")
    p.add_argument("--nf-csv-astropt", type=str, required=True, help="Path to AstroPT NF scores (CSV)")
    p.add_argument("--nf-csv-astroclip", type=str, required=True, help="Path to AstroCLIP NF scores (CSV)")

    p.add_argument("--output-all", type=str, required=True,
                   help="Path to merged CSV (all objects after join).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to write top-K lists + pareto set (optional).")
    p.add_argument("--output-filtered", type=str, default=None,
                   help="Path to save the final selected (filtered) list.")
    p.add_argument("--output-ids", type=str, default=None,
                   help="Path to save the IDs of the selected list.")
    p.add_argument("--summary-csv", type=str, default=None,
                   help="Optional path for strategy comparison summary (CSV).")

    # --- selection mode ---
    p.add_argument("--mode", choices=["mismatch-only", "density-only", "multimodal", "pareto"],
                   default="multimodal",
                   help="Which strategy to output as 'selected' list.")

    # --- fusion score for multimodal ---
    p.add_argument("--fusion", choices=["min", "geo", "avg", "rank_product"],
                   default="geo",
                   help="Fusion score for multimodal ranking.")

    # --- thresholds (percentiles) for filtering ---
    p.add_argument("--t-img", type=float, default=0.0, help="Percentile threshold for image density outliers.")
    p.add_argument("--t-spec", type=float, default=0.0, help="Percentile threshold for spec density outliers.")
    p.add_argument("--t-mis", type=float, default=0.0, help="Percentile threshold for mismatch.")

    # --- list sizes ---
    p.add_argument("--top-k", type=int, default=200, help="Top-K to export per strategy.")
    p.add_argument("--join", choices=["inner", "left"], default="inner",
                   help="How to join cosine and NF tables. inner recommended.")

    # --- pareto ---
    p.add_argument("--pareto-top-k", type=int, default=None,
                   help="If set, keep only top-K from Pareto set.")

    return p.parse_args()


def load_embeddings_and_compute_cosine_rank(path: str, model_name: str) -> pd.DataFrame:
    """
    Load .pt embedding file (list of dicts), compute image-spectrum cosine similarity, and rank.
    Returns DF: object_id, model, cosine_similarity, rank_cosine
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    
    # Load data
    # Assuming list of dicts based on inspection
    print(f"Loading {model_name} embeddings from {path} ...")
    data = torch.load(path, map_location="cpu")
    
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"File {path} does not contain a non-empty list.")
    
    # Detect keys
    # Keys might vary.
    # AION: embedding_hsc, embedding_spectrum
    # AstroPT/Clip: embedding_images, embedding_spectra (plural)
    first = data[0]
    keys = set(first.keys())
    
    img_key = None
    spec_key = None
    
    # Heuristics for Image
    if "embedding_images" in keys: img_key = "embedding_images"
    elif "embedding_hsc" in keys: img_key = "embedding_hsc"
    
    # Heuristics for Spec
    if "embedding_spectra" in keys: spec_key = "embedding_spectra"
    elif "embedding_spectrum" in keys: spec_key = "embedding_spectrum"
    
    if not img_key or not spec_key:
        raise ValueError(f"Could not identify img/spec keys in {model_name}. Keys found: {keys}")
    
    # Convert list of dicts to lists
    object_ids = []
    emb_img_list = []
    emb_spec_list = []
    
    for item in data:
        object_ids.append(str(item["object_id"]))
        # Items are tensors or lists. Convert to tensor.
        emb_img_list.append(item[img_key])
        emb_spec_list.append(item[spec_key])
        
    # Stack tensors
    # They should be 1D tensors of same dim.
    # Note: data[i][img_key] might be shape (D,).
    t_img = torch.stack(emb_img_list)
    t_spec = torch.stack(emb_spec_list)
    
    # Normalize? Cosine similarity does normalization internally if we use F.cosine_similarity.
    # But usually better to normalize explicitly if we do dot product or debug.
    # F.cosine_similarity(dim=1)
    
    print(f"Computing cosine for {len(object_ids)} objects ({model_name})...")
    cos_sim = F.cosine_similarity(t_img, t_spec, dim=1).numpy()
    
    # Build DataFrame
    df = pd.DataFrame({
        "object_id": object_ids,
        "cosine_similarity": cos_sim,
        "model": model_name
    })
    
    # Rank: 1 = lowest cosine (most mismatched)
    # Ascending sort
    df["rank_cosine"] = df["cosine_similarity"].rank(ascending=True, method="min")
    
    return df


def normalize_modality(key: str) -> Optional[str]:
    """Map arbitrary embedding keys to 'img' or 'spec'."""
    k = key.lower()
    if "hsc" in k or "image" in k:
        return "img"
    if "spec" in k:
        return "spec"
    return None


def load_nf_csv_and_pivot(path: str, model_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"NF CSV not found: {path}")
    df = pd.read_csv(path)
    req = {"object_id", "embedding_key", "anomaly_sigma", "rank"}
    if not req.issubset(df.columns):
        raise ValueError(f"NF CSV missing columns. Found={list(df.columns)} Required={sorted(req)}")

    df = df.copy()
    df["object_id"] = df["object_id"].astype(str)
    
    # Normalize modality
    df["modality"] = df["embedding_key"].apply(normalize_modality)
    # Drop unknown modalities
    df = df.dropna(subset=["modality"])

    # Deduplicate: keep min rank (most anomalous) per (object, modality)
    df = df.sort_values("rank", ascending=True)
    df = df.drop_duplicates(subset=["object_id", "modality"], keep="first")

    # Pivot
    # from long: object_id, modality, anomaly_sigma, rank
    # to wide: object_id, nf_img_sigma, nf_spec_sigma, rank_img, rank_spec
    pivot_sigma = df.pivot(index="object_id", columns="modality", values="anomaly_sigma").add_prefix("nf_").add_suffix("_sigma")
    pivot_rank = df.pivot(index="object_id", columns="modality", values="rank").add_prefix("rank_")
    
    wide = pivot_sigma.join(pivot_rank).reset_index()
    
    # Add model name
    wide["model"] = model_name
    return wide


# -----------------------------
# Core math
# -----------------------------

def rank_to_percentile(rank: pd.Series, N: int) -> pd.Series:
    # rank=1 -> 1.0 (most anomalous); rank=N -> 1/N (least anomalous)
    return (N - rank + 1.0) / float(N)


def safe_N(series: pd.Series, fallback: int) -> int:
    mx = series.max()
    if pd.isna(mx):
        return fallback
    return int(max(mx, fallback))


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentiles + multimodal fusion scores + stability metrics."""
    df = df.copy()

    N = len(df)
    N_cos = safe_N(df["rank_cosine"], N)
    N_img = safe_N(df["rank_img"], N)
    N_spec = safe_N(df["rank_spec"], N)

    df["p_mis"] = rank_to_percentile(df["rank_cosine"], N_cos)
    df["p_img"] = rank_to_percentile(df["rank_img"], N_img)
    df["p_spec"] = rank_to_percentile(df["rank_spec"], N_spec)

    # Fusion scores
    df["score_mm_min"] = df["p_mis"] * np.minimum(df["p_img"], df["p_spec"])
    df["score_mm_geo"] = df["p_mis"] * np.sqrt(df["p_img"] * df["p_spec"])
    df["score_mm_avg"] = df["p_mis"] * (df["p_img"] + df["p_spec"]) / 2.0
    df["rank_product"] = df["rank_cosine"] * df["rank_img"] * df["rank_spec"]

    # --- Improvement 2: Agreement-weighted score ---
    # Consensus: geometric mean of the three percentiles
    # a_i = (p_mis * p_img * p_spec)^(1/3)
    df["consensus_geo"] = np.cbrt(df["p_mis"] * df["p_img"] * df["p_spec"])
    
    # Robust score: score_mm_geo * consensus
    # This penalizes objects that are high in multimodal score but low in consensus (i.e. high variance)
    df["score_mm_robust"] = df["score_mm_geo"] * df["consensus_geo"]

    # --- Improvement 3: Multimodal Uplift ---
    # We need a percentile for the joint score to compare with unimodal percentiles.
    # Let's derive it from the main multimodal score (geo).
    # Rank desc because higher score = more anomalous
    df["rank_mm_geo"] = df["score_mm_geo"].rank(ascending=False, method="min")
    N_mm = len(df) # or safe_N? safe to assume N here
    df["p_joint"] = rank_to_percentile(df["rank_mm_geo"], N_mm)

    # Uplift: difference between joint percentile and best single-density percentile
    # "Multimodal helps if p_joint > max(p_img,    # Uplift delta
    df["uplift_mm"] = df["p_joint"] - np.maximum(df["p_img"], df["p_spec"])

    # --- Cross-Model Consensus ---
    # User Request: Consensus "entre modèles". 
    # a_i = geometric mean of percentiles from different models.
    # We use p_joint (percentile of the multimodal score) as the representative 'p' for the model.
    # If p not available (e.g. object missing in one model), ignore or penalize?
    # Group by object_id
    
    if "model" in df.columns and df["model"].nunique() > 1:
        # Geometric mean of p_joint across models for each object
        # log-mean-exp
        # p_joint is in [0, 1]. Avoid log(0)
        eps = 1e-9
        
        def gmean_pjoint(g):
            p = g["p_joint"]
            return np.exp(np.log(p + eps).mean())
        
        # This gives one value per object
        cons_series = df.groupby("object_id")["p_joint"].transform(lambda x: np.exp(np.log(x + eps).mean()))
        df["consensus_cross_model"] = cons_series
        
        # Robust score 2.0: MM score * Cross-Model Consensus
        # This penalizes objects that are anomalous in AION but normal in AstroPT
        df["score_mm_robust_cross_model"] = df["score_mm_geo"] * df["consensus_cross_model"]
    else:
        # Fallback for single model: consensus is 1.0 (or just self)
        df["consensus_cross_model"] = 1.0
        df["score_mm_robust_cross_model"] = df["score_mm_robust"] # fallback to cross-view robust

    return df


def compute_robustness(
    df_selected: pd.DataFrame, 
    df_full: pd.DataFrame, 
    targets: List[str], 
    q: float
) -> float:
    """
    Robustness@q: Fraction of objects in `df_selected` (the top-K list)
    that appear in the `top-q` percentile of the `targets` rankings.
    
    targets: list of rank columns in df_full (e.g. ['rank_cosine', 'rank_img', 'rank_spec'])
    q: percentile threshold (e.g. 0.01 for top 1%)
    """
    if df_selected.empty:
        return 0.0
    
    # IDs in the selected list
    selected_ids = set(df_selected["object_id"])
    n_sel = len(selected_ids)
    
    # Get the rows for these IDs from full df to check their ranks in other cols
    # (Assuming df_selected has necessary columns, but safe to look up in df_full)
    sub = df_full[df_full["object_id"].isin(selected_ids)]
    
    # Target columns corresponding to ranks:
    # rank_cosine -> p_mis
    # rank_img -> p_img
    # rank_spec -> p_spec
    
    target_map = {
        "rank_cosine": "p_mis",
        "rank_img": "p_img",
        "rank_spec": "p_spec"
    }
    
    total_score = 0.0
    # Denominator: total number of (object, model) checks performed per target column.
    # This is simply len(sub).
    denominator = len(sub)
    if denominator == 0: 
        return 0.0
    
    for rank_col in targets:
        p_col = target_map.get(rank_col)
        if not p_col:
            continue
            
        # Threshold: p >= 1 - q
        # e.g. q=0.01 -> p >= 0.99
        t = 1.0 - q
        
        # Count rows (object-model pairs) that satisfy the condition
        n_passed = (sub[p_col] >= t).sum()
        
        # Fraction
        total_score += (n_passed / denominator)
        
    return total_score / len(targets)


# -----------------------------
# Pareto front
# -----------------------------

def pareto_front(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Return Pareto-optimal rows for maximizing cols (p_mis, p_img, p_spec).
    O(N^2) but fine for ~40k; still OK in pandas/numpy with vectorization tricks.
    """
    X = df[cols].to_numpy(dtype=np.float32)
    n = X.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    # Simple O(N^2) domination check
    for i in range(n):
        if is_dominated[i]:
            continue
        # A point i is dominated if there exists j with X[j]>=X[i] all dims and > in at least one dim
        ge = (X >= X[i]).all(axis=1)
        gt = (X > X[i]).any(axis=1)
        dom = ge & gt
        dom[i] = False
        if dom.any():
            is_dominated[i] = True

    return df.loc[~is_dominated].copy()


# -----------------------------
# Strategy lists
# -----------------------------

def topk_mismatch(df: pd.DataFrame, k: int) -> pd.DataFrame:
    # lowest cosine = most mismatch; your rank_cosine already reflects that
    return df.sort_values("rank_cosine", ascending=True).head(k).copy()


def topk_density(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Density-only list: we want things that are rare in at least one modality.
    Two useful variants exist; here we export BOTH:
      - image-only ranking (p_img)
      - spec-only ranking (p_spec)
      - union ranking by max(p_img, p_spec)
    """
    out = df.copy()
    out["p_any"] = np.maximum(out["p_img"], out["p_spec"])
    return out.sort_values("p_any", ascending=False).head(k).copy()


def topk_multimodal(df: pd.DataFrame, k: int, fusion: str) -> pd.DataFrame:
    out = df.copy()

    if fusion == "min":
        key, asc = "score_mm_min", False
    elif fusion == "geo":
        key, asc = "score_mm_geo", False
    elif fusion == "avg":
        key, asc = "score_mm_avg", False
    elif fusion == "rank_product":
        key, asc = "rank_product", True
    else:
        raise ValueError(f"Unknown fusion: {fusion}")

    out = out.sort_values(key, ascending=asc).head(k).copy()
    out["score_selected"] = out[key]
    return out


def apply_3way_filter(df: pd.DataFrame, t_img: float, t_spec: float, t_mis: float) -> pd.Series:
    """
    Clean definition you asked for:
      (p_img >= t_img) AND (p_spec >= t_spec) AND (p_mis >= t_mis)
    Thresholds of 0 disable the corresponding constraint.
    """
    m = pd.Series(True, index=df.index)
    if t_img > 0:
        m &= (df["p_img"] >= t_img)
    if t_spec > 0:
        m &= (df["p_spec"] >= t_spec)
    if t_mis > 0:
        m &= (df["p_mis"] >= t_mis)
    return m


# -----------------------------
# “No-label” proxy evaluation
# -----------------------------

def proxy_rates(df_sub: pd.DataFrame, t_img_eval: float, t_spec_eval: float, t_mis_eval: float) -> Dict[str, float]:
    """
    Proxy categories, ML-first and fully reproducible:
      - matching_problem: p_mis >= t_mis_eval
      - artifact_like:
          (p_img >= t_img_eval and p_spec < t_spec_eval) OR
          (p_spec >= t_spec_eval and p_img < t_img_eval)
      - astrophysical_candidate: (p_img >= t_img_eval) AND (p_spec >= t_spec_eval)

    Defaults recommended: eval thresholds around 0.99 (top 1%) or 0.95 (top 5%).
    """
    if df_sub.empty:
        return dict(
            n=0,
            rate_matching_problem=np.nan,
            rate_artifact_like=np.nan,
            rate_astrophysical_candidate=np.nan,
            median_cosine=np.nan,
            median_p_mis=np.nan,
            median_p_img=np.nan,
            median_p_spec=np.nan,
        )

    p_mis = df_sub["p_mis"].to_numpy()
    p_img = df_sub["p_img"].to_numpy()
    p_spec = df_sub["p_spec"].to_numpy()

    matching = (p_mis >= t_mis_eval)
    artifact = ((p_img >= t_img_eval) & (p_spec < t_spec_eval)) | ((p_spec >= t_spec_eval) & (p_img < t_img_eval))
    astro = (p_img >= t_img_eval) & (p_spec >= t_spec_eval)

    return dict(
        n=int(len(df_sub)),
        rate_matching_problem=float(matching.mean()),
        rate_artifact_like=float(artifact.mean()),
        rate_astrophysical_candidate=float(astro.mean()),
        median_cosine=float(np.median(df_sub["cosine_similarity"].to_numpy())),
        median_p_mis=float(np.median(p_mis)),
        median_p_img=float(np.median(p_img)),
        median_p_spec=float(np.median(p_spec)),
    )


def overlap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    if a.empty or b.empty:
        return 0.0
    sa = set(a["object_id"].astype(str).tolist())
    sb = set(b["object_id"].astype(str).tolist())
    return len(sa & sb) / float(min(len(sa), len(sb)))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    
    # Prepare output dir if needed
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path("outputs_anomaly_detection")
        if not args.output_filtered and not args.output_ids:
             out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs configuration
    # (name, emb_path, nf_path)
    inputs = [
        ("aion", args.aion_embeddings, args.nf_csv_aion),
        ("astropt", args.astropt_embeddings, args.nf_csv_astropt),
        ("astroclip", args.astroclip_embeddings, args.nf_csv_astroclip)
    ]
    
    list_df_cos = []
    list_df_nf = []
    
    print("\n=== Loading Data ===")
    for model_name, emb_path, nf_path in inputs:
        print(f"Processing model: {model_name}")
        
        # 1. Embeddings -> Cosine DF
        df_c = load_embeddings_and_compute_cosine_rank(emb_path, model_name)
        list_df_cos.append(df_c)
        
        # 2. NF CSV -> Pivot DF
        df_n = load_nf_csv_and_pivot(nf_path, model_name)
        list_df_nf.append(df_n)
        
    # Concatenate
    df_cos = pd.concat(list_df_cos, ignore_index=True)
    df_nf = pd.concat(list_df_nf, ignore_index=True)
    
    # Join on [object_id, model]
    print(f"\nMerging... Cosine rows={len(df_cos)}, NF rows={len(df_nf)}")
    df = pd.merge(df_cos, df_nf, on=["object_id", "model"], how=args.join)
    
    if len(df) == 0:
        raise RuntimeError(f"Join produced 0 rows. Check IDs overlap between embeddings and NF scores.")

    # Fill missing values if Left Join used
    if args.join == "left":
        # Fill ranks with N (per model? or just max)
        # For simplicity, filling with max+1 or handling via safe_N
        # (compute_scores handles N per group, so we assume non-nan but if nan it might break)
        pass 

    print(f"Total objects after merge: {len(df)}")
    
    # Compute scores
    print("Computing metrics...")
    df = compute_scores(df)

    # Save merged "all"
    Path(args.output_all).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_all, index=False)
    print(f"Saved merged table to {args.output_all}")

    # Build lists
    # Note: top-k is usually per-model.
    # get_topk helper handles grouping by model if exists.
    
    def get_topk(d, key, asc=True, k=200):
        if "model" in d.columns:
            return d.groupby("model", group_keys=False).apply(lambda x: x.sort_values(key, ascending=asc).head(k))
        else:
            return d.sort_values(key, ascending=asc).head(k)

    K = int(args.top_k)

    df_mis = get_topk(df, "rank_cosine", True, K)
    
    # Density: max(p_img, p_spec) -> p_any. Sort desc.
    df["p_any"] = np.maximum(df["p_img"], df["p_spec"])
    df_den = get_topk(df, "p_any", False, K)

    # Multimodal
    fusion_key_map = {
        "min": ("score_mm_min", False),
        "geo": ("score_mm_geo", False),
        "avg": ("score_mm_avg", False),
        "rank_product": ("rank_product", True)
    }
    mm_key, mm_asc = fusion_key_map[args.fusion]
    df_mm = get_topk(df, mm_key, mm_asc, K)
    df_mm["score_selected"] = df_mm[mm_key]

    # Filtered
    filt = apply_3way_filter(df, args.t_img, args.t_spec, args.t_mis)
    df_mm_filtered_base = df.loc[filt].copy()
    if len(df_mm_filtered_base) > 0:
        df_mm_filtered = get_topk(df_mm_filtered_base, mm_key, mm_asc, K)
    else:
        df_mm_filtered = df_mm_filtered_base
    if not df_mm_filtered.empty:
        df_mm_filtered["score_selected"] = df_mm_filtered[mm_key]
    else:
        # Avoid column error if empty
        pass

    # Pareto set
    df_par = pareto_front(df, cols=["p_mis", "p_img", "p_spec"])
    if args.pareto_top_k is not None and len(df_par) > 0:
        df_par = topk_multimodal(df_par, int(args.pareto_top_k), fusion="geo")

    # Determine "Selected" list based on mode
    if args.mode == "mismatch-only":
        selected = df_mis
        tag = "selected_mismatch_only"
    elif args.mode == "density-only":
        selected = df_den
        tag = "selected_density_only_any"
    elif args.mode == "multimodal":
        use_filtered = (args.t_img > 0) or (args.t_spec > 0) or (args.t_mis > 0)
        selected = df_mm_filtered if use_filtered else df_mm
        tag = f"selected_multimodal_{args.fusion}" + ("_filtered3way" if use_filtered else "")
    elif args.mode == "pareto":
        selected = df_par
        tag = "selected_pareto"
    else:
        selected = df_mm
    
    # Outputs requested by user
    if args.output_filtered:
        selected.to_csv(args.output_filtered, index=False)
        print(f"Saved selected list to {args.output_filtered}")
        
    if args.output_ids:
        # Just IDs
        with open(args.output_ids, "w") as f:
            f.write("object_id\n")
            for oid in selected["object_id"]:
                f.write(f"{oid}\n")
        print(f"Saved IDs to {args.output_ids}")

    # Standard strategy summary (optional)
    if args.output_dir or args.summary_csv:
        # ... (Metrics calculation as before)
        # Use simple global thresholds or per-model? 
        # Metric functions treat everything as one distribution unless splitting.
        # But 'compute_robustness' handles comparing rank within model.
        # 'proxy_rates' is just percentiles.
        
        t_eval = 0.99
        
        def get_metrics(df_sub: pd.DataFrame, name: str) -> Dict[str, float]:
            base = proxy_rates(df_sub, t_eval, t_eval, t_eval)
            targets = ["rank_cosine", "rank_img", "rank_spec"]
            r1  = compute_robustness(df_sub, df, targets, q=0.01)
            r5  = compute_robustness(df_sub, df, targets, q=0.05)
            r10 = compute_robustness(df_sub, df, targets, q=0.10)
            up_mean = df_sub["uplift_mm"].mean() if "uplift_mm" in df_sub.columns else np.nan
            up_pos  = (df_sub["uplift_mm"] > 0).mean() if "uplift_mm" in df_sub.columns else np.nan
            return {
                **base,
                "robustness_at_1pct": float(r1),
                "robustness_at_5pct": float(r5),
                "robustness_at_10pct": float(r10),
                "uplift_mean": float(up_mean),
                "uplift_positive_rate": float(up_pos),
            }

        rates_mis = get_metrics(df_mis, "mismatch_only")
        rates_den = get_metrics(df_den, "density_only_any")
        rates_mm  = get_metrics(df_mm, f"multimodal_{args.fusion}")
        rates_mm_filt = get_metrics(df_mm_filtered, f"multimodal_{args.fusion}_filtered3way")

        ov_mis_den = overlap(df_mis, df_den)
        ov_mis_mm = overlap(df_mis, df_mm)
        ov_den_mm = overlap(df_den, df_mm)

        summary = pd.DataFrame([
            {"strategy": "mismatch_only", **rates_mis},
            {"strategy": "density_only_any", **rates_den},
            {"strategy": f"multimodal_{args.fusion}", **rates_mm},
            {"strategy": f"multimodal_{args.fusion}_filtered3way", **rates_mm_filt},
        ])
        
        summary["overlap_with_mismatch_only"] = [1.0, ov_mis_den, ov_mis_mm, overlap(df_mm_filtered, df_mis)]
        summary["overlap_with_density_only_any"] = [ov_mis_den, 1.0, ov_den_mm, overlap(df_mm_filtered, df_den)]

        summary_path = Path(args.summary_csv) if args.summary_csv else (out_dir / "strategy_summary.csv")
        summary.to_csv(summary_path, index=False)

        print("\n=== Strategy comparison (proxy) ===")
        print(summary.to_string(index=False))
        
        # Save per-model lists if output-dir is set
        if args.output_dir:
            def _save_list(dfx: pd.DataFrame, name: str) -> None:
                p = out_dir / f"top{K}_{name}.csv"
                dfx.to_csv(p, index=False)
                (out_dir / f"top{K}_{name}_ids.txt").write_text("object_id\n" + "\n".join(dfx["object_id"].astype(str)) + "\n")

            _save_list(df_mis, "mismatch_only")
            _save_list(df_den, "density_only_any")
            _save_list(df_mm, f"multimodal_{args.fusion}")
            _save_list(df_mm_filtered, f"multimodal_{args.fusion}_filtered3way")
            
    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
