#!/usr/bin/env python3
"""
Script to predict physical parameters from AION, AstroPT, and AstroCLIP embeddings.
Models: Ridge (Linear Baseline), LightGBM (Non-linear).
Metrics: R², RMSE, PR (Participation Ratio), EPD (Effective Predictive Dimensionality), 
         PES (Predictive Efficiency Score), CWP (Coverage-Weighted Performance).
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple, Dict, List, Optional, Any


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from astropy.io import fits
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from lightgbm import LGBMRegressor
import shap

# Embedding keys
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

ASTROCLIP_EMBEDDING_KEYS = [
    "embedding_images",
    "embedding_spectra",
    "embedding_joint",
]

# --- Publication Style Settings (Matched to plot_paper_displacement.py) ---
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

COLORS = {
    "AION": "#1f77b4",     # Blue
    "AstroPT": "#ff7f0e",  # Orange
    "AstroCLIP": "#2ca02c", # Green
    "Random": "#7f7f7f"    # Gray
}

def load_embeddings(path: Path) -> List[dict]:
    """Load embedding records from a .pt file."""
    print(f"Loading embeddings from {path}...")
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unsupported embeddings format: {type(data)}")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def load_fits_catalog(path: Path, id_col_name: Optional[str] = None) -> Tuple[Dict, List[str], str]:
    """Load FITS catalog and return dict mapping object_id to row data."""
    print(f"Loading catalog from {path}...")
    with fits.open(path) as hdul:
        data = hdul[1].data
        columns = hdul[1].columns.names
        
        catalog_dict = {}
        id_column = None
        
        if id_col_name:
            if id_col_name in columns:
                id_column = id_col_name
            else:
                raise ValueError(f"Requested ID column '{id_col_name}' not found in catalog. Available: {columns}")
        else:
            # Find ID column (Auto-detect)
            for priority_col in ['TARGETID', 'targetid', 'TargetID']:
                if priority_col in columns:
                    id_column = priority_col
                    break
            if id_column is None:
                for col in columns:
                    if col.lower() in ['object_id', 'objid', 'id']:
                        id_column = col
                        break
        
        if id_column is None:
            raise ValueError(f"Could not find object ID column. Available: {columns}")
            
        print(f"Using '{id_column}' as object ID column")
        
        # Build dictionary
        # optimization: check if id is string or int, convert only if necessary for consistency
        for row in data:
            obj_id = str(row[id_column]).strip()
            catalog_dict[obj_id] = {col: row[col] for col in columns}
            
        # Identify numeric columns
        numeric_columns = []
        for col in columns:
            if col == id_column: continue
            col_format = hdul[1].columns[col].format
            if any(fmt in col_format.upper() for fmt in ['E', 'D', 'I', 'J', 'K', 'F']):
                numeric_columns.append(col)
                continue
            try:
                # heuristic check
                float(data[0][col])
                numeric_columns.append(col)
            except (ValueError, TypeError, IndexError):
                continue
                
        return catalog_dict, numeric_columns, id_column

def merge_data(
    records: List[dict],
    catalog: Dict,
    target_param: str,
    embedding_key: str,
    model_name: str,
    subset_ids: Optional[set] = None # Only keep these IDs if provided
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Merge embeddings with catalog target parameter.
    Returns X (embeddings), y (targets), ids.
    """
    rec_dict = {str(r.get("object_id", "")): r for r in records}
    
    # Filter by subset_ids if provided to enforce shared split logic
    if subset_ids is not None:
        all_ids = [res_id for res_id in rec_dict.keys() if res_id in subset_ids]
    else:
        all_ids = list(rec_dict.keys())
    
    if "" in all_ids: all_ids.remove("")
    all_ids.sort()
    
    X_list = []
    y_list = []
    valid_ids = []
    
    for obj_id in all_ids:
        rec = rec_dict[obj_id]
        emb_vec = None
        
        # Joint embedding handling for AstroPT/AstroCLIP if key missing or for consistency
        if embedding_key == "embedding_joint" and model_name in ["AstroPT", "AstroCLIP"]:
             joint_direct = rec.get("embedding_joint")
             if joint_direct is not None:
                 emb_vec = joint_direct.detach().cpu().numpy() if isinstance(joint_direct, torch.Tensor) else np.asarray(joint_direct)
             else:
                 img = rec.get("embedding_images")
                 spec = rec.get("embedding_spectra")
                 if img is not None and spec is not None:
                     img = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
                     spec = spec.detach().cpu().numpy() if isinstance(spec, torch.Tensor) else np.asarray(spec)
                     emb_vec = np.concatenate([img, spec])
        else:
            val = rec.get(embedding_key)
            if val is not None:
                emb_vec = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
        
        if emb_vec is None:
            continue
            
        if obj_id not in catalog:
            continue
        try:
            target_val = float(catalog[obj_id][target_param])
            if np.isnan(target_val) or np.isinf(target_val):
                continue
        except (ValueError, TypeError, KeyError):
            continue
            
        X_list.append(emb_vec)
        y_list.append(target_val)
        valid_ids.append(obj_id)
        
    if not X_list:
        return np.array([]), np.array([]), []
        
    return np.stack(X_list), np.array(y_list), valid_ids

def calculate_participation_ratio(shap_values_global: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Calculate Participation Ratio (PR) and PR90 from Global SHAP.
    PR = (sum(phi)^2) / (D * sum(phi^2))
    """
    phi = np.abs(shap_values_global).mean(axis=0) # (N_features,)
    
    sum_phi = np.sum(phi)
    if sum_phi == 0:
        return 0.0, 0.0, phi
        
    sum_phi_sq = np.sum(phi**2)
    if sum_phi_sq == 0:
        return 0.0, 0.0, phi
        
    D = len(phi)
    pr = (sum_phi**2) / (D * sum_phi_sq)
    
    sorted_phi = np.sort(phi)[::-1]
    cumsum_phi = np.cumsum(sorted_phi)
    threshold = 0.90 * sum_phi
    n_features_90 = np.searchsorted(cumsum_phi, threshold) + 1
    pr90 = n_features_90 / D
    
    return pr, pr90, phi

def bootstrap_pr(shap_values: np.ndarray, n_boot: int = 50) -> Tuple[float, float]:
    """Bootstrap uncertainty for PR."""
    prs = []
    # shap_values shape: (N_samples, N_features)
    # We want to bootstrap SAMPLES
    n_samples = shap_values.shape[0]
    if n_samples < 50: return 0.0, 0.0 # Not enough samples to bootstrap meaningfully
    
    for _ in range(n_boot):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        # Compute global importance for this bootstrap sample
        # Note: We recompute mean(|SHAP|) for the bootstrap sample
        sample_shap = shap_values[idx]
        pr, _, _ = calculate_participation_ratio(sample_shap)
        prs.append(pr)
    
    return np.mean(prs), np.std(prs)

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    run_shap: bool = True
) -> Dict[str, Any]:
    """
    Train Linear (Ridge) and Non-Linear (LightGBM) models.
    Calculate metrics.
    """
    results = {}
    
    # --- 1. Linear Probe (Ridge) ---
    ridge = Ridge(random_state=random_state)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    results["r2_ridge"] = r2_ridge
    
    # --- 2. Non-Linear Probe (LightGBM) ---
    lgbm = LGBMRegressor(n_jobs=-1, random_state=random_state, verbose=-1)
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    
    # Core Performance Metrics (Keep B1)
    r2_lgbm = r2_score(y_test, y_pred_lgbm)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
    mae = mean_absolute_error(y_test, y_pred_lgbm)
    
    results["r2"] = r2_lgbm
    results["rmse"] = rmse
    results["mae"] = mae
    results["y_test"] = y_test
    results["y_pred"] = y_pred_lgbm
    
    # Metric D4: Linear Decodability Gap
    results["linear_gap"] = r2_lgbm - r2_ridge

    # --- 3. SHAP Analysis (B2 & D Metrics) ---
    if run_shap:
        # Use a subset of test data for SHAP to save time if large
        explainer = shap.TreeExplainer(lgbm)
        
        # Taking a representative sample from Test set for SHAP
        # (Using test set is better for evaluation of trained model logic on unseen data, 
        # though often training data is used for interpretation. Let's use Test.)
        shap_X = X_test 
        if len(shap_X) > 2000:
            shap_X = shap_X[:2000]
            
        try:
            shap_values = explainer.shap_values(shap_X)
            
            # Implementation B2 Fix 1: Normalize SHAP values per sample
            # Avoids domination by high-variance samples
            # shape: (N_samples, N_features)
            row_sums = np.abs(shap_values).sum(axis=1, keepdims=True) + 1e-12
            shap_values_norm = shap_values / row_sums
            
            # Calculate PR on NORMALIZED values
            pr, pr90, phi = calculate_participation_ratio(shap_values_norm)
            
            # Bootstrap uncertainty
            pr_mean, pr_std = bootstrap_pr(shap_values_norm)
            
            dim = X_train.shape[1]
            eps = 1e-3
            
            results["pr"] = pr
            results["pr_std"] = pr_std
            results["pr90"] = pr90
            results["phi"] = phi # Store vector? Might be too large for CSV. Don't return in flat dict.
            
            # D Metrics
            # D1: EPD
            results["epd"] = pr * dim
            
            # D2: PES (Predictive Efficiency Score)
            # PES = R2 / (PR + eps)
            results["pes"] = r2_lgbm / (pr + eps)
            
            # D3: CWP (Coverage-Weighted Performance)
            # CWP = R2 * PR
            results["cwp"] = r2_lgbm * pr
            
        except Exception as e:
            print(f"    SHAP Failed: {e}")
            results["pr"] = np.nan
            results["pes"] = np.nan
            results["cwp"] = np.nan
            
    return results

def get_random_baseline(n_samples, dim, seed):
    """Generate random Gaussian embeddings."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, dim))

def plot_scatter_compilation(df: pd.DataFrame, output_dir: Path):
    """Compile scatter plots (True vs Pred) for all models."""

    unique_params = df['target_param'].unique()
    sns.set_context("paper", font_scale=1.5)
    
    rows_labels = ["Images", "Spectra", "Joint"]
    cols_labels = ["AION", "AstroPT", "AstroCLIP", "Random"]
    
    grid_mapping = [
        # Images
        [("AION", "embedding_hsc"), ("AstroPT", "embedding_images"), ("AstroCLIP", "embedding_images"), ("Random", "embedding_random")],
        # Spectra
        [("AION", "embedding_spectrum"), ("AstroPT", "embedding_spectra"), ("AstroCLIP", "embedding_spectra"), ("Random", "embedding_random")],
        # Joint
        [("AION", "embedding_hsc_desi"), ("AstroPT", "embedding_joint"), ("AstroCLIP", "embedding_joint"), ("Random", "embedding_random")]
    ]
    
    for param in unique_params:
        if len(df[df['target_param'] == param]) == 0: continue
        
        # Global limits
        y_all = []
        for r in range(3):
            for c in range(4):
                m, k = grid_mapping[r][c]
                row = df[(df['model_dataset'] == m) & (df['embedding_key'] == k) & (df['target_param'] == param)]
                if not row.empty:
                    y_all.extend(row.iloc[0]["y_test"])
                    y_all.extend(row.iloc[0]["y_pred"])
        
        if not y_all: continue
        y_all = np.array(y_all)
        g_min, g_max = np.min(y_all), np.max(y_all)
        span = g_max - g_min
        g_min -= 0.05 * span
        g_max += 0.05 * span
        
        fig = plt.figure(figsize=(24, 18), constrained_layout=True)
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        for r in range(3):
            for c in range(4):
                m, k = grid_mapping[r][c]
                ax = fig.add_subplot(gs[r, c])
                
                res_row = df[(df['model_dataset'] == m) & (df['embedding_key'] == k) & (df['target_param'] == param)]
                
                # Special handling for Random: It usually only runs once (maybe not for every modality?). 
                # If Random was run per-param, we check.
                # If we only have ONE random run, replicate it visually or skip rows 1,2 if duplicates.
                # For now assuming we ran Random once and tagged it generally.
                if m == "Random" and res_row.empty:
                    # Try generic random key if specific one failed
                    res_row = df[(df['model_dataset'] == "Random") & (df['target_param'] == param)]
                
                if res_row.empty:
                    ax.axis('off')
                    continue
                
                res = res_row.iloc[0]
                y_test = res["y_test"]
                y_pred = res["y_pred"]
                
                ax.scatter(y_test, y_pred, alpha=0.3, s=5, c='k', edgecolors='none')
                ax.plot([g_min, g_max], [g_min, g_max], 'r--')
                
                stats = f"R²={res['r2']:.2f}\nRMSE={res['rmse']:.2f}"
                if 'pes' in res and not np.isnan(res['pes']):
                    stats += f"\nPES={res['pes']:.1f}"
                
                ax.text(0.05, 0.95, stats, transform=ax.transAxes, va='top', bbox=dict(fc='white', alpha=0.8))
                
                ax.set_ylim(g_min, g_max)
                ax.set_xlim(g_min, g_max)
                
                if r == 0: ax.set_title(cols_labels[c], fontweight='bold')
                if c == 0: ax.set_ylabel(rows_labels[r], fontweight='bold')
                
        fig.suptitle(f"Predictions: {param}", fontsize=20)
        safe = param.replace("_", "-")
        plt.savefig(output_dir / f"scatter_{safe}.png")
        plt.close()

def plot_pareto(df: pd.DataFrame, output_dir: Path):
    """Plot R² vs PR (Pareto Frontier)."""
    if 'pr' not in df.columns: return

    # Filter out Random for main plot or treat differently
    # df_main = df[df['model_dataset'] != "Random"]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="pr", y="r2", hue="model_dataset", style="embedding_key", s=100, alpha=0.8)
    
    # Add labels for parameters? Too messy.
    plt.xlabel("Participation Ratio (PR) - Dimensional Usage")
    plt.ylabel("Performance ($R^2$)")
    plt.title("Performance vs. Efficiency (Pareto Frontier)")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "pareto_r2_pr.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aion-embeddings", required=True)
    parser.add_argument("--astropt-embeddings", required=True)
    parser.add_argument("--astroclip-embeddings", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--params", nargs="+", help="Specific parameters")
    parser.add_argument("--all-params", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-col", help="Override catalog ID column name")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load All Data
    aion = load_embeddings(Path(args.aion_embeddings))
    astropt = load_embeddings(Path(args.astropt_embeddings))
    astroclip = load_embeddings(Path(args.astroclip_embeddings))
    catalog, numeric_cols, id_col = load_fits_catalog(Path(args.catalog), id_col_name=args.id_col)
    
    if args.params:
        params = args.params
    elif args.all_params:
        params = numeric_cols
    else:
        print("No parameters specified. Use --params or --all-params.")
        return

    # Prepare Tasks
    datasets = [
        ("AION", aion, AION_EMBEDDING_KEYS),
        ("AstroPT", astropt, ASTROPT_EMBEDDING_KEYS),
        ("AstroCLIP", astroclip, ASTROCLIP_EMBEDDING_KEYS)
    ]
    
    results = []
    
    for param in params:
        # Handle aliases
        aliases = {"redshift": "Z", "mass": "LOGM", "sfr": "LOGSFR"}
        if param in aliases and aliases[param] in numeric_cols:
            param = aliases[param]
            
        print(f"\n=== Parameter: {param} ===")
        
        # SHARED SPLIT LOGIC (F1)
        # 1. Identify all valid IDs in catalog for this param
        valid_catalog_ids = [oid for oid, row in catalog.items() 
                             if isinstance(row.get(param), (int, float)) 
                             and not np.isnan(row.get(param)) 
                             and not np.isinf(row.get(param))]
        
        if len(valid_catalog_ids) < 100:
            print("Not enough valid IDs. Skipping.")
            continue
            
        # 2. Split IDs
        train_ids, test_ids = train_test_split(valid_catalog_ids, test_size=0.2, random_state=args.seed)
        train_id_set = set(train_ids)
        test_id_set = set(test_ids)
        
        print(f"  Shared Split: Train={len(train_ids)}, Test={len(test_ids)}")
        
        # Iterate Models
        for name, records, keys in datasets:
            for key in keys:
                print(f"  Processing {name} - {key}...")
                
                # Get Train Data
                X_train, y_train, _ = merge_data(records, catalog, param, key, name, subset_ids=train_id_set)
                # Get Test Data
                X_test, y_test, _ = merge_data(records, catalog, param, key, name, subset_ids=test_id_set)
                
                if len(X_train) < 50 or len(X_test) < 50:
                    print(f"    Skipping {name}-{key}: Insufficient overlap with split (Tr={len(X_train)}, Te={len(X_test)})")
                    continue
                    
                metrics = train_and_evaluate(X_train, y_train, X_test, y_test, args.seed)
                metrics.update({
                    "target_param": param,
                    "model_dataset": name,
                    "embedding_key": key
                })
                results.append(metrics)
                print(f"    R²={metrics.get('r2', -9):.3f}, PR={metrics.get('pr', 0):.3f}, PES={metrics.get('pes', 0):.1f}")

        # RANDOM BASELINE (F3)
        print("  Processing Random Baseline...")
        # Use simple Gaussian 512D (typical size)
        dim_random = 512
        X_rand_tr = get_random_baseline(len(train_ids), dim_random, args.seed)
        y_rand_tr = np.array([catalog[oid][param] for oid in train_ids]) # We need to ensure alignment? 
        # Wait, get_random_baseline generates random arrays. We need targets aligned with IDs.
        # Since train_ids is a list, we can just extract targets in that order.
        # And generate random X of matching length.
        
        X_rand_te = get_random_baseline(len(test_ids), dim_random, args.seed + 1)
        y_rand_te = np.array([catalog[oid][param] for oid in test_ids])
        
        met_rand = train_and_evaluate(X_rand_tr, y_rand_tr, X_rand_te, y_rand_te, args.seed)
        met_rand.update({
            "target_param": param,
            "model_dataset": "Random",
            "embedding_key": "embedding_random"
        })
        results.append(met_rand)
        print(f"    R²={met_rand.get('r2'):.3f} (Random)")

    if not results:
        print("No results.")
        return
        
    df = pd.DataFrame(results)
    
    # Save Raw
    # Drop large columns for CSV
    df_save = df.drop(columns=['y_test', 'y_pred', 'phi'], errors='ignore')
    df_save.to_csv(output_dir / "prediction_results.csv", index=False)
    
    # Aggregation (E2)
    # Group by Model+Key, average across Params
    agg_cols = ["r2", "pr", "epd", "pes", "cwp", "linear_gap"]
    agg = df_save.groupby(["model_dataset", "embedding_key"])[agg_cols].agg(["mean", "std"])
    agg.to_csv(output_dir / "aggregated_results.csv")
    print(f"\nSaved aggregated results to {output_dir / 'aggregated_results.csv'}")

    # Plotting
    plot_scatter_compilation(df, output_dir)
    plot_pareto(df, output_dir)
    
if __name__ == "__main__":
    main()
