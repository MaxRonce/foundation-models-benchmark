#!/usr/bin/env python3
"""
Script to predict physical parameters from foundation model embeddings.
Models: Ridge (Linear Baseline), LightGBM (Non-linear).
Metrics: R², RMSE, PR, EPD, PES, CWP.
"""
import sys
import yaml
import argparse
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
from lightgbm import LGBMRegressor
import shap
from tqdm import tqdm

from fmb.paths import load_paths, FMBPaths
from fmb.data.utils import load_embeddings_file
from fmb.viz.style import set_style

# Apply style
set_style()

def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_catalog(path: Path, id_col_override: Optional[str] = None) -> Tuple[Dict, List[str], str]:
    """Load FITS catalog."""
    print(f"Loading catalog from {path}...")
    with fits.open(path) as hdul:
        data = hdul[1].data
        columns = hdul[1].columns.names
        
        id_column = id_col_override
        if not id_column:
            # Auto-detect ID
            for cand in ['TARGETID', 'targetid', 'TargetID', 'object_id', 'objid']:
                if cand in columns:
                    id_column = cand
                    break
        
        if not id_column:
            raise ValueError(f"Could not find ID column. Available: {columns}")
            
        print(f"Using '{id_column}' as ID column.")
        
        catalog_dict = {}
        for row in data:
            oid = str(row[id_column]).strip()
            # Store all columns provided
            catalog_dict[oid] = {col: row[col] for col in columns}
            
        # Detect numeric columns
        numeric_cols = []
        for col in columns:
            if col == id_column: continue
            try:
                # Basic check on first element
                v = data[0][col]
                if isinstance(v, (int, float, np.number)):
                    numeric_cols.append(col)
            except: pass
            
        return catalog_dict, numeric_cols, id_column

def merge_data(
    records: List[dict],
    catalog: Dict,
    target_param: str,
    embedding_key: str,
    subset_ids: Optional[set] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Merge embeddings with catalog targets."""
    # Build dictionary for fast lookup
    rec_dict = {str(r.get("object_id") or r.get("targetid", "")): r for r in records}
    
    if subset_ids is not None:
        target_ids = sorted(list(subset_ids))
    else:
        target_ids = sorted(list(rec_dict.keys()))
        
    X_list = []
    y_list = []
    valid_ids = []
    
    for obj_id in target_ids:
        if obj_id not in rec_dict or obj_id not in catalog:
            continue
            
        rec = rec_dict[obj_id]
        
        # Handle Embeddings
        # Special logic for 'embedding_joint' if not present in file 
        # (Assuming extract_embedding_matrices logic or just load what's there)
        # Here we do simple lookup, computing joint on fly if needed
        emb_vec = None
        val = rec.get(embedding_key)
        
        if val is None and embedding_key == "embedding_joint":
             # Try compute
             img = rec.get("embedding_images") or rec.get("embedding_hsc")
             spec = rec.get("embedding_spectra") or rec.get("embedding_spectrum")
             if img is not None and spec is not None:
                 if not isinstance(img, np.ndarray): img = np.array(img).flatten()
                 if not isinstance(spec, np.ndarray): spec = np.array(spec).flatten()
                 emb_vec = np.concatenate([img, spec])
        elif val is not None:
             emb_vec = val
             if not isinstance(emb_vec, np.ndarray):
                 if isinstance(emb_vec, torch.Tensor):
                     emb_vec = emb_vec.detach().cpu().numpy().flatten()
                 else:
                     emb_vec = np.array(emb_vec).flatten()
                     
        if emb_vec is None: continue
        
        # Handle Target
        target_val = catalog[obj_id].get(target_param)
        try:
            target_val = float(target_val)
            if np.isnan(target_val) or np.isinf(target_val): continue
        except: continue
        
        X_list.append(emb_vec)
        y_list.append(target_val)
        valid_ids.append(obj_id)
        
    if not X_list:
        return np.array([]), np.array([]), []
        
    return np.stack(X_list), np.array(y_list), valid_ids

def calculate_pr(shap_values: np.ndarray) -> Tuple[float, float, float]:
    """Calculate Participation Ratio and PR90."""
    # Per-sample normalization (B2 fix)
    row_sums = np.abs(shap_values).sum(axis=1, keepdims=True) + 1e-12
    shap_norm = shap_values / row_sums
    
    phi = np.abs(shap_norm).mean(axis=0)
    
    sum_phi = np.sum(phi)
    sum_phi_sq = np.sum(phi**2)
    if sum_phi_sq == 0: return 0.0, 0.0, 0.0
    
    D = len(phi)
    pr = (sum_phi**2) / (D * sum_phi_sq)
    
    # PR90
    sorted_phi = np.sort(phi)[::-1]
    cumsum = np.cumsum(sorted_phi)
    thresh = 0.90 * sum_phi
    n90 = np.searchsorted(cumsum, thresh) + 1
    pr90 = n90 / D
    
    return pr, pr90, phi

def bootstrap_pr(shap_values: np.ndarray, n_boot: int = 50) -> Tuple[float, float]:
    """Bootstrap uncertainty for PR."""
    prs = []
    n_samples = shap_values.shape[0]
    if n_samples < 50: return 0.0, 0.0 
    
    # Pre-normalize once to be safe or re-normalize per boot?
    # Logic: normalize SHAP per sample first, then bootstrap samples.
    row_sums = np.abs(shap_values).sum(axis=1, keepdims=True) + 1e-12
    shap_norm = shap_values / row_sums
    
    for _ in range(n_boot):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        sample_shap = shap_norm[idx]
        # Re-calc phi for this boot
        phi = np.abs(sample_shap).mean(axis=0)
        
        sum_phi = np.sum(phi)
        sum_phi_sq = np.sum(phi**2)
        if sum_phi_sq == 0:
            prs.append(0.0)
        else:
             D = len(phi)
             pr = (sum_phi**2) / (D * sum_phi_sq)
             prs.append(pr)
    
    return np.mean(prs), np.std(prs)

def train_and_evaluate(X_tr, y_tr, X_te, y_te, seed: int, run_shap: bool) -> Dict:
    results = {}
    
    # Ridge
    ridge = Ridge(random_state=seed)
    ridge.fit(X_tr, y_tr)
    pred_ridge = ridge.predict(X_te)
    results["r2_ridge"] = r2_score(y_te, pred_ridge)
    
    # LightGBM
    lgbm = LGBMRegressor(n_jobs=-1, random_state=seed, verbose=-1)
    lgbm.fit(X_tr, y_tr)
    pred_lgbm = lgbm.predict(X_te)
    
    results["r2"] = r2_score(y_te, pred_lgbm)
    results["rmse"] = np.sqrt(mean_squared_error(y_te, pred_lgbm))
    results["mae"] = mean_absolute_error(y_te, pred_lgbm)
    results["y_test"] = y_te.tolist()
    results["y_pred"] = pred_lgbm.tolist()
    
    # SHAP
    if run_shap:
        try:
            explainer = shap.TreeExplainer(lgbm)
            # Subsample for SHAP
            X_shap = X_te[:2000] if len(X_te) > 2000 else X_te
            shap_values = explainer.shap_values(X_shap)
            
            pr, pr90, phi = calculate_pr(shap_values)
            results["pr"] = pr
            results["pr90"] = pr90
            results["phi"] = phi
            
            # Bootstrap
            pr_mean, pr_std = bootstrap_pr(shap_values, n_boot=50) 
            results["pr_std"] = pr_std

            # Metrics
            dim = X_tr.shape[1]
            eps = 1e-3
            results["epd"] = pr * dim
            results["pes"] = results["r2"] / (pr + eps)
            results["cwp"] = results["r2"] * pr
            results["linear_gap"] = results["r2"] - results["r2_ridge"]
            
        except Exception as e:
            print(f"SHAP Error: {e}")
            
    return results

def get_random_embeddings(n, dim, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim))

def plot_scatter(df: pd.DataFrame, out_dir: Path):
    """Generate scatter plots."""
    params = df["target_param"].unique()
    for p in params:
        sub = df[df["target_param"] == p]
        # Grid plot loop similar to original...
        # Simplified for now to save space, assuming separate Analysis class or function
        pass 
    # (Keeping original plotting logic is complex for in-place edit, skipping detailed re-implementation in this step for brevity locally testing first)
    # Actually, I should keep it runnable. I'll include a simple plotter.

# --- Optimized Plotting ---
class ResultPlotter:
    def __init__(self, df: pd.DataFrame, out_dir: Path):
        self.df = df
        self.out_dir = out_dir
        
    def plot_all(self):
        self.plot_scatter()
        self.plot_pareto()
        
    def plot_scatter(self):
        params = self.df["target_param"].unique()
        for p in params:
            dd = self.df[self.df["target_param"] == p]
            if dd.empty: continue
            
            # Simple grid
            g = sns.FacetGrid(dd, col="model", row="modality", height=3, aspect=1)
            g.map_dataframe(self._scatter_on, "y_test", "y_pred")
            g.set_titles("{col_name} | {row_name}")
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle(f"Prediction: {p}")
            g.savefig(self.out_dir / f"scatter_{p}.png")
            plt.close()
            
    def _scatter_on(self, y_test, y_pred, color=None, label=None, data=None):
         # Need to unwrap lists if specific format
         # But Seaborn handles dataframes. y_test is list in cell. Explode?
         # Data structure is one row per experiment.
         # For plotting we need points.
         # Explode:
         row = data.iloc[0] # Should be unique per facet
         yt, yp = np.array(row["y_test"]), np.array(row["y_pred"])
         r2 = row["r2"]
         plt.scatter(yt, yp, alpha=0.1, s=1, color='k')
         
         mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
         plt.plot([mn, mx], [mn, mx], 'r--')
         plt.text(0.05, 0.9, f"R2={r2:.2f}", transform=plt.gca().transAxes)

    def plot_pareto(self):
        if "pr" not in self.df.columns: return
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=self.df, x="pr", y="r2", hue="model", style="modality", s=100)
        plt.title("Performance (R2) vs Efficiency (PR)")
        plt.savefig(self.out_dir / "pareto.png")
        plt.close()

def run_analysis(
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    slurm: bool = False
):
    """Main execution function."""
    paths = load_paths()
    if not config_path:
        config_path = paths.repo_root / "src/fmb/configs/analysis/regression.yaml"
        
    cfg = load_config(config_path)
    
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = paths.analysis / "regression"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Catalog
    # Catalog path: either in config or default location in data
    # Assuming catalog is in data folder.
    # Config might specify 'catalog_path' relative to data or absolute.
    cat_path = paths.dataset / cfg.get("catalog_filename", "euclid_desi_catalog.fits") # Default?
    # Or search for generic catalog
    if not cat_path.exists():
        # Try finding *catalog*.fits in data
        candidates = list(paths.dataset.glob("*catalog*.fits"))
        if candidates: cat_path = candidates[0]
        else:
             print(f"Catalog not found in {paths.dataset}")
             return
             
    catalog, num_cols, id_col = load_catalog(cat_path, cfg.get("catalog_id_column"))
    
    # 2. Determine Targets
    targets = cfg.get("targets", [])
    col_map = cfg.get("column_mapping", {})
    
    # 3. Load Models
    models_to_run = cfg.get("models", ["AION", "AstroPT", "AstroCLIP"])
    emb_files = []
    
    # Auto-detect embeddings
    for m in models_to_run:
        # Search paths.embeddings
        candidates = list(paths.embeddings.glob(f"*{m.lower()}*.pt"))
        if not candidates:
            # Try subfolder
            cand2 = list((paths.embeddings / m.lower()).glob("*.pt"))
            candidates.extend(cand2)
            
        if candidates:
            emb_files.append((m, candidates[0]))
        else:
            print(f"[warn] No embeddings found for {m}")
            
    if not emb_files:
        print("No embeddings found.")
        return
        
    # Load all embeddings into memory
    loaded_data = []
    for m, p in emb_files:
        recs = load_embeddings_file(p)
        loaded_data.append((m, recs))
        
    results = []
    seed = cfg.get("seed", 42)
    
    for t_name in targets:
        col = col_map.get(t_name, t_name) # Map to catalog column
        if col not in num_cols:
             # Try uppercase
             if col.upper() in num_cols: col = col.upper()
             else:
                 print(f"Skipping target {t_name} (Col {col} not found)")
                 continue
                 
        print(f"\nAnalyzing Target: {t_name} (Col: {col})")
        
        # Split logic: find common valid IDs for FAIR comparison?
        # Or split per dataset?
        # Strategy: Use only IDs valid in catalog for this param.
        valid_ids = [k for k,v in catalog.items() if isinstance(v.get(col), (int, float, np.number)) and not np.isnan(float(v.get(col)))]
        
        train_ids, test_ids = train_test_split(valid_ids, test_size=cfg.get("test_size", 0.2), random_state=seed)
        train_set = set(train_ids)
        test_set = set(test_ids)
        
        for m_name, recs in loaded_data:
            # Determine keys available
            if not recs: continue
            sample = recs[0]
            keys = [k for k in sample.keys() if k.startswith("embedding_")]
            
            for k in keys:
                print(f"  Model: {m_name}, Key: {k}")
                X_tr, y_tr, _ = merge_data(recs, catalog, col, k, subset_ids=train_set)
                X_te, y_te, _ = merge_data(recs, catalog, col, k, subset_ids=test_set)
                
                if len(X_tr) < 50:
                    print(f"    Insufficient data (Tr={len(X_tr)}).")
                    continue
                    
                metrics = train_and_evaluate(X_tr, y_tr, X_te, y_te, seed, cfg.get("run_shap", True))
                metrics.update({
                    "target_param": t_name,
                    "model": m_name,
                    "modality": k.replace("embedding_", "")
                })
                results.append(metrics)
                print(f"    R2={metrics['r2']:.3f}")
                
        # Random Baseline (Restored!)
        print("  Processing Random Baseline...")
        dim_random = 512
        X_rand_tr = get_random_embeddings(len(train_ids), dim_random, seed)
        y_rand_tr = np.array([catalog[oid][col] for oid in train_ids])
        
        X_rand_te = get_random_embeddings(len(test_ids), dim_random, seed + 1)
        y_rand_te = np.array([catalog[oid][col] for oid in test_ids])
        
        met_rand = train_and_evaluate(X_rand_tr, y_rand_tr, X_rand_te, y_rand_te, seed, cfg.get("run_shap", True))
        met_rand.update({
            "target_param": t_name,
            "model": "Random",
            "modality": "embedding_random"
        })
        results.append(met_rand)
        print(f"    R2={met_rand.get('r2'):.3f} (Random)")
        
    # Save & Plot
    if results:
        df = pd.DataFrame(results)
        df_clean = df.drop(columns=["y_test", "y_pred"], errors="ignore")
        df_clean.to_csv(out_path / "results_summary.csv", index=False)
        print(f"Results saved to {out_path}")
        
        plotter = ResultPlotter(df, out_path)
        plotter.plot_all()

if __name__ == "__main__":
    # Barebones CLI for direct test
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file")
    args = parser.parse_args()
    cfg_p = Path(args.config) if args.config else None
    run_analysis(cfg_p)
