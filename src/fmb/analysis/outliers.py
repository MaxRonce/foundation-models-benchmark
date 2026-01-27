"""
Anomaly Analysis Module.
Analyzes and visualizes the results from the multimodal anomaly detection pipeline.
"""
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fmb.paths import load_paths

# Try to import venn
try:
    from matplotlib_venn import venn2, venn3
    HAS_VENN = True
except ImportError:
    HAS_VENN = False

class AnomalyAnalyzer:
    def __init__(self, input_csv: Path, output_dir: Path):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_csv}")
        print(f"Loading analysis data from {self.input_csv}...")
        return pd.read_csv(self.input_csv, low_memory=False)

    def plot_correlations(self, metrics: List[str] = ["rank_cosine", "rank_mm_geo", "p_joint"]) -> None:
        """Plot Spearman correlation of ranks between models."""
        if "model" not in self.df.columns:
            print("[warn] 'model' column missing, skipping correlation analysis.")
            return

        print("Generating correlation plots...")
        for m in metrics:
            if m not in self.df.columns:
                continue
            
            pivot = self.df.pivot(index="object_id", columns="model", values=m)
            if pivot.shape[1] < 2:
                continue
                
            plt.figure(figsize=(8, 6))
            corr = pivot.corr(method="spearman")
            sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=0, vmax=1)
            plt.title(f"Spearman Correlation of '{m}' between Models")
            plt.tight_layout()
            out_file = self.output_dir / f"corr_{m}.png"
            plt.savefig(out_file)
            plt.close()
            print(f"  Saved {out_file}")

    def plot_scatter_ranks(self) -> None:
        """Scatter plot of Image vs Spectrum percentiles."""
        print("Generating rank scatter plots...")
        plt.figure(figsize=(9, 8))
        
        hue = "model" if "model" in self.df.columns else None
        
        sns.scatterplot(
            data=self.df, 
            x="p_img", 
            y="p_spec", 
            hue=hue, 
            alpha=0.3, 
            s=15,
            palette="viridis" if hue else None
        )
            
        plt.xlabel("Image Anomaly Percentile (1.0 = Most Anomalous)")
        plt.ylabel("Spectrum Anomaly Percentile (1.0 = Most Anomalous)")
        plt.title("Image vs Spectrum Anomaly Percentiles")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        out_file = self.output_dir / "scatter_p_img_p_spec.png"
        plt.savefig(out_file)
        plt.close()
        print(f"  Saved {out_file}")

    def plot_uplift_distribution(self) -> None:
        """Histogram of Multimodal Uplift."""
        if "uplift_mm" not in self.df.columns:
            return

        print("Generating uplift distribution plot...")
        plt.figure(figsize=(8, 6))
        hue = "model" if "model" in self.df.columns else None
        
        sns.histplot(
            data=self.df, 
            x="uplift_mm", 
            hue=hue, 
            element="step", 
            kde=True, 
            bins=50,
            common_norm=False
        )
        
        plt.axvline(0, color="k", linestyle="--", alpha=0.5)
        
        # Add explanatory annotations
        # 1. Near Zero: Robust
        plt.text(0.02, plt.gca().get_ylim()[1]*0.9, 
                 "Robust Multimodal\nAnomalies", 
                 fontsize=9, color="green", ha="left")
                 
        # 2. Negative: Suppressed
        plt.text(-0.2, plt.gca().get_ylim()[1]*0.9, 
                 "Suppressed\nUnimodal Artifacts", 
                 fontsize=9, color="red", ha="right")

        plt.title("Impact of Multimodal Fusion (Uplift Distribution)")
        plt.xlabel("Uplift Score\n(Negative values indicate objects filtered out because modalities disagree)")
        plt.ylabel("Count")
        
        # Move legend if needed, but default is usually fine.
        # Ensure the legend title is clear.
        # We don't call plt.legend() because it overwrites the hue legend.
        
        plt.tight_layout()
        
        out_file = self.output_dir / "uplift_hist.png"
        plt.savefig(out_file)
        plt.close()
        print(f"  Saved {out_file}")

    def plot_overlaps(self, top_k: int = 200) -> None:
        """Venn diagram of Top-K candidates across models."""
        if "model" not in self.df.columns:
            return
            
        models = sorted(self.df["model"].unique())
        if len(models) not in [2, 3]:
            print(f"[info] Skipping Venn diagram (suitable for 2 or 3 models, found {len(models)})")
            return

        if not HAS_VENN:
            print("[warn] matplotlib-venn not installed, skipping Venn diagram.")
            return

        print(f"Generating Venn diagram for Top-{top_k} anomalies...")
        
        # Identify top-k objects per model based on rank_mm_geo (or score_mm_geo)
        # Using sorted head if ranks are not pre-computed 1..N
        # Assume score_mm_geo descending is best.
        
        sets = {}
        for m in models:
            sub = self.df[self.df["model"] == m]
            if "score_mm_geo" in sub.columns:
                top_objs = set(sub.sort_values("score_mm_geo", ascending=False).head(top_k)["object_id"])
            elif "rank_mm_geo" in sub.columns:
                top_objs = set(sub.sort_values("rank_mm_geo", ascending=True).head(top_k)["object_id"])
            else:
                print(f"[warn] No suitable ranking col for model {m}")
                continue
            sets[m] = top_objs
            
        plt.figure(figsize=(8, 8))
        if len(models) == 2:
            venn2([sets[models[0]], sets[models[1]]], set_labels=models)
        elif len(models) == 3:
            venn3([sets[models[0]], sets[models[1]], sets[models[2]]], set_labels=models)
            
        plt.title(f"Overlap of Top-{top_k} Multimodal Anomalies")
        
        out_file = self.output_dir / f"venn_top{top_k}_models.png"
        plt.savefig(out_file)
        plt.close()
        print(f"  Saved {out_file}")

    def run_all(self, top_k: int = 200) -> None:
        self.plot_correlations()
        self.plot_scatter_ranks()
        self.plot_uplift_distribution()
        self.plot_overlaps(top_k=top_k)


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser(description="Analyze Anomaly Detection Results")
    parser.add_argument("--input_csv", type=str, help="Path to all_scores.csv (default: auto-detect from runs/outliers/multimodal/all_scores.csv)")
    parser.add_argument("--top-k", type=int, default=200, help="Top-K threshold for overlaps")
    
    args = parser.parse_args(argv)
    
    paths = load_paths()
    
    # Auto-detect input input if not provided
    if args.input_csv:
        in_path = Path(args.input_csv)
    else:
        in_path = paths.outliers / "multimodal" / "all_scores.csv"
        
    if not in_path.exists():
        print(f"[error] Input file not found: {in_path}")
        print("Please run 'python -m fmb.cli detect multimodal' first.")
        return

    # Output directory: runs/analysis/outliers
    out_dir = paths.runs_root / "analysis" / "outliers"
    
    analyzer = AnomalyAnalyzer(in_path, out_dir)
    analyzer.run_all(top_k=args.top_k)
    print(f"\n[success] Analysis plots saved to {out_dir}")

if __name__ == "__main__":
    main()
