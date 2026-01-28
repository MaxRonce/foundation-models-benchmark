"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.outliers.advanced_analysis
Description: Cross-model outlier comparison analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fmb.paths import load_paths
from fmb.viz.style import apply_style

KEYS = {
    "AION": {
        "Images": "embedding_hsc",
        "Spectra": "embedding_spectrum",
        "Joint": "embedding_hsc_desi",
    },
    "AstroPT": {
        "Images": "embedding_images",
        "Spectra": "embedding_spectra",
        "Joint": "embedding_joint",
    },
    "AstroCLIP": {
        "Images": "embedding_images",
        "Spectra": "embedding_spectra",
        "Joint": "embedding_joint",
    },
}


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Score file not found: {path}")
    df = pd.read_csv(path)
    df["object_id"] = df["object_id"].astype(str)
    return df


def run_analysis(
    aion_scores: Union[str, Path, None] = None,
    astropt_scores: Union[str, Path, None] = None,
    astroclip_scores: Union[str, Path, None] = None,
    save_prefix: Union[str, Path, None] = None,
):
    """
    Main entry point for advanced analysis visualization.
    """
    # 1. Apply Style
    apply_style()

    # 2. Resolve Paths
    paths = load_paths()
    out_dir = paths.analysis / "advanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_prefix is None:
        save_prefix = out_dir / "paper_advanced"
    else:
        save_prefix = Path(save_prefix)
        if not save_prefix.parent.exists():
            save_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Defaults for input files if not provided
    # We look for standard names in paths.outliers
    if aion_scores is None:
        aion_scores = paths.outliers / "anomaly_scores_aion.csv"
    else:
        aion_scores = Path(aion_scores)

    if astropt_scores is None:
        astropt_scores = paths.outliers / "anomaly_scores_astropt.csv"
    else:
        astropt_scores = Path(astropt_scores)

    if astroclip_scores is None:
        astroclip_scores = paths.outliers / "anomaly_scores_astroclip.csv"
    else:
        astroclip_scores = Path(astroclip_scores)

    # 3. Load Data
    print("Loading scores...")
    print(f"  AION:      {aion_scores}")
    print(f"  AstroPT:   {astropt_scores}")
    print(f"  AstroCLIP: {astroclip_scores}")

    try:
        dfs = {
            "AION": load_data(aion_scores),
            "AstroPT": load_data(astropt_scores),
            "AstroCLIP": load_data(astroclip_scores),
        }
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all score files exist or provide paths via arguments.")
        sys.exit(1)

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
        # Check if key exists (some runs might be partial)
        if key not in df["embedding_key"].unique():
            print(f"Warning: Key '{key}' not found in {model} scores. Skipping.")
            continue

        mod_data = df[df["embedding_key"] == key][["object_id", "rank"]].rename(
            columns={"rank": f"{model}-{mod}"}
        )
        if big_merged is None:
            big_merged = mod_data
        else:
            big_merged = big_merged.merge(mod_data, on="object_id")

    if big_merged is None or big_merged.empty:
        print(
            "Error: No common objects found after merge. Check object_ids consistency."
        )
        return

    n_total = len(big_merged)
    print(f"Matched {n_total} objects.")

    # Update systems list based on what was actually merged
    valid_cols = [c for c in big_merged.columns if c != "object_id"]
    # Re-order to keep generic order if possible
    final_cols = []
    for model, mod in systems:
        col = f"{model}-{mod}"
        if col in valid_cols:
            final_cols.append(col)

    # --- 1. Spearman Clustermap (Rank Correlation) ---
    print("Computing Spearman Correlations...")
    corr_matrix = big_merged[final_cols].corr(method="spearman")

    plt.figure()
    g = sns.clustermap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        dendrogram_ratio=(0.1, 0.1),
    )  # , cbar_pos=(.02, .32, .03, .2))
    g.fig.suptitle("Hierarchical Clustering of Systems (Spearman Correlation)", y=1.02)
    s_path = f"{save_prefix}_spearman_clustermap.png"
    g.savefig(s_path, dpi=300)
    print(f"Saved {s_path}")

    # --- 2. Jaccard Index Clustermap (Top 1% Intersection) ---
    print("Computing Jaccard Indices...")
    top_1_thr = n_total * 0.01
    jaccard_matrix = pd.DataFrame(index=final_cols, columns=final_cols, dtype=float)

    top_1_sets = {}
    for col in final_cols:
        top_1_sets[col] = set(big_merged[big_merged[col] <= top_1_thr]["object_id"])

    for col1 in final_cols:
        for col2 in final_cols:
            s1 = top_1_sets[col1]
            s2 = top_1_sets[col2]
            iou = (
                len(s1.intersection(s2)) / len(s1.union(s2))
                if len(s1.union(s2)) > 0
                else 0
            )
            jaccard_matrix.loc[col1, col2] = iou

    plt.figure()
    g2 = sns.clustermap(
        jaccard_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        dendrogram_ratio=(0.1, 0.1),
    )  # , cbar_pos=(.02, .32, .03, .2))
    g2.fig.suptitle(
        r"Hierarchical Clustering of Top 1\% Anomalies (Jaccard Index)", y=1.02
    )
    j_path = f"{save_prefix}_jaccard_clustermap.png"
    g2.savefig(j_path, dpi=300)
    print(f"Saved {j_path}")

    # --- 3. Disagreement Analysis (AION-Joint vs AstroPT-Joint) ---
    sys1 = "AION-Joint"
    sys2 = "AstroPT-Joint"

    if sys1 in final_cols and sys2 in final_cols:
        print("Analyzing Disagreements...")

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
        ax.hexbin(
            x_pct, y_pct, gridsize=50, cmap="Greys", mincnt=1, bins="log", alpha=0.6
        )

        # Highlight Controversial
        ax.scatter(
            x_pct[mask_a],
            y_pct[mask_a],
            s=10,
            c="red",
            label=f"Anomaly in {sys1}\nNormal in {sys2}",
            alpha=0.8,
        )
        ax.scatter(
            x_pct[mask_b],
            y_pct[mask_b],
            s=10,
            c="blue",
            label=f"Anomaly in {sys2}\nNormal in {sys1}",
            alpha=0.8,
        )

        ax.set_xlabel(f"{sys1} Rank (\%)")
        ax.set_ylabel(f"{sys2} Rank (\%)")
        ax.set_title(f"Disagreement Analysis: {sys1} vs {sys2}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Annotate regions
        ax.axvline(x=1, color="red", linestyle=":", alpha=0.5)
        ax.axhline(y=1, color="blue", linestyle=":", alpha=0.5)

        plt.tight_layout()
        d_path = f"{save_prefix}_disagreement_scatter.png"
        fig.savefig(d_path, dpi=300)
        print(f"Saved {d_path}")

        # Export Disagreements
        disagreements = big_merged[mask_a | mask_b][["object_id", sys1, sys2]].copy()
        disagreements["type"] = np.where(
            mask_a[mask_a | mask_b], f"{sys1}_Anomaly", f"{sys2}_Anomaly"
        )
        csv_path = f"{save_prefix}_disagreement_objects.csv"
        disagreements.to_csv(csv_path, index=False)
        print(f"Exported {len(disagreements)} controversial objects to {csv_path}")
    else:
        print(f"Skipping disagreement analysis: {sys1} or {sys2} missing.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aion-scores", default=None)
    parser.add_argument("--astropt-scores", default=None)
    parser.add_argument("--astroclip-scores", default=None)
    parser.add_argument("--save-prefix", default=None)
    args = parser.parse_args(argv)

    run_analysis(
        aion_scores=args.aion_scores,
        astropt_scores=args.astropt_scores,
        astroclip_scores=args.astroclip_scores,
        save_prefix=args.save_prefix,
    )


if __name__ == "__main__":
    main()
