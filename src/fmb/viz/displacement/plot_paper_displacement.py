"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.displacement.plot_paper_displacement
Description: FMB module: fmb.viz.displacement.plot_paper_displacement
"""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

# --- Publication Style Settings ---
try:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )
except Exception:
    print("Warning: LaTeX not available, falling back to STIX fonts.")
    plt.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

plt.rcParams.update(
    {
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
    }
)

# Mapping of embedding types to keys for each model
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

COLORS = {
    "AION": "#1f77b4",  # Blue
    "AstroPT": "#ff7f0e",  # Orange
    "AstroCLIP": "#2ca02c",  # Green
}


def load_data(path: Path) -> pd.DataFrame:
    """Load anomaly scores from CSV."""
    df = pd.read_csv(path)
    df["object_id"] = df["object_id"].astype(str)
    return df


def plot_panel(
    ax, source_name, target_name, source_ranks, target_ranks, n_total, color
):
    """Plot a single displacement panel."""
    # Normalize to percentile (0-100)
    top_1_threshold = n_total * 0.01
    subset_mask = source_ranks <= top_1_threshold

    if subset_mask.sum() == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    ranks_pct = (target_ranks[subset_mask] / n_total) * 100

    # Plot Histogram
    ax.hist(
        ranks_pct,
        bins=40,
        range=(0, 100),
        color=color,
        edgecolor="black",
        alpha=0.7,
        linewidth=0.5,
        zorder=3,
    )

    # Add markers for Top 1% and Top 10%
    ax.axvline(x=1, color="red", linestyle="--", linewidth=1, zorder=4)
    ax.axvline(x=10, color="orange", linestyle="--", linewidth=1, zorder=4)

    # Shade regions
    ax.axvspan(0, 1, color="red", alpha=0.1, zorder=1)
    ax.axvspan(1, 10, color="orange", alpha=0.1, zorder=1)

    # Calculate stats
    retained_1 = (
        (target_ranks[subset_mask] <= top_1_threshold).sum() / subset_mask.sum() * 100
    )
    retained_10 = (
        (target_ranks[subset_mask] <= (n_total * 0.1)).sum() / subset_mask.sum() * 100
    )

    # Add text box
    stats_text = (
        f"\\textbf{{Retained}}:\n"
        f"Top 1\\%: {retained_1:.1f}\\%\n"
        f"Top 10\\%: {retained_10:.1f}\\%"
    )
    ax.text(
        0.95,
        0.92,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"
        ),
        zorder=5,
        fontsize=7,
    )

    ax.set_title(
        rf"\textbf{{{source_name}}} $\rightarrow$ \textbf{{{target_name}}}", pad=5
    )
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-model/multi-type displacement analysis grid"
    )
    parser.add_argument("--aion-scores", required=True, help="AION scores CSV")
    parser.add_argument("--astropt-scores", required=True, help="AstroPT scores CSV")
    parser.add_argument(
        "--astroclip-scores", required=True, help="AstroCLIP scores CSV"
    )
    parser.add_argument(
        "--save", default="paper_displacement_grid.png", help="Output filename"
    )

    args = parser.parse_args(argv)

    # Load all datasets
    print("Loading scores...")
    dfs = {
        "AION": load_data(Path(args.aion_scores)),
        "AstroPT": load_data(Path(args.astropt_scores)),
        "AstroCLIP": load_data(Path(args.astroclip_scores)),
    }

    embedding_types = ["Images", "Spectra", "Joint"]
    comparisons = [("AION", "AstroPT"), ("AstroPT", "AstroCLIP"), ("AstroCLIP", "AION")]

    fig, axes = plt.subplots(3, 3, figsize=(10, 9), sharex=True, sharey="row")

    for row_idx, emb_type in enumerate(embedding_types):
        print(f"Processing {emb_type}...")

        # Extract relevant subset for each model
        subsets = {}
        for model_name, df in dfs.items():
            key = KEYS[model_name][emb_type]
            subsets[model_name] = df[df["embedding_key"] == key].copy()
            if len(subsets[model_name]) == 0:
                print(f"Warning: No data for {model_name} with key {key}")

        # Common merge to ensure we compare the same objects
        merged = subsets["AION"][["object_id", "rank"]].rename(
            columns={"rank": "rank_aion"}
        )
        merged = merged.merge(
            subsets["AstroPT"][["object_id", "rank"]].rename(
                columns={"rank": "rank_astropt"}
            ),
            on="object_id",
        )
        merged = merged.merge(
            subsets["AstroCLIP"][["object_id", "rank"]].rename(
                columns={"rank": "rank_astroclip"}
            ),
            on="object_id",
        )

        n_total = len(merged)
        print(f"  Matched {n_total} objects for {emb_type}")

        for col_idx, (src, tgt) in enumerate(comparisons):
            ax = axes[row_idx, col_idx]
            src_col = f"rank_{src.lower()}"
            tgt_col = f"rank_{tgt.lower()}"

            plot_panel(
                ax,
                src,
                tgt,
                merged[src_col].values,
                merged[tgt_col].values,
                n_total,
                COLORS[src],
            )

            # Labels
            if col_idx == 0:
                ax.set_ylabel(rf"\textbf{{{emb_type}}}" + "\nCount")
            if row_idx == 2:
                ax.set_xlabel("Percentile Rank in Target")

    plt.tight_layout()

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
