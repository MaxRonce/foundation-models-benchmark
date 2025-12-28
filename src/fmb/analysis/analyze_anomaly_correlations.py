#!/usr/bin/env python3
"""
Script to analyze the correlation and overlap of anomaly scores between AION and AstroPT models.

This script:
1. Loads anomaly scores from AION and AstroPT.
2. Matches them based on object_id and embedding type.
3. Calculates Pearson and Spearman correlations of the ranks.
4. Performs a quantile overlap analysis with varying granularities.
5. Generates a report, CSV statistics, and visualization plots.

Usage:
    python analyze_anomaly_correlations.py \
        --aion-scores /home/ronceray/AION/outputs/anomaly_scores.csv \
        --astropt-scores /home/ronceray/AION/outputs/anomaly_scores_astropt.csv \
        --output-dir /home/ronceray/AION/outputs/analysis
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import itertools

# Pretty names for all keys
PRETTY_NAMES = {
    "embedding_hsc": "AION Images",
    "embedding_spectrum": "AION Spectra",
    "embedding_hsc_desi": "AION Joint",
    "embedding_images": "AstroPT Images",
    "embedding_spectra": "AstroPT Spectra",
    "embedding_joint": "AstroPT Joint",
}

def load_data(path: Path) -> pd.DataFrame:
    """Load anomaly scores from CSV."""
    try:
        df = pd.read_csv(path)
        # Ensure object_id is string for consistent merging
        df['object_id'] = df['object_id'].astype(str)
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

def analyze_correlations(
    aion_df: pd.DataFrame,
    astropt_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Analyze correlations and overlaps for ALL pairwise embedding combinations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine datasets to make it easier to pair any key with any key
    # Add a 'source' column to distinguish if needed, but keys are unique enough usually.
    # Actually, AION and AstroPT keys are distinct sets.
    # Let's just concatenate them into one big dataframe for easier handling
    full_df = pd.concat([aion_df, astropt_df], ignore_index=True)
    
    # Get all unique keys
    all_keys = sorted(full_df['embedding_key'].unique())
    print(f"Found {len(all_keys)} unique embedding keys: {all_keys}")
    
    stats_list = []
    
    # Open report file
    report_path = output_dir / "correlation_report_all_pairs.txt"
    with open(report_path, "w") as report:
        report.write("=" * 80 + "\n")
        report.write("ANOMALY SCORE CORRELATION ANALYSIS (ALL PAIRS)\n")
        report.write("=" * 80 + "\n\n")
        
        # Generate all unique pairs
        pairs = list(itertools.combinations(all_keys, 2))
        print(f"Analyzing {len(pairs)} pairs...")
        
        for key1, key2 in pairs:
            name1 = PRETTY_NAMES.get(key1, key1)
            name2 = PRETTY_NAMES.get(key2, key2)
            pair_name = f"{name1} vs {name2}"
            
            report.write(f"Analyzing {pair_name}...\n")
            
            # Filter data
            subset1 = full_df[full_df['embedding_key'] == key1].copy()
            subset2 = full_df[full_df['embedding_key'] == key2].copy()
            
            # Merge on object_id
            merged = pd.merge(
                subset1, 
                subset2, 
                on='object_id', 
                suffixes=('_1', '_2'),
                how='inner'
            )
            
            n_samples = len(merged)
            report.write(f"  Matched Samples: {n_samples}\n")
            
            if n_samples < 10:
                report.write("  WARNING: Too few samples for analysis. Skipping.\n\n")
                continue
                
            # 1. Correlations
            pearson_corr, p_val = pearsonr(merged['rank_1'], merged['rank_2'])
            spearman_corr, s_p_val = spearmanr(merged['rank_1'], merged['rank_2'])
            
            report.write(f"  Pearson Correlation (Ranks):  {pearson_corr:.4f} (p={p_val:.4g})\n")
            report.write(f"  Spearman Correlation (Ranks): {spearman_corr:.4f} (p={s_p_val:.4g})\n")
            
            # 2. Quantile Overlap Analysis & Enrichment
            granularities = [2, 4, 10, 20, 50, 100]
            
            for N in granularities:
                # Create bins
                merged['bin_1'] = pd.qcut(merged['rank_1'], N, labels=False)
                merged['bin_2'] = pd.qcut(merged['rank_2'], N, labels=False)
                
                # Calculate overlap for each bin
                overlaps = []
                enrichments = []
                
                for i in range(N):
                    in_bin_1 = merged['bin_1'] == i
                    in_bin_2 = merged['bin_2'] == i
                    
                    # Intersection
                    intersection = (in_bin_1 & in_bin_2).sum()
                    total_in_bin = in_bin_1.sum()
                    
                    overlap_pct = (intersection / total_in_bin) * 100 if total_in_bin > 0 else 0
                    overlaps.append(overlap_pct)
                    
                    # Enrichment Factor
                    # Expected overlap if random = 1/N * 100%
                    # Enrichment = Observed / Expected
                    expected_pct = (1/N) * 100
                    enrichment = overlap_pct / expected_pct if expected_pct > 0 else 0
                    enrichments.append(enrichment)
                    
                    # Add to stats list
                    stats_list.append({
                        'pair_name': pair_name,
                        'key1': key1,
                        'key2': key2,
                        'granularity': N,
                        'bin_index': i,
                        'bin_percentile_start': (i / N) * 100,
                        'bin_percentile_end': ((i + 1) / N) * 100,
                        'overlap_pct': overlap_pct,
                        'enrichment_factor': enrichment,
                        'pearson_corr': pearson_corr,
                        'spearman_corr': spearman_corr
                    })
                
                # Report top bin (most anomalous)
                if N == 10: # Only report deciles in text
                    report.write(f"  Top 10% Overlap: {overlaps[0]:.1f}% (Enrichment: {enrichments[0]:.2f}x)\n")

            # 3. Rank Displacement Analysis (Top 1% Focus)
            # Take Top 1% of Model 1, see where they land in Model 2
            top_1_pct_threshold = n_samples * 0.01
            top_1_subset = merged[merged['rank_1'] <= top_1_pct_threshold]
            
            if len(top_1_subset) > 0:
                median_rank_2 = top_1_subset['rank_2'].median()
                median_rank_pct_2 = (median_rank_2 / n_samples) * 100
                
                # How many of Top 1% (Model 1) are in Top 1% (Model 2)?
                in_top_1_2 = (top_1_subset['rank_2'] <= top_1_pct_threshold).sum()
                in_top_1_pct = (in_top_1_2 / len(top_1_subset)) * 100
                
                # How many are in Top 10% (Model 2)?
                top_10_pct_threshold = n_samples * 0.10
                in_top_10_2 = (top_1_subset['rank_2'] <= top_10_pct_threshold).sum()
                in_top_10_pct = (in_top_10_2 / len(top_1_subset)) * 100
                
                report.write(f"  Rank Displacement (Focus on Top 1% of {name1}):\n")
                report.write(f"    Median Rank in {name2}: {median_rank_2:.0f} ({median_rank_pct_2:.1f}%)\n")
                report.write(f"    Retained in Top 1%:  {in_top_1_pct:.1f}%\n")
                report.write(f"    Retained in Top 10%: {in_top_10_pct:.1f}%\n")
                
                # Save displacement data for plotting
                # We can't save all points in the main stats CSV, maybe just generate plot directly here?
                # Or save to a separate file? Let's generate plot directly to save IO complexity
                plot_rank_displacement(merged, name1, name2, output_dir)
                plot_rank_scatter(merged, name1, name2, output_dir)

            report.write("-" * 40 + "\n\n")

    # Save stats to CSV
    stats_df = pd.DataFrame(stats_list)
    stats_csv_path = output_dir / "overlap_stats_all_pairs.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved statistics to {stats_csv_path}")
    print(f"Saved report to {report_path}")
    
    # Generate Summary Visualizations
    plot_summary_overlaps(stats_df, output_dir)

    # Generate Combined Grid Plots
    # We need to re-generate the merged data for plotting since we didn't store it all in memory
    # (To avoid memory issues if datasets are huge, though 40k is small)
    # Let's just re-loop for plotting to be safe and clean
    plot_combined_grids(full_df, all_keys, output_dir)


def plot_combined_grids(full_df: pd.DataFrame, all_keys: list[str], output_dir: Path) -> None:
    """Generate combined grid plots for all pairs."""
    pairs = list(itertools.combinations(all_keys, 2))
    n_pairs = len(pairs)
    
    # Calculate grid dimensions
    cols = 4
    rows = (n_pairs + cols - 1) // cols
    
    # 1. Combined Scatter Grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    
    print("Generating combined scatter grid...")
    for idx, (key1, key2) in enumerate(pairs):
        ax = axes[idx]
        name1 = PRETTY_NAMES.get(key1, key1)
        name2 = PRETTY_NAMES.get(key2, key2)
        
        subset1 = full_df[full_df['embedding_key'] == key1]
        subset2 = full_df[full_df['embedding_key'] == key2]
        
        merged = pd.merge(subset1, subset2, on='object_id', suffixes=('_1', '_2'))
        
        # Hexbin plot
        hb = ax.hexbin(merged['rank_1'], merged['rank_2'], gridsize=30, cmap='inferno', bins='log', mincnt=1)
        
        ax.set_title(f'{name1}\nvs {name2}', fontsize=10)
        if idx % cols == 0:
            ax.set_ylabel('Rank 2', fontsize=8)
        if idx >= (rows - 1) * cols:
            ax.set_xlabel('Rank 1', fontsize=8)
            
        # Add y=x line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'w--', alpha=0.5)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / "combined_scatter_grid.png", dpi=150)
    plt.close()
    
    # 2. Combined Displacement Grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    
    print("Generating combined displacement grid...")
    for idx, (key1, key2) in enumerate(pairs):
        ax = axes[idx]
        name1 = PRETTY_NAMES.get(key1, key1)
        name2 = PRETTY_NAMES.get(key2, key2)
        
        subset1 = full_df[full_df['embedding_key'] == key1]
        subset2 = full_df[full_df['embedding_key'] == key2]
        
        merged = pd.merge(subset1, subset2, on='object_id', suffixes=('_1', '_2'))
        n_samples = len(merged)
        
        # Top 1% of Model 1
        top_1_threshold = n_samples * 0.01
        subset = merged[merged['rank_1'] <= top_1_threshold]
        
        if len(subset) > 0:
            ranks_pct = (subset['rank_2'] / n_samples) * 100
            ax.hist(ranks_pct, bins=30, range=(0, 100), color='skyblue', edgecolor='black', alpha=0.7)
            
            ax.axvline(x=1, color='red', linestyle='--', linewidth=1)
            ax.axvline(x=10, color='orange', linestyle='--', linewidth=1)
            
            ax.set_title(f'Top 1% {name1}\nin {name2}', fontsize=10)
            if idx % cols == 0:
                ax.set_ylabel('Count', fontsize=8)
            if idx >= (rows - 1) * cols:
                ax.set_xlabel('Rank % in Model 2', fontsize=8)
            ax.tick_params(labelsize=8)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / "combined_displacement_grid.png", dpi=150)
    plt.close()

def plot_rank_scatter(df: pd.DataFrame, name1: str, name2: str, output_dir: Path) -> None:
    """Generate a hexbin scatter plot of ranks."""
    # Clean names for filename
    safe_name1 = name1.replace(" ", "_").replace("(", "").replace(")", "")
    safe_name2 = name2.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"scatter_{safe_name1}_vs_{safe_name2}.png"
    
    plt.figure(figsize=(8, 7))
    # Log scale ranks for better visualization of top anomalies (low rank values)
    # +1 to avoid log(0)
    plt.hexbin(df['rank_1'], df['rank_2'], gridsize=50, cmap='inferno', bins='log', mincnt=1)
    plt.colorbar(label='Count (log)')
    
    plt.xlabel(f'Rank in {name1} (Lower = More Anomalous)')
    plt.ylabel(f'Rank in {name2} (Lower = More Anomalous)')
    plt.title(f'Rank Correlation: {name1} vs {name2}')
    
    # Add y=x line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'w--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=100)
    plt.close()

def plot_rank_displacement(df: pd.DataFrame, name1: str, name2: str, output_dir: Path) -> None:
    """
    Plot histogram of ranks in Model 2 for the Top 1% of Model 1.
    """
    n_samples = len(df)
    top_1_threshold = n_samples * 0.01
    
    # Select Top 1% in Model 1
    subset = df[df['rank_1'] <= top_1_threshold]
    
    # Clean names
    safe_name1 = name1.replace(" ", "_").replace("(", "").replace(")", "")
    safe_name2 = name2.replace(" ", "_").replace("(", "").replace(")", "")
    filename = f"displacement_{safe_name1}_vs_{safe_name2}.png"
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of ranks in Model 2 (normalized to percentile)
    ranks_pct = (subset['rank_2'] / n_samples) * 100
    
    plt.hist(ranks_pct, bins=50, range=(0, 100), color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.axvline(x=1, color='red', linestyle='--', label='Top 1% Threshold')
    plt.axvline(x=10, color='orange', linestyle='--', label='Top 10% Threshold')
    plt.axvline(x=50, color='gray', linestyle=':', label='Median')
    
    plt.xlabel(f'Percentile Rank in {name2}')
    plt.ylabel(f'Count (from Top 1% of {name1})')
    plt.title(f'Where do Top 1% {name1} Anomalies fall in {name2}?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=100)
    plt.close()

def plot_summary_overlaps(stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary plots for overlap analysis."""
    
    pairs = stats_df['pair_name'].unique()
    granularities = stats_df['granularity'].unique()
    
    # 1. Correlation Matrix Heatmap
    unique_keys = sorted(list(set(stats_df['key1'].unique()) | set(stats_df['key2'].unique())))
    n_keys = len(unique_keys)
    corr_matrix = np.zeros((n_keys, n_keys))
    
    np.fill_diagonal(corr_matrix, 1.0)
    pair_stats = stats_df.drop_duplicates(subset=['key1', 'key2'])
    
    for _, row in pair_stats.iterrows():
        i = unique_keys.index(row['key1'])
        j = unique_keys.index(row['key2'])
        val = row['spearman_corr']
        corr_matrix[i, j] = val
        corr_matrix[j, i] = val
        
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Spearman Correlation')
    
    pretty_labels = [PRETTY_NAMES.get(k, k) for k in unique_keys]
    plt.xticks(range(n_keys), pretty_labels, rotation=45, ha='right')
    plt.yticks(range(n_keys), pretty_labels)
    
    for i in range(n_keys):
        for j in range(n_keys):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            plt.text(j, i, f"{val:.2f}", ha='center', va='center', color=color)
            
    plt.title('Anomaly Rank Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close()
    
    # 2. Enrichment Plot (Top 10%)
    # Bar chart of Enrichment Factor for the Top 10% bin across all pairs
    N = 10
    if N in granularities:
        subset = stats_df[(stats_df['granularity'] == N) & (stats_df['bin_index'] == 0)].sort_values('enrichment_factor', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(subset['pair_name'], subset['enrichment_factor'], color='teal')
        plt.axvline(x=1.0, color='red', linestyle='--', label='Random Chance (1.0x)')
        
        plt.xlabel('Enrichment Factor (Observed / Random)')
        plt.title('Top 10% Anomaly Enrichment by Pair')
        plt.legend()
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_dir / "enrichment_bar_plot.png", dpi=150)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze anomaly score correlations.")
    parser.add_argument(
        "--aion-scores", 
        default="/home/ronceray/AION/outputs/anomaly_scores.csv",
        help="Path to AION anomaly scores CSV"
    )
    parser.add_argument(
        "--astropt-scores", 
        default="/home/ronceray/AION/outputs/anomaly_scores_astropt.csv",
        help="Path to AstroPT anomaly scores CSV"
    )
    parser.add_argument(
        "--output-dir", 
        default="/home/ronceray/AION/outputs/analysis",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    print("Loading data...")
    aion_df = load_data(Path(args.aion_scores))
    astropt_df = load_data(Path(args.astropt_scores))
    
    print(f"Loaded {len(aion_df)} AION rows and {len(astropt_df)} AstroPT rows.")
    
    analyze_correlations(aion_df, astropt_df, Path(args.output_dir))
    print("Done!")

if __name__ == "__main__":
    main()
