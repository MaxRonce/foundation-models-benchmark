"""
Script to analyze the anomaly rank distribution of neighbors of given anomalies.
For a list of query anomalies, it finds their nearest neighbors in the embedding space
and plots the distribution of the neighbors' anomaly ranks.

Usage:
    python -m scratch.analyze_neighbor_ranks \
        --input /path/to/embeddings.pt \
        --scores /path/to/scores.csv \
        --csv query_anomalies.csv \
        --n-similar 10 \
        --save outputs/neighbor_ranks.png
"""
import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Visualize style
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
except Exception:
    pass

from scratch.find_similar_anomalies import load_records, get_embedding_matrices, find_neighbors
from scratch.display_outlier_images import read_object_ids


def load_scores(path: Path) -> pd.DataFrame:
    print(f"Loading scores from {path}...")
    df = pd.read_csv(path)
    # Ensure object_id is string
    df['object_id'] = df['object_id'].astype(str)
    return df


def plot_rank_distribution(
    ranks: np.ndarray,
    total_objects: int,
    embedding_key: str,
    save_path: Path
):
    """Plots histogram of percentile ranks."""
    percentiles = (ranks / total_objects) * 100
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot histogram
    counts, bins, patches = ax.hist(
        percentiles, 
        bins=50, 
        range=(0, 100), 
        color='#1f77b4', 
        edgecolor='black', 
        alpha=0.7, 
        zorder=3
    )
    
    # Stats
    top1 = np.sum(percentiles <= 1.0) / len(percentiles) * 100
    top10 = np.sum(percentiles <= 10.0) / len(percentiles) * 100
    
    # Text box
    stats_text = (
        f"Neighbors count: {len(ranks)}\n"
        f"Top 1%: {top1:.1f}%\n"
        f"Top 10%: {top10:.1f}%"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    ax.set_title(f"Neighbor Anomaly Ranks ({embedding_key.replace('_', ' ')})")
    ax.set_xlabel("Anomaly Percentile (Lower is more anomalous)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze neighbor anomaly ranks."
    )
    parser.add_argument("--input", required=True, help="Path to embeddings .pt file")
    parser.add_argument("--scores", required=True, help="Path to anomaly scores CSV")
    
    # ID selection arguments
    parser.add_argument("--object_ids", nargs="+", help="List of object IDs to query")
    parser.add_argument("--csv", nargs="+", help="CSV files containing object_id column")
    
    parser.add_argument("--n-similar", type=int, default=10, help="Number of similar objects to find per query")
    parser.add_argument("--save", type=str, default="neighbor_ranks.png", help="Base path to save output plots")
    
    args = parser.parse_args(argv)
    
    # 1. Collect Query IDs
    query_ids = []
    if args.object_ids:
        query_ids.extend(args.object_ids)
    if args.csv:
        csv_paths = [Path(p) for p in args.csv]
        file_ids = read_object_ids(csv_paths, limit=None)
        query_ids.extend(file_ids)
        
    seen = set()
    unique_query_ids = []
    for q in query_ids:
        if q not in seen:
            unique_query_ids.append(q)
            seen.add(q)
    query_ids = unique_query_ids
    
    if not query_ids:
        raise SystemExit("No object IDs provided.")
        
    print(f"Analyzing neighbors for {len(query_ids)} query objects...")

    # 2. Load Data
    records = load_records(Path(args.input))
    scores_df = load_scores(Path(args.scores))
    
    matrices, all_ids = get_embedding_matrices(records)
    
    # 3. Process each embedding type
    for key, embedding_matrix in matrices.items():
        print(f"\nProcessing embedding type: {key}")
        
        # Correct key for scores matching
        # find_similar_anomalies maps to "embedding_joint" for computed joint
        # But scores might be "embedding_hsc_desi". 
        # Also AstroPT output scores might use "embedding_joint".
        
        # Let's check what keys are in scores_df
        available_score_keys = scores_df['embedding_key'].unique()
        
        target_score_key = key
        
        # Mapping attempts if direct match fail
        if target_score_key not in available_score_keys:
             if target_score_key == "embedding_joint" and "embedding_hsc_desi" in available_score_keys:
                 # Assume AION joint equivalent
                 target_score_key = "embedding_hsc_desi"
             elif target_score_key == "embedding_joint" and "embedding_joint" in available_score_keys:
                 # AstroPT joint
                 pass
        
        if target_score_key not in available_score_keys:
            print(f"Warning: Key '{target_score_key}' not found in scores CSV. Available: {available_score_keys}. Skipping.")
            continue

        # Get subset of scores
        subset_scores = scores_df[scores_df['embedding_key'] == target_score_key].set_index('object_id')
        total_objects = len(subset_scores)
        
        # Find Neighbors
        # This returns [Q1, N1_1... N1_k, Q2, N2_1...]
        ordered_ids = find_neighbors(query_ids, all_ids, embedding_matrix, args.n_similar)
        
        if not ordered_ids:
            print("No neighbors found.")
            continue
            
        # Extract just the neighbors (and maybe Query itself if we want to confirm it is low rank?)
        # Use wants "ranks of the neighbors".
        # The list structure is: Q, N1, N2... Nk, Q, ...
        # chunk size = n_similar + 1
        
        chunk_size = args.n_similar + 1
        neighbor_ids = []
        
        for i in range(0, len(ordered_ids), chunk_size):
            chunk = ordered_ids[i : i + chunk_size]
            # chunk[0] is query
            neighbors = chunk[1:]
            neighbor_ids.extend(neighbors)
            
        # Get ranks
        neighbor_ids = [nid for nid in neighbor_ids if nid in subset_scores.index]
        if not neighbor_ids:
            print("No neighbors found in score file.")
            continue
            
        neighbor_ranks = subset_scores.loc[neighbor_ids, 'rank'].values
        
        # Plot
        suffix = key.replace("embedding_", "")
        base_save = Path(args.save)
        new_filename = f"{base_save.stem}_{suffix}{base_save.suffix}"
        save_path = base_save.parent / new_filename
        
        plot_rank_distribution(neighbor_ranks, total_objects, key, save_path)

if __name__ == "__main__":
    main()
