"""
Similarity Search & Neighbor Analysis.
Common logic for finding similar anomalies and analyzing their properties.
"""
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Sequence
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fmb.paths import load_paths
from fmb.data.utils import (
    read_object_ids, 
    collect_samples, 
    collect_samples_with_index, 
    load_index,
    load_embeddings_file,
    extract_embedding_matrices
)
from fmb.viz.similarity import plot_vertical_panels
from fmb.data.load_display_data import EuclidDESIDataset

def find_nearest_neighbors(
    query_ids: List[str], 
    all_ids: List[str], 
    emb_matrix: torch.Tensor, 
    k: int = 5
) -> List[str]:
    """
    Returns flat list of [Q1, N1..Nk, Q2, N1..Nk].
    """
    id_map = {oid: i for i, oid in enumerate(all_ids)}
    valid_q_idxs = []
    valid_q_ids = []
    
    for q in query_ids:
        if q in id_map:
            valid_q_idxs.append(id_map[q])
            valid_q_ids.append(q)
        else:
            print(f"[warn] Query ID {q} not found in embeddings.")

    if not valid_q_idxs:
        return []

    q_vecs = emb_matrix[valid_q_idxs] # (M, D)
    
    # Sim (M, N)
    sim = torch.mm(q_vecs, emb_matrix.t())
    
    # Top k+1+padding (to skip self)
    search_k = min(len(all_ids), k + 10)
    top_v, top_i = torch.topk(sim, k=search_k, dim=1)
    
    results = []
    for i, qid in enumerate(valid_q_ids):
        # Always put Query first
        batch = [qid]
        
        candidates = top_i[i].tolist()
        found = 0
        for idx in candidates:
            # Skip self
            cand_id = all_ids[idx]
            if cand_id == qid:
                continue
            batch.append(cand_id)
            found += 1
            if found == k:
                break
        
        results.extend(batch)
        
    return results

def visualize_similarity(
    query_ids: List[str],
    tasks: List[Tuple[str, Path]],
    n_similar: int,
    output_path: Path,
    cache_dir: str
):
    """Run similarity search and visualize results for multiple models."""
    if not tasks:
        print("No tasks provided.")
        return

    all_annotated = []
    row_lbls = []
    
    # Pre-check all paths
    valid_tasks = []
    for m, p in tasks:
        if not p.exists():
            print(f"[warn] Embedding file not found for {m}: {p}")
        else:
            valid_tasks.append((m, p))
            
    if not valid_tasks:
        print("No valid embedding files found.")
        return
        
    # We load dataset once to save time if possible?
    # Actually collect_samples re-loads, but we can instantiation dataset once.
    ds = EuclidDESIDataset(split="all", cache_dir=cache_dir)
    
    for model_name, emb_path in valid_tasks:
        print(f"\nProcessing Model: {model_name}...")
        records = load_embeddings_file(emb_path)
        matrices, all_ids = extract_embedding_matrices(records)
        
        for mod_key, mat in matrices.items():
            mod_pretty = mod_key.replace("embedding_", "").capitalize()
            # Special case renaming
            if mod_key == "embedding_hsc": mod_pretty = "Image"
            if mod_key == "embedding_spectra": mod_pretty = "Spectrum"
            if mod_key == "embedding_joint": mod_pretty = "Joint"
            
            print(f"  Searching in modality: {mod_pretty} ({mod_key})")
            
            ordered_ids = find_nearest_neighbors(query_ids, all_ids, mat, k=n_similar)
            if not ordered_ids: 
                print("    No neighbors found.")
                continue
            
            samples = collect_samples(ds, ordered_ids, verbose=False)
            
            # Map back to ordered list (handle missing)
            s_map = {str(s.get("object_id") or s.get("targetid")): s for s in samples}
            
            for i, oid in enumerate(ordered_ids):
                # We need to construct rows.
                # ordered_ids contains [Q1, N1..Nk, Q2, N1..Nk...]
                pass

            # Rename for grid
            cols = n_similar + 1
            
            # Should have exactly len(ordered_ids) samples
            current_batch = []
            for oid in ordered_ids:
                if oid in s_map:
                    current_batch.append(s_map[oid])
                else:
                    current_batch.append({"object_id": oid})
            
            annotated_batch = []
            for i, s in enumerate(current_batch):
                new_s = s.copy()
                orig = str(new_s.get("object_id",""))
                col_idx = i % cols
                if col_idx == 0:
                    prefix = "[QUERY]"
                else:
                    prefix = f"[#{col_idx}]"
                new_s["object_id"] = f"{prefix} {orig}"
                annotated_batch.append(new_s)
            
            all_annotated.extend(annotated_batch)
            
            # Add label for each query row
            num_queries = len(ordered_ids) // cols
            lbl = f"{model_name}\n{mod_pretty}"
            for _ in range(num_queries):
                row_lbls.append(lbl)

    if not all_annotated:
        print("No results to visualize.")
        return
        
    print(f"\nGenerating combined visualization at {output_path}...")
    plot_vertical_panels(
        all_annotated, 
        cols=n_similar+1, 
        save_path=output_path, 
        show=False,
        row_labels=row_lbls
    )

def analyze_neighbor_ranks(
    query_ids: List[str],
    tasks: List[Tuple[str, Path, Path]],
    n_similar: int,
    output_dir: Path
):
    """Analyze rank distribution of neighbors for multiple models and combine into one plot."""
    if not tasks:
        print("No tasks provided.")
        return
        
    # Pre-check
    valid_tasks = []
    for m, ep, sp in tasks:
        if not ep.exists():
            print(f"[warn] Embeddings not found for {m}: {ep}")
            continue
        if not sp.exists():
            print(f"[warn] Scores not found for {m}: {sp}")
            continue
        valid_tasks.append((m, ep, sp))
        
    if not valid_tasks:
        print("No valid tasks found.")
        return

    plot_data = []

    for model_name, emb_path, scores_path in valid_tasks:
        print(f"\nProcessing Model: {model_name}...")
        records = load_embeddings_file(emb_path)
        matrices, all_ids = extract_embedding_matrices(records)
        
        print(f"  Loading scores from {scores_path}...")
        scores_df = pd.read_csv(scores_path)
        scores_df["object_id"] = scores_df["object_id"].astype(str)
        
        for mod_key, mat in matrices.items():
            mod_pretty = mod_key.replace("embedding_", "").capitalize()
            print(f"  Analyzing neighbors for {mod_pretty} ({mod_key})...")
            
            # Match score key logic
            target_key = mod_key
            if target_key not in scores_df["embedding_key"].values:
                if "hsc" in target_key and "embedding_hsc" in scores_df["embedding_key"].values: target_key = "embedding_hsc"
                if "spectra" in target_key and "embedding_spectrum" in scores_df["embedding_key"].values: target_key = "embedding_spectrum"
                if "joint" in target_key and "embedding_hsc_desi" in scores_df["embedding_key"].values: target_key = "embedding_hsc_desi"
                
            sub_scores = scores_df[scores_df["embedding_key"] == target_key].set_index("object_id")
            if sub_scores.empty:
                continue
                
            ordered_ids = find_nearest_neighbors(query_ids, all_ids, mat, k=n_similar)
            if not ordered_ids: continue
            
            neighbor_ranks = []
            for i in range(0, len(ordered_ids), n_similar + 1):
                block = ordered_ids[i:i+n_similar+1]
                neighbors = block[1:] 
                for nid in neighbors:
                    if nid in sub_scores.index:
                        r = sub_scores.loc[nid, "rank"]
                        neighbor_ranks.append(r)
            
            if not neighbor_ranks:
                continue
                
            ranks = np.array(neighbor_ranks)
            total_obj = len(sub_scores)
            percentiles = (ranks / total_obj) * 100.0
            
            plot_data.append({
                "title": f"{model_name}\n{mod_pretty}",
                "data": percentiles,
                "count": len(ranks)
            })

    if not plot_data:
        print("No data collected for plotting.")
        return

    # Create Combined Plot
    n_plots = len(plot_data)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows + 1), constrained_layout=True)
    axes = axes.flatten()
    
    # Global title?
    fig.suptitle(f"Neighbor Anomaly Ranks (Query N={len(query_ids)})", fontsize=16)

    for i, item in enumerate(plot_data):
        ax = axes[i]
        data = item["data"]
        
        ax.hist(data, bins=50, range=(0, 100), color='#1f77b4', edgecolor='black', alpha=0.7)
        ax.set_title(item["title"], fontsize=10)
        ax.set_xlabel("Anomaly Percentile")
        if i % cols == 0:
            ax.set_ylabel("Count")
            
        # Stats
        top1 = np.mean(data <= 1.0) * 100
        stats = f"N={item['count']}\nTop1%={top1:.1f}%"
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, ha='right', va='top', 
                bbox=dict(facecolor='white', alpha=0.9, pad=2), fontsize=8)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    out_file = output_dir / "neighbor_ranks_combined.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"\n[success] Combined plot saved to: {out_file}")

# --- CLI Entry Points ---

def main_search(argv: List[str] = None):
    # Argparse wrapper needed for cleaner integration maybe?
    # Or just use args passed from Typer
    pass # Implemented in CLI directly via calls to visualize_similarity

def main(argv: List[str] = None):
    # This main handles both? Or we expose separate functions.
    pass
