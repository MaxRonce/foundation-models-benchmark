#!/usr/bin/env python3
"""
Script to compute cosine similarity between image and spectrum embeddings
for the same object in AstroPT embeddings.

This helps quantify the alignment between the two modalities for each object.

Usage:
    python -m scratch.compute_cosine_similarity_astropt \
        --input /path/to/astropt_embeddings.pt \
        --output cosine_similarities.csv \
        --histogram cosine_sim_histogram.png
"""
import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_records(path: Path) -> list[dict]:
    """Load embedding records from a .pt file."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns:
        Cosine similarity value between -1 and 1
    """
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    
    # Compute dot product
    return np.dot(vec1_norm, vec2_norm)


def extract_cosine_similarities(records: list[dict]) -> tuple[list[str], list[float]]:
    """
    Extract cosine similarities between image and spectrum embeddings.
    
    Returns:
        - object_ids: List of object IDs
        - similarities: List of cosine similarity values
    """
    object_ids = []
    similarities = []
    
    skipped_missing = 0
    skipped_invalid = 0
    
    for rec in records:
        obj_id = rec.get("object_id", "")
        img_emb = rec.get("embedding_images")
        spec_emb = rec.get("embedding_spectra")
        
        # Skip if either embedding is missing
        if img_emb is None or spec_emb is None:
            skipped_missing += 1
            continue
        
        # Convert to numpy arrays
        if isinstance(img_emb, torch.Tensor):
            img_emb = img_emb.detach().cpu().numpy()
        else:
            img_emb = np.asarray(img_emb)
            
        if isinstance(spec_emb, torch.Tensor):
            spec_emb = spec_emb.detach().cpu().numpy()
        else:
            spec_emb = np.asarray(spec_emb)
        
        # Flatten if needed
        img_emb = img_emb.flatten()
        spec_emb = spec_emb.flatten()
        
        # Check for invalid values
        if np.any(np.isnan(img_emb)) or np.any(np.isnan(spec_emb)):
            skipped_invalid += 1
            continue
        if np.any(np.isinf(img_emb)) or np.any(np.isinf(spec_emb)):
            skipped_invalid += 1
            continue
        
        # Compute cosine similarity
        similarity = compute_cosine_similarity(img_emb, spec_emb)
        
        object_ids.append(obj_id)
        similarities.append(float(similarity))
    
    if skipped_missing > 0:
        print(f"⚠️  Skipped {skipped_missing} records with missing embeddings")
    if skipped_invalid > 0:
        print(f"⚠️  Skipped {skipped_invalid} records with invalid (NaN/Inf) embeddings")
    
    return object_ids, similarities


def save_to_csv(path: Path, object_ids: list[str], similarities: list[float]) -> None:
    """Save cosine similarities to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w") as f:
        f.write("object_id,cosine_similarity\n")
        for obj_id, sim in zip(object_ids, similarities):
            f.write(f"{obj_id},{sim:.6f}\n")
    
    print(f"💾 Saved {len(similarities)} cosine similarities to {path}")


def plot_histogram(similarities: list[float], save_path: Path) -> None:
    """Plot histogram of cosine similarities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    similarities_array = np.array(similarities)
    
    # Plot histogram
    ax.hist(similarities_array, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    
    # Add statistics as vertical lines
    mean_sim = np.mean(similarities_array)
    median_sim = np.median(similarities_array)
    
    ax.axvline(mean_sim, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_sim:.3f}")
    ax.axvline(median_sim, color="green", linestyle="--", linewidth=2, label=f"Median: {median_sim:.3f}")
    
    # Labels and title
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Cosine Similarity between Image and Spectrum Embeddings\n(AstroPT)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = (
        f"N = {len(similarities_array)}\n"
        f"Mean = {mean_sim:.3f}\n"
        f"Median = {median_sim:.3f}\n"
        f"Std = {np.std(similarities_array):.3f}\n"
        f"Min = {np.min(similarities_array):.3f}\n"
        f"Max = {np.max(similarities_array):.3f}"
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print(f"📊 Saved histogram to {save_path}")


def print_statistics(similarities: list[float]) -> None:
    """Print statistical summary of cosine similarities."""
    similarities_array = np.array(similarities)
    
    print("\n" + "=" * 70)
    print("📈 Cosine Similarity Statistics")
    print("=" * 70)
    print(f"  Number of objects:     {len(similarities_array)}")
    print(f"  Mean similarity:       {np.mean(similarities_array):.4f}")
    print(f"  Median similarity:     {np.median(similarities_array):.4f}")
    print(f"  Std deviation:         {np.std(similarities_array):.4f}")
    print(f"  Min similarity:        {np.min(similarities_array):.4f}")
    print(f"  Max similarity:        {np.max(similarities_array):.4f}")
    print()
    print("  Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(similarities_array, p)
        print(f"    {p:3d}%: {val:.4f}")
    print("=" * 70)
    
    # Identify low similarity objects
    threshold = np.percentile(similarities_array, 5)  # Bottom 5%
    low_sim_count = np.sum(similarities_array < threshold)
    print(f"\n💡 {low_sim_count} objects have similarity below 5th percentile ({threshold:.4f})")
    print("   These may indicate misalignment between image and spectrum modalities")


def find_extremes(
    object_ids: list[str],
    similarities: list[float],
    n_top: int = 10,
    n_bottom: int = 10,
) -> None:
    """Print objects with highest and lowest cosine similarities."""
    similarities_array = np.array(similarities)
    indices = np.argsort(similarities_array)
    
    print(f"\n🔝 Top {n_top} most similar objects (image-spectrum alignment):")
    for i in range(min(n_top, len(indices))):
        idx = indices[-(i + 1)]
        print(f"   {i+1:2d}. Object {object_ids[idx]}: {similarities_array[idx]:.4f}")
    
    print(f"\n🔻 Bottom {n_bottom} least similar objects (potential misalignment):")
    for i in range(min(n_bottom, len(indices))):
        idx = indices[i]
        print(f"   {i+1:2d}. Object {object_ids[idx]}: {similarities_array[idx]:.4f}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between image and spectrum embeddings (AstroPT)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to AstroPT embeddings .pt file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path to save cosine similarities",
    )
    parser.add_argument(
        "--histogram",
        default=None,
        help="Optional path to save histogram plot (PNG)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top/bottom objects to display (default: 10)",
    )
    
    args = parser.parse_args(argv)
    
    print("=" * 70)
    print("Cosine Similarity Analysis: Image vs Spectrum Embeddings (AstroPT)")
    print("=" * 70)
    
    # Load embeddings
    print(f"\n[1/4] Loading embeddings from {args.input}...")
    records = load_records(Path(args.input))
    print(f"  Loaded {len(records)} records")
    
    # Compute cosine similarities
    print("\n[2/4] Computing cosine similarities...")
    object_ids, similarities = extract_cosine_similarities(records)
    print(f"  Computed similarities for {len(similarities)} objects")
    
    # Save to CSV
    print("\n[3/4] Saving results...")
    save_to_csv(Path(args.output), object_ids, similarities)
    
    # Generate histogram if requested
    if args.histogram:
        plot_histogram(similarities, Path(args.histogram))
    
    # Print statistics
    print("\n[4/4] Analysis")
    print_statistics(similarities)
    find_extremes(object_ids, similarities, n_top=args.top_n, n_bottom=args.top_n)
    
    print("\n✅ Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
