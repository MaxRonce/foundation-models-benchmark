#!/usr/bin/env python3
"""
Script to detect anomalies based on cosine similarity between image and spectrum embeddings.
Objects with lower cosine similarity (least aligned) are considered anomalies.

Supports both AstroPT and AION embedding formats automatically.

Usage:
    python -m scratch.detect_cosine_anomalies \
        --input /path/to/embeddings.pt \
        --output outliers.csv \
        --threshold-percent 1.0

    python -m scratch.detect_cosine_anomalies \
        --input /path/to/embeddings.pt \
        --output outliers.csv \
        --threshold-count 100
"""
import argparse
import sys
from pathlib import Path
from typing import Sequence, Optional, Tuple, List, Dict

import numpy as np
import torch

# Define known key pairs for image and spectrum embeddings
KNOWN_KEY_PAIRS = [
    ("embedding_images", "embedding_spectra"),  # AstroPT
    ("embedding_hsc", "embedding_spectrum"),    # AION
    ("embedding_hsc_desi", "embedding_spectrum"), # AION alternative
]

def load_records(path: Path) -> List[Dict]:
    """Load embedding records from a .pt file."""
    print(f"Loading embeddings from {path}...")
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Handle case where it might be a dict of lists or a single record
        # But commonly we expect a list of dicts. 
        # If it's a dict with keys like 'object_id': [...], 'embedding': [...]
        # that would be a different format. 
        # For now assuming list of dicts or single dict record.
        return [data]
    
    raise ValueError(f"Unsupported embeddings format: {type(data)}. Expected list or dict.")

def detect_keys(record: Dict) -> Tuple[str, str]:
    """
    Detect which keys to use for image and spectrum embeddings.
    Returns (image_key, spectrum_key).
    """
    keys = set(record.keys())
    
    for img_key, spec_key in KNOWN_KEY_PAIRS:
        if img_key in keys and spec_key in keys:
            print(f"✅ Predicted keys: Image='{img_key}', Spectrum='{spec_key}'")
            return img_key, spec_key
            
    # If no known pair found, try to find keys containing 'image'/'hsc' and 'spectr'
    img_candidates = [k for k in keys if 'image' in k.lower() or 'hsc' in k.lower()]
    spec_candidates = [k for k in keys if 'spectr' in k.lower()]
    
    if len(img_candidates) == 1 and len(spec_candidates) == 1:
        print(f"⚠️  Guessed keys: Image='{img_candidates[0]}', Spectrum='{spec_candidates[0]}'")
        return img_candidates[0], spec_candidates[0]
        
    raise ValueError(
        f"Could not automatically detect embedding keys in {list(keys)}. "
        "Please specify them using --keys image_key,spectrum_key"
    )

def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2
    
    # Compute dot product
    return float(np.dot(vec1_norm, vec2_norm))

def process_embeddings(
    records: List[Dict], 
    img_key: str, 
    spec_key: str
) -> Tuple[List[str], np.ndarray]:
    """
    Extract embeddings and compute similarities.
    Returns list of object_ids and array of similarities.
    """
    object_ids = []
    similarities = []
    
    skipped = 0
    
    for i, rec in enumerate(records):
        obj_id = rec.get("object_id", str(i))
        img_emb = rec.get(img_key)
        spec_emb = rec.get(spec_key)
        
        if img_emb is None or spec_emb is None:
            skipped += 1
            continue
            
        # Convert to numpy
        if isinstance(img_emb, torch.Tensor):
            img_emb = img_emb.detach().cpu().numpy()
        else:
            img_emb = np.asarray(img_emb)
            
        if isinstance(spec_emb, torch.Tensor):
            spec_emb = spec_emb.detach().cpu().numpy()
        else:
            spec_emb = np.asarray(spec_emb)
            
        # Flatten
        img_emb = img_emb.flatten()
        spec_emb = spec_emb.flatten()
        
        # Check invalid
        if np.any(np.isnan(img_emb)) or np.any(np.isnan(spec_emb)):
            skipped += 1
            continue
            
        sim = compute_cosine_similarity(img_emb, spec_emb)
        
        object_ids.append(obj_id)
        similarities.append(sim)
        
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} records due to missing or invalid data/keys.")
        
    return object_ids, np.array(similarities)

def save_anomalies(
    path: Path, 
    anomalies: List[Tuple[str, float]], 
    total_count: int
) -> None:
    """Save anomalies to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w") as f:
        f.write("object_id,cosine_similarity,rank\n")
        for i, (obj_id, sim) in enumerate(anomalies):
            f.write(f"{obj_id},{sim:.6f},{i+1}\n")
            
    print(f"💾 Saved {len(anomalies)} anomalies to {path}")
    if len(anomalies) > 0:
        print(f"   (Top anomaly: {anomalies[0][0]} with similarity {anomalies[0][1]:.6f})")

def main():
    parser = argparse.ArgumentParser(
        description="Detect anomalies based on cosine similarity between image and spectrum embeddings."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to input .pt file")
    parser.add_argument("--output", required=True, type=Path, help="Path to output CSV file")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--threshold-percent", type=float, help="Percentage of anomalies to flag (0-100)")
    group.add_argument("--threshold-count", type=int, help="Fixed number of anomalies to flag")
    
    parser.add_argument("--keys", help="Comma-separated keys for image,spectrum embeddings")
    
    args = parser.parse_args()
    
    # 1. Load Data
    records = load_records(args.input)
    if not records:
        print("❌ Error: No records found in input file.")
        sys.exit(1)
        
    # 2. Determine Keys
    if args.keys:
        try:
            img_key, spec_key = args.keys.split(",")
            img_key, spec_key = img_key.strip(), spec_key.strip()
        except ValueError:
            print("❌ Error: --keys must be in format 'image_key,spectrum_key'")
            sys.exit(1)
    else:
        try:
            img_key, spec_key = detect_keys(records[0])
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
            
    # 3. Compute Similarities
    print("Computing cosine similarities...")
    object_ids, similarities = process_embeddings(records, img_key, spec_key)
    
    if len(object_ids) == 0:
        print("❌ Error: No valid object pairs found.")
        sys.exit(1)
        
    # 4. Identify Anomalies
    n_total = len(similarities)
    
    if args.threshold_count:
        n_anomalies = args.threshold_count
    else:
        n_anomalies = int(n_total * (args.threshold_percent / 100.0))
        n_anomalies = max(1, n_anomalies) # Ensure at least 1
        
    if n_anomalies > n_total:
        print(f"⚠️  Requested {n_anomalies} anomalies but only {n_total} objects exist. returning all.")
        n_anomalies = n_total
        
    # Sort by similarity (ascending - lowest similarity first)
    sorted_indices = np.argsort(similarities)
    anomaly_indices = sorted_indices[:n_anomalies]
    
    anomalies = [
        (object_ids[idx], similarities[idx]) 
        for idx in anomaly_indices
    ]
    
    # 5. Stats
    mean_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    
    print("\n" + "="*50)
    print(f"Analysis Summary:")
    print(f"  Total objects: {n_total}")
    print(f"  Mean similarity: {mean_sim:.4f}")
    print(f"  Min similarity:  {min_sim:.4f}")
    print(f"  Flagging bottom {n_anomalies} objects ({n_anomalies/n_total*100:.2f}%)")
    print("="*50 + "\n")
    
    # 6. Save
    save_anomalies(args.output, anomalies, n_total)
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
