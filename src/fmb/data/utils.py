"""
Data Utilities for FMB.
Helpers for reading object IDs and collecting samples from the dataset.
"""
import csv
from pathlib import Path
from typing import List, Dict, Optional, Sequence
import torch
import numpy as np
from tqdm import tqdm

from fmb.data.load_display_data import EuclidDESIDataset

def read_object_ids(csv_paths: Sequence[Path], limit: Optional[int] = None, verbose: bool = False) -> List[str]:
    """Read object IDs from a list of CSV files."""
    ids = []
    for p in csv_paths:
        if not p.exists():
            print(f"[warn] CSV not found: {p}")
            continue
        try:
            with open(p, "r") as f:
                reader = csv.DictReader(f)
                if "object_id" not in reader.fieldnames:
                     # Check if it's single column no header or different name?
                     # For now strict.
                     print(f"[warn] 'object_id' column missing in {p}")
                     continue
                for row in reader:
                    ids.append(str(row["object_id"]))
                    if limit and len(ids) >= limit:
                        break
        except Exception as e:
            print(f"[error] Failed to read {p}: {e}")
            
        if limit and len(ids) >= limit:
            break
            
    if verbose:
        print(f"Loaded {len(ids)} object IDs.")
    return ids

def load_index(path: Path) -> Dict[str, tuple]:
    """Load index CSV mapping object_id -> (split, index)."""
    mapping = {}
    print(f"Loading index from {path}...")
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = str(row["object_id"])
            split = row["split"]
            idx = int(row["index"])
            mapping[oid] = (split, idx)
    return mapping

def collect_samples(dataset: EuclidDESIDataset, object_ids: List[str], verbose: bool = False) -> List[Dict]:
    """
    Collect samples from dataset matching object_ids.
    WARNING: Linear scan if no index is used. Very slow for large datasets.
    """
    target_set = set(object_ids)
    samples = []
    
    # We scan the dataset
    # Optim: if dataset provides a way to get ID quickly without loading full sample?
    # EuclidDESIDataset loads full sample.
    # But usually we call this on a small list of query IDs.
    
    if verbose:
        print(f"Scanning dataset ({len(dataset)} samples) for {len(target_set)} targets...")
        
    found_map = {}
    
    # Heuristic: iterate full dataset once
    # Or rely on dataset having an internal index? It doesn't.
    
    for i in tqdm(range(len(dataset)), desc="Scanning Dataset", disable=not verbose):
        # We need to access just the ID if possible to be fast
        # But dataset[i] does heavy loading (images etc). 
        # This function is inherently slow without an auxiliary index.
        # But we must support it as fallback.
        
        # Accessing raw HF dataset might be faster for just ID check?
        # self.dataset[i]['object_id']
        raw_sample = dataset.dataset[i]
        oid = str(raw_sample.get("object_id") or raw_sample.get("targetid"))
        
        if oid in target_set:
            # Now load full processed sample
            found_map[oid] = dataset[i]
            if len(found_map) == len(target_set):
                break
                
    # preserve order
    for oid in object_ids:
        if oid in found_map:
            samples.append(found_map[oid])
            
    return samples

def collect_samples_with_index(cache_dir: str, object_ids: List[str], index_map: Dict[str, tuple], verbose: bool = False) -> List[Dict]:
    """Collect samples efficiently using a pre-computed index."""
    samples = []
    
    # Group by split to minimize dataset reloads
    by_split = {}
    for oid in object_ids:
        if oid in index_map:
            split, idx = index_map[oid]
            if split not in by_split: by_split[split] = []
            by_split[split].append((oid, idx))
            
    gathered = {}
    
    for split, items in by_split.items():
        if verbose: print(f"Loading split '{split}' for {len(items)} samples...")
        ds = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        for oid, idx in items:
            gathered[oid] = ds[idx]
            
    # preserve order
    for oid in object_ids:
        if oid in gathered:
            samples.append(gathered[oid])
            
    return samples

def prepare_rgb_image(sample: Dict) -> np.ndarray:
    """Extract RGB image from sample as (H, W, 3/1) numpy array."""
    # From tensor (C, H, W) to (H, W, C) numpy
    img_t = sample.get("rgb_image")
    if img_t is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
        
    img_np = img_t.permute(1, 2, 0).cpu().numpy()
    # Clip 0..1
    img_np = np.clip(img_np, 0, 1)
    return img_np
