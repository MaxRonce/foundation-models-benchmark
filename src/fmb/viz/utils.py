"""
Visualization utilities for the FMB project.
Handles sample collection, image preparation, and index loading.
"""

import csv
import math
from pathlib import Path
from typing import Sequence, Dict, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from fmb.data.load_display_data import EuclidDESIDataset
from fmb.paths import load_paths

def load_viz_style():
    """Load matplotlib style from centralized YAML config."""
    import yaml
    import matplotlib.pyplot as plt
    
    style_path = load_paths().repo_root / "src/fmb/configs/viz_style.yaml"
    if style_path.exists():
        with open(style_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Flatten and update rcParams
        def flatten(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
            
        plt.rcParams.update(flatten(config))
        print(f"Loaded visualization style from {style_path}")
    else:
        print(f"Warning: Style config not found at {style_path}, using defaults.")


def load_index(index_path: Path) -> Dict[str, Tuple[str, int]]:
    """Load a CSV index mapping object_id to split and dataset index."""
    mapping: Dict[str, Tuple[str, int]] = {}
    with index_path.open() as handle:
        reader = csv.DictReader(handle)
        required = {"object_id", "split", "index"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Index file must contain columns {required}")
        for row in reader:
            oid = str(row["object_id"]).strip()
            split = row["split"].strip()
            try:
                idx = int(row["index"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid index for object_id={oid}: {row['index']}") from exc
            if oid:
                mapping[oid] = (split, idx)
    return mapping


def collect_samples(
    dataset: EuclidDESIDataset,
    target_ids: Sequence[str],
    verbose: bool = False,
) -> list[dict]:
    """Scan a dataset to collect samples matching a set of object IDs."""
    wanted = {str(oid): None for oid in target_ids}
    collected: list[dict] = []
    remaining = set(wanted.keys())
    iterator = dataset
    if verbose:
        iterator = tqdm(dataset, desc="Scanning dataset", unit="sample")
    for sample in iterator:
        oid = str(sample.get("object_id"))
        if oid in remaining:
            collected.append(sample)
            remaining.remove(oid)
            if not remaining:
                break
    missing = set(wanted.keys()) - {str(s.get("object_id")) for s in collected}
    if missing and verbose:
        print(f"Warning: {len(missing)} object IDs not found: {sorted(list(missing))[:5]}...")
    return collected


def collect_samples_with_index(
    cache_dir: str,
    object_ids: Sequence[str],
    index_map: Dict[str, Tuple[str, int]],
    verbose: bool = False,
) -> list[dict]:
    """Use an index map to efficiently collect samples from multiple splits."""
    from collections import defaultdict

    grouped: Dict[str, list[Tuple[str, int]]] = defaultdict(list)
    for oid in object_ids:
        if oid not in index_map:
            continue
        grouped[index_map[oid][0]].append((oid, index_map[oid][1]))

    samples_by_id: Dict[str, dict] = {}
    for split, entries in grouped.items():
        if verbose:
            print(f"Loading split '{split}' to fetch {len(entries)} samples")
        dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        
        # Determine strict offset if using global indices (heuristic)
        # indices in 'test' split start high (e.g. 84) but dataset is small (e.g. 16)
        # We assume indices are global 0..N across train+test
        min_idx = min(idx for _, idx in entries)
        offset = 0
        if min_idx >= len(dataset):
             # Likely global indices where this split is appended after another
             # Heuristic: if smallest index is >= size, try shifting by the smallest index found in this group
             # or better, just re-scan/match by ID if possible? 
             # No, 'EuclidDESIDataset' doesn't support fast ID lookup.
             # The user's CSV implies: train 0-83, test 84-99. 
             # dataset(test) has relative indices 0-15.
             # So for test, we subtract... something. 
             # We can try to infer the offset from the first item if we assume contiguous blocks
             pass

        # Better approach: 
        # If absolute index is out of bounds, check if subtracting the 'train' size helps?
        # But we don't know the train size easily here without loading it.
        # Let's rely on the fact that indices in the CSV are 0-based GLOBAL.
        # We need to assume standard order: train then test.
        
        # Just-in-time "scan" fallback if index is out of range?
        # Or hardcode the offset logic if split == 'test'?
        
        # Let's use the loaded dataset size to infer modulo or offset
        # IF split is 'test' and indices are > len(dataset), subtract (MinIndex or something)
        # Actually, the most reliable way without external knowledge is to assume 
        # the CSV indices are simply 'global' and we need 'local'.
        
        # We will iterate and try to adjust if determining an offset is obvious.
        # For this specific case (Euclid+DESI), we know Train is first.
        
        # HACK/FIX: If split is 'test' and max_index >= len(dataset), subtract the min_index found in that block?
        # Risk: what if we just request one sample?
        
        # Safer fix: if index out of bounds, try idx % len(dataset)? 
        # No, that's dangerous.
        
        # Let's subtract the offset based on the CSV structure. 
        # The CSV has 'train' 0..83, 'test' 84..99.
        # We can detect the start index of the split from the entries themselves.
        sorted_entries = sorted(entries, key=lambda x: x[1])
        base_offset = 0
        if split == 'test' and sorted_entries[0][1] >= len(dataset):
             # Heuristic: assume strict contiguity and test starts after train
             # We can't know the exact offset without loading train, BUT we can guess
             # if the indices are clustered.
             # Let's try to normalize by the first index of the block? 
             # No, that fails if we randomly sample.
             
             # Fallback: Just search for the object ID if index fails? No, dataset doesn't support search.
             pass

        # Correct logic: The 'index' in the CSV is likely useless for direct access if it's global
        # UNLESS we know the global ordering.
        # Given the previous error "index 95 out of range for split 'test' (size 16)", 
        # it's clear we need to map global 84 -> local 0.
        # If split is test, try subtracting (indices[0] of test?) or just use the local relative index?
        # The CSV is: 84 -> test. we need 84 - 84 = 0.
        
        # Let's assume the CSV *should* have had local indices, or we must correct them.
        # We can compute the shift from the minimum index in the group IF we assume the group covers the start.
        # But we effectively just need to shift 'train' by 0 and 'test' by [size of train].
        # We don't know size of train here.
        
        # Wait, `collect_samples` (the other function) scans by ID. 
        # Maybe we should just fallback to `collect_samples` if indices look wrong?
        
        # Let's implement a robust "try-except" block.
        first_idx = sorted_entries[0][1]
        use_offset = 0
        if sorted_entries[-1][1] >= len(dataset):
            # Indices are out of bounds. Try to shift.
            # Assuming the chunk of indices is contiguous-ish and starts relatively at 0
            if split == 'test':
                 # In this dataset, test comes after train.
                 # The offset is likely `first_idx` if we are grabbing the full set, 
                 # but we might be grabbing a subset.
                 # Let's try to infer offset from the CSV itself?
                 # No, simply use the difference between the requested index and the dataset length? Impossible.
                 
                 # PROPER FIX:
                 # Load 'train' to get its length if split is 'test'.
                 if split == 'test':
                     # This is slightly inefficient but safe
                     ds_train = EuclidDESIDataset(split='train', cache_dir=cache_dir, verbose=False)
                     use_offset = len(ds_train)
        
        for oid, idx in entries:
            local_idx = idx - use_offset
            if local_idx < 0 or local_idx >= len(dataset):
                # Only check fallback if strictly needed
                if use_offset == 0 and split == 'test':
                      # Try "lazy" offset discovery: if 84 is requested and size is 16, 
                      # and we didn't calculate offset yet.
                      # We can try to guess the offset is `idx` if it's the first one? No.
                      pass
                
                if verbose:
                    print(f"Warning: index {local_idx} (orig {idx}) out of range for split '{split}'")
                continue
            
            sample = dataset[local_idx]
            # Verify ID match to be sure
            sample_id = str(sample.get("object_id") or sample.get("targetid"))
            if sample_id != str(oid):
                if verbose:
                    print(f"ID Mismatch at {local_idx}: expected {oid}, got {sample_id}. Fallback scanning...")
                # Fallback: scan for it (slow but correct)
                # This handles scrambled indices
                found = False
                for i, s in enumerate(dataset):
                    if str(s.get("object_id") or s.get("targetid")) == str(oid):
                        sample = s
                        found = True
                        break
                if not found:
                    continue

            samples_by_id[oid] = sample

    missing = [oid for oid in object_ids if oid not in samples_by_id]
    if missing and verbose:
        print(f"Warning: {len(missing)} object IDs missing in index/dataset: {missing[:5]}...")

    return [samples_by_id[oid] for oid in object_ids if oid in samples_by_id]


def prepare_rgb_image(sample: dict) -> np.ndarray:
    """Prepare an RGB image for display from a sample dictionary."""
    rgb = sample.get("rgb_image")
    if rgb is None:
        raise ValueError("Sample missing 'rgb_image'")
    
    if isinstance(rgb, torch.Tensor):
        tensor = rgb.detach().cpu()
        if tensor.dim() == 3:
            # Handle (C, H, W) -> (H, W, C)
            if tensor.shape[0] in (1, 3):
                array = tensor.permute(1, 2, 0).numpy()
            else:
                array = tensor.numpy()
        elif tensor.dim() == 2:
            array = tensor.numpy()
        else:
            raise ValueError(f"Unexpected tensor shape for rgb_image: {tuple(tensor.shape)}")
    else:  # assume numpy array
        array = np.asarray(rgb)
        # Handle (C, H, W) -> (H, W, C) if needed
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
            
    # Normalize if needed (assume float [0, 1] or int [0, 255])
    if array.dtype.kind in ('f', 'u'):
        if array.max() > 1.01:
            array = array.astype(float) / 255.0
            
    array = np.clip(array, 0.0, 1.0)
    return array
