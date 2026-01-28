"""
Foundation Models Benchmark (FMB)

Module: fmb.data.utils
Description: Embedding loading and image preprocessing utilities
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from fmb.data.load_display_data import EuclidDESIDataset


def read_object_ids(
    csv_paths: Sequence[Path], limit: Optional[int] = None, verbose: bool = False
) -> List[str]:
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


def collect_samples(
    dataset: EuclidDESIDataset, object_ids: List[str], verbose: bool = False
) -> List[Dict]:
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
        print(
            f"Scanning dataset ({len(dataset)} samples) for {len(target_set)} targets..."
        )

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


def collect_samples_with_index(
    cache_dir: str,
    object_ids: List[str],
    index_map: Dict[str, tuple],
    verbose: bool = False,
) -> List[Dict]:
    """Collect samples efficiently using a pre-computed index."""
    samples = []

    # Group by split to minimize dataset reloads
    by_split = {}
    for oid in object_ids:
        if oid in index_map:
            split, idx = index_map[oid]
            if split not in by_split:
                by_split[split] = []
            by_split[split].append((oid, idx))

    gathered = {}

    for split, items in by_split.items():
        if verbose:
            print(f"Loading split '{split}' for {len(items)} samples...")
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


# --- Embedding Loading Utilities ---


def load_embeddings_file(path: Path) -> List[Dict]:
    """Load raw embeddings list of dicts."""
    print(f"Loading embeddings from {path}...")
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unknown format in {path}")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []


def extract_embedding_matrices(
    records: List[Dict],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """
    Extract tensors for all available modalities.
    Returns map {modality_key: Tensor(N, D)} and list of object_ids.
    """
    import torch.nn.functional as F

    if not records:
        return {}, []

    sample = records[0]
    keys = []

    # Heuristics for modality keys
    if "embedding_images" in sample and "embedding_spectra" in sample:
        # AstroPT/Clip style
        keys = ["embedding_images", "embedding_spectra", "embedding_joint"]
    elif "embedding_hsc" in sample:
        # AION style
        keys = ["embedding_hsc", "embedding_spectrum"]
        if "embedding_hsc_desi" in sample:
            keys.append("embedding_hsc_desi")

    # Fallback/Filtering (only keys present in sample)
    final_keys = []
    # If explicit keys derived, verify them
    if keys:
        final_keys = [k for k in keys if k in sample]
    else:
        final_keys = [k for k in sample.keys() if k.startswith("embedding_")]

    # Ensure joint calculation if missing but components exist?
    # For AstroCLIP/PT, usually 'embedding_joint' is saved.
    # If not, existing scripts handled it specifically.
    # For generic loader, we keep it simple: return what is there.

    print(f"  Detected modalities: {final_keys}")

    vectors_map = {k: [] for k in final_keys}
    oids = []

    for r in records:
        oid = str(r.get("object_id") or r.get("targetid", ""))
        if not oid:
            continue

        current = {}
        valid = True
        for k in final_keys:
            v = r.get(k)
            if v is None:
                valid = False
                break
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            current[k] = v.flatten().float()

        if valid:
            oids.append(oid)
            for k in final_keys:
                vectors_map[k].append(current[k])

    # Stack and Normalize
    matrices = {}
    for k, vlist in vectors_map.items():
        mat = torch.stack(vlist)
        mat = F.normalize(mat, p=2, dim=1)
        matrices[k] = mat

    return matrices, oids
