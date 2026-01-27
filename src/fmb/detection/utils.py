"""
Detection Utilities.
Data loading, processing, and I/O helpers for outlier detection.
"""
import csv
import random
from pathlib import Path
from typing import Sequence, Dict, Tuple, List, Optional, Iterable, Union

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Default keys per model type
KEYS_AION = ["embedding_hsc_desi", "embedding_hsc", "embedding_spectrum"]
KEYS_ASTROPT = ["embedding_images", "embedding_spectra", "embedding_joint"]
KEYS_ASTROCLIP = ["embedding_images", "embedding_spectra", "embedding_joint"]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: Path) -> list[dict]:
    """Load embeddings list/dict from .pt file."""
    print(f"[utils] Loading {path} ...")
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def extract_embeddings(records: Sequence[dict], key: str) -> tuple[np.ndarray, list[str]]:
    """Extract vectors and object_ids for a specific key."""
    vectors: list[np.ndarray] = []
    object_ids: list[str] = []
    
    found_any = False
    for rec in records:
        tensor = rec.get(key)
        if tensor is None:
            continue
        found_any = True
        
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy().copy()
        else:
            array = np.asarray(tensor).copy()
        
        if array.ndim > 1:
            array = array.flatten()
            
        vectors.append(array)
        object_id = rec.get("object_id", "")
        object_ids.append(str(object_id))

    if not found_any:
        return np.array([]), []

    stacked = np.stack(vectors, axis=0)
    return stacked, object_ids


def filter_nonfinite_rows(
    tensor: torch.Tensor,
    object_ids: Sequence[str],
) -> tuple[torch.Tensor, list[str]]:
    """Remove rows containing NaN or Inf."""
    mask = torch.isfinite(tensor).all(dim=1)
    if mask.all():
        return tensor, list(object_ids)
    
    filtered_tensor = tensor[mask]
    filtered_ids = [obj for obj, keep in zip(object_ids, mask.tolist()) if keep]
    dropped = len(object_ids) - len(filtered_ids)
    if dropped > 0:
        print(f"[warn] dropped {dropped} rows containing NaN/inf values")
        
    return filtered_tensor, filtered_ids


def clip_embeddings_by_sigma(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Clip values to mean +/- sigma*std."""
    if sigma is None or sigma <= 0:
        return tensor
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    lower = mean - sigma * std
    upper = mean + sigma * std
    return torch.clamp(tensor, min=lower, max=upper)


def standardize_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize (z-score) the tensor."""
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    standardized = (tensor - mean) / std
    return standardized, mean, std


def apply_pca(tensor: torch.Tensor, n_components: int) -> tuple[torch.Tensor, object]:
    """Reduce dimensions using PCA."""
    from sklearn.decomposition import PCA
    print(f"      [PCA] Fitting PCA with n_components={n_components} on shape {tensor.shape}...")
    data_np = tensor.cpu().numpy()
    pca = PCA(n_components=n_components)
    transformed_np = pca.fit_transform(data_np)
    
    explained = pca.explained_variance_ratio_.sum()
    print(f"      [PCA] Explained variance: {explained:.4f}")
    
    return torch.from_numpy(transformed_np).float(), pca


def compute_sigma(values: np.ndarray) -> np.ndarray:
    """Compute sigma deviation (outlier score) from array of values."""
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mean) / std


def collate_rows(
    object_ids: Sequence[str],
    embedding_key: str,
    log_probs: np.ndarray,
) -> list[dict]:
    """Prepare result rows for CSV."""
    neg_log_probs = -log_probs
    sigma_scores = compute_sigma(neg_log_probs)
    order = np.argsort(-sigma_scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    
    rows: list[dict] = []
    for idx, object_id in enumerate(object_ids):
        rows.append(
            {
                "object_id": object_id,
                "embedding_key": embedding_key,
                "log_prob": float(log_probs[idx]),
                "neg_log_prob": float(neg_log_probs[idx]),
                "anomaly_sigma": float(sigma_scores[idx]),
                "rank": int(ranks[idx]),
            },
        )
    return rows


def save_scores_csv(path: Path, rows: Iterable[dict]) -> None:
    """Save results to CSV."""
    if not rows:
        return
    fieldnames = ["object_id", "embedding_key", "log_prob", "neg_log_prob", "anomaly_sigma", "rank"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
