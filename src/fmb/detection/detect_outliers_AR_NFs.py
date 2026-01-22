#!/usr/bin/env python3
"""
Script to detect outliers using Autoregressive Normalizing Flows (Masked Autoregressive Flows).
Supports detecting anomalies from multiple embedding sources (AION, AstroPT, AstroCLIP).
Generates separate anomaly score CSVs for each source.

Usage:
    python -m scratch.detect_outliers_AR_NFs \\
        --aion-embeddings /path/to/aion.pt \\
        --output-prefix outputs/anomaly_scores
"""
import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, Sequence, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import normflows as nf
except ImportError as exc:
    raise SystemExit(
        "The 'normflows' package is required. Install it with 'pip install normflows'.",
    ) from exc


# Default keys per model type
KEYS_AION = ["embedding_hsc_desi", "embedding_hsc", "embedding_spectrum"]
KEYS_ASTROPT = ["embedding_images", "embedding_spectra", "embedding_joint"]
KEYS_ASTROCLIP = ["embedding_images", "embedding_spectra", "embedding_joint"]


def load_records(path: Path) -> list[dict]:
    print(f"[info] Loading {path} ...")
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def extract_embeddings(records: Sequence[dict], key: str) -> tuple[np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    object_ids: list[str] = []
    
    # Pre-check if key exists in first record to avoid iteration if missing
    if records and key not in records[0]:
         # Some files might have mixed records, but usually keys are consistent.
         # We'll scan anyway, but this effectively skips if key is totally absent.
         pass

    found_any = False
    for rec in records:
        tensor = rec.get(key)
        if tensor is None:
            continue
        found_any = True
        
        if isinstance(tensor, torch.Tensor):
            # Force a CPU copy and a numpy copy to break any shared memory storage
            array = tensor.detach().cpu().numpy().copy()
        else:
            array = np.asarray(tensor).copy()
        
        # Flatten if needed (e.g. 1, D -> D)
        if array.ndim > 1:
            array = array.flatten()
            
        vectors.append(array)
        object_id = rec.get("object_id", "")
        object_ids.append(str(object_id))

    if not found_any:
        # Return empty if key not found, caller decides if it's an error
        return np.array([]), []

    stacked = np.stack(vectors, axis=0)
    if stacked.ndim != 2:
         # If we somehow got here with weird shapes
         raise ValueError(f"Embeddings for key '{key}' have inconsistent or wrong shapes: {stacked.shape}")
         
    return stacked, object_ids


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ar_flow(
    dim: int,
    hidden_features: int,
    num_transforms: int,
) -> nn.Module:
    base = nf.distributions.base.DiagGaussian(dim)
    flows: list[nn.Module] = []
    
    for _ in range(num_transforms):
        # Masked Affine Autoregressive Flow
        # features=dim, hidden_features=hidden_features
        flows.append(nf.flows.MaskedAffineAutoregressive(features=dim, hidden_features=hidden_features))
        # Permutation to mix dimensions
        flows.append(nf.flows.Permute(dim, mode="swap"))
        # ActNorm for stability
        flows.append(nf.flows.ActNorm(dim))
        
    return nf.NormalizingFlow(base, flows)


def train_flow(
    flow: nn.Module,
    data: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
    grad_clip: float | None = None,
    weight_decay: float = 0.0,
) -> None:
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
    )
    flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    flow.train()
    
    from tqdm import tqdm
    
    # Epoch loop with progress bar
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Epochs", unit="epoch")
    for epoch in epoch_pbar:
        total_loss = 0.0
        total_items = 0
        skipped_batches = 0
        
        # Batch loop with progress bar
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, unit="batch")
        for (batch,) in batch_pbar:
            batch = batch.to(device)
            loss = flow.forward_kld(batch)
            
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                continue
                
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
                
            optimizer.step()
            
            # Update batch pbar with current batch loss
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            total_loss += loss.item() * batch.size(0)
            total_items += batch.size(0)
            
        avg_loss = total_loss / max(total_items, 1) if total_items > 0 else float("nan")
        
        # Update epoch pbar description with latest average loss
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
        
        if not np.isfinite(avg_loss):
             tqdm.write(f"[error] Epoch {epoch}: Loss is NaN/Inf. Stopping training for this key.")
             break

        if skipped_batches == len(loader):
             tqdm.write(f"[warn] Epoch {epoch}: All batches skipped (NaN/Inf).")
             break

        if log_every > 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            # Write to stdout as well to keep permanent record
            # Use write() to avoid interfering with progress bars
            tqdm.write(f"[{flow.__class__.__name__}] epoch {epoch:03d}/{epochs:03d} | loss={avg_loss:.4f}")
            
    flow.eval()


def compute_log_probs(flow: nn.Module, data: torch.Tensor, device: torch.device) -> np.ndarray:
    loader = DataLoader(TensorDataset(data), batch_size=2048, shuffle=False)
    log_probs_list = []
    
    flow.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            lp = flow.log_prob(batch)
            log_probs_list.append(lp.cpu().numpy())
            
    return np.concatenate(log_probs_list)


def compute_sigma(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mean) / std


def filter_nonfinite_rows(
    tensor: torch.Tensor,
    object_ids: Sequence[str],
) -> tuple[torch.Tensor, list[str]]:
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
    if sigma is None or sigma <= 0:
        return tensor
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    lower = mean - sigma * std
    upper = mean + sigma * std
    clipped = torch.clamp(tensor, min=lower, max=upper)
    return clipped


def collate_rows(
    object_ids: Sequence[str],
    embedding_key: str,
    log_probs: np.ndarray,
) -> list[dict[str, str | float | int]]:
    neg_log_probs = -log_probs
    sigma_scores = compute_sigma(neg_log_probs)
    order = np.argsort(-sigma_scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    
    rows: list[dict[str, str | float | int]] = []
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


def save_scores_csv(path: Path, rows: Iterable[dict[str, str | float | int]]) -> None:
    fieldnames = ["object_id", "embedding_key", "log_prob", "neg_log_prob", "anomaly_sigma", "rank"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def standardize_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    standardized = (tensor - mean) / std
    return standardized, mean, std


def process_file_and_keys(
    fpath: Path,
    possible_keys: List[str],
    args: argparse.Namespace,
    model_name: str
) -> List[dict]:
    
    if not fpath.exists():
        print(f"[error] File not found: {fpath}")
        return []

    records = load_records(fpath)
    all_rows = []
    
    # Identify which keys actually exist
    valid_keys = []
    # Peek at first record
    if records:
        first_rec = records[0]
        for k in possible_keys:
            if k in first_rec:
                valid_keys.append(k)
    
    if not valid_keys:
        print(f"[warn] None of the expected keys {possible_keys} found in {fpath}. Skipping.")
        return []
        
    extracted_data = {}
    for key in valid_keys:
        print(f"      Extracting key: '{key}' ...")
        embeddings_array, object_ids = extract_embeddings(records, key)
        extracted_data[key] = (embeddings_array, object_ids)
        
    # Free the huge records object from memory immediately
    del records
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[debug] Memory after releasing records: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    for key in valid_keys:
        print(f"\n---> Processing {model_name} key: '{key}'")
        embeddings_array, object_ids = extracted_data[key]
        
        # Free from dictionary to allow GC if needed, though we iterate
        # extracted_data[key] = None 

        
        if len(embeddings_array) == 0:
            print(f"[warn] No embeddings extracted for {key}")
            continue

        embeddings_tensor = torch.from_numpy(embeddings_array).float()
        embeddings_tensor, object_ids = filter_nonfinite_rows(embeddings_tensor, object_ids)
        
        if len(embeddings_tensor) < 2:
            print(f"[warn] Not enough data for {key}")
            continue
            
        if not args.no_standardize:
            embeddings_tensor, _, _ = standardize_tensor(embeddings_tensor)
            
        embeddings_tensor = clip_embeddings_by_sigma(embeddings_tensor, args.clip_sigma)
        
        # Build Flow
        dim = embeddings_tensor.shape[1]
        flow = build_ar_flow(dim, args.hidden_features, args.num_transforms)
        
        print(f"[{key}] Training AR Flow (MAF) dim={dim}, hidden={args.hidden_features}...")
        train_flow(
            flow=flow,
            data=embeddings_tensor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=torch.device(args.device),
            log_every=args.log_every,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
        )
        
        log_probs = compute_log_probs(flow, embeddings_tensor, torch.device(args.device))
        rows = collate_rows(object_ids, key, log_probs)
        all_rows.extend(rows)
        
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Autoregressive Flows (MAF) for anomaly detection on multiple embeddings."
    )
    
    # Inputs
    parser.add_argument("--aion-embeddings", type=str, help="Path to AION embeddings file")
    parser.add_argument("--astropt-embeddings", type=str, help="Path to AstroPT embeddings file")
    parser.add_argument("--astroclip-embeddings", type=str, help="Path to AstroCLIP embeddings file")
    
    # Output
    parser.add_argument("--output-prefix", default="outputs/anomaly_scores", 
                        help="Prefix for output CSVs (e.g. 'outputs/scores' -> 'outputs/scores_aion.csv')")
    parser.add_argument("--output-csv", help="Legacy argument, ignored if inputs provided separately, or used as prefix fallback.")

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-transforms", type=int, default=6, help="Number of MAF steps")
    parser.add_argument("--hidden-features", type=int, default=256, help="Hidden width")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--clip-sigma", type=float, default=8.0)
    parser.add_argument("--pca-components", type=int, default=0, 
                        help="Reduce dimensions with PCA to this many components (0 = disabled). Recommended for high-dim data.")

    return parser.parse_args()

def apply_pca(tensor: torch.Tensor, n_components: int) -> tuple[torch.Tensor, object]:
    from sklearn.decomposition import PCA
    print(f"      [PCA] Fitting PCA with n_components={n_components} on shape {tensor.shape}...")
    # sklearn PCA expects numpy
    data_np = tensor.cpu().numpy()
    pca = PCA(n_components=n_components)
    transformed_np = pca.fit_transform(data_np)
    
    explained = pca.explained_variance_ratio_.sum()
    print(f"      [PCA] Explained variance: {explained:.4f}")
    
    return torch.from_numpy(transformed_np).float(), pca

def process_file_and_keys(
    fpath: Path,
    possible_keys: List[str],
    args: argparse.Namespace,
    model_name: str
) -> List[dict]:
    
    if not fpath.exists():
        print(f"[error] File not found: {fpath}")
        return []

    records = load_records(fpath)
    all_rows = []
    
    # --- ASTROPT SPECIAL HANDLING ---
    # If we are processing AstroPT and 'embedding_joint' is requested/possible but not in records,
    # try to synthesize it from images and spectra.
    if model_name == "astropt" and records:
        has_images = "embedding_images" in records[0]
        has_spectra = "embedding_spectra" in records[0]
        has_joint = "embedding_joint" in records[0]
        
        if has_images and has_spectra and not has_joint:
            print("      [info] AstroPT: 'embedding_joint' not found. Synthesizing from images + spectra...")
            count_syn = 0
            for rec in records:
                img = rec.get("embedding_images")
                spec = rec.get("embedding_spectra")
                if img is not None and spec is not None:
                    # Ensure numpy or torch
                    if isinstance(img, torch.Tensor): img = img.cpu().numpy()
                    if isinstance(spec, torch.Tensor): spec = spec.cpu().numpy()
                    
                    # Flatten if necessary (though usually 1D here)
                    if img.ndim > 1: img = img.flatten()
                    if spec.ndim > 1: spec = spec.flatten()
                    
                    joint = np.concatenate([img, spec])
                    rec["embedding_joint"] = joint
                    count_syn += 1
            print(f"      [info] Synthesized joint embeddings for {count_syn} records.")

    # Identify which keys actually exist
    valid_keys = []
    # Peek at first record
    if records:
        first_rec = records[0]
        for k in possible_keys:
            if k in first_rec:
                valid_keys.append(k)
    
    if not valid_keys:
        print(f"[warn] None of the expected keys {possible_keys} found in {fpath}. Skipping.")
        return []
        
    extracted_data = {}
    for key in valid_keys:
        print(f"      Extracting key: '{key}' ...")
        embeddings_array, object_ids = extract_embeddings(records, key)
        extracted_data[key] = (embeddings_array, object_ids)
        
    # Free the huge records object from memory immediately
    del records
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[debug] Memory after releasing records: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    for key in valid_keys:
        print(f"\n---> Processing {model_name} key: '{key}'")
        embeddings_array, object_ids = extracted_data[key]
        
        if len(embeddings_array) == 0:
            print(f"[warn] No embeddings extracted for {key}")
            continue

        embeddings_tensor = torch.from_numpy(embeddings_array).float()
        embeddings_tensor, object_ids = filter_nonfinite_rows(embeddings_tensor, object_ids)
        
        if len(embeddings_tensor) < 2:
            print(f"[warn] Not enough data for {key}")
            continue
            
        if not args.no_standardize:
            embeddings_tensor, _, _ = standardize_tensor(embeddings_tensor)
            
        embeddings_tensor = clip_embeddings_by_sigma(embeddings_tensor, args.clip_sigma)
        
        # Apply PCA if requested
        if args.pca_components > 0:
            if args.pca_components >= embeddings_tensor.shape[1]:
                 print(f"[info] Skipping PCA: requested {args.pca_components} >= current dim {embeddings_tensor.shape[1]}")
            else:
                 embeddings_tensor, _ = apply_pca(embeddings_tensor, args.pca_components)
                 # Re-standardize after PCA? Usually PCA output is centered but variances differ.
                 # Normalizing Flows often like unit variance, so let's re-standardize
                 embeddings_tensor, _, _ = standardize_tensor(embeddings_tensor)
        
        # Build Flow
        dim = embeddings_tensor.shape[1]
        flow = build_ar_flow(dim, args.hidden_features, args.num_transforms)
        
        print(f"[{key}] Training AR Flow (MAF) dim={dim}, hidden={args.hidden_features}...")
        train_flow(
            flow=flow,
            data=embeddings_tensor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=torch.device(args.device),
            log_every=args.log_every,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
        )
        
        log_probs = compute_log_probs(flow, embeddings_tensor, torch.device(args.device))
        rows = collate_rows(object_ids, key, log_probs)
        all_rows.extend(rows)
        
    return all_rows


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_seed)
    
    tasks = []
    if args.aion_embeddings:
        tasks.append(("aion", args.aion_embeddings, KEYS_AION))
    if args.astropt_embeddings:
        tasks.append(("astropt", args.astropt_embeddings, KEYS_ASTROPT))
    if args.astroclip_embeddings:
        tasks.append(("astroclip", args.astroclip_embeddings, KEYS_ASTROCLIP))
        
    if not tasks and args.output_csv:
        print("[error] No embedding inputs provided. Use --aion-embeddings, --astropt-embeddings, etc.")
        return

    for name, path_str, keys_list in tasks:
        print(f"\\n=== Processing {name.upper()} Embeddings ===")
        rows = process_file_and_keys(Path(path_str), keys_list, args, name)
        
        if rows:
            out_path = Path(f"{args.output_prefix}_{name}.csv")
            save_scores_csv(out_path, rows)
            print(f"[success] Saved {len(rows)} scores to {out_path}")
        else:
            print(f"[warn] No results generated for {name}.")

if __name__ == "__main__":
    main()
