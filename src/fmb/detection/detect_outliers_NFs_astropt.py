"""
Script to detect outliers using Normalizing Flows (NFs).
It trains a Normalizing Flow model on the embeddings to learn the density distribution.
Objects with low log-likelihoods are considered anomalies.
Produces a CSV with anomaly scores for each object.

Adapted for AstroPT embeddings (images + spectra).

Usage:
    python -m scratch.detect_outliers_NFs_astropt \
        --input /path/to/astropt_embeddings.pt \
        --output-csv scratch/outputs/anomaly_scores_astropt.csv \
        --epochs 250
"""
import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import normflows as nf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'normflows' package is required. Install it with 'pip install normflows'.",
    ) from exc


EMBEDDING_KEYS = [
    "embedding_joint",
    "embedding_images",
    "embedding_spectra",
]


def load_records(path: Path) -> list[dict]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def extract_embeddings(records: Sequence[dict], key: str) -> tuple[np.ndarray, list[str]]:
    vectors: list[np.ndarray] = []
    object_ids: list[str] = []
    for rec in records:
        if key == "embedding_joint":
            img = rec.get("embedding_images")
            spec = rec.get("embedding_spectra")
            if img is None or spec is None:
                continue
            
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            else:
                img = np.asarray(img)
                
            if isinstance(spec, torch.Tensor):
                spec = spec.detach().cpu().numpy()
            else:
                spec = np.asarray(spec)
                
            array = np.concatenate([img, spec])
        else:
            tensor = rec.get(key)
            if tensor is None:
                continue
            if isinstance(tensor, torch.Tensor):
                array = tensor.detach().cpu().numpy()
            else:
                array = np.asarray(tensor)
        
        vectors.append(array)
        object_id = rec.get("object_id", "")
        object_ids.append(str(object_id))
        
    if not vectors:
        raise ValueError(f"No embeddings found for key '{key}'")
    stacked = np.stack(vectors, axis=0)
    if stacked.ndim != 2:
        raise ValueError(
            f"Expected embeddings for key '{key}' to be 2D (N, D), got shape {stacked.shape}",
        )
    return stacked, object_ids


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_flow(
    dim: int,
    hidden_features: int,
    num_transforms: int,
) -> nn.Module:
    if dim < 2 or dim % 2 != 0:
        raise RuntimeError(
            f"Embedding dimension {dim} is not supported for RealNVP-style coupling (needs an even dimension ≥ 2).",
        )
    base = nf.distributions.base.DiagGaussian(dim)
    flows: list[nn.Module] = []
    cond_dim = dim // 2
    transformed_dim = dim - cond_dim
    for _ in range(num_transforms):
        net = nf.nets.MLP(
            [cond_dim, hidden_features, hidden_features, transformed_dim * 2],
            init_zeros=True,
        )
        flows.append(nf.flows.AffineCouplingBlock(net, scale_map="sigmoid"))
        flows.append(nf.flows.Permute(dim, mode="swap"))
        flows.append(nf.flows.ActNorm((dim,)))
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
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_items = 0
        skipped_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            log_prob = flow.log_prob(batch)
            if not torch.isfinite(log_prob).all():
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                continue
            loss = -log_prob.mean()
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
            total_items += batch.size(0)
        avg_loss = total_loss / max(total_items, 1)
        if skipped_batches:
            print(
                f"[warn] epoch {epoch:03d}: skipped {skipped_batches}/{len(loader)} batches "
                "due to non-finite log_prob or loss",
            )
            if skipped_batches == len(loader):
                print(
                    "[error] all batches failed in this epoch; stopping early. "
                    "Try lowering --lr or decreasing --clip-sigma for stronger clipping.",
                )
                break
        if epoch == 1 or epoch == epochs or (log_every > 0 and epoch % log_every == 0):
            print(f"[{flow.__class__.__name__}] epoch {epoch:03d}/{epochs:03d} | loss={avg_loss:.4f}")
    flow.eval()


def compute_log_probs(flow: nn.Module, data: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return flow.log_prob(data.to(device)).cpu().numpy()


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
    print(f"[warn] dropped {dropped} rows containing NaN/inf values before training")
    if len(filtered_tensor) == 0:
        raise SystemExit("All rows were removed due to non-finite values; cannot train flow.")
    return filtered_tensor, filtered_ids


def clip_embeddings_by_sigma(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma is None or sigma <= 0:
        return tensor
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    lower = mean - sigma * std
    upper = mean + sigma * std
    clipped = torch.clamp(tensor, min=lower, max=upper)
    num_rows_clipped = int((clipped != tensor).any(dim=1).sum().item())
    if num_rows_clipped > 0:
        print(f"[info] clipped {num_rows_clipped} rows outside ±{sigma:.1f}σ to stabilize flow training")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a normalizing-flow density model on embeddings and compute anomaly scores (AstroPT version).",
    )
    parser.add_argument("--input", required=True, help="Path to embeddings .pt file")
    parser.add_argument("--output-csv", required=True, help="CSV path to store per-object anomaly scores")
    parser.add_argument(
        "--embedding-key",
        choices=EMBEDDING_KEYS,
        nargs="+",
        help="Embedding key(s) to analyse. Defaults to every available key in the input file.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs for the flow model")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size during flow training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for flow training")
    parser.add_argument("--num-transforms", type=int, default=6, help="Number of coupling blocks in the flow")
    parser.add_argument("--hidden-features", type=int, default=256, help="Hidden width in coupling networks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device to use")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Log average loss every N epochs (always logs on the first and last epoch)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Gradient clipping value (L2 norm). Set to 0 to disable.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay to apply in the Adam optimizer.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable per-feature standardization before training the flow.",
    )
    parser.add_argument(
        "--clip-sigma",
        type=float,
        default=8.0,
        help="Clip embeddings to mean ± sigma·std to avoid NF instabilities. Set <=0 to disable.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_random_seed(args.random_seed)
    records = load_records(Path(args.input))

    requested_keys = args.embedding_key or EMBEDDING_KEYS
    processed_rows: list[dict[str, str | float | int]] = []
    available_keys: list[str] = []

    for key in requested_keys:
        try:
            embeddings_array, object_ids = extract_embeddings(records, key)
        except ValueError as err:
            print(f"[skip] {err}")
            continue
        available_keys.append(key)
        embeddings_tensor = torch.from_numpy(embeddings_array).float()
        embeddings_tensor, object_ids = filter_nonfinite_rows(embeddings_tensor, object_ids)
        if embeddings_tensor.ndim != 2:
            raise SystemExit(f"Expected embeddings for '{key}' to have shape (N, D), got {embeddings_tensor.shape}")
        if len(embeddings_tensor) < 2:
            raise SystemExit(f"Need at least 2 samples for '{key}' to train a flow; got {len(embeddings_tensor)}")
        if not args.no_standardize:
            embeddings_tensor, mean, std = standardize_tensor(embeddings_tensor)
            print(
                f"[{key}] standardized embeddings (feature mean≈{mean.mean():.4f}, feature std≈{std.mean():.4f})",
            )
        embeddings_tensor = clip_embeddings_by_sigma(embeddings_tensor, args.clip_sigma)
        embeddings_tensor, object_ids = filter_nonfinite_rows(embeddings_tensor, object_ids)
        if len(embeddings_tensor) < 2:
            raise SystemExit(f"Need at least 2 samples for '{key}' after cleaning; got {len(embeddings_tensor)}")
        flow = build_flow(
            dim=embeddings_tensor.shape[1],
            hidden_features=args.hidden_features,
            num_transforms=args.num_transforms,
        )
        print(
            f"[{key}] training flow with dim={embeddings_tensor.shape[1]}, "
            f"num_transforms={args.num_transforms}, hidden={args.hidden_features}",
        )
        train_flow(
            flow=flow,
            data=embeddings_tensor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=torch.device(args.device),
            log_every=args.log_every,
            grad_clip=None if args.grad_clip is not None and args.grad_clip <= 0 else args.grad_clip,
            weight_decay=args.weight_decay,
        )
        log_probs = compute_log_probs(flow, embeddings_tensor, torch.device(args.device))
        rows = collate_rows(object_ids, key, log_probs)
        processed_rows.extend(rows)
        sigma_values = [row["anomaly_sigma"] for row in rows]
        print(
            f"[{key}] anomaly sigma stats | min={min(sigma_values):.3f} "
            f"median={np.median(sigma_values):.3f} max={max(sigma_values):.3f}",
        )

    if not processed_rows:
        raise SystemExit("No embeddings were processed; double-check the requested embedding keys.")

    save_scores_csv(Path(args.output_csv), processed_rows)
    num_objects = len({row["object_id"] for row in processed_rows})
    formatted_keys = ", ".join(available_keys)
    print(
        f"Saved anomaly scores for {len(processed_rows)} rows ({num_objects} objects × {formatted_keys}) "
        f"to '{args.output_csv}'.",
    )


if __name__ == "__main__":
    main()
