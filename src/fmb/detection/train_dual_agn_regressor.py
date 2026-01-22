"""
Script to train a classifier/regressor to identify Dual AGN candidates from embeddings.
It uses a labeled set of Dual AGN objects to train a neural network on the embedding space.
Produces a scored CSV of all objects in the embedding file.

Usage:
    python -m scratch.train_dual_agn_regressor \
        --embeddings /path/to/embeddings.pt \
        --dual-csv /path/to/dual_agn_catalog.csv \
        --output scratch/outputs/dual_agn_scores.csv \
        --epochs 50
"""
import argparse
import csv
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scratch.detect_outliers import EMBEDDING_KEYS, load_records, stack_embeddings  # type: ignore[import]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_dual_ids(path: Path) -> set[str]:
    dual_ids: set[str] = set()
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "object_id" not in (reader.fieldnames or []):
            raise SystemExit(f"CSV '{path}' must contain an 'object_id' column")
        for row in reader:
            oid = row.get("object_id")
            if oid:
                dual_ids.add(str(oid).strip())
    return dual_ids


class DualAGNRegressor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - simple wrapper
        return self.net(x).squeeze(-1)


def prepare_dataset(
    records: list[dict],
    embedding_key: str,
    dual_ids: set[str],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    embeddings = stack_embeddings(records, embedding_key)
    object_ids = [str(rec.get("object_id", "")) for rec in records]
    labels = np.array([1.0 if oid in dual_ids else 0.0 for oid in object_ids], dtype=np.float32)
    tensor = torch.from_numpy(embeddings).float()
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
    tensor = (tensor - mean) / std
    return tensor, torch.from_numpy(labels), object_ids


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: float,
) -> None:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")


def compute_predictions(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def write_predictions(
    path: Path,
    object_ids: Sequence[str],
    labels: Sequence[float],
    scores: Sequence[float],
) -> None:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    ranking = np.argsort(scores)[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object_id", "label", "score", "rank"])
        for rank, idx in enumerate(ranking, start=1):
            writer.writerow([object_ids[idx], f"{labels[idx]:.0f}", f"{scores[idx]:.6f}", rank])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural regressor on embedding space to score Dual AGN candidates.",
    )
    parser.add_argument("--embeddings", required=True, help="Path to embeddings .pt file")
    parser.add_argument("--dual-csv", required=True, help="CSV containing object_id column for positives")
    parser.add_argument("--embedding-key", choices=EMBEDDING_KEYS, default="embedding_hsc_desi")
    parser.add_argument("--output", required=True, help="Where to write the scored CSV")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction for validation set")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.random_seed)

    embeddings_path = Path(args.embeddings)
    dual_path = Path(args.dual_csv)
    output_path = Path(args.output)

    records = load_records(embeddings_path)
    dual_ids = read_dual_ids(dual_path)
    features, labels, object_ids = prepare_dataset(records, args.embedding_key, dual_ids)

    num_positive = int(labels.sum().item())
    if num_positive == 0:
        raise SystemExit("No dual AGN IDs overlap with the embeddings file; cannot train.")
    pos_weight = float((len(labels) - num_positive) / max(num_positive, 1))
    print(f"Loaded {len(labels)} samples ({num_positive} positives, pos_weight={pos_weight:.2f})")

    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    val_size = int(len(indices) * args.val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_dataset = TensorDataset(features[train_idx], labels[train_idx])
    val_dataset = TensorDataset(features[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = DualAGNRegressor(features.shape[1], args.hidden_dim, args.dropout)
    device = torch.device(args.device)
    train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
    )

    scores = compute_predictions(model, features, device=device)
    write_predictions(output_path, object_ids, labels.numpy(), scores)
    print(f"Saved ranked candidates to {output_path}")


if __name__ == "__main__":
    main()
