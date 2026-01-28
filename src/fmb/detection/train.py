"""
Foundation Models Benchmark (FMB)

Module: fmb.detection.train
Description: Training script for Normalizing Flows
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_flow_model(
    flow: nn.Module,
    data: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    log_every: int = 25,
    grad_clip: float = 5.0,
    weight_decay: float = 1e-5,
) -> None:
    """Train the flow model on the provided data."""
    device_obj = torch.device(device)
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
    )

    flow.to(device_obj)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)
    flow.train()

    # Epoch loop with progress bar
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Epochs", unit="epoch")

    for epoch in epoch_pbar:
        total_loss = 0.0
        total_items = 0
        skipped_batches = 0

        # Batch loop - hide progress for individual batches unless very slow
        tqdm(loader, desc=f"Epoch {epoch}", leave=False, unit="batch", disable=True)

        for (batch,) in loader:
            batch = batch.to(device_obj)

            # Loss is negative log likelihood (or forward KLD for some flows)
            try:
                # Try standard log_prob first (for RealNVP)
                # Note: normflows uses -log_prob as loss usually
                log_prob = flow.log_prob(batch)
                loss = -log_prob.mean()
            except Exception:
                # Fallback for some wrappers (like MAF in older versions maybe?)
                # standardized wrapper usually ensures log_prob exists
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

            total_loss += loss.item() * batch.size(0)
            total_items += batch.size(0)

        avg_loss = total_loss / max(total_items, 1) if total_items > 0 else float("nan")

        # Update epoch pbar
        epoch_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        if not np.isfinite(avg_loss):
            tqdm.write(
                f"[error] Epoch {epoch}: Loss is NaN/Inf. Stopping training for this key."
            )
            break

        if skipped_batches == len(loader):
            tqdm.write(f"[warn] Epoch {epoch}: All batches skipped (NaN/Inf).")
            break

        if log_every > 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            tqdm.write(
                f"[{flow.__class__.__name__}] epoch {epoch:03d}/{epochs:03d} | loss={avg_loss:.4f}"
            )

    flow.eval()


def compute_log_probs(flow: nn.Module, data: torch.Tensor, device: str) -> np.ndarray:
    """Compute log probabilities (scores) for the data."""
    device_obj = torch.device(device)
    loader = DataLoader(TensorDataset(data), batch_size=2048, shuffle=False)
    log_probs_list = []

    flow.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device_obj)
            lp = flow.log_prob(batch)
            log_probs_list.append(lp.cpu().numpy())

    return np.concatenate(log_probs_list)
