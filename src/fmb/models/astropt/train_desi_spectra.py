"""
Distributed training entry-point for AstroPT on DESI spectra.

This script mirrors the structure of the generic AstroPT training loop while
replacing all imaging components with the DESI spectrum loader introduced in
scripts/euclid_desi_dataset/desi_spectrum_dataloader.py. It supports single GPU
debug runs and multi-GPU distributed data parallel (DDP) launches via torchrun.

USAGE EXAMPLES:
===============

Single GPU training:
-------------------
python scripts/train_desi_spectra.py --batch-size 16 --compile

Multi-GPU training (2 GPUs):
---------------------------
torchrun --standalone --nproc_per_node=2 scripts/train_desi_spectra.py \
    --batch-size 32 \
    --grad-accum 2 \
    --compile

CONFIGURATION:
==============

To increase training iterations (default: 500,000):
---------------------------------------------------
Edit the TrainingConfig dataclass below and change:
    max_iters: int = 500000  # <- Increase this value (e.g., 1000000 for longer training)

Training runs for a FIXED NUMBER OF ITERATIONS, not epochs.
Approximate epochs = max_iters / (dataset_size / batch_size / grad_accum / num_gpus)

Other important parameters:
--------------------------
- --batch-size: Batch size per GPU (default: 16)
- --grad-accum: Gradient accumulation steps (default: 4)
- --compile: Enable torch.compile for faster training
- --log-wandb: Enable Weights & Biases logging
- --eval-interval: How often to run validation (default: 500 iterations)
- --checkpoint-interval: How often to save checkpoints (default: 10,000 iterations)

Multi-GPU notes:
---------------
- Use torchrun with --nproc_per_node=N where N is the number of GPUs
- Effective batch size = batch_size × num_gpus × grad_accum
- Training speed scales nearly linearly with number of GPUs
- Only the master process (GPU 0) prints logs and saves checkpoints
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
import glob
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _WANDB_AVAILABLE = False

from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
# scripts/euclid_desi_dataset/desi_spectrum_dataloader.py 
from scripts.euclid_desi_dataset.desi_spectrum_dataloader import (
    DESISpectraDataset,
    spectra_collate,
)


# ---------------------------------------------------------------------------
# Helper dataclass for runtime configuration.
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Container that gathers together the main hyperparameters."""

    out_dir: str = "/pbs/throng/training/astroinfo2025/work/jzoubian/logs/astropt_desi_spectra_2"
    data_dir: str = "/pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset" 
    train_split: str | None = None
    val_split: str | None = None
    test_split: str | None = None
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50
    checkpoint_interval: int = 10_000
    always_save_checkpoint: bool = False
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_workers: int = 8
    block_size: int = 256
    patch_size: int = 10
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5
    max_iters: int = 500000
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    log_via_wandb: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 42
    auto_resume: bool = True


# ---------------------------------------------------------------------------
# Utility functions.
# ---------------------------------------------------------------------------
def parse_args() -> TrainingConfig:
    """Collect CLI overrides for the training configuration."""

    parser = argparse.ArgumentParser(
        description="Train AstroPT on DESI spectra with DDP support."
    )
    parser.add_argument("--out-dir", default=TrainingConfig.out_dir)
    parser.add_argument("--data-dir", default=TrainingConfig.data_dir)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=TrainingConfig.gradient_accumulation_steps,
        help="Gradient accumulation steps per optimization step.",
    )
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--block-size", type=int, default=TrainingConfig.block_size)
    parser.add_argument("--patch-size", type=int, default=TrainingConfig.patch_size)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        default=TrainingConfig.log_via_wandb,
        help="Enable Weights & Biases logging if available.",
    )
    parser.add_argument("--wandb-project", default=TrainingConfig.wandb_project)
    parser.add_argument("--wandb-run-name", default=TrainingConfig.wandb_run_name)
    parser.add_argument("--val-fraction", type=float, default=TrainingConfig.val_fraction)
    parser.add_argument("--test-fraction", type=float, default=TrainingConfig.test_fraction)
    parser.add_argument("--split-seed", type=int, default=TrainingConfig.split_seed)
    parser.add_argument("--auto-resume", dest="auto_resume", action="store_true", default=TrainingConfig.auto_resume)
    parser.add_argument("--no-auto-resume", dest="auto_resume", action="store_false")
    args = parser.parse_args()

    config = TrainingConfig()
    config.out_dir = args.out_dir
    config.data_dir = args.data_dir
    config.train_split = args.train_split
    config.val_split = args.val_split
    config.test_split = args.test_split
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.grad_accum
    config.num_workers = args.num_workers
    config.block_size = args.block_size
    config.patch_size = args.patch_size
    config.compile = args.compile
    config.log_via_wandb = args.log_wandb and _WANDB_AVAILABLE
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_run_name
    config.val_fraction = args.val_fraction
    config.test_fraction = args.test_fraction
    config.split_seed = args.split_seed
    config.auto_resume = args.auto_resume
    return config


def maybe_init_wandb(config: TrainingConfig, run_name: str | None) -> None:
    """Kick off a Weights & Biases session if requested."""
    if not config.log_via_wandb or not _WANDB_AVAILABLE:
        return
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=config.__dict__,
    )


def cleanup_wandb() -> None:
    """Ensure WANDB run is closed cleanly."""
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


def find_latest_checkpoint(out_dir: str) -> str | None:
    """Return the most recent checkpoint path if any exist."""
    pattern = os.path.join(out_dir, "ckpt_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def setup_ddp(config: TrainingConfig) -> tuple[bool, int, int, torch.device]:
    """Initialise DDP process group metadata."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
    else:
        ddp_rank = 0
        ddp_world_size = 1
        device = torch.device(config.device)

    return ddp, ddp_rank, ddp_world_size, device


def prepare_spectra_batch(
    batch: dict[str, Any],
    patch_size: int,
    block_size: int,
    device: torch.device,
    target_dtype: torch.dtype,
) -> dict[str, Any] | None:
    """
    Convert the collated DESI batch into the tensors expected by the model.

    We chunk each spectrum into consecutive windows of `patch_size` pixels,
    pad incomplete windows, and trim to `block_size` tokens so that the GPT
    model receives aligned inputs and position indices.
    """
    flux = batch["flux"].to(device=device, dtype=torch.float32)
    B, L = flux.shape
    pad = (patch_size - (L % patch_size)) % patch_size
    if pad:
        flux = F.pad(flux, (0, pad))

    patches = flux.view(B, -1, patch_size)
    token_count = patches.size(1)

    if block_size > 0:
        token_count = min(token_count, block_size)
        patches = patches[:, :token_count]

    if token_count < 2:
        return None

    positions = torch.arange(token_count, device=device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand(B, -1)

    inputs = patches[:, :-1].to(dtype=target_dtype)
    targets = patches[:, 1:].to(dtype=target_dtype)
    input_positions = positions[:, :-1]

    if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        return None

    meta_fields = {}
    for key in ("targetid", "redshift", "norm"):
        if key in batch:
            meta_fields[key] = batch[key]

    return {
        "X": {
            "spectra": inputs,
            "spectra_positions": input_positions,
        },
        "Y": {
            "spectra": targets,
        },
        "meta": meta_fields,
    }


def prepare_dataset_splits(
    config: TrainingConfig, master_process: bool
) -> tuple[DESISpectraDataset, DESISpectraDataset | None, DESISpectraDataset | None]:
    """Load the DESI dataset and prepare train/val/test splits."""

    if config.val_fraction + config.test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    if config.train_split and config.val_split and config.test_split:
        train_dataset = DESISpectraDataset(
            data_dir=config.data_dir,
            split=config.train_split,
        )
        val_dataset = DESISpectraDataset(
            data_dir=config.data_dir,
            split=config.val_split,
        )
        test_dataset = DESISpectraDataset(
            data_dir=config.data_dir,
            split=config.test_split,
        )
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    else:
        base_dataset = DESISpectraDataset(
            data_dir=config.data_dir,
            split=config.train_split,
        )
        hf_dataset = base_dataset.dataset
        total = len(hf_dataset)
        if total == 0:
            raise RuntimeError("Loaded dataset is empty; cannot create splits.")

        val_frac = config.val_fraction
        test_frac = config.test_fraction
        temp_frac = val_frac + test_frac

        if temp_frac > 0:
            split_dict = hf_dataset.train_test_split(
                test_size=temp_frac,
                seed=config.split_seed,
            )
            train_hf = split_dict["train"]
            temp_hf = split_dict["test"]

            if test_frac > 0:
                if temp_frac == 0:
                    raise ValueError("temp_frac is zero despite non-zero test_frac")
                relative_test = test_frac / temp_frac
                val_test_split = temp_hf.train_test_split(
                    test_size=relative_test,
                    seed=config.split_seed,
                )
                val_hf = val_test_split["train"]
                test_hf = val_test_split["test"]
            else:
                val_hf = temp_hf
                test_hf = None
        else:
            train_hf = hf_dataset
            val_hf = None
            test_hf = None

        train_dataset = DESISpectraDataset(
            data_dir=config.data_dir,
            hf_dataset=train_hf,
        )
        val_dataset = (
            DESISpectraDataset(data_dir=config.data_dir, hf_dataset=val_hf)
            if val_hf is not None
            else None
        )
        test_dataset = (
            DESISpectraDataset(data_dir=config.data_dir, hf_dataset=test_hf)
            if test_hf is not None
            else None
        )
    if master_process:
        train_len = len(train_dataset)
        val_len = len(val_dataset) if val_dataset is not None else 0
        test_len = len(test_dataset) if test_dataset is not None else 0
        print(
            f"Dataset sizes -> total: {train_len + val_len + test_len:,} | "
            f"train: {train_len:,} | val: {val_len:,} | test: {test_len:,}"
        )
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: DESISpectraDataset,
    val_dataset: DESISpectraDataset | None,
    test_dataset: DESISpectraDataset | None,
    config: TrainingConfig,
    ddp: bool,
    world_size: int,
    rank: int,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Initialise dataloaders for each split."""

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if ddp
        else None
    )

    def build_sampler(dataset, shuffle):
        if not ddp or dataset is None:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )

    val_sampler = build_sampler(val_dataset, shuffle=False)
    test_sampler = build_sampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(not ddp),
        num_workers=config.num_workers,
        collate_fn=spectra_collate,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=spectra_collate,
            pin_memory=True,
            drop_last=False,
            sampler=val_sampler,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=spectra_collate,
            pin_memory=True,
            drop_last=False,
            sampler=test_sampler,
        )
        if test_dataset is not None
        else None
    )
    return train_loader, val_loader, test_loader


def estimate_tokens_per_iter(
    config: TrainingConfig, world_size: int, modalities: list[ModalityConfig]
) -> int:
    """Compute logging-friendly estimate for tokens processed each step."""
    tokens_per_iter = (
        config.gradient_accumulation_steps
        * world_size
        * config.batch_size
        * config.block_size
        * len(modalities)
    )
    return tokens_per_iter


def main() -> None:
    """Entry point."""
    config = parse_args()

    ddp, ddp_rank, world_size, device = setup_ddp(config)
    master_process = ddp_rank == 0

    # Seed management for reproducibility across ranks.
    torch.manual_seed(1337 + ddp_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Construct modality registry for spectra-only training.
    modalities = [
        ModalityConfig(
            name="spectra",
            input_size=config.patch_size,
            patch_size=config.patch_size,
            pos_input_size=1,
            loss_weight=1.0,
            embed_pos=True,
        ),
    ]
    modality_registry = ModalityRegistry(modalities)

    train_dataset, val_dataset, test_dataset = prepare_dataset_splits(
        config, master_process
    )

    if ddp:
        # When running with DDP every rank performs gradient accumulation locally.
        # We divide by world size so the effective number of micro-steps matches
        # the single-GPU configuration.
        assert config.gradient_accumulation_steps % world_size == 0, (
            "Gradient accumulation steps must be divisible by the world size."
        )
        config.gradient_accumulation_steps //= world_size

    # Build the PyTorch dataloaders, wiring in DistributedSampler instances when
    # the script is launched under torchrun so that each rank sees unique data.
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        config,
        ddp,
        world_size,
        ddp_rank,
    )
    tokens_per_iter = estimate_tokens_per_iter(config, world_size, modalities)

    run_stamp = time.strftime("%Y%m%d-%H%M%S")
    model_tag = f"L{config.n_layer}_H{config.n_head}_E{config.n_embd}_BS{config.block_size}"

    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print(f"Training logs will be written to: {config.out_dir}")
        print(f"Estimated tokens per iteration: {tokens_per_iter:,}")

    # Configure model and optimiser.
    gpt_config = GPTConfig(
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        bias=config.bias,
        dropout=config.dropout,
        attn_type="causal",
    )
    # Instantiate the AstroPT transformer and move it onto the worker's device.
    model = GPT(gpt_config, modality_registry)

    resume_iter = 0
    resume_best_val = float("inf")
    resume_optimizer_state = None
    if config.auto_resume:
        latest_ckpt = find_latest_checkpoint(config.out_dir)
        if latest_ckpt is not None and os.path.isfile(latest_ckpt):
            if master_process:
                print(f"Resuming from checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            if any(k.startswith(unwanted_prefix) for k in state_dict.keys()):
                cleaned_state = {}
                for key, value in state_dict.items():
                    new_key = key[len(unwanted_prefix):] if key.startswith(unwanted_prefix) else key
                    cleaned_state[new_key] = value
                state_dict = cleaned_state
            model.load_state_dict(state_dict)
            resume_iter = checkpoint.get("iter_num", 0)
            resume_best_val = checkpoint.get("best_val_loss", float("inf"))
            resume_optimizer_state = checkpoint.get("optimizer")
        elif master_process and config.auto_resume:
            print("No checkpoint found; starting from scratch.")

    model = model.to(device)
    if config.compile:
        # torch.compile can fuse kernels for sizeable speed-ups on modern GPUs.
        model = torch.compile(model)  # type: ignore[attr-defined]

    if ddp:
        # Wrap the model with DDP to synchronise gradients across ranks.
        model = DDP(model, device_ids=[device], find_unused_parameters=False)

    # Configure the fused AdamW optimiser that ships with AstroPT.
    base_model = model.module if isinstance(model, DDP) else model
    optimizer = base_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device.type,
    )

    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        # Ensure optimizer tensors are on the correct device after loading.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    target_dtype = dtype_map.get(config.dtype, torch.float32)
    use_amp = target_dtype in {torch.bfloat16, torch.float16}
    # GradScaler only kicks in for float16; bfloat16 tolerates large gradients.
    scalar = torch.cuda.amp.GradScaler(enabled=(target_dtype == torch.float16))
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=target_dtype)
        if use_amp
        else nullcontext()
    )

    maybe_init_wandb(config, config.wandb_run_name)

    best_val_loss = resume_best_val
    iter_num = resume_iter
    leftover_tokens = 0
    loss_meter = 0.0
    start_time = time.time()

    # We track a logical epoch counter so DDP samplers reshuffle between passes.
    try:
        epoch = iter_num // len(train_loader)
    except TypeError:
        epoch = 0
    while iter_num < config.max_iters:
        if ddp:
            assert isinstance(train_loader.sampler, DistributedSampler)
            train_loader.sampler.set_epoch(epoch)
            if val_loader is not None and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            model.train()
            # Convert the raw DESI spectra into modality-aware tensors for AstroPT.
            batch_tokens = prepare_spectra_batch(
                batch,
                patch_size=config.patch_size,
                block_size=config.block_size,
                device=device,
                target_dtype=target_dtype,
            )
            if batch_tokens is None:
                continue
            inputs = batch_tokens["X"]
            targets = batch_tokens["Y"]

            with autocast_ctx:
                # Forward pass returns predictions and loss; we track the loss scalar.
                _, loss = model(inputs, targets=targets)

            # Gradient accumulation splits the effective batch across micro-steps.
            loss = loss / config.gradient_accumulation_steps
            scalar.scale(loss).backward()

            leftover_tokens += 1
            iter_num += 1
            loss_meter += loss.item() * config.gradient_accumulation_steps

            if leftover_tokens % config.gradient_accumulation_steps == 0:
                # Optionally clip gradients before the optimiser step to stabilise training.
                if config.grad_clip > 0:
                    scalar.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scalar.step(optimizer)
                scalar.update()
                optimizer.zero_grad(set_to_none=True)
                leftover_tokens = 0

            if master_process and iter_num % config.log_interval == 0:
                # Emit console (and optional WandB) logs so we can monitor convergence.
                elapsed = time.time() - start_time
                tokens = iter_num * config.batch_size * config.block_size
                speed = tokens / elapsed if elapsed > 0 else 0.0
                print(
                    f"iter {iter_num:6d} | loss {loss_meter / config.log_interval:.4f} "
                    f"| tokens/sec {speed:,.0f}"
                )
                if config.log_via_wandb:
                    wandb.log(
                        {
                            "train/loss": loss_meter / config.log_interval,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/tokens_per_sec": speed,
                        },
                        step=iter_num,
                    )
                loss_meter = 0.0

            if iter_num % config.eval_interval == 0:
                def run_eval(loader: DataLoader | None) -> float:
                    if loader is None:
                        return float("inf")
                    model.eval()
                    losses = []
                    with torch.no_grad():
                        loader_iter = iter(loader)
                        for _ in range(config.eval_iters):
                            try:
                                eval_batch = next(loader_iter)
                            except StopIteration:
                                loader_iter = iter(loader)
                                eval_batch = next(loader_iter)

                            eval_tokens = prepare_spectra_batch(
                                eval_batch,
                                patch_size=config.patch_size,
                                block_size=config.block_size,
                                device=device,
                                target_dtype=target_dtype,
                            )
                            if eval_tokens is None:
                                continue
                            with autocast_ctx:
                                _, eval_loss = model(
                                    eval_tokens["X"], targets=eval_tokens["Y"]
                                )
                                losses.append(eval_loss.item())
                    model.train()
                    if not losses:
                        return float("inf")
                    return float(sum(losses) / len(losses))

                val_loss = run_eval(val_loader)
                if master_process:
                    print(f"[eval] iter {iter_num} | val loss {val_loss:.4f}")

                    loss_path = os.path.join(config.out_dir, "loss.txt")
                    log_header = "iter_num,train_loss,val_loss,lr\n"
                    train_loss_value = loss_meter / max(1, config.log_interval)
                    with open(loss_path, "a") as loss_file:
                        if loss_file.tell() == 0:
                            loss_file.write(log_header)
                        loss_file.write(
                            f"{iter_num},{train_loss_value:.6f},{val_loss:.6f},{optimizer.param_groups[0]['lr']:.6e}\n"
                        )

                    if os.path.exists(loss_path):
                        loss_df = pd.read_csv(loss_path)
                        if len(loss_df) > 1:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(loss_df["iter_num"], loss_df["train_loss"], label="train")
                            ax.plot(loss_df["iter_num"], loss_df["val_loss"], label="val")
                            ax.set_yscale("log")
                            ax.set_xlabel("iteration")
                            ax.set_ylabel("loss")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            fig.tight_layout()
                            fig.savefig(os.path.join(config.out_dir, "loss.png"), dpi=150)
                            plt.close(fig)

                    if val_dataset is not None:
                        sample_loader = DataLoader(
                            val_dataset,
                            batch_size=min(4, config.batch_size),
                            shuffle=False,
                            num_workers=0,
                            collate_fn=spectra_collate,
                        )
                        try:
                            sample_batch = next(iter(sample_loader))
                        except StopIteration:
                            sample_batch = None
                        if sample_batch is not None:
                            sample_tokens = prepare_spectra_batch(
                                sample_batch,
                                patch_size=config.patch_size,
                                block_size=config.block_size,
                                device=device,
                                target_dtype=target_dtype,
                            )
                            if sample_tokens is not None:
                                sample_inputs = sample_tokens["X"]
                                sample_targets = sample_tokens["Y"]
                                eval_ctx = (
                                    torch.amp.autocast(device_type=device.type, dtype=target_dtype)
                                    if use_amp
                                    else nullcontext()
                                )
                                with torch.no_grad():
                                    with eval_ctx:
                                        preds, _ = model(sample_inputs, targets=sample_targets)
                                preds_tensor = (
                                    preds["spectra"].detach().to(dtype=torch.float32).cpu()
                                )
                                target_tensor = (
                                    sample_targets["spectra"].detach().to(dtype=torch.float32).cpu()
                                )
                                num_examples = min(4, preds_tensor.size(0))
                                if num_examples > 0:
                                    fig, axes = plt.subplots(
                                        num_examples,
                                        1,
                                        figsize=(10, 3 * num_examples),
                                        sharex=True,
                                    )
                                    if num_examples == 1:
                                        axes = [axes]
                                    meta = sample_tokens.get("meta", {})
                                    target_ids = meta.get("targetid")
                                    for idx in range(num_examples):
                                        pred_seq = preds_tensor[idx].reshape(-1).numpy()
                                        target_seq = target_tensor[idx].reshape(-1).numpy()
                                        x = np.arange(target_seq.shape[0])
                                        axes[idx].plot(x, target_seq, label="target", color="tab:blue")
                                        axes[idx].plot(x, pred_seq, label="recon", color="tab:orange", alpha=0.8)
                                        axes[idx].set_ylabel("flux")
                                        label = (
                                            f"targetid={target_ids[idx]}" if target_ids is not None else "sample"
                                        )
                                        axes[idx].set_title(label)
                                        axes[idx].grid(True, alpha=0.3)
                                        axes[idx].legend()
                                    axes[-1].set_xlabel("spectral pixel")
                                    fig.tight_layout()
                                    recon_path = os.path.join(
                                        config.out_dir,
                                        f"recon_{model_tag}_{run_stamp}_iter{iter_num:06d}.png",
                                    )
                                    fig.savefig(recon_path, dpi=150)
                                    if config.log_via_wandb:
                                        wandb.log({"reconstruction": wandb.Image(recon_path)}, step=iter_num)
                                    plt.close(fig)

                    if config.log_via_wandb:
                        wandb.log({"val/loss": val_loss}, step=iter_num)
                    if val_loss < best_val_loss or config.always_save_checkpoint:
                        best_val_loss = val_loss
                        checkpoint = {
                            "model": base_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iter_num": iter_num,
                            "config": config.__dict__,
                            "best_val_loss": best_val_loss,
                        }
                        ckpt_path = os.path.join(
                            config.out_dir,
                            f"ckpt_best_{model_tag}_{run_stamp}.pt",
                        )
                        torch.save(checkpoint, ckpt_path)
                        print(f"Saved checkpoint to {ckpt_path}")

            if (
                master_process
                and config.checkpoint_interval > 0
                and iter_num > 0
                and iter_num % config.checkpoint_interval == 0
            ):
                periodic_ckpt = {
                    "model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "config": config.__dict__,
                    "best_val_loss": best_val_loss,
                }
                ckpt_path = os.path.join(
                    config.out_dir,
                    f"ckpt_{model_tag}_{run_stamp}_iter{iter_num:06d}.pt",
                )
                torch.save(periodic_ckpt, ckpt_path)
                print(f"[checkpoint] saved periodic checkpoint to {ckpt_path}")

            if iter_num >= config.max_iters:
                break

        epoch += 1

    if ddp:
        destroy_process_group()
    cleanup_wandb()


if __name__ == "__main__":
    main()
