"""
Fine-tune AstroCLIP image encoder(s) on one or several huggingface parquet datasets.

Example:
    python -m fmb.models.astroclip.finetune_image_encoder \\
        --parquet-path hf://datasets/.../train.parquet \\
        --checkpoint hackathon2025/data/astroclip.ckpt \\
        --output-path outputs/image_encoder_ft.pt \\
        --output-ckpt outputs/astroclip_image_ft.ckpt \\
        --epochs 5 --batch-size 128 --device cuda --amp
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# CHANGED imports
from fmb.data.astroclip_parquet import ParquetDataSource
from fmb.models.astroclip.core.astroclip import AstroClipModel, CLIPLoss
from fmb.paths import load_paths

def parse_args() -> argparse.Namespace:
    paths = load_paths()
    
    # Config file support
    # We parse config first to set defaults
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument("--config", help="Chemin vers un fichier de config YAML.")
    early_args, _ = early_parser.parse_known_args()
    
    defaults = {}
    if early_args.config:
        import yaml
        with open(early_args.config, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
            defaults.update(yaml_config)

    parser = argparse.ArgumentParser(description="Fine-tune AstroCLIP image encoder.")
    parser.add_argument("--config", help="Chemin vers un fichier de config YAML.")
    parser.set_defaults(**defaults)

    parser.add_argument(
        "--parquet-path",
        dest="parquet_paths",
        nargs="+",
        required=False,
        help="Chemins vers les parquets (hf:// supporté). Ignoré si --use-arrow est utilisé.",
    )
    parser.add_argument(
        "--use-arrow",
        action="store_true",
        help="Charger depuis le cache Arrow local (format HuggingFace) au lieu de parquets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(paths.cache),  # Note: user might want dataset_path here if it points to arrow cache
        help="Répertoire du cache HuggingFace (utilisé avec --use-arrow).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split à charger depuis le cache Arrow ('train' ou 'test').",
    )
    parser.add_argument(
        "--learnable-scale",
        action="store_true",
        help="Rendre la température CLIP (logit_scale) apprenable au lieu de fixe.",
    )
    parser.add_argument(
        "--finetune-spectrum",
        action="store_true",
        help="Fine-tuner également l'encodeur de spectre (sinon il reste gelé).",
    )
    
    # Checkpoint is required unless provided in config
    if "checkpoint" in defaults:
        parser.add_argument("--checkpoint", required=False, help="Checkpoint AstroCLIP Lightning.")
    else:
        parser.add_argument("--checkpoint", required=True, help="Checkpoint AstroCLIP Lightning.")
    
    # Defaults relative to retrained_weights_path
    default_out_pt = str(paths.retrained_weights / "astroclip_image_encoder.pt")
    default_out_ckpt = str(paths.retrained_weights / "astroclip_finetuned.ckpt")
    
    parser.add_argument("--output-path", default=default_out_pt, help="Fichier de sortie du nouvel encodeur d'images (.pt).")
    parser.add_argument(
        "--output-ckpt",
        default=default_out_ckpt,
        help="Chemin optionnel pour sauvegarder un checkpoint AstroCLIP complet avec l'encodeur fine-tuné.",
    )
    parser.add_argument("--device", default="cuda", help="Device d'entraînement (cuda ou cpu).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--slice-length", type=int, default=7700)
    parser.add_argument("--image-size", type=int, default=144)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction utilisée pour la validation (0 pour désactiver).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Activer l'entraînement AMP.")
    parser.add_argument("--disable-augment", action="store_true", help="Désactiver les augmentations d'images.")
    parser.add_argument(
        "--spectrum-norm",
        choices=["zscore", "minmax", "none"],
        default="none",
        help="Normalisation appliquée aux spectres avant l'encodage.",
    )
    parser.add_argument(
        "--include-wavelength",
        action="store_true",
        help="Empiler flux et longueur d'onde sur deux canaux (assurez-vous que le spectrum encoder l'accepte).",
    )
    parser.add_argument(
        "--focus-high-z",
        action="store_true",
        help="Lors du sampling parquet, privilégier les galaxies à haut redshift.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="Nombre d'itérations de warmup pour le scheduler (0 = 10% des steps).")
    parser.add_argument("--patience", type=int, default=3, help="Patience pour l'early stopping (epochs).")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Amélioration minimale requise pour reset la patience.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Clip des gradients (<=0 pour désactiver).")
    parser.add_argument("--accumulate-steps", type=int, default=1, help="Nombre d'itérations pour accumuler les gradients.")
    parser.add_argument("--log-interval", type=int, default=20, help="Intervalle d'affichage des logs (nombre de batchs).")
    parser.add_argument(
        "--unfreeze-backbone-blocks",
        type=int,
        default=0,
        help="Nombre de blocs DINO à dégeler (0 = ne pas dégeler la backbone).",
    )
    parser.add_argument(
        "--history-json",
        default=None,
        help="Chemin pour sauvegarder l'historique d'entraînement (JSON). Défaut: même dossier que output-path.",
    )
    parser.add_argument(
        "--history-plot",
        default=None,
        help="Chemin pour sauvegarder la courbe des pertes/metrics. Défaut: même dossier que output-path.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_logit_scale(model: AstroClipModel) -> float:
    logit_scale = model.logit_scale
    if isinstance(logit_scale, torch.Tensor):
        return float(logit_scale.detach().cpu().exp().item())
    return float(math.exp(logit_scale))


def load_dataframe(
    parquet_paths: Optional[Iterable[str]],
    max_samples: Optional[int],
    seed: int,
    focus_high_z: bool,
    image_size: int,
    batch_size: int,
    use_arrow: bool = False,
    cache_dir: str = "/n03data/ronceray/datasets",
    split: str = "train",
) -> pd.DataFrame:
    """Load dataset from Arrow cache or Parquet files.
    
    Args:
        parquet_paths: Paths to parquet files (ignored if use_arrow=True)
        max_samples: Maximum number of samples to load
        seed: Random seed
        focus_high_z: Prioritize high-redshift samples
        image_size: Target image size
        batch_size: Batch size (for compatibility)
        use_arrow: If True, load from local Arrow cache instead of parquet
        cache_dir: Directory containing Arrow cache
        split: Which split to load from Arrow cache
    """
    if use_arrow:
        # Load from local Arrow cache (HuggingFace format)
        # CHANGED: Import from fmb.data.astroclip_loader
        from fmb.data.astroclip_loader import (
            load_local_arrow_dataset,
            convert_dataset_to_astroclip_format,
        )
        
        print(f"Loading from Arrow cache: {cache_dir} (split={split})")
        df = load_local_arrow_dataset(
            cache_dir=cache_dir,
            split=split,
            max_samples=max_samples,
            seed=seed,
        )
        
        # Convert to AstroCLIP format
        df = convert_dataset_to_astroclip_format(df, image_size=image_size)
        
    else:
        # Original parquet loading logic
        if not parquet_paths:
            raise ValueError("Either --use-arrow or --parquet-path must be specified")
        
        frames = []
        for path in parquet_paths:
            ds = ParquetDataSource(
                parquet_path=path,
                focus_high_z=focus_high_z,
                sample_size=max_samples,
                image_size=image_size,
                batch_size=batch_size,
                enable_cache=False,
            )
            df = ds.load()
            frames.append(df)

        if not frames:
            raise ValueError("Aucun parquet n'a été chargé. Vérifiez les chemins fournis.")

        df = pd.concat(frames, ignore_index=True)
        if max_samples is not None and max_samples < len(df):
            df = df.sample(max_samples, random_state=seed).reset_index(drop=True)

    required_columns = ["spectrum", "redshift", "image"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est absente du DataFrame final.")
    
    return df.reset_index(drop=True)


def train_val_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if val_ratio <= 0 or len(df) < 2:
        return df, None

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    split_idx = int(len(df) * (1 - val_ratio))
    if split_idx <= 0 or split_idx >= len(df):
        return df, None

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def _normalise_spectrum(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return tensor
    if mode == "zscore":
        mean = tensor.mean()
        std = tensor.std(unbiased=False).clamp(min=1e-6)
        return (tensor - mean) / std
    if mode == "minmax":
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val).clamp(min=1e-6)
        return (tensor - min_val) / scale
    raise ValueError(f"Mode de normalisation inconnu: {mode}")


class AstroClipFineTuneDataset(Dataset):
    """Dataset that pads/trims spectra, normalises them and returns tensors for training."""

    def __init__(
        self,
        df: pd.DataFrame,
        slice_length: int,
        spectrum_norm: str,
        include_wavelength: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.slice_length = int(slice_length)
        self.spectrum_norm = spectrum_norm
        self.include_wavelength = include_wavelength

    def __len__(self) -> int:
        return len(self.df)

    def _pad_or_trim(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if tensor.numel() < self.slice_length:
            pad_len = self.slice_length - tensor.numel()
            tensor = torch.cat([tensor, torch.zeros(pad_len, dtype=torch.float32)])
        else:
            tensor = tensor[: self.slice_length]
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        spec = row["spectrum"]
        if isinstance(spec, dict):
            flux = np.asarray(spec["flux"])
            wavelength = np.asarray(spec.get("wavelength"))
        else:
            arr = np.asarray(spec)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                flux = arr[:, 0]
                wavelength = arr[:, 1]
            else:
                flux = arr.squeeze()
                wavelength = np.linspace(0, 1, flux.shape[0], dtype=np.float32)

        flux_tensor = self._pad_or_trim(flux)
        flux_tensor = _normalise_spectrum(flux_tensor, self.spectrum_norm)
        wavelength_tensor = self._pad_or_trim(wavelength)

        spectrum = flux_tensor.unsqueeze(-1)
        if self.include_wavelength:
            spectrum = torch.stack([flux_tensor, wavelength_tensor], dim=-1)

        image_tensor = row["image"]
        image = image_tensor if isinstance(image_tensor, torch.Tensor) else torch.as_tensor(image_tensor)
        image = image.float()

        return {
            "image": image,
            "spectrum": spectrum,
        }


def build_dataloader(
    df: pd.DataFrame,
    batch_size: int,
    slice_length: int,
    spectrum_norm: str,
    include_wavelength: bool,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = AstroClipFineTuneDataset(df, slice_length, spectrum_norm, include_wavelength)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def maybe_unfreeze_backbone(image_encoder: nn.Module, num_blocks: int) -> None:
    if num_blocks <= 0:
        return
    backbone = getattr(image_encoder, "backbone", None)
    if backbone is None or not hasattr(backbone, "blocks"):
        print("Impossible de dégeler la backbone: attribut 'backbone.blocks' introuvable.")
        return
    blocks = list(backbone.blocks)
    num_blocks = min(num_blocks, len(blocks))
    if num_blocks <= 0:
        return
    for block in blocks[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    if hasattr(backbone, "norm"):
        for param in backbone.norm.parameters():
            param.requires_grad = True
    if hasattr(backbone, "patch_embed"):
        for param in backbone.patch_embed.parameters():
            param.requires_grad = True
    print(f"Dégel de {num_blocks} blocs de la backbone AstroDINO.")


@dataclass
class HistoryEntry:
    epoch: int
    train_loss: float
    train_cosine: float
    val_loss: Optional[float] = None
    val_cosine: Optional[float] = None


def evaluate(
    image_encoder: nn.Module,
    spectrum_encoder: nn.Module,
    loader: DataLoader,
    criterion: CLIPLoss,
    device: torch.device,
    logit_scale: float,
) -> Dict[str, float]:
    image_encoder.eval()
    spectrum_encoder.eval()

    total_loss = 0.0
    total_cosine = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            spectrum = batch["spectrum"].to(device, non_blocking=True)

            image_features = image_encoder(images)
            spectrum_features = spectrum_encoder(spectrum)

            loss = criterion(image_features, spectrum_features, logit_scale)
            cosine = F.cosine_similarity(
                F.normalize(image_features, dim=-1),
                F.normalize(spectrum_features, dim=-1),
                dim=-1,
            ).mean()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_cosine += cosine.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return {"loss": float("nan"), "cosine": float("nan")}

    return {
        "loss": total_loss / total_samples,
        "cosine": total_cosine / total_samples,
    }


def _cosine_scheduler(total_steps: int, warmup_steps: int) -> List[float]:
    schedule = []
    for step in range(total_steps):
        if step < warmup_steps:
            schedule.append(step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            schedule.append(0.5 * (1.0 + math.cos(math.pi * progress)))
    return schedule


def _plot_history(history: List[HistoryEntry], output: Path) -> None:
    epochs = [entry.epoch for entry in history]
    train_loss = [entry.train_loss for entry in history]
    train_cos = [entry.train_cosine for entry in history]
    val_epochs = [entry.epoch for entry in history if entry.val_loss is not None]
    val_loss = [entry.val_loss for entry in history if entry.val_loss is not None]
    val_cos_epochs = [entry.epoch for entry in history if entry.val_cosine is not None]
    val_cos = [entry.val_cosine for entry in history if entry.val_cosine is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train")
    if val_loss:
        axes[0].plot(val_epochs, val_loss, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_cos, label="Train")
    if val_cos:
        axes[1].plot(val_cos_epochs, val_cos, label="Val")
    axes[1].set_title("Cosine similarity")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def export_full_checkpoint(model: AstroClipModel, output_ckpt: Path) -> None:
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": model.hparams,
        },
        output_ckpt,
    )


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = parse_args() if args is None else args
    set_seed(parsed.seed)

    device = torch.device(parsed.device)
    df = load_dataframe(
        parsed.parquet_paths,
        parsed.max_samples,
        parsed.seed,
        focus_high_z=parsed.focus_high_z,
        image_size=parsed.image_size,
        batch_size=parsed.batch_size,
        use_arrow=parsed.use_arrow,
        cache_dir=parsed.cache_dir,
        split=parsed.split,
    )

    train_df, val_df = train_val_split(df, parsed.val_ratio, parsed.seed)
    train_loader = build_dataloader(
        train_df,
        parsed.batch_size,
        parsed.slice_length,
        parsed.spectrum_norm,
        parsed.include_wavelength,
        shuffle=True,
        num_workers=parsed.num_workers,
    )
    if val_df is not None:
        val_loader = build_dataloader(
            val_df,
            parsed.batch_size,
            parsed.slice_length,
            parsed.spectrum_norm,
            parsed.include_wavelength,
            shuffle=False,
            num_workers=parsed.num_workers,
        )
    else:
        val_loader = None


    # Define unsafe load context manager
    from contextlib import contextmanager
    @contextmanager
    def unsafe_torch_load_context():
        original_load = torch.load
        def unsafe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                 kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = unsafe_load
        try:
            yield
        finally:
            torch.load = original_load

    # Load initial model
    with unsafe_torch_load_context():
        model = AstroClipModel.load_from_checkpoint(parsed.checkpoint, map_location=device)
    
    image_encoder = model.image_encoder.to(device)
    spectrum_encoder = model.spectrum_encoder.to(device)
    
    # Handle Spectrum Encoder training
    if parsed.finetune_spectrum:
        spectrum_encoder.train()
        for param in spectrum_encoder.parameters():
            param.requires_grad = True
        print("Note: Spectrum Encoder is TRAINABLE (Fine-tuning both modalities).")
    else:
        spectrum_encoder.eval()
        for param in spectrum_encoder.parameters():
            param.requires_grad = False
        print("Note: Spectrum Encoder is FROZEN.")

    maybe_unfreeze_backbone(image_encoder, parsed.unfreeze_backbone_blocks)

    # Setup parameters to optimize
    trainable_params = list([p for p in image_encoder.parameters() if p.requires_grad])
    if parsed.finetune_spectrum:
        trainable_params.extend([p for p in spectrum_encoder.parameters() if p.requires_grad])
    
    # Handle logit scale (temperature)
    if parsed.learnable_scale:
        if isinstance(model.logit_scale, torch.Tensor):
            model.logit_scale.requires_grad = True
        else:
            # Create parameter if it was a float/buffer
            import math
            initial_val = model.logit_scale if isinstance(model.logit_scale, float) else model.logit_scale.item()
            model.logit_scale = torch.nn.Parameter(torch.tensor(initial_val, device=device))
        
        trainable_params.append(model.logit_scale)
        print("Note: CLIP temperature/scale is LEARNABLE.")
    else:
        # Ensure it's frozen
        if isinstance(model.logit_scale, torch.Tensor):
            model.logit_scale.requires_grad = False
        print("Note: CLIP temperature/scale is FIXED.")

    optimizer = torch.optim.AdamW(trainable_params, lr=parsed.lr, weight_decay=parsed.weight_decay)

    total_steps = (len(train_loader) * parsed.epochs) // max(1, parsed.accumulate_steps)
    warmup_steps = parsed.warmup_steps or max(1, int(0.1 * total_steps))
    scheduler_factors = _cosine_scheduler(total_steps, warmup_steps)

    criterion = CLIPLoss()
    
    # Helper to get current scale (tensor if learnable, float if fixed)
    def get_current_scale():
        if parsed.learnable_scale:
            return model.logit_scale
        # Return float value for fixed scale
        if isinstance(model.logit_scale, torch.Tensor):
             return model.logit_scale.exp().item()
        return math.exp(model.logit_scale)

    scaler = GradScaler(enabled=parsed.amp and device.type == "cuda")
    history: List[HistoryEntry] = []

    best_val_loss = float("inf")
    patience_counter = 0

    global_step = 0

    for epoch in range(1, parsed.epochs + 1):
        image_encoder.train()
        cumulative_loss = 0.0
        cumulative_cosine = 0.0
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            spectrum = batch["spectrum"].to(device, non_blocking=True)

            with autocast(enabled=scaler.is_enabled()):
                image_features = image_encoder(images)
                spectrum_features = spectrum_encoder(spectrum)
                
                # Get current scale (might change if learnable)
                current_scale = get_current_scale()
                loss = criterion(image_features, spectrum_features, current_scale)
                
                cosine = F.cosine_similarity(
                    F.normalize(image_features, dim=-1),
                    F.normalize(spectrum_features, dim=-1),
                    dim=-1,
                ).mean()

            batch_size = images.size(0)
            cumulative_loss += loss.item() * batch_size
            cumulative_cosine += cosine.item() * batch_size
            sample_count += batch_size

            loss = loss / parsed.accumulate_steps

            scaler.scale(loss).backward()

            if batch_idx % parsed.accumulate_steps == 0:
                if parsed.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn_utils.clip_grad_norm_(trainable_params, parsed.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if global_step < len(scheduler_factors):
                    factor = scheduler_factors[global_step]
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = parsed.lr * factor
                global_step += 1

            if batch_idx % parsed.log_interval == 0:
                scale_val = model.logit_scale.item() if isinstance(model.logit_scale, torch.Tensor) else model.logit_scale
                loss_val = loss.item() * parsed.accumulate_steps # Restore full loss scale
                print(
                    f"[Epoch {epoch}] Batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss_val:.4f} cosine={cosine.item():.4f} "
                    f"scale={scale_val:.2f}"
                )

        train_loss = cumulative_loss / max(1, sample_count)
        train_cos = cumulative_cosine / max(1, sample_count)

        val_metrics = {"loss": None, "cosine": None}
        if val_loader is not None:
            # Pass current scale to evaluate
            val_metrics = evaluate(image_encoder, spectrum_encoder, val_loader, criterion, device, get_current_scale())

        history.append(
            HistoryEntry(
                epoch=epoch,
                train_loss=train_loss,
                train_cosine=train_cos,
                val_loss=val_metrics["loss"],
                val_cosine=val_metrics["cosine"],
            )
        )

        val_loss_for_patience = val_metrics["loss"] if val_metrics["loss"] is not None else train_loss
        improved = val_loss_for_patience + parsed.min_delta < best_val_loss
        if improved:
            best_val_loss = val_loss_for_patience
            patience_counter = 0
            patience_counter = 0
            
            # Save ALL trainable components
            checkpoint_payload = {
                "image_encoder": {k: v.detach().cpu() for k, v in image_encoder.state_dict().items()},
            }
            if parsed.finetune_spectrum:
                 checkpoint_payload["spectrum_encoder"] = {
                     k: v.detach().cpu() for k, v in spectrum_encoder.state_dict().items()
                 }
            
            # Handle logit scale
            if parsed.learnable_scale:
                scale_val = model.logit_scale.detach().cpu() if isinstance(model.logit_scale, torch.Tensor) else model.logit_scale
                checkpoint_payload["logit_scale"] = scale_val
                
            torch.save(checkpoint_payload, parsed.output_path)
            print(f"[Epoch {epoch}] Nouveau meilleur modèle sauvegardé dans {parsed.output_path}")
        else:
            patience_counter += 1
            if patience_counter >= parsed.patience:
                print("Patience atteinte, arrêt anticipé.")
                break

    # Reload best state
    print(f"Reloading best model from {parsed.output_path}...")
    best_payload = torch.load(parsed.output_path, map_location="cpu")
    
    # Check if it's the new format (dict with 'image_encoder') or old (just state dict)
    if "image_encoder" in best_payload:
        image_encoder.load_state_dict(best_payload["image_encoder"])
        if parsed.finetune_spectrum and "spectrum_encoder" in best_payload:
            spectrum_encoder.load_state_dict(best_payload["spectrum_encoder"])
        if parsed.learnable_scale and "logit_scale" in best_payload:
            # Restore scale
            saved_scale = best_payload["logit_scale"]
            if isinstance(model.logit_scale, torch.nn.Parameter):
                with torch.no_grad():
                    model.logit_scale.copy_(saved_scale if isinstance(saved_scale, torch.Tensor) else torch.tensor(saved_scale))
    else:
        # Fallback for old format (only image encoder)
        image_encoder.load_state_dict(best_payload)

    if parsed.output_ckpt:
        with unsafe_torch_load_context():
            export_model = AstroClipModel.load_from_checkpoint(parsed.checkpoint, map_location="cpu")
        
        # Load best weights into export model
        export_model.image_encoder.load_state_dict(image_encoder.state_dict())
        if parsed.finetune_spectrum:
             export_model.spectrum_encoder.load_state_dict(spectrum_encoder.state_dict())
        
        # Update scale in export model
        if parsed.learnable_scale:
             if isinstance(export_model.logit_scale, torch.nn.Parameter):
                 with torch.no_grad():
                     export_model.logit_scale.copy_(model.logit_scale)
             else:
                 # It was fixed, now we want to save the learned value? 
                 # If export model has fixed scale (float), we might need to convert it to param or save value
                 # But AstroClipModel constructor logic might reset it.
                 # Let's assumes strict consistency isn't critical for 'export_full_checkpoint' as long as state_dict is right
                 pass 
                 
        export_full_checkpoint(export_model, Path(parsed.output_ckpt))
        print(f"[OK] Checkpoint complet sauvegardé dans {parsed.output_ckpt}")

    history_path = Path(parsed.history_json) if parsed.history_json else Path(parsed.output_path).with_suffix(".history.json")
    history_payload = [
        {
            "epoch": entry.epoch,
            "train_loss": entry.train_loss,
            "train_cosine": entry.train_cosine,
            "val_loss": entry.val_loss,
            "val_cosine": entry.val_cosine,
        }
        for entry in history
    ]
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history_payload, indent=2))
    print(f"[OK] Historique sauvegardé dans {history_path}")

    plot_path = Path(parsed.history_plot) if parsed.history_plot else Path(parsed.output_path).with_suffix(".history.png")
    _plot_history(history, plot_path)
    print(f"[OK] Courbe d'entraînement sauvegardée dans {plot_path}")


if __name__ == "__main__":
    main()
