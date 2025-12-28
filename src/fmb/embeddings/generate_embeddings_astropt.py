#!/usr/bin/env python3
"""
Run inference with the trained astroPT multimodal model from
`astroPT/scripts/train_spectra_images.py` and export embeddings for the whole
Euclid+DESI dataset.

Outputs a single .pt file containing a list of dicts:
  {"object_id": ..., "redshift": ..., "embedding_images": <tensor>, "embedding_spectra": <tensor>}

Example:
  python -m scratch.generate_embeddings_astropt \\
    --checkpoint /n03data/ronceray/models/astropt/ckpt_final.pt \\
    --split all \\
    --cache-dir /n03data/ronceray/datasets \\
    --output-dir /n03data/ronceray/embeddings \\
    --batch-size 16
"""
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from astropt.model import GPT, GPTConfig
from astroPT.scripts.train_spectra_images import (  # type: ignore
    EuclidDESIMultimodalDataset,
    TrainingConfig,
    create_modality_registry,
)
from astroPT.scripts.euclid_desi_dataset.multimodal_dataloader import (  # type: ignore
    multimodal_collate_fn,
    prepare_multimodal_batch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export astroPT embeddings for Euclid+DESI.")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint produced by train_spectra_images.py (e.g., ckpt_final.pt).",
    )
    p.add_argument("--split", type=str, default="all", help="Dataset split(s): train,test,all,...")
    p.add_argument("--cache-dir", type=str, default="/n03data/ronceray/datasets", help="Euclid HF cache/dataset root.")
    p.add_argument("--output-dir", type=str, default="/n03data/ronceray/embeddings", help="Directory to save embeddings.")
    p.add_argument("--output-name", type=str, default="astropt_embeddings.pt", help="Filename for the saved embeddings.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--reduction",
        choices=["mean", "exp_decay", "last", "none"],
        default="mean",
        help="How to pool token embeddings per modality.",
    )
    p.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (defaults to CUDA if available).")
    return p.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_cfg = TrainingConfig(**ckpt["config"])
    modality_registry = create_modality_registry(train_cfg)

    # Rebuild model and load weights
    gpt_conf = GPTConfig(attn_type="causal", **ckpt["model_args"])
    model = GPT(gpt_conf, modality_registry)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, modality_registry, train_cfg


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[load] checkpoint: {ckpt_path}")
    model, modality_registry, train_cfg = load_model(ckpt_path, device)

    dataset = EuclidDESIMultimodalDataset(
        split=args.split,
        image_size=train_cfg.image_size,
        spectrum_length=train_cfg.spectrum_length,
        cache_dir=args.cache_dir,
        verbose=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=device.type == "cuda",
    )
    print(f"[data] Loaded {len(dataset)} samples; batches: {len(loader)}")

    records: Dict[str, Dict] = {}
    progress = tqdm(loader, desc="Encoding", unit="batch")

    with torch.inference_mode():
        for batch in progress:
            inputs = prepare_multimodal_batch(
                batch,
                image_patch_size=train_cfg.image_patch_size,
                spectrum_patch_size=train_cfg.spectrum_patch_size,
                device=device,
                modality_registry=modality_registry,
            )
            if not inputs:
                continue

            embeds = model.generate_embeddings(inputs, reduction=args.reduction)

            if "images" in embeds and "image_object_ids" in batch:
                ids: List = batch["image_object_ids"]
                redshifts = batch.get("image_redshifts", [])
                for i, obj_id in enumerate(ids):
                    rec = records.setdefault(str(obj_id), {"object_id": obj_id})
                    if i < len(redshifts):
                        rec["redshift"] = float(redshifts[i])
                    rec["embedding_images"] = embeds["images"][i].detach().cpu()

            if "spectra" in embeds and "spectrum_object_ids" in batch:
                ids_spec: List = batch["spectrum_object_ids"]
                redshifts_spec = batch.get("spectrum_redshifts", [])
                for i, obj_id in enumerate(ids_spec):
                    rec = records.setdefault(str(obj_id), {"object_id": obj_id})
                    if i < len(redshifts_spec):
                        rec["redshift"] = float(redshifts_spec[i])
                    rec["embedding_spectra"] = embeds["spectra"][i].detach().cpu()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_name
    torch.save(list(records.values()), out_path)

    print(f"[done] Saved {len(records)} embedding records to {out_path}")


if __name__ == "__main__":
    main()
