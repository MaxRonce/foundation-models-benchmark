"""Command-line utility to compute AstroCLIP embeddings and export them to a NPZ file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from hackathon2025.tools.inference import EmbeddingComputer, ParquetDataSource, StreamingDataSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute AstroCLIP embeddings and export them to a NPZ file.")
    parser.add_argument(
        "--parquet-path",
        default=None,
        help="Chemin vers un parquet local ou hf://datasets/... (si absent, utilisation du dataset AstroCLIP train).",
    )
    parser.add_argument(
        "--focus-high-z",
        action="store_true",
        help="Prioriser les galaxies à haut redshift lors de l'échantillonnage (parquet uniquement).",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint AstroCLIP (.ckpt) contenant les encodeurs.")
    parser.add_argument("--device", default="cuda", help="Device pour l'inférence (ex: cuda, cpu).")
    parser.add_argument("--sample-size", type=int, default=512, help="Nombre d'échantillons à charger.")
    parser.add_argument("--image-size", type=int, default=144, help="Dimension des images (resize/crop).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size utilisé pour l'inférence.")
    parser.add_argument("--slice-length", type=int, default=7700, help="Longueur des spectres après padding/troncature.")
    parser.add_argument("--output", required=True, help="Chemin du fichier .npz à créer.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Forcer l'utilisation du dataset AstroCLIP (streaming) même si --parquet-path est fourni.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Ignorer les caches locaux pour la lecture des données et le calcul des embeddings.",
    )
    return parser.parse_args()


def _build_data_source(args: argparse.Namespace):
    if args.parquet_path and not args.streaming:
        return ParquetDataSource(
            parquet_path=args.parquet_path,
            focus_high_z=args.focus_high_z,
            sample_size=args.sample_size,
            image_size=args.image_size,
            batch_size=args.batch_size,
            enable_cache=not args.disable_cache,
        )
    return StreamingDataSource(
        sample_size=args.sample_size,
        image_size=args.image_size,
        batch_size=args.batch_size,
        enable_cache=not args.disable_cache,
    )


def _tensor_to_chw_array(tensor: Any) -> np.ndarray:
    t = torch.as_tensor(tensor)
    if t.ndim == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 3:
        raise ValueError(f"Image tensor attendu de dimension 3, obtenu {tuple(t.shape)}")
    return t.detach().cpu().numpy()


def _pack_dataframe(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    images = np.stack([_tensor_to_chw_array(img) for img in df["image"]])
    redshift = df["redshift"].to_numpy(dtype=np.float32)
    pair_ids = df.index.to_numpy(dtype=np.int64)
    payload: Dict[str, np.ndarray] = {
        "images": images,
        "redshift": redshift,
        "pair_id": pair_ids,
    }
    if "targetid" in df.columns:
        payload["targetid"] = df["targetid"].to_numpy(dtype=np.int64)
    return payload


def main(args: argparse.Namespace | None = None) -> None:
    parsed = parse_args() if args is None else args

    output_path = Path(parsed.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = _build_data_source(parsed)
    df = source.load()

    embedder = EmbeddingComputer(parsed.checkpoint, parsed.device)
    embeddings = embedder.build_embeddings(
        df=df,
        batch_size=parsed.batch_size,
        slice_length=parsed.slice_length,
        source_signature=source.signature(),
        use_cache=not parsed.disable_cache,
        export_path=output_path,
    )

    df_payload = _pack_dataframe(df)

    metadata = {
        "parquet_path": parsed.parquet_path,
        "checkpoint": parsed.checkpoint,
        "device": parsed.device,
        "sample_size": parsed.sample_size,
        "image_size": parsed.image_size,
        "batch_size": parsed.batch_size,
        "slice_length": parsed.slice_length,
        "source_signature": source.signature(),
        "streaming": bool(parsed.streaming or not parsed.parquet_path),
    }

    payload = {
        **embeddings,
        **df_payload,
        "metadata": np.array(json.dumps(metadata)),
    }

    np.savez_compressed(output_path, **payload)
    print(f"Embeddings sauvegardés dans {output_path}")


if __name__ == "__main__":
    main()
