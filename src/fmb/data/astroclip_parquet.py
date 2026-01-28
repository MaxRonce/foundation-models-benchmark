"""
Foundation Models Benchmark (FMB)

Module: fmb.data.astroclip_parquet
Description: Parquet data source for AstroCLIP
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import EntryNotFoundError
from PIL import Image

# Define constants locally if needed, or rely on env
DEFAULT_CACHE_DIR = Path(os.getcwd()) / ".cache"
CACHE_DIR = Path(os.environ.get("ASTROCLIP_CACHE_DIR", DEFAULT_CACHE_DIR))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def zscore_image_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """Apply a per-channel z-score normalisation."""
    tensor = torch.as_tensor(image_tensor, dtype=torch.float32)
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.shape}")
    dims = (-1, -2) if tensor.ndim == 4 else (-2, -1)
    mean = tensor.mean(dim=dims, keepdim=True)
    std = tensor.std(dim=dims, keepdim=True, unbiased=False).clamp(min=1e-6)
    return (tensor - mean) / std


def _cache_path(prefix: str, **kwargs: Any) -> Path:
    payload = json.dumps(kwargs, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return CACHE_DIR / f"{prefix}_{digest}"


def resolve_parquet_path(raw_path: str) -> str:
    """Resolve a local or HuggingFace parquet path into a local filename."""
    if not raw_path:
        raise ValueError("Chemin parquet vide.")

    raw_path = raw_path.strip()
    if raw_path.startswith("hf://"):
        path_no_scheme = raw_path[len("hf://") :]
        if not path_no_scheme.startswith("datasets/"):
            raise ValueError("Chemin HuggingFace invalide (attendu hf://datasets/...).")

        parts = path_no_scheme.split("/")
        if len(parts) < 4:
            raise ValueError("Chemin HuggingFace incomplet (repo et fichier attendus).")

        repo_id = "/".join(parts[1:3])
        inner_path = "/".join(parts[3:])
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=inner_path,
                repo_type="dataset",
                cache_dir=str(CACHE_DIR / "hf_cache"),
            )
        except EntryNotFoundError as exc:
            files = list_repo_files(repo_id=repo_id, repo_type="dataset")
            guesses = [f for f in files if inner_path.split("/")[-1] in f]
            hint = f" Exemples trouvés: {guesses[:5]}" if guesses else ""
            raise FileNotFoundError(f"{inner_path} introuvable sur Hugging Face.{hint}") from exc

    path_obj = Path(raw_path).expanduser()
    if not path_obj.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path_obj}")
    return str(path_obj)


class DataSource(ABC):
    """Abstract loader returning DataFrames with image/spectrum/redshift pairs."""

    def __init__(
        self,
        sample_size: Optional[int],
        image_size: int,
        batch_size: int,
        enable_cache: bool = True,
    ) -> None:
        self.sample_size = int(sample_size) if sample_size is not None else None
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.enable_cache = enable_cache

    def load(self) -> pd.DataFrame:
        """Load the data frame, optionally using the on-disk cache."""
        if not self.enable_cache:
            return self._load_impl()

        cache_path = _cache_path(
            "df",
            source=self.__class__.__name__,
            sample=self.sample_size if self.sample_size is not None else "all",
            image=self.image_size,
            batch=self.batch_size,
            signature=self.signature(),
        ).with_suffix(".pkl")

        if cache_path.exists():
            return pd.read_pickle(cache_path)

        df = self._load_impl()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        return df

    @abstractmethod
    def signature(self) -> str:
        """Identifier for cache invalidation."""

    @abstractmethod
    def _load_impl(self) -> pd.DataFrame:
        """Concrete loader implementation."""


class ParquetDataSource(DataSource):
    """Load samples from a parquet dataset stored locally or on the Hub."""

    def __init__(
        self,
        parquet_path: str,
        focus_high_z: bool,
        sample_size: Optional[int],
        image_size: int,
        batch_size: int,
        enable_cache: bool = False,
    ) -> None:
        super().__init__(sample_size, image_size, batch_size, enable_cache=enable_cache)
        self.parquet_path = parquet_path
        self.focus_high_z = focus_high_z
        self._transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
            ]
        )

    def signature(self) -> str:
        return f"path={self.parquet_path}|focus={self.focus_high_z}"

    def _preprocess_image(self, blob: Dict[str, Any]) -> torch.Tensor:
        img = Image.open(io.BytesIO(blob["bytes"])).convert("RGB")
        return self._transform(img)

    def _load_impl(self) -> pd.DataFrame:
        resolved_path = resolve_parquet_path(self.parquet_path)
        df = pd.read_parquet(resolved_path)

        if "image" not in df.columns and "RGB_image" in df.columns:
            df["image"] = df["RGB_image"].apply(self._preprocess_image)
        elif "image" not in df.columns:
            raise ValueError("Le parquet doit contenir une colonne 'image' ou 'RGB_image'.")

        if "redshift" not in df.columns:
            raise ValueError("La colonne 'redshift' est absente du parquet.")

        df = df.dropna(subset=["redshift"]).reset_index(drop=True)

        if self.sample_size is not None and len(df) > self.sample_size:
            if self.focus_high_z:
                df = df.nlargest(self.sample_size, "redshift").reset_index(drop=True)
            else:
                df = df.sample(self.sample_size, random_state=42).sort_index().reset_index(drop=True)

        df["pair_id"] = np.arange(len(df))
        return df
