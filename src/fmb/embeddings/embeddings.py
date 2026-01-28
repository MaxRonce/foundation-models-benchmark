"""
Foundation Models Benchmark (FMB)

Module: fmb.embeddings.embeddings
Description: Shared embedding generation utilities
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
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import EntryNotFoundError
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from astroclip.data.datamodule import AstroClipCollator
from astroclip.models import AstroClipModel

HACKATHON_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = HACKATHON_ROOT / ".cache"
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


def batch_to_records(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a batch from the streaming dataloader into record dictionaries."""
    records: List[Dict[str, Any]] = []
    spec_batch = batch["spectrum"]
    target_batch = batch.get("targetid")
    for idx in range(batch["image"].shape[0]):
        spec_sample = spec_batch[idx]
        flux: np.ndarray
        wavelength: np.ndarray

        if isinstance(spec_sample, dict):
            flux = np.asarray(spec_sample["flux"])
            wavelength = np.asarray(spec_sample["wavelength"])
        else:
            spec_tensor = torch.as_tensor(spec_sample).cpu()
            if spec_tensor.ndim == 1:
                flux = spec_tensor.numpy()
                wavelength = np.linspace(0, 1, spec_tensor.shape[0])
            elif spec_tensor.ndim == 2:
                if spec_tensor.shape[0] == 2:
                    flux = spec_tensor[0].numpy()
                    wavelength = spec_tensor[1].numpy()
                elif spec_tensor.shape[1] == 2:
                    flux = spec_tensor[:, 0].numpy()
                    wavelength = spec_tensor[:, 1].numpy()
                elif spec_tensor.shape[1] == 1:
                    flux = spec_tensor[:, 0].numpy()
                    wavelength = np.linspace(0, 1, spec_tensor.shape[0])
                elif spec_tensor.shape[0] == 1:
                    flux = spec_tensor[0].numpy()
                    wavelength = np.linspace(0, 1, spec_tensor.shape[1])
                else:
                    raise ValueError(f"Format de spectre inattendu: {spec_tensor.shape}")
            else:
                raise ValueError(f"Spectre NDIM={spec_tensor.ndim} non pris en charge")

        target_value = int(target_batch[idx]) if target_batch is not None else -1
        image_tensor = torch.as_tensor(batch["image"][idx]).cpu().float()

        records.append(
            {
                "image": image_tensor,
                "redshift": float(batch["redshift"][idx]),
                "targetid": target_value,
                "spectrum": {"flux": flux, "wavelength": wavelength},
            }
        )
    return records


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


class StreamingDataSource(DataSource):
    """Load samples from the official streaming AstroCLIP dataset on HuggingFace."""

    def __init__(self, sample_size: int, image_size: int, batch_size: int, enable_cache: bool = True) -> None:
        if sample_size is None:
            raise ValueError("StreamingDataSource requiert un sample_size explicite.")
        super().__init__(sample_size, image_size, batch_size, enable_cache=enable_cache)

    def signature(self) -> str:
        return f"hf_train|images={self.image_size}|batch={self.batch_size}"

    def _load_impl(self) -> pd.DataFrame:
        dataset = load_dataset("EiffL/AstroCLIP", streaming=True, split="train").with_format("torch")
        collator = AstroClipCollator(center_crop=self.image_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collator, drop_last=False)

        records: List[Dict[str, Any]] = []
        for batch in loader:
            records.extend(batch_to_records(batch))
            if self.sample_size is not None and len(records) >= self.sample_size:
                break

        if not records:
            raise RuntimeError("Impossible de récupérer des exemples depuis le stream Hugging Face.")

        df = pd.DataFrame(records[: self.sample_size] if self.sample_size is not None else records)
        df["pair_id"] = np.arange(len(df))
        return df


class AstroClipPairDataset(Dataset):
    """Dataset that pads or trims spectra and returns tensors ready for AstroCLIP."""

    def __init__(self, df: pd.DataFrame, slice_length: int = 1024) -> None:
        self.df = df.reset_index(drop=True)
        self.slice_length = int(slice_length)

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
            flux_arr = np.asarray(spec["flux"])
            wave_arr = np.asarray(spec.get("wavelength"))
        else:
            spec_tensor = torch.as_tensor(spec)
            if spec_tensor.ndim == 1:
                flux_arr = spec_tensor.numpy()
                wave_arr = np.linspace(0, 1, spec_tensor.shape[0])
            elif spec_tensor.ndim == 2:
                if spec_tensor.shape[0] == 2:
                    flux_arr = spec_tensor[0].numpy()
                    wave_arr = spec_tensor[1].numpy()
                elif spec_tensor.shape[1] == 2:
                    flux_arr = spec_tensor[:, 0].numpy()
                    wave_arr = spec_tensor[:, 1].numpy()
                elif spec_tensor.shape[1] == 1:
                    flux_arr = spec_tensor[:, 0].numpy()
                    wave_arr = np.linspace(0, 1, spec_tensor.shape[0])
                elif spec_tensor.shape[0] == 1:
                    flux_arr = spec_tensor[0].numpy()
                    wave_arr = np.linspace(0, 1, spec_tensor.shape[1])
                else:
                    raise ValueError(f"Format de spectre inattendu: {spec_tensor.shape}")
            else:
                raise ValueError(f"Spectre NDIM={spec_tensor.ndim} non pris en charge")

        flux = self._pad_or_trim(flux_arr)
        wave = self._pad_or_trim(wave_arr)
        spectrum = flux.unsqueeze(-1)

        image_tensor = row["image"]
        image = image_tensor.detach().clone().float() if isinstance(image_tensor, torch.Tensor) else torch.as_tensor(
            image_tensor, dtype=torch.float32
        )

        redshift = torch.tensor(row["redshift"], dtype=torch.float32)
        sample = {
            "spectrum": spectrum,
            "image": image,
            "redshift": redshift,
            "wavelength": wave,
        }
        if "pair_id" in row:
            sample["pair_id"] = torch.tensor(row["pair_id"], dtype=torch.long)
        return sample


def approx_distance_mpc(redshift: float) -> float:
    """Approximate cosmological distance in Mpc using Hubble law."""
    hubble_km_s_mpc = 70.0
    c_km_s = 299_792.458
    return (c_km_s / hubble_km_s_mpc) * float(redshift)


class EmbeddingComputer:
    """Compute and persist AstroCLIP embeddings for image/spectrum pairs."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model: Optional[AstroClipModel] = None

    def _load_model(self) -> AstroClipModel:
        if self._model is None:
            try:
                model = AstroClipModel.load_from_checkpoint(self.checkpoint_path)
            except KeyError as exc:
                if "pytorch-lightning_version" not in str(exc):
                    raise
                ckpt_data = torch.load(self.checkpoint_path, map_location="cpu")
                if "pytorch-lightning_version" not in ckpt_data:
                    ckpt_data["pytorch-lightning_version"] = "2.3.3"
                    patched_dir = CACHE_DIR / "patched_checkpoints"
                    patched_dir.mkdir(exist_ok=True)
                    patched_path = patched_dir / (Path(self.checkpoint_path).name + ".patched")
                    torch.save(ckpt_data, patched_path)
                    model = AstroClipModel.load_from_checkpoint(str(patched_path))
                else:
                    raise
            model = model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    def _cache_file(
        self,
        batch_size: int,
        slice_length: int,
        source_signature: str,
        df_len: int,
    ) -> Path:
        return _cache_path(
            "embeddings",
            checkpoint=self.checkpoint_path,
            device=self.device,
            batch=batch_size,
            slice_length=slice_length,
            source=source_signature,
            size=df_len,
        ).with_suffix(".npz")

    def load_cached_embeddings(
        self,
        batch_size: int,
        slice_length: int,
        source_signature: str,
        df_len: int,
    ) -> Optional[Dict[str, Any]]:
        cache_path = self._cache_file(batch_size, slice_length, source_signature, df_len)
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            return {key: cached[key] for key in cached.files}
        return None

    def invalidate_cache_entry(
        self,
        batch_size: int,
        slice_length: int,
        source_signature: str,
        df_len: int,
    ) -> None:
        cache_path = self._cache_file(batch_size, slice_length, source_signature, df_len)
        if cache_path.exists():
            cache_path.unlink()

    def build_embeddings(
        self,
        df: pd.DataFrame,
        batch_size: int,
        slice_length: int,
        source_signature: str,
        *,
        use_cache: bool = True,
        export_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Compute embeddings and optional PCA projection."""
        cache_path = self._cache_file(batch_size, slice_length, source_signature, len(df))

        if use_cache and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            return {key: cached[key] for key in cached.files}

        dataset = AstroClipPairDataset(df, slice_length=slice_length)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

        model = self._load_model()

        cos_sims: List[torch.Tensor] = []
        img_embeds: List[torch.Tensor] = []
        spec_embeds: List[torch.Tensor] = []
        flux_batches: List[torch.Tensor] = []
        wave_batches: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                image_tensor = batch["image"].to(self.device)
                spectrum_tensor = batch["spectrum"].to(self.device)

                image_embeddings = model(image_tensor, input_type="image")
                spectrum_embeddings = model(spectrum_tensor, input_type="spectrum")

                similarity = torch.nn.functional.cosine_similarity(image_embeddings, spectrum_embeddings, dim=1)

                cos_sims.append(similarity.cpu())
                img_embeds.append(image_embeddings.cpu())
                spec_embeds.append(spectrum_embeddings.cpu())
                flux_batches.append(batch["spectrum"].squeeze(-1).cpu())
                wave_batches.append(batch["wavelength"].cpu())

        cos_sims_t = torch.cat(cos_sims).numpy()
        img_embeds_t = torch.cat(img_embeds).numpy()
        spec_embeds_t = torch.cat(spec_embeds).numpy()
        flux_t = torch.cat(flux_batches).numpy()
        wave_t = torch.cat(wave_batches).numpy()

        joint_embedding = 0.5 * (img_embeds_t + spec_embeds_t)

        pca = PCA(n_components=2, random_state=42).fit(joint_embedding)
        joint_pca = pca.transform(joint_embedding)
        explained_ratio = pca.explained_variance_ratio_

        payload = {
            "cosine_similarity": cos_sims_t,
            "image_embeddings": img_embeds_t,
            "spectrum_embeddings": spec_embeds_t,
            "joint_pca": joint_pca,
            "pca_variance": explained_ratio,
            "flux": flux_t,
            "wavelength": wave_t,
        }

        np.savez_compressed(cache_path, **payload)
        if export_path is not None:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(export_path, **payload)
        return payload


def clear_cache() -> None:
    """Remove cached DataFrame and embedding artefacts."""
    if not CACHE_DIR.exists():
        return
    for file in CACHE_DIR.glob("**/*"):
        if file.is_file():
            file.unlink()
