"""Local-aware codec manager that supports offline operation."""

from __future__ import annotations

import json
import inspect
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from aion.codecs.manager import CodecManager
from aion.codecs.config import MODALITY_CODEC_MAPPING, CodecType, HF_REPO_ID
from aion.modalities import Modality


class LocalCodecManager(CodecManager):
    """Codec manager that can load codecs from a local snapshot or HF repo."""

    def __init__(self, repo: str | Path, device: str | torch.device = "cpu") -> None:
        super().__init__(device=device)
        self.repo = Path(repo) if Path(repo).exists() else repo
        self._codec_cache = {}

    @staticmethod
    def _config_path(repo: str | Path, modality: type[Modality]) -> Path:
        base = Path(repo)
        return base / "codecs" / modality.name / "config.json"

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_codec(self, modality_type: type[Modality]):
        if modality_type in self._codec_cache:
            return self._codec_cache[modality_type]

        if modality_type not in MODALITY_CODEC_MAPPING:
            raise ValueError(f"No codec mapping found for modality {modality_type}")
        codec_class: CodecType = MODALITY_CODEC_MAPPING[modality_type]

        if isinstance(self.repo, Path):
            config_path = self._config_path(self.repo, modality_type)
            if config_path.exists():
                config = self._load_config(config_path)
                repo_ref = str(self.repo)
            else:
                config_path = hf_hub_download(
                    HF_REPO_ID, f"codecs/{modality_type.name}/config.json"
                )
                config = self._load_config(Path(config_path))
                repo_ref = HF_REPO_ID
        else:
            config_path = hf_hub_download(
                self.repo, f"codecs/{modality_type.name}/config.json"
            )
            config = self._load_config(Path(config_path))
            repo_ref = self.repo

        # Filter config keys to match codec __init__ signature
        init_params = set(inspect.signature(codec_class.__init__).parameters.keys())
        init_params.discard("self")
        filtered_config = {k: v for k, v in config.items() if k in init_params}

        codec = codec_class.from_pretrained(
            repo_ref, modality=modality_type, **filtered_config
        )
        codec = codec.eval()
        self._codec_cache[modality_type] = codec
        return codec
