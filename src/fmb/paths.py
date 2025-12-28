from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import datetime as dt
import yaml

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _expand_vars(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))

def _p(v: str | Path) -> Path:
    return Path(_expand_vars(str(v))).resolve()

@dataclass(frozen=True)
class FMBPaths:
    repo_root: Path
    data: Path
    embeddings: Path
    checkpoints: Path
    runs: Path
    cache: Path

    def ensure(self) -> "FMBPaths":
        for p in [self.data, self.embeddings, self.checkpoints, self.runs, self.cache]:
            p.mkdir(parents=True, exist_ok=True)
        return self

    def embeddings_dir(self, model: str) -> Path:
        d = self.embeddings / model
        d.mkdir(parents=True, exist_ok=True)
        return d

    def checkpoints_dir(self, model: str) -> Path:
        d = self.checkpoints / model
        d.mkdir(parents=True, exist_ok=True)
        return d

    def new_run_dir(self, tag: str) -> Path:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = self.runs / f"{stamp}_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        return d

def load_paths(config_path: Optional[Path] = None, *, ensure: bool = True) -> FMBPaths:
    repo_root = _repo_root()

    if config_path is None:
        env_cfg = os.environ.get("FMB_PATHS_CONFIG")
        if env_cfg:
            config_path = Path(env_cfg)

    if config_path is None:
        config_path = repo_root / "configs" / "storage.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Paths config not found: {config_path}. "
            "Create it (e.g. configs/storage.yaml) or set FMB_PATHS_CONFIG."
        )

    cfg = yaml.safe_load(config_path.read_text())

    storage_root = cfg.get("storage_root")
    if storage_root:
        storage_root = _p(storage_root)
    else:
        storage_root = repo_root

    data = _p(cfg.get("data_root", storage_root / "data"))
    emb  = _p(cfg.get("emb_root", storage_root / "embeddings"))
    ckpt = _p(cfg.get("ckpt_root", storage_root / "checkpoints"))
    runs = _p(cfg.get("runs_root", storage_root / "runs"))
    cache = _p(cfg.get("cache_root", storage_root / "cache"))

    paths = FMBPaths(
        repo_root=repo_root,
        data=data,
        embeddings=emb,
        checkpoints=ckpt,
        runs=runs,
        cache=cache,
    )
    return paths.ensure() if ensure else paths
