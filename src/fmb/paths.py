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
    
    # Specific specialized paths
    dataset: Path
    dataset_train: Path
    dataset_test: Path
    dataset_hf_id: str
    base_weights: Path

    
    # Model specific base weights (can be configured separately)
    base_weights_aion: Path
    base_weights_astropt: Path
    base_weights_astroclip: Path

    retrained_weights: Path
    nfs_weights: Path
    outliers: Path
    analysis: Path
    
    # Kept for backward compat or generic usage if needed
    embeddings: Path
    cache: Path
    
    # Fallback/base roots
    storage_root: Path
    runs_root: Path
    
    def ensure(self) -> "FMBPaths":
        # Create directories that are meant to be output directories
        # dataset and base_weights are input dirs usually, so we might not want to mkdir them blindly?
        # But if they are just roots, it's safer to ensure they exist or warn.
        # Let's ensure output dirs.
        for p in [self.retrained_weights, self.nfs_weights, self.outliers, self.analysis, self.embeddings, self.cache, self.runs_root]:
            p.mkdir(parents=True, exist_ok=True)
        return self

    def embeddings_dir(self, model: str) -> Path:
        d = self.embeddings / model
        d.mkdir(parents=True, exist_ok=True)
        return d
    
    # Helper for generic "run" outputs if needed
    def new_run_dir(self, tag: str) -> Path:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = self.runs_root / f"{stamp}_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        return d

_CACHED_PATHS: Optional[FMBPaths] = None

def load_paths(config_path: Optional[Path] = None, *, ensure: bool = True) -> FMBPaths:
    global _CACHED_PATHS
    if _CACHED_PATHS is not None and config_path is None:
        return _CACHED_PATHS

    repo_root = _repo_root()

    if config_path is None:
        env_cfg = os.environ.get("FMB_PATHS_CONFIG")
        if env_cfg:
            config_path = Path(env_cfg)

    # New config location: src/fmb/configs/
    if config_path is None:
        configs_dir = repo_root / "src" / "fmb" / "configs"
        # Try paths_local.yaml first (user-specific), then paths.template.yaml (default)
        candidates = [
            configs_dir / "paths_local.yaml",
            configs_dir / "paths.template.yaml",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break
    
    if config_path is None or not config_path.exists():
         # Fallback to template
         config_path = repo_root / "src" / "fmb" / "configs" / "paths.template.yaml"
    
    # Load config if valid
    cfg = {}
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}

    # 1. Storage Root (Default Base)
    storage_root_str = cfg.get("storage_root")
    if storage_root_str:
        storage_root = _p(storage_root_str)
    else:
        storage_root = repo_root

    # Helper to resolve path: check absolute, or relative to storage_root
    def resolve(key: str, default_subpath: str) -> Path:
        val = cfg.get(key)
        if val:
            return _p(val)
        return storage_root / default_subpath

    def resolve_optional(key: str, default_abs: Path) -> Path:
        val = cfg.get(key)
        if val:
            return _p(val)
        return default_abs

    # 2. Resolve requested paths
    dataset_path = resolve("dataset_path", "data")
    dataset_path_train = resolve_optional("dataset_path_train", dataset_path / "train")
    dataset_path_test = resolve_optional("dataset_path_test", dataset_path / "test")
    dataset_hf_id = cfg.get("dataset_hf_id", "msiudek/astroPT_euclid_Q1_desi_dr1_dataset")

    base_weights_path = resolve("base_weights_path", "checkpoints/base")
    
    # Resolve model specific weights, defaulting to base_weights/<model_name>
    base_weights_aion = resolve_optional("base_weights_path_aion", base_weights_path / "aion")
    base_weights_astropt = resolve_optional("base_weights_path_astropt", base_weights_path / "astropt")
    base_weights_astroclip = resolve_optional("base_weights_path_astroclip", base_weights_path / "astroclip")

    retrained_weights_path = resolve("retrained_weights_path", "checkpoints/retrained")
    nfs_weights_path = resolve("nfs_weights_path", "checkpoints/nfs")
    outliers_path = resolve("outliers_path", "outputs/outliers")
    analysis_path = resolve("analysis_path", "outputs/analysis")
    
    # Generic/Legacy
    emb_path = resolve("embeddings_path", "embeddings") # Also support embeddings_path
    if "emb_root" in cfg:
        emb_path = _p(cfg["emb_root"])
    
    cache_path = resolve("cache_root", "cache")
    runs_path = resolve("runs_root", "runs")

    paths = FMBPaths(
        repo_root=repo_root,
        storage_root=storage_root,
        dataset=dataset_path,
        dataset_train=dataset_path_train,
        dataset_test=dataset_path_test,
        dataset_hf_id=dataset_hf_id,
        base_weights=base_weights_path,
        base_weights_aion=base_weights_aion,
        base_weights_astropt=base_weights_astropt,
        base_weights_astroclip=base_weights_astroclip,
        retrained_weights=retrained_weights_path,
        nfs_weights=nfs_weights_path,
        outliers=outliers_path,
        analysis=analysis_path,
        embeddings=emb_path,
        cache=cache_path,
        runs_root=runs_path,
    )

    if ensure:
        paths.ensure()
    
    _CACHED_PATHS = paths
    return paths
