from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import datetime as dt
import importlib.util

@dataclass(frozen=True)
class FMBPaths:
    data: Path
    embeddings: Path
    checkpoints: Path
    runs: Path
    cache: Path

    def ensure(self) -> "FMBPaths":
        for p in [self.data, self.embeddings, self.checkpoints, self.runs, self.cache]:
            p.mkdir(parents=True, exist_ok=True)
        return self

    def new_run_dir(self, tag: str) -> Path:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        d = self.runs / f"{stamp}_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        return d

def load_paths_from_py(py_path: Path) -> FMBPaths:
    spec = importlib.util.spec_from_file_location("fmb_paths_local", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import paths file: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    paths = FMBPaths(
        data=Path(mod.DATA_ROOT),
        embeddings=Path(mod.EMB_ROOT),
        checkpoints=Path(mod.CKPT_ROOT),
        runs=Path(mod.RUNS_ROOT),
        cache=Path(mod.CACHE_ROOT),
    )
    return paths.ensure()
