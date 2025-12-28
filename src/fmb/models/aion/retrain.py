from __future__ import annotations
from pathlib import Path
import subprocess
from fmb.paths import get_paths

def run(cfg: dict, run_dir: Path) -> None:
    p = get_paths()
    script = p.external / "AION" / cfg["entrypoint"]  # ex: "scratch/retrain_euclid_codec.py"
    cmd = [cfg.get("python", "python"), str(script)] + cfg.get("args", [])
    run_dir.mkdir(parents=True, exist_ok=True)

    # log command
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")

    # execute
    subprocess.run(cmd, check=True, cwd=str(p.external / "AION"))
