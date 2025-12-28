from __future__ import annotations
import typer
from pathlib import Path
import yaml

from fmb.paths import new_run_dir

app = typer.Typer(no_args_is_help=True)

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@app.command()
def retrain(model: str, config: Path):
    cfg = load_yaml(config)
    run_dir = new_run_dir(f"retrain_{model}")
    typer.echo(f"[retrain] model={model} config={config} run={run_dir}")

    if model == "aion":
        from fmb.models.aion.retrain import run
    elif model == "astroclip":
        from fmb.models.astroclip.retrain import run
    elif model == "astropt":
        from fmb.models.astropt.retrain import run
    else:
        raise typer.BadParameter("model must be one of: aion, astroclip, astropt")

    run(cfg, run_dir)

@app.command()
def embed(model: str, config: Path):
    cfg = load_yaml(config)
    run_dir = new_run_dir(f"embed_{model}")
    typer.echo(f"[embed] model={model} config={config} run={run_dir}")

    if model == "aion":
        from fmb.models.aion.embed import run
    elif model == "astroclip":
        from fmb.models.astroclip.embed import run
    elif model == "astropt":
        from fmb.models.astropt.embed import run
    else:
        raise typer.BadParameter("model must be one of: aion, astroclip, astropt")

    run(cfg, run_dir)

@app.command()
def detect(method: str, config: Path):
    cfg = load_yaml(config)
    run_dir = new_run_dir(f"detect_{method}")
    typer.echo(f"[detect] method={method} config={config} run={run_dir}")

    if method == "cosine":
        from fmb.detection.cosine import run
    elif method == "nfs":
        from fmb.detection.nfs import run
    else:
        raise typer.BadParameter("method must be one of: cosine, nfs")

    run(cfg, run_dir)

@app.command()
def analyze(task: str, config: Path):
    cfg = load_yaml(config)
    run_dir = new_run_dir(f"analyze_{task}")
    typer.echo(f"[analyze] task={task} config={config} run={run_dir}")

    if task == "metrics":
        from fmb.analysis.metrics import run
    elif task == "regression":
        from fmb.analysis.regression import run
    elif task == "similarity":
        from fmb.analysis.similarity import run
    else:
        raise typer.BadParameter("task must be one of: metrics, regression, similarity")

    run(cfg, run_dir)
