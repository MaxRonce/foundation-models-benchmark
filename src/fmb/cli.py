from __future__ import annotations
import typer
import subprocess
import os
import sys
from pathlib import Path

# --- Path setup for external dependencies ---
repo_root = Path(__file__).resolve().parents[2]
external_paths = [
    repo_root / "external" / "AION",
    repo_root / "external" / "astroPT" / "src",
    repo_root / "external" / "AstroCLIP",
]

for p in external_paths:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
# --------------------------------------------

from typing import Optional, List
import yaml

from fmb.paths import load_paths


app = typer.Typer(
    help="FMB: Foundation Models Benchmark CLI - Refactored Pipeline",
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)

def run_slurm(sbatch_file: str, name: str, extra_args: List[str]):
    """Shorthand to submit a slurm job with optional extra args (passed to the sbatch)."""
    # Note: forwarding extra args to sbatch is tricky; usually sbatch scripts are fixed.
    # But for now we just submit the sbatch.
    cmd = ["sbatch", f"slurm/{sbatch_file}"]
    if extra_args:
        # Some users might want to pass args to sbatch, but here we keep it simple.
        typer.echo(f"  Note: Extra arguments {extra_args} are NOT forwarded to sbatch automatically.")
    
    typer.echo(f" Submitting Slurm job for {name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        typer.echo(f" Job submitted: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Error submitting Slurm job: {e.stderr}")
    except FileNotFoundError:
        typer.echo("❌ 'sbatch' command not found. Are you on a Slurm cluster?")

def forward_args(ctx: typer.Context):
    """Clean up sys.argv to only contain the extra arguments for underlying argparse."""
    # Typer consumes the command and arguments it knows.
    # We want to give the rest to the script's main().
    sys.argv = [sys.argv[0]] + ctx.args

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def retrain(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Model to retrain (aion, astropt, astroclip)"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to YAML config file"),
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a Slurm job instead of running locally")
):
    """Stage 01: Retrain foundation models or codecs."""
    if slurm:
        run_slurm(f"01_retrain/{model}.sbatch", f"retrain {model}", ctx.args)
        return

    typer.echo(f"Running retrain for {model} locally...")
    
    # Build sys.argv for the underlying script
    sys.argv = [sys.argv[0]]
    if config:
        sys.argv.extend(["--config", config])
    # Add any extra args from ctx.args
    sys.argv.extend(ctx.args)
    
    # Use new simplified entry points
    if model == "aion" or model == "aion_codec":
        # AION now only has the Euclid<->HSC adapter U-Net training
        from fmb.models.aion.retrain_euclid_hsc_adapter_unet import main as run_task
    elif model == "astropt":
        from fmb.models.astropt.retrain import main as run_task
    elif model == "astroclip":
        from fmb.models.astroclip.finetune import main as run_task
    else:
        typer.echo(f"❌ Unknown model: {model}")
        raise typer.Exit(1)
    
    run_task()

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def embed(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Model to use for embeddings (aion, astropt, astroclip)"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to model-specific configuration YAML"),
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a Slurm job instead of running locally")
):
    """Stage 02: Generate embeddings from foundation models."""
    # Combine --config with extra args
    extra_args = list(ctx.args)
    if config:
        extra_args.extend(["--config", config])

    if slurm:
        run_slurm(f"02_embeddings/{model}.sbatch", f"embedding {model}", extra_args)
        return

    typer.echo(f"Generating embeddings for {model} locally...")
    
    # Update sys.argv to pass extra args to the script
    # We keep the script name (argv[0]) and append our processed args
    import sys
    sys.argv = [sys.argv[0]] + extra_args

    if model == "aion":
        from fmb.embeddings.generate_embeddings_aion import main as run_task
    elif model == "astropt":
        # ... (Already imported above in original file? No, it was in the block)
        from fmb.embeddings.generate_embeddings_astropt import main as run_task
    elif model == "astroclip":
        from fmb.embeddings.generate_embeddings_astroclip import main as run_task
    else:
        typer.echo(f"❌ Unknown model: {model}")
        raise typer.Exit(1)
    
    run_task()



# --- Detect Commands ---
detect_app = typer.Typer(help="Stage 03: Detect anomalies using embeddings.")
app.add_typer(detect_app, name="detect")

@detect_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def outliers(
    ctx: typer.Context,
    method: str = typer.Option("nfs", "--method", help="Method to use: 'nfs' (Normalizing Flows)"),
    config: Optional[str] = typer.Option(None, "--config", help="Path to config YAML"),
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a Slurm job")
):
    """
    Run Normalizing Flow-based outlier detection.
    Wrapper around fmb.detection.run.
    """
    if slurm:
        # We might need a specific sbatch file for this new command
        # For now, let's assume we use the generic one or a new one
        typer.echo("Slurm submission for 'detect outliers' not yet fully configured with new script. Running locally.")
        # run_slurm(f"03_detection/nfs.sbatch", "detect outliers", ctx.args)
        # return

    typer.echo(f"🔍 Running outlier detection ({method}) locally...")
    
    # Forward args to the new run script
    # We construct the argv manually to map options
    from fmb.detection import run
    
    # Extract known options from context if passed, or just forward everything
    # But Typer consumes config/method. We need to pass them to argparse if they are needed.
    # The run.py uses argparse.
    
    # Reconstruct argv for argparse
    run_args = []
    if config:
        run_args.extend(["--config", config])
    
    # Pass through other arguments (like --aion-embeddings)
    run_args.extend(ctx.args)
    
    run.main(run_args)


# Legacy Detect (Commented out to replace with Group)
# @app.command()
# def detect(
#     ctx: typer.Context,
#     method: str = typer.Argument(..., help="Detection method (cosine, nfs, iforest)"),
#     slurm: bool = typer.Option(False, "--slurm", help="Submit as a Slurm job instead of running locally")
# ):
#     ...

@app.command()
def analyze(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="Analysis task (predict_params, tsne)"),
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a Slurm job instead of running locally")
):
    """Stage 04: Analyze embeddings, predict physics, or visualize."""
    if slurm:
        run_slurm(f"04_analysis/{task}.sbatch", f"analysis {task}", ctx.args)
        return

    typer.echo(f"📊 Running analysis {task} locally...")
    forward_args(ctx)

    if task == "predict_params":
        from fmb.analysis.predict_physical_params import main as run_task
    elif task == "tsne":
        from fmb.viz.plot_paper_tsne_comparison import main as run_task
    else:
        typer.echo(f"❌ Unknown task: {task}")
        raise typer.Exit(1)
    
    run_task()

@app.command()
def display(
    ctx: typer.Context,
    split: str = typer.Option("train", "--split", help="Dataset split to load (train, test, all)"),
    index: int = typer.Option(0, "--index", help="Index of sample to display"),
    save: Optional[str] = typer.Option(None, "--save", help="Path to save the figure"),
    show_bands: bool = typer.Option(False, "--show-bands", help="Display spectrum/SED and individual bands"),
    no_gui: bool = typer.Option(False, "--no-gui", help="Don't open GUI window (save only)")
):
    """Load and display dataset samples."""
    typer.echo(f"📊 Loading dataset split '{split}'...")
    forward_args(ctx)
    
    # Build arguments for the display script
    sys.argv = [sys.argv[0], "--split", split, "--index", str(index)]
    if save:
        sys.argv.extend(["--save", save])
    if show_bands:
        sys.argv.append("--show-bands")
    if no_gui:
        sys.argv.append("--no-gui")
    
    from fmb.data.load_display_data import main as display_main
    display_main()

@app.command()
def paths(
    data: bool = typer.Option(False, "--data", help="Print DATA_ROOT only"),
    embeddings: bool = typer.Option(False, "--embeddings", help="Print EMB_ROOT only"),
    checkpoints: bool = typer.Option(False, "--checkpoints", help="Print CKPT_ROOT only"),
    runs: bool = typer.Option(False, "--runs", help="Print RUNS_ROOT only")
):
    """Display current path configuration."""
    P = load_paths()
    if data:
        typer.echo(P.dataset)
    elif embeddings:
        typer.echo(P.embeddings)
    elif checkpoints:
        typer.echo(P.base_weights)
    elif runs:
        typer.echo(P.runs_root)
    else:
        typer.echo(f"DATA_ROOT:   {P.dataset}")
        typer.echo(f"EMB_ROOT:    {P.embeddings}")
        typer.echo(f"CKPT_ROOT:   {P.base_weights}")
        typer.echo(f"RUNS_ROOT:   {P.runs_root}")
        typer.echo(f"CACHE_ROOT:  {P.cache}")


# --- Viz Commands ---
viz_app = typer.Typer(help="Stage 05: Visualizations for publication and inspection.")
app.add_typer(viz_app, name="viz")

@viz_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def paper_umap(
    ctx: typer.Context,
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a Slurm job instead of running locally")
):
    """Generate the publication-ready combined UMAP figure (AstroPT, AION, AstroCLIP)."""
    if slurm:
        run_slurm("05_viz/paper_umap.sbatch", "viz paper_umap", ctx.args)
        return

    typer.echo("Generating publication combined UMAP plot locally...")
    forward_args(ctx)
    from fmb.viz.plot_paper_combined_umap import main as run_task
    run_task()


if __name__ == "__main__":
    app()
