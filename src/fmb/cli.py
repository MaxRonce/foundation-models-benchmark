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

@detect_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def cosine(
    ctx: typer.Context,
    aion_embeddings: Optional[str] = typer.Option(None, "--aion-embeddings"),
    astropt_embeddings: Optional[str] = typer.Option(None, "--astropt-embeddings"),
    astroclip_embeddings: Optional[str] = typer.Option(None, "--astroclip-embeddings"),
):
    """
    Compute Cosine Similarity (Image vs Spectrum).
    Output: runs/outliers/cosine_scores_{model}.csv
    """
    from fmb.detection import cosine
    # Construct args manually (Typer -> argparse)
    args = []
    if aion_embeddings: args.extend(["--aion-embeddings", aion_embeddings])
    if astropt_embeddings: args.extend(["--astropt-embeddings", astropt_embeddings])
    if astroclip_embeddings: args.extend(["--astroclip-embeddings", astroclip_embeddings])
    
    cosine.main(args)

@detect_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def multimodal(
    ctx: typer.Context,
    top_k: int = typer.Option(200, "--top-k", help="Number of top anomalies to export per model (ranked by fusion score)."),
    fusion: str = typer.Option("geo", "--fusion", help="Fusion method: 'geo' (Geometric Mean), 'min' (Minimum), 'avg' (Average)."),
    t_img: float = typer.Option(0.99, "--t-img", help="Filter: Only keep objects in top P percentile of Image Density Anomaly."),
    t_spec: float = typer.Option(0.99, "--t-spec", help="Filter: Only keep objects in top P percentile of Spectrum Density Anomaly."),
    t_mis: float = typer.Option(0.99, "--t-mis", help="Filter: Only keep objects in top P percentile of Cosine Mismatch."),
):
    """
    Combine & Filter Anomalies (Multimodal Fusion).
    
    Generates a final list of anomalies by combining:
    1. Cosine Mismatch (Image vs Spectrum)
    2. Image Density Outliers (Normalizing Flows)
    3. Spectrum Density Outliers (Normalizing Flows)
    
    Outputs are saved to: runs/outliers/multimodal/
    """
    from fmb.detection import multimodal
    
    args = [
        "--top-k", str(top_k),
        "--fusion", fusion,
        "--t-img", str(t_img),
        "--t-spec", str(t_spec),
        "--t-mis", str(t_mis),
    ]
    multimodal.main(args)



# --- Analyze Commands ---
analyze_app = typer.Typer(help="Stage 04: Analyze embeddings and detection results.")
app.add_typer(analyze_app, name="analyze")

@analyze_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def predict_params(
    ctx: typer.Context,
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a Slurm job instead of running locally")
):
    """Predict physical parameters (redshift, etc.) from embeddings."""
    if slurm:
        run_slurm("04_analysis/predict_params.sbatch", "analysis predict_params", ctx.args)
        return

    typer.echo(f"Running physical parameter prediction locally...")
    forward_args(ctx)
    from fmb.analysis.predict_physical_params import main as run_task
    run_task()

@analyze_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def tsne(
    ctx: typer.Context,
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a Slurm job instead of running locally")
):
    """Generate t-SNE comparison plots."""
    if slurm:
        run_slurm("04_analysis/tsne.sbatch", "analysis tsne", ctx.args)
        return

    typer.echo(f"Running t-SNE analysis locally...")
    forward_args(ctx)
    from fmb.viz.plot_paper_tsne_comparison import main as run_task
    run_task()

@analyze_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def outliers(
    ctx: typer.Context,
    input_csv: str = typer.Option(None, "--input-csv", help="Path to all_scores.csv"),
    top_k: int = typer.Option(200, "--top-k", help="Top-K threshold"),
    slurm: bool = typer.Option(False, "--slurm", help="Submit as a Slurm job")
):
    """
    Analyze Multimodal Anomaly Results (Correlations, Uplift, Overlap).
    Input: runs/outliers/multimodal/all_scores.csv
    Output: runs/analysis/outliers/
    """
    if slurm:
        # submit slurm
        typer.echo("Slurm not configured for analysis outliers yet.")
        pass

    typer.echo("Analyzing anomaly results...")
    
    from fmb.analysis import outliers
    
    args = []
    if input_csv: args.extend(["--input_csv", input_csv])
    if top_k: args.extend(["--top-k", str(top_k)])
    
    outliers.main(args)

@analyze_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def similarity(
    ctx: typer.Context,
    emb_path: Optional[str] = typer.Option(None, "--embeddings", help="Path to embeddings .pt file (optional, auto-detected if omitted)"),
    model: Optional[str] = typer.Option(None, "--model", help="Name of the model (or 'all'). Default: all"),
    queries: Optional[List[str]] = typer.Option(None, "--query", help="Object ID(s) to query"),
    query_csv: Optional[str] = typer.Option(None, "--query-csv", help="CSV with object_id column to query"),
    n_similar: int = typer.Option(5, "--n-similar", help="Number of neighbors"),
    save: Optional[str] = typer.Option(None, "--save", help="Output image path"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir"),
):
    """
    Visual Similarity Search.
    Finds and displays nearest neighbors (Images + Spectra).
    """
    from fmb.analysis import similarity
    from fmb.data.utils import read_object_ids
    from fmb.paths import load_paths
    from pathlib import Path
    
    paths = load_paths()
    
    # 1. Resolve Queries
    q_ids = []
    if queries: q_ids.extend(queries)
    if query_csv:
        q_ids.extend(read_object_ids([Path(query_csv)]))
        
    if not q_ids:
        typer.echo("❌ No query IDs provided.")
        raise typer.Exit(1)
        
    # 2. Resolve Tasks (Model, Path)
    tasks = []
    
    if emb_path:
        # Explicit path provided
        tsk_name = model if model else "CustomModel"
        tasks.append((tsk_name, Path(emb_path)))
    else:
        # Auto-detect
        emb_root = paths.embeddings
        if not emb_root.exists():
            typer.echo(f"❌ Embeddings root not found: {emb_root}")
            raise typer.Exit(1)
            
        candidates = []
        # Simple heuristic: scan directory for known .pt files
        pt_files = list(emb_root.glob("*embeddings*.pt"))
        
        # Also check subdirectories if organized by model?
        # paths.embeddings might contain 'aion/embeddings.pt', etc.
        # But commonly they are in root of run.
        
        for p in pt_files:
            fname = p.name.lower()
            m_name = "Unknown"
            if "astropt" in fname: m_name = "AstroPT"
            elif "astroclip" in fname: m_name = "AstroCLIP"
            elif "aion" in fname: m_name = "AION"
            else: m_name = fname.replace("embeddings", "").replace(".pt", "").strip("_").capitalize()
            candidates.append((m_name, p))
            
        # Filter
        if model and model.lower() != "all":
            tasks = [t for t in candidates if t[0].lower() == model.lower()]
            if not tasks:
                 typer.echo(f"❌ No embeddings found matching model '{model}'. Found: {[c[0] for c in candidates]}")
                 raise typer.Exit(1)
        else:
            tasks = candidates

    if not tasks:
         typer.echo(f"❌ No embedding files found in {paths.embeddings}")
         raise typer.Exit(1)
         
    # 3. Resolve Paths
    if save:
        out_path = Path(save)
    else:
        # Default save location
        analysis_dir = paths.analysis / "similarity"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        if len(tasks) == 1:
            name = tasks[0][0].lower()
            out_path = analysis_dir / f"similarity_{name}.png"
        else:
            out_path = analysis_dir / "similarity_combined.png"
            
    if not cache_dir:
        cache_dir = str(paths.dataset)

    typer.echo(f"🔍 Finding similar objects for {len(q_ids)} queries...")
    typer.echo(f"   Tasks: {[t[0] for t in tasks]}")
    typer.echo(f"   Output: {out_path}")
    
    similarity.visualize_similarity(
        query_ids=q_ids,
        tasks=tasks,
        n_similar=n_similar,
        output_path=out_path,
        cache_dir=cache_dir
    )

@analyze_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def neighbor_ranks(
    ctx: typer.Context,
    emb_path: Optional[str] = typer.Option(None, "--embeddings", help="Path to embeddings .pt file (optional, auto-detected if omitted)"),
    scores_path: Optional[str] = typer.Option(None, "--scores", help="Path to anomaly_scores.csv (optional, auto-detected if omitted)"),
    model: Optional[str] = typer.Option(None, "--model", help="Filter by model name (default: all)"),
    queries: Optional[List[str]] = typer.Option(None, "--query", help="Object ID(s) to query"),
    query_csv: Optional[str] = typer.Option(None, "--query-csv", help="CSV with object_id column to query"),
    n_similar: int = typer.Option(10, "--n-similar", help="Number of neighbors"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Output directory"),
):
    """
    Analyze Rank Distribution of Neighbors.
    Checks if neighbors of anomalies are also anomalies.
    """
    from fmb.analysis import similarity
    from fmb.data.utils import read_object_ids
    from fmb.paths import load_paths
    from pathlib import Path
    
    paths = load_paths()
    
    # 1. Resolve Queries
    q_ids = []
    if queries: q_ids.extend(queries)
    if query_csv:
        q_ids.extend(read_object_ids([Path(query_csv)]))
        
    if not q_ids:
        typer.echo("❌ No query IDs provided.")
        raise typer.Exit(1)

    # 2. Resolve Tasks (Model, EmbPath, ScorePath)
    tasks = []
    
    if emb_path and scores_path:
        # Explicit
        m_name = model if model else "CustomModel"
        tasks.append((m_name, Path(emb_path), Path(scores_path)))
    else:
        # Auto-detect
        emb_root = paths.embeddings
        score_root = paths.outliers # Usually where anomaly_scores_*.csv live
        
        candidates = []
        pt_files = list(emb_root.glob("*embeddings*.pt"))
        
        for p in pt_files:
            fname = p.name.lower()
            m_name = "Unknown"
            if "astropt" in fname: m_name = "AstroPT"
            elif "astroclip" in fname: m_name = "AstroCLIP"
            elif "aion" in fname: m_name = "AION"
            else: m_name = fname.replace("embeddings", "").replace(".pt", "").strip("_").capitalize()
            
            # Try to find matching score
            slug = m_name.lower().replace(" ", "")
            possible_names = [
                f"anomaly_scores_{slug}.csv",
                f"scores_{slug}.csv",
                f"{slug}_scores.csv"
            ]
            
            found_score = None
            for sname in possible_names:
                sp = score_root / sname
                if sp.exists():
                    found_score = sp
                    break
            
            if found_score:
                candidates.append((m_name, p, found_score))
            else:
                pass
                
        # Filter
        if model and model.lower() != "all":
            tasks = [t for t in candidates if t[0].lower() == model.lower()]
            if not tasks:
                 typer.echo(f"❌ No valid tasks (embedding+scores) found matching model '{model}'.")
                 raise typer.Exit(1)
        else:
            tasks = candidates

    if not tasks:
        typer.echo("❌ No valid tasks found. Ensure embeddings(.pt) and scores(.csv) exist and match.")
        typer.echo(f"   Embeddings: {paths.embeddings}")
        typer.echo(f"   Scores: {paths.outliers}")
        raise typer.Exit(1)

    # 3. Resolve Output
    if out_dir:
        out_path = Path(out_dir)
    else:
        out_path = paths.analysis / "neighbors"
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"📊 Analyzing neighbor ranks for {len(q_ids)} queries...")
    typer.echo(f"   Tasks: {[(t[0], str(t[2].name)) for t in tasks]}")
    typer.echo(f"   Output: {out_path}")
    
    similarity.analyze_neighbor_ranks(
        query_ids=q_ids,
        tasks=tasks,
        n_similar=n_similar,
        output_dir=out_path
    )

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
