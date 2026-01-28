# SLURM Job Submission Scripts

This directory contains SLURM batch scripts for running the FMB pipeline on HPC clusters.

## Directory Structure

```
slurm/
├── 01_retrain/          # Stage 1: Model adaptation
├── 02_embeddings/       # Stage 2: Embedding extraction
├── 03_detection/        # Stage 3: Anomaly detection
├── 04_analysis/         # Stage 4: Analysis tasks
├── 05_viz/              # Stage 5: Visualizations
└── logs/                # Job output logs (created automatically)
```

## Usage

### Direct Submission

```bash
# Submit a specific stage
sbatch slurm/01_retrain/aion.sbatch
sbatch slurm/02_embeddings/aion.sbatch
sbatch slurm/03_detection/nfs.sbatch
```

### Via CLI (Recommended)

```bash
# CLI automatically submits the appropriate SLURM script
fmb retrain aion --slurm
fmb embed aion --slurm
fmb detect outliers --slurm
```


## Configuration

If run on Candide@IAP : All scripts expect:
- Environment: `.venv` (virtualenv at repository root)
- Modules: `gcc/13.4.0`, `cuda/12.8`, `intelpython/3-2025.1.0`
- Working directory: `$HOME/foundation-models-benchmark`

Modify the `#SBATCH` directives to adjust:
- `--time` - Walltime limit
- `--mem` - Memory allocation
- `--gres=gpu:N` - Number of GPUs
- `--partition` - Cluster partition

## Logs

Logs are written to `slurm/logs/<job_name>_<job_id>.out` (and `.err` for some jobs).

Create the logs directory if it doesn't exist:
```bash
mkdir -p slurm/logs
```
