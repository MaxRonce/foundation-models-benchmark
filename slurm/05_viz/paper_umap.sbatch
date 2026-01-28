#!/bin/bash
#SBATCH --job-name=viz_umap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/viz_umap_%j.out

set -euo pipefail

echo "Starting UMAP visualization at $(date)"

module purge
module load intelpython/3-2025.1.0

cd "$HOME/foundation-models-benchmark"
source .venv/bin/activate
python3 --version

# Generate combined UMAP plot
python -m fmb.cli viz paper-umap

echo "Finished at $(date)"
