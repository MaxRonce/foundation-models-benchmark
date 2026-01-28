#!/bin/bash
#SBATCH --job-name=embed_astropt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/embed_astropt_%j.out

set -euo pipefail

echo "Starting embeddings extraction for AstroPT at $(date)"

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

cd "$HOME/foundation-models-benchmark"
source .venv/bin/activate
python3 --version

# Generate embeddings via CLI
python -m fmb.cli embed astropt --config configs/embeddings/astropt.yaml

echo "Finished at $(date)"
