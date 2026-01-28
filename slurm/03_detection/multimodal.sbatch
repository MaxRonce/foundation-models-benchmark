#!/bin/bash
#SBATCH --job-name=detect_multimodal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/detect_multimodal_%j.out

set -euo pipefail

echo "Starting multimodal anomaly fusion at $(date)"

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

cd "$HOME/foundation-models-benchmark"
source .venv/bin/activate
python3 --version

# Run multimodal fusion via CLI
python -m fmb.cli detect multimodal --top-k 200 --fusion geo

echo "Finished at $(date)"
