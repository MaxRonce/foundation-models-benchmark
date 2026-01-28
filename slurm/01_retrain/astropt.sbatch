#!/bin/bash
#SBATCH --job-name=retrain_astropt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/retrain_astropt_%j.out

set -euo pipefail

echo "Starting job $SLURM_JOB_ID on host $(hostname) at $(date)"

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

cd "$HOME/foundation-models-benchmark"
source .venv/bin/activate
python3 --version

# Retrain AstroPT via CLI
python -m fmb.cli retrain astropt --config configs/retrain/astropt.yaml

echo "Finished job $SLURM_JOB_ID at $(date)"
