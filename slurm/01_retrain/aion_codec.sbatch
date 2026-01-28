#!/bin/bash
#SBATCH --job-name=retrain_aion_codec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/retrain_aion_codec_%j.out
#SBATCH --nodelist=n36
#SBATCH --partition=pscompl

set -euo pipefail

echo "Starting job $SLURM_JOB_ID on host $(hostname) at $(date)"

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

cd "$HOME/foundation-models-benchmark"
source .venv/bin/activate
python3 --version

# Retrain AION codec via CLI
python -m fmb.cli retrain aion_codec --config configs/retrain/aion.yaml

echo "Finished job $SLURM_JOB_ID at $(date)"
