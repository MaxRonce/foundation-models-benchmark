#!/bin/bash
#SBATCH --job-name=analyze_params
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/analyze_params_%j.out

set -euo pipefail

echo "Starting physical parameters prediction at $(date)"

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

cd "$HOME/foundation-models-benchmark"
source .venv/bin/activate
python3 --version

# Run physical parameters prediction via CLI
python -m fmb.cli analyze predict_params

echo "Finished at $(date)"
