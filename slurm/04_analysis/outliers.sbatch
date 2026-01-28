#!/bin/bash
#SBATCH --job-name=analyze_outliers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/analyze_outliers_%j.out

set -euo pipefail

echo "Starting outlier analysis at $(date)"

module purge
module load intelpython/3-2025.1.0

cd "$HOME/foundation-models-benchmark"
source .venv/bin/activate
python3 --version

# Analyze anomaly results
python -m fmb.cli analyze outliers

echo "Finished at $(date)"
