#!/bin/bash
#SBATCH --job-name=anomalies_nfs_autoregressiv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=anomalies_nfs_autoregressive_%j.out


set -euo pipefail

echo "Starting job $SLURM_JOB_ID on host $(hostname) at $(date)"

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

cd "$HOME/AION"
source .venv/bin/activate
python3 --version

# Where to save the finetuned codec and diagnostics
RUN_DIR="outputs/retrained_euclid_codec_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

python -m scratch.detect_outliers_NFs_autoregressive_astropt   \
  --input /n03data/ronceray/embeddings/astropt_embeddings.pt     \
  --output-csv outputs/anomaly_scores_astropt_autoregressive.csv    \
  --epochs 250    \
  --num-transforms 8     \
  --hidden-features 256


echo "Finished job $SLURM_JOB_ID at $(date)"
