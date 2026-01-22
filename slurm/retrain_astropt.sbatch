#!/bin/bash
#SBATCH --job-name=euclid_codec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_retrain_euclid_codec_%j.out

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

python -m astroPT.scripts.train_spectra_images \
  --cache-dir /n03data/ronceray/datasets \
  --out-dir /n03data/ronceray/models/astropt \
  --batch-size 8 \
  --grad-accum 4 

echo "Finished job $SLURM_JOB_ID at $(date)"
