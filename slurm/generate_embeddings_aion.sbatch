#!/bin/bash
#SBATCH --job-name=generate_aion_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_generate_aion_embeddings_%j.out

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


# Assuming adapter is in standard location or provided by user. 
# PLEASE UPDATE --adapter-checkpoint TO YOUR TRAINED MODEL PATH
python -m fmb.embeddings.generate_embeddings_aion \
  --split all \
  --batch-size 20 \
  --cache-dir /n03data/ronceray/datasets \
  --model-dir /n03data/ronceray/huggingface/aion \
  --adapter-checkpoint /n03data/ronceray/models/aion/adapters_final.pt \
  --output /n03data/ronceray/embeddings/aion_embeddings.pt

echo "Finished job $SLURM_JOB_ID at $(date)"
