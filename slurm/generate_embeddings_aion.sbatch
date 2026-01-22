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

python -m scratch.generate_embeddings \
  --split all \
  --batch-size 20 \
  --cache-dir /n03data/ronceray/datasets \
  --model-dir /n03data/ronceray/huggingface/aion \
  --codec-dir /home/ronceray/AION/outputs/retrained_euclid_codec_428529 \
  --output /n03data/ronceray/embeddings/aion_embeddings_spec_image_with_retrained_codec_428529.pt

echo "Finished job $SLURM_JOB_ID at $(date)"
