#!/bin/bash
#SBATCH --job-name=astropt_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_astropt_embeddings_%j.out

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
RUN_DIR="outputs/astropt_embeddings_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

python -m scratch.generate_embeddings_astropt \
  --checkpoint /n03data/ronceray/models/astropt/ckpt_final.pt \
  --split all \
  --cache-dir /n03data/ronceray/datasets \
  --output-dir /n03data/ronceray/embeddings \
  --output-name astropt_embeddings.pt \
  --batch-size 16 \
  --num-workers 4


echo "Finished job $SLURM_JOB_ID at $(date)"
