#!/bin/bash
#SBATCH --job-name=fmb_embed_aion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/embed_aion_%j.out
#SBATCH --error=slurm/logs/embed_aion_%j.err

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

# source .venv/bin/activate

# Utilise fmb.cli embed aion (anciennement python -m fmb.embeddings.generate_embeddings)
# Les chemins par défaut sont gérés par configs/paths_local.py si non fournis.

python -m fmb.cli embed aion \
  --split all \
  --batch-size 20 \
  --output "$(python -m fmb.cli paths --embeddings)/aion/embeddings_all.pt"
