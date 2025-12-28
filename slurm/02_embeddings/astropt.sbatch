#!/bin/bash
#SBATCH --job-name=fmb_embed_astropt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/embed_astropt_%j.out
#SBATCH --error=slurm/logs/embed_astropt_%j.err

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

# source .venv/bin/activate

# Utilise fmb.cli embed astropt
OUTPUT_FILE="$(python -m fmb.cli paths --embeddings)/astropt/embeddings_train.pt"
mkdir -p "$(dirname "$OUTPUT_FILE")"

python -m fmb.cli embed astropt \
    --output "$OUTPUT_FILE" \
    --batch-size 32
