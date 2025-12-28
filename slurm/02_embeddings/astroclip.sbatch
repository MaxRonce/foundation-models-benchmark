#!/bin/bash
#SBATCH --job-name=fmb_embed_astroclip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/embed_astroclip_%j.out
#SBATCH --error=slurm/logs/embed_astroclip_%j.err

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

# Utilise le script fmb.cli embed astroclip
# Sauvegarde en .npz conforme à l'original

CHECKPOINT="checkpoints/astroclip.ckpt"
OUTPUT_FILE="$(python -m fmb.cli paths --embeddings)/astroclip/embeddings.npz"

mkdir -p "$(dirname "$OUTPUT_FILE")"

python -m fmb.cli embed astroclip \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT_FILE" \
    --sample-size 10000 \
    --batch-size 128 \
    --device cuda
