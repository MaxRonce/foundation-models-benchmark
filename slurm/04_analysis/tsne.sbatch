#!/bin/bash
#SBATCH --job-name=fmb_viz_tsne
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/viz_tsne_%j.out
#SBATCH --error=slurm/logs/viz_tsne_%j.err

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load intelpython/3-2025.1.0

# Utilise fmb.cli analyze tsne
INPUT_DIR="$(python -m fmb.cli paths --embeddings)"
OUTPUT_DIR="$(python -m fmb.cli paths --runs)/viz_tsne"

mkdir -p "$OUTPUT_DIR"

python -m fmb.cli analyze tsne \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"
