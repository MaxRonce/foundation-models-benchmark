#!/bin/bash
#SBATCH --job-name=fmb_detect_nfs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/detect_nfs_%j.out
#SBATCH --error=slurm/logs/detect_nfs_%j.err

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

# Utilise fmb.cli detect nfs
INPUT_FILE="$(python -m fmb.cli paths --embeddings)/aion/embeddings_train.pt"
RUN_DIR="$(python -m fmb.cli paths --runs)/detect_nfs"

mkdir -p "$RUN_DIR"

python -m fmb.cli detect nfs \
    --input "$INPUT_FILE" \
    --output "$RUN_DIR/nfs_outliers.csv" \
    --epochs 50 \
    --lr 1e-4
