#!/bin/bash
#SBATCH --job-name=fmb_analyze_params
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/analyze_params_%j.out
#SBATCH --error=slurm/logs/analyze_params_%j.err

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

# Utilise fmb.cli analyze predict_params
INPUT_FILE="$(python -m fmb.cli paths --embeddings)/aion/embeddings_train.pt"
RUN_DIR="$(python -m fmb.cli paths --runs)/analysis_params"

mkdir -p "$RUN_DIR"

python -m fmb.cli analyze predict_params \
    --input "$INPUT_FILE" \
    --output-dir "$RUN_DIR" \
    --batch-size 128
