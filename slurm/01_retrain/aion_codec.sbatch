#!/bin/bash
#SBATCH --job-name=fmb_retrain_aion_codec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/retrain_aion_codec_%j.out
#SBATCH --error=slurm/logs/retrain_aion_codec_%j.err

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

# source .venv/bin/activate

RUN_DIR="$(python -m fmb.cli paths --checkpoints)/aion/codec_retrained_$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

python -m fmb.cli retrain aion_codec \
  --split all \
  --max-entries 0 \
  --batch-size 32 \
  --epochs 30 \
  --lr 1e-5 \
  --resize 96 \
  --crop-size 96 \
  --output "$RUN_DIR" \
  --num-workers 0 \
  --max-abs 1000 \
  --grad-clip 1 \
  --scheduler cosine
