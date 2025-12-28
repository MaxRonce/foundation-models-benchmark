#!/bin/bash
#SBATCH --job-name=fmb_retrain_astropt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/retrain_astropt_%j.out
#SBATCH --error=slurm/logs/retrain_astropt_%j.err

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

# source .venv/bin/activate

OUT_DIR="$(python -m fmb.cli paths --checkpoints)/astropt"
mkdir -p "$OUT_DIR"

python -m fmb.cli retrain astropt \
    --batch-size 16 \
    --grad-accum 4 \
    --compile \
    --out-dir "$OUT_DIR" \
    --max-iters 10000
