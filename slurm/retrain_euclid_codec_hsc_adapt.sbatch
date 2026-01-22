#!/bin/bash
#SBATCH --job-name=euclid_codec_hsc_adapt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --output=slurm_retrain_euclid_codec_hsc_adapt_%j.out
#SBATCH --nodelist=n36
#SBATCH --partition=pscompl

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
RUN_DIR="outputs/retrained_euclid_codec_adapted_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

 python -m scratch.retrain_euclid_hsc_adapter   \
  --cache-dir /n03data/ronceray/datasets   \
  --split all     \
  --max-entries 0  \
  --batch-size 8     \
  --epochs 15   \
  --lr 1e-4     \
  --resize 96    \
  --crop-size 96   \
  --output $RUN_DIR     \
  --num-workers 0    \
  --max-abs 100
            



echo "Finished job $SLURM_JOB_ID at $(date)"
