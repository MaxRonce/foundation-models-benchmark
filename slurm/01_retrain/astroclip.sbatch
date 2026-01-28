#!/bin/bash
#SBATCH --job-name=fmb_retrain_astroclip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/retrain_astroclip_%j.out
#SBATCH --error=slurm/logs/retrain_astroclip_%j.err

set -euo pipefail

# Entraînement AstroCLIP
# Utilise configs/paths_local.py pour DATA_ROOT et CKPT_ROOT

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load gcc/13.4.0
module load cuda/12.8
module load intelpython/3-2025.1.0

# Activation de l'environnement (adapter si nécessaire)
# source .venv/bin/activate

# Dossier de sortie basé sur FMBPaths
# Par défaut, finetune_image_encoder.py demande --output-path explicite.
# On le dérive du système de chemins de FMB.

CHECKPOINT="checkpoints/astroclip.ckpt" # À adapter
OUTPUT_DIR="checkpoints/astroclip_finetuned"
mkdir -p "$OUTPUT_DIR"

python -m fmb.cli retrain astroclip \
    --use-arrow \
    --cache-dir "$(python -m fmb.cli paths --data)" \
    --checkpoint "$CHECKPOINT" \
    --output-path "$OUTPUT_DIR/image_encoder_ft.pt" \
    --output-ckpt "$OUTPUT_DIR/astroclip_full_ft.ckpt" \
    --epochs 5 \
    --batch-size 128 \
    --amp
