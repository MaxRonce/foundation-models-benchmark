#!/bin/bash
#SBATCH --job-name=fmb_detect_cosine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/detect_cosine_%j.out
#SBATCH --error=slurm/logs/detect_cosine_%j.err

set -euo pipefail

export PYTHONPATH=$PYTHONPATH:$PWD/src

module purge
module load intelpython/3-2025.1.0

# Détection d'anomalies par similarité cosinus via CLI
# Prend en entrée les embeddings AION ou AstroPT (.pt ou .npz)

INPUT_FILE="$(python -m fmb.cli paths --embeddings)/aion/embeddings_train.pt"
OUTPUT_CSV="$(python -m fmb.cli paths --runs)/detect_cosine/outliers.csv"

mkdir -p "$(dirname "$OUTPUT_CSV")"

python -m fmb.cli detect cosine \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_CSV" \
    --threshold-percent 1.0
