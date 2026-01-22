#!/bin/bash
#SBATCH --job-name=pred_physical_params
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_pred_physical_params_%j.out

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
RUN_DIR="outputs/results"
mkdir -p "$RUN_DIR"

python -m scratch.predict_physical_params  \
    --aion-embeddings /n03data/ronceray/embeddings/aion_embeddings_spec_image_with_retrained_codec_428529.pt \
    --astropt-embeddings /n03data/ronceray/embeddings/astropt_embeddings.pt \
    --catalog /home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_physical_param.fits \
    --output-dir /home/ronceray/AION/results/physical_params_prediction/all_params \
    --all-data  \

python3 scratch/predict_physical_params.py \
    --aion-embeddings /n03data/ronceray/embeddings/aion_embeddings_spec_image_with_retrained_codec_428529.pt \
    --astropt-embeddings /n03data/ronceray/embeddings/astropt_embeddings.pt \
    --catalog /home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_physical_param.fits \
    --output-dir /home/ronceray/AION/results/physical_params_prediction/all_params_all_data \
    --all-data  

echo "Finished job $SLURM_JOB_ID at $(date)"
