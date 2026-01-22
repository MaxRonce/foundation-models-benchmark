#!/bin/bash
#SBATCH --job-name=slurm_correlate_embeddings_tsne
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_correlate_embeddings_tsne_%j.out

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

python -m scratch.visualize_multimodel_embeddings_tsne  \
    --aion-embeddings /n03data/ronceray/embeddings/aion_embeddings_spec_image_with_retrained_codec_428529.pt   \
    --astropt-embeddings /n03data/ronceray/embeddings/astropt_embeddings.pt \
    --catalog /home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_physical_param.fits \
    --output-dir "$RUN_DIR/tsne_viz_better" \
    --tsne-cache /home/ronceray/AION/outputs/results/tsne_coords.pt  \
    --tsne-preset balanced \
    --all-params \
    --random-state 42
    

echo "Finished job $SLURM_JOB_ID at $(date)"
