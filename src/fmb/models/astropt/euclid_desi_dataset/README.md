


# Euclid-DESI Multimodal Dataset Scripts

This directory contains all scripts related to multimodal training on Euclid images + DESI spectra.

> **Note:** The work in this directory was developed by **Malgorzata Siudek**.

## Directory Structure

```
euclid_desi_dataset/
├── README.md                           # This file
├── multimodal_dataloader.py           # Main dataloader for images + spectra
├── test_multimodal_dataloader.py      # Test script for dataloader functionality
└── embeddings/                        # Embedding generation and analysis
    ├── extract_multimodal_embeddings.py    # Extract embeddings from trained models
    └── multimodal_embedding_analysis.py    # Comprehensive embedding analysis
```

## Main Scripts

### Data Loading
- **`multimodal_dataloader.py`**: Core dataloader class for handling Euclid RGB images and DESI spectra simultaneously
- **`test_multimodal_dataloader.py`**: Test script to verify dataloader functionality

### Embedding Analysis (embeddings/)
- **`extract_multimodal_embeddings.py`**: Extract embeddings from trained multimodal models
- **`multimodal_embedding_analysis.py`**: Analyze embedding quality, correlations, and downstream tasks

## Usage Examples

### Training
```bash
# Main training script (in parent scripts/ directory)
torchrun --standalone --nproc_per_node=2 scripts/train_spectra_images.py

# Production training with PBS
qsub scripts/launch_multimodal_full_training.sh
```

### Embedding Extraction
```bash
python scripts/euclid_desi_dataset/embeddings/extract_multimodal_embeddings.py \
    --checkpoint /path/to/model/ckpt_best.pt \
    --output_dir /path/to/embeddings/ \
    --split test_batch_1 \
    --batch_size 32
```

### Embedding Analysis
```bash
python scripts/euclid_desi_dataset/embeddings/multimodal_embedding_analysis.py \
    --embeddings_file /path/to/embeddings.npz \
    --output_dir /path/to/analysis/
```

## Data Flow

1. **Training**: `train_spectra_images.py` → uses `multimodal_dataloader.py`
2. **Embedding Extraction**: `extract_multimodal_embeddings.py` → produces embeddings
3. **Analysis**: `multimodal_embedding_analysis.py` → analyzes embeddings

## Key Features

- **Multimodal Architecture**: Joint training on images (224×224 RGB) and spectra (7781 wavelength points)
- **Patch-based Processing**: 16×16 patches for images, size-10 patches for spectra
- **Balanced Loss Weighting**: Automatically balanced based on sequence lengths
- **Robust Extraction**: Handles duplicates and validates TARGETID matching
- **Comprehensive Analysis**: PCA, correlations, downstream tasks, visualizations