# Data Module

Data loading and preprocessing for the FMB benchmark.

## Overview

This module provides loaders for the Euclid+DESI astronomical dataset in various formats (Arrow, Parquet) with PyTorch dataset wrappers for training.

## Key Components

### Dataset Loaders

#### `load_display_data.py`
**EuclidDESIDataset**: Main dataset class for loading Euclid images + DESI spectra from HuggingFace cache.

```python
from fmb.data.load_display_data import EuclidDESIDataset

dataset = EuclidDESIDataset(split="train", cache_dir="./data")
sample = dataset[0]  # Returns: images, spectra, redshift, object_id
```

**Features:**
- Loads from local Arrow cache (msiudek/astroPT_euclid_Q1_desi_dr1_dataset)
- Supports train/test splits
- Returns: VIS, NISP-Y, NISP-J, NISP-H images + spectrum + metadata

---

#### `astroclip_loader.py`
Arrow dataset loader specialized for AstroCLIP fine-tuning.

**Key Functions:**
- `load_local_arrow_dataset()` - Load from HuggingFace cache
- `prepare_image_from_euclid_bands()` - Multi-band image processing
- `convert_dataset_to_astroclip_format()` - Format converter

---

#### `astroclip_parquet.py`
Parquet-based data source for AstroCLIP (legacy support).

**Classes:**
- `ParquetDataSource` - Load from local or HF parquet files
- Handles HF URIs: `hf://datasets/repo/file.parquet`

---

### PyTorch Wrappers

#### `datasets.py`
PyTorch Dataset wrappers for each foundation model.

**Classes:**
- `AIONDataset` - AION multimodal training
- `AstroPTDataset` - AstroPT images + spectra
- `AstroCLIPDataset` - AstroCLIP vision-language pairs

**Usage:**
```python
from fmb.data.datasets import AIONDataset, FMBDataConfig

config = FMBDataConfig(split="train", image_size=224)
dataset = AIONDataset(config)
```

---

### Utilities

#### `utils.py`
Embedding loading and image preprocessing utilities.

**Key Functions:**
- `load_embeddings_file(path)` - Load .pt/.npz/.pkl embeddings
- `prepare_rgb_image(sample)` - Extract and normalize RGB images
- `load_euclid_multiband_image(sample)` - Handle multi-band images

---

#### `index_dataset.py`
Indexing utilities for fast object ID lookups.

---

## Data Flow

```
HuggingFace Cache (Arrow files)
         ↓
EuclidDESIDataset (raw data)
         ↓
Model-specific Dataset wrapper (AIONDataset, etc.)
         ↓
DataLoader → Training
```

## Dataset Structure

Each sample contains:
- **Images**: 4 bands (VIS, NISP-Y, J, H) as tensors
- **Spectrum**: Flux array (7781 wavelength bins)
- **Metadata**: object_id, targetid, redshift
- **RGB Image**: Pre-combined 3-channel image (optional)

## Configuration

Dataset paths are managed via `paths_local.yaml`:
```yaml
dataset_path: /path/to/data
dataset_path_train: /path/to/train
dataset_path_test: /path/to/test
```
