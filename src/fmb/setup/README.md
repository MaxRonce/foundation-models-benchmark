# Setup Module

Environment validation and automated data/weights downloads.

## Key Components

### Environment Checks

#### `check_environment_aion.py`
Validate AION dependencies and setup.

```bash
fmb setup check-env aion
```

**Checks**:
- Python packages (torch, aion, etc.)
- CUDA availability
- Model weights existence
- Dataset paths

---

#### `check_environment_astroclip.py`
Validate AstroCLIP dependencies.

```bash
fmb setup check-env astroclip
```

**Checks**:
- AstroCLIP package installation
- HuggingFace transformers
- Dataset cache
- Config files

---

### Data Downloads

#### `download_data.py`
Download Euclid+DESI dataset from HuggingFace.

```bash
fmb setup download-data --split train
```

**Features**:
- Automatic HF authentication
- Progress tracking
- Resumable downloads
- Split selection (train/test/all)

**Target**: `data/msiudek__astroPT_euclid_Q1_desi_dr1_dataset__<split>/`

---

## Configuration

Setup uses `paths_local.yaml` for all path management.

Example:
```yaml
# Data
dataset_path: C:\data\euclid_desi
dataset_hf_id: msiudek/astroPT_euclid_Q1_desi_dr1_dataset

# Weights
base_weights: C:\data\weights\base
base_weights_aion: C:\data\weights\base\aion

# Index
dataset_index: C:\data\index\euclid_index.csv
```

## Troubleshooting

### HuggingFace Authentication
```bash
huggingface-cli login
```


## Offline Mode

For clusters without internet:

1. Download on a machine with internet
2. Copy `data/` directory to cluster
3. Update `paths_local.yaml` with correct paths
4. Run environment checks to verify
