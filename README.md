# Foundation Models Benchmark (FMB)

**Official code repository for:**
> **Benchmarking Foundation Models for Unsupervised Discovery in Large Multimodal Astrophysical Datasets**  


This repository provides a scalable pipeline for benchmarking pretrained multimodal foundation models—AION, AstroPT, and AstroCLIP—for unsupervised anomaly detection in matched imaging–spectroscopy data from the Euclid and DESI astronomical surveys.

## Overview

We introduce a modular framework for:
- **Lightweight adaptation** of foundation models to Euclid+DESI data
- **Embedding extraction** from three representation paradigms: autoregressive modeling (AstroPT), contrastive alignment (AstroCLIP), and predictive transformers (AION)
- **Scalable anomaly detection** via density estimation and multimodal fusion
- **Cross-model ranking analysis** to understand representation-relative anomaly definitions
- **Predictive probing** to evaluate decodability and effective dimensionality
- **Embeddings analysis and visualisation** Many visualisation tools are provided to analyse embeddings and their structure / relationships

## Installation

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (recommended)
- Conda or Mamba package manager

### Setup

1. **Clone the repository with submodules:**
   ```bash
   git clone --recursive https://github.com/MaxRonce/foundation-models-benchmark.git
   cd foundation-models-benchmark
   ```

2. **Create and activate environment:**
   ```bash
   conda env create -f environment.yml
   conda activate fmb
   ```
   
   Or if you don't have an `environment.yml`, create a new environment:
   ```bash
   conda create -n fmb python=3.12
   conda activate fmb
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Configure paths:**
   
   Create `src/fmb/configs/paths_local.yaml` based on `paths.template.yaml`:
   ```yaml
   storage_root: "/path/to/your/data"
   dataset_path: "/path/to/euclid_desi_dataset"
   # ... (see paths.template.yaml for all options)
   ```

5. **Verify installation:**
   ```bash
   fmb paths
   ```

## Pipeline

The benchmark follows a structured pipeline:

### Stage 0: Setup & Data Indexing
```bash
# Download pretrained weights
python src/fmb/setup/download_weights_aion.py
python src/fmb/setup/download_weights_astroclip.py

# Check environment
python src/fmb/setup/check_environment_aion.py
python src/fmb/setup/check_environment_astroclip.py

# Create dataset index (object_id → split/index mapping)
fmb data index --splits all

# Display sample (optional)
fmb display --split all --show-bands --no-gui --save runs/images/test.png
```

### Stage 1: Lightweight Adaptation (Retraining)
```bash
fmb retrain aion --config configs/retrain/aion.yaml
fmb retrain astropt --config configs/retrain/astropt.yaml
fmb retrain astroclip --config configs/retrain/astroclip.yaml
```

### Stage 2: Embedding Extraction
```bash
fmb embed aion --config configs/embeddings/aion.yaml
fmb embed astropt --config configs/embeddings/astropt.yaml
fmb embed astroclip --config configs/embeddings/astroclip.yaml
```

### Stage 3: Anomaly Detection
```bash
# Per-modality density estimation (Normalizing Flows)
fmb detect outliers

# Multimodal fusion and ranking
fmb detect multimodal --top-k 200 --fusion geo
```

### Stage 4: Analysis
```bash
# Anomaly correlation and uplift analysis
fmb analyze outliers

# Visual similarity search
fmb analyze similarity --query <object_id>
fmb analyze neighbor-ranks --query <object_id>

# Physical parameter regression
fmb analyze regression --config configs/analysis/regression.yaml

# Embedding displacement analysis
fmb analyze displacement
```

### Stage 5: Visualization
```bash
# UMAP embeddings
fmb viz paper-umap

# Advanced analysis figures (Spearman, Jaccard, Disagreements)
fmb viz advanced-analysis

# Outlier grid visualization
fmb viz outlier-grid --csv runs/outliers/anomaly_scores_aion.csv --cols 4

# Single object detailed view
fmb viz single-object --object-id <object_id>
```

## Project Structure

```
.
├── src/fmb/              # Core package
│   ├── models/           # Model adaptation implementations
│   ├── embeddings/       # Embedding extraction
│   ├── detection/        # Anomaly detection (NFS, cosine, multimodal fusion)
│   ├── analysis/         # Regression, displacement, similarity
│   ├── viz/              # Publication-ready visualizations
│   ├── data/             # Data loading and indexing
│   ├── paths.py          # Centralized path management
│   └── cli.py            # Unified CLI
├── configs/              # YAML configurations
├── external/             # Foundation model repositories (submodules)
│   ├── AION/
│   ├── astroPT/
│   └── AstroCLIP/
├── slurm/                # HPC job scripts
└── tests/                # Unit tests
```

## Testing

Run the test suite:
```bash
python -m unittest discover tests
```

## HPC/Slurm Integration

Many commands support `--slurm` flag for cluster submission:
```bash
fmb retrain astropt --config configs/retrain/astropt.yaml --slurm
fmb embed aion --slurm
```

## Acknowledgements

This work builds upon the following foundation models:

- **AstroPT**: [https://github.com/Smith42/astroPT](https://github.com/Smith42/astroPT) & [https://github.com/astroinfo-hacks/astroPT](https://github.com/astroinfo-hacks/astroPT)
- **AION**: [https://github.com/PolymathicAI/AION](https://github.com/PolymathicAI/AION)
- **AstroCLIP**: [https://github.com/PolymathicAI/AstroCLIP](https://github.com/PolymathicAI/AstroCLIP)

We thank the developers of these models for making their code publicly available.

## License

See LICENSE file for details.
