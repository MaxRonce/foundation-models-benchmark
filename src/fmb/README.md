# FMB - Foundation Models Benchmark

Core package for benchmarking foundation models on astronomical data.

## Package Structure

```
fmb/
├── cli.py              # Command-line interface (main entry point)
├── paths.py            # Centralized path management from config.yaml
├── data/               # Data loading and preprocessing
├── models/             # Foundation model implementations
├── detection/          # Anomaly detection methods
├── analysis/           # Embedding analysis and benchmarking
├── viz/                # Publication-ready visualizations
├── embeddings/         # Embedding generation scripts
├── setup/              # Environment checks and data downloads
└── configs/            # YAML configuration files
```

## Module Overview

### [data/](./data)
Data loaders for Euclid+DESI datasets, HuggingFace integration, and PyTorch dataset wrappers.

### [models/](./models)
Foundation model implementations and modified script for re training:
- **AION**: Multimodal foundation model
- **AstroPT**: Transformer-based model for spectra and images
- **AstroCLIP**: Vision-language model for astronomy

### [detection/](./detection)
Anomaly detection using Normalizing Flows and cosine mismatch detection + multimodal selection.

### [analysis/](./analysis)
Benchmark analyses:
- Physical parameter regression
- Anomaly displacement
- Visual similarity analysis

### [viz/](./viz)
Visualization tools for UMAP, outliers, and similarity plots.

### [embeddings/](./embeddings)
Scripts to generate embeddings from trained foundation models.

### [setup/](./setup)
Environment validation and automated data/weights downloads.

## Configuration

All paths are managed via `src/fmb/configs/paths_local.yaml`. See [paths.py](./paths.py) for details.
