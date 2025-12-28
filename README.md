# Foundation Models Benchmark (FMB)

This repository provides a standardized framework for benchmarking multimodal foundation models—including AION, astroPT, and AstroCLIP—specialized for Euclid imaging and DESI spectroscopic data.

## System Architecture

The project is structured as a modular Python package (`fmb`) designed to facilitate reproducible research through centralized path management and a unified command-line interface (CLI).

```text
.
├── src/fmb/              # Core source code
│   ├── models/           # Training and fine-tuning implementations
│   ├── embeddings/       # Embedding extraction and processing
│   ├── detection/        # Anomaly detection methodologies (Cosine similarity, Normalizing Flows)
│   ├── analysis/         # Physical parameter estimation
│   ├── viz/              
│   ├── data/             # Data preprocessing pipelines
│   ├── paths.py          
│   └── cli.py            
├── slurm/                
├── configs/              
├── external/             
├── data/                 
├── embeddings/           
└── checkpoints/          
```

## Installation and Setup

1. Clone the repository and initialize submodules:
   ```bash
   git clone --recursive https://github.com/MaxRonce/foundation-models-benchmark.git
   cd foundation-models-benchmark
   ```
2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Path Configuration

The framework utilizes a modular path management system. Users must define local storage roots in `configs/paths_local.py` (which is excluded from version control):

```python
# configs/paths_local.py
DATA_ROOT = "/n03data/ronceray/datasets"
EMB_ROOT = "/n03data/ronceray/embeddings"
CKPT_ROOT = "/n03data/ronceray/models"
RUNS_ROOT = "/n03data/ronceray/runs"
CACHE_ROOT = "/n03data/ronceray/cache"
```

The current configuration can be validated using:
```bash
python -m fmb.cli paths
```

## Command-Line Interface

The `fmb` CLI provides a unified interface for all pipeline stages, ensuring consistency across execution environments. It supports direct argument passing to the underlying research scripts.

### Model Retraining and Fine-tuning
```bash
python -m fmb.cli retrain aion_codec --epochs 10 --batch-size 32
```

### Embedding Extraction
```bash
python -m fmb.cli embed aion --split all
```

### Anomaly Detection
```bash
python -m fmb.cli detect cosine --threshold-percent 1.0
```

### Statistical Analysis and Visualization
```bash
python -m fmb.cli analyze tsne
```

## HPC Integration (Slurm)

To submit a batch job:
```bash
sbatch slurm/02_embeddings/aion.sbatch
```

Alternatively, jobs can be submitted directly via the CLI:
```bash
python -m fmb.cli embed aion --slurm
```

## Research Context and Acknowledgments
This framework evolved from collaborative development during the AstroInfo 2025 hackathon.
- **AION**: A multimodal foundation model supporting modality-specific and fused latent spaces.
- **astroPT**: A transformer-based architecture for joint spectral and imaging encoding.
- **AstroCLIP**: A contrastive learning framework for cross-modal alignment.
