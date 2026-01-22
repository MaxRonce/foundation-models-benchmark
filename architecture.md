# Foundation Models Benchmark (FMB) Project Architecture

This document describes the file and folder organization of the project following the migration to a standardized structure.

## Overview

The project follows a standardized Python structure with all source code located in `src/fmb`.

```
foundation-models-benchmark/
├── src/                # Main source code
│   └── fmb/            # Main Python package
│       ├── cli.py      # Unified pipelines Entry Point (CLI)
│       ├── data/       # Data loaders and utilities
│       ├── models/     # Model definitions and training scripts
│       ├── analysis/   # Post-training analysis scripts
│       └── ...
├── slurm/              # Submission scripts for the cluster
├── external/           # Submodules and intact external code (AstroPT, etc.)
├── temp/               # Temporary code or code currently being migrated (to be cleaned up)
└── pyproject.toml      # Project configuration and dependencies
```

## Directory Details

### `src/fmb/` (Project Core)

*   **`cli.py`** : Unified Command Line Interface. It is the entry point for launching training and processing tasks.
    *   Usage : `python -m fmb.cli --help`

*   **`models/`** : Contains model implementations and specific training scripts.
    *   **`astropt/`** 
        *   `retrain_spectra_images.py` : Multimodal training script (Images + Spectra).
        *   `euclid_desi_dataset/` : Specific management of the Euclid + DESI dataset. TODO refactor this
    *   **`astroclip/`** : AstroCLIP Model (Contrastive Learning).
        *   `finetune_image_encoder.py` : Image encoder fine-tuning script.
        *   `core/` : Fundamental modules (`astroclip.py`, `specformer.py`, `modules.py`).

*   **`data/`** : Model-specific data loading utilities.
    *   `astroclip_loader.py` : Loading from local Arrow cache.
    *   `astroclip_parquet.py` : Loading from Parquet files.
    *   `load_display_data_hsc.py` : (Inherited) Loader for HSC data, used to test AION.

### `slurm/` (Deployment)

Contains bash scripts for submitting jobs to the cluster (via `sbatch`). These scripts are organized around the major calculation steps of the pipeline:

1.  **`01_retrain/`** : Scripts to retrain or fine-tune foundation models.
2.  **`02_embeddings/`** : Scripts to generate embeddings from the trained models.
3.  **`03_detection/`** : Scripts to run anomaly detection algorithms on the embeddings.
4.  **`04_analysis/`** : Scripts to predict physical parameters or visualize results.

### Debugging & Execution Philosophy

*   **Ease of Debugging**: Every part of the code is designed to be easily runnable from a terminal.
*   **Direct Execution**: You can run any script directly using python (e.g., `python src/fmb/models/astropt/retrain.py ...`).
*   **CLI Entrypoint**: The `fmb-cli` (or `python -m fmb.cli`) serves **only** as a convenient wrapper to launch these scripts using python. It does not contain heavy logic itself but dispatch execution to the appropriate module. This ensures consistency between local execution, debugging, and SLURM usage.

### `external/`

Contains cloned or integrated external dependencies that are not part of the active code maintained in `fmb`, but are necessary for operation (e.g., original `astroPT` for reference or legacy imports).

### Difference between `external/` and `src/fmb/models/`

*   **`external/`** contains the **original** upstream codebases (submodules). It is used as a reference or library source but its code is not meant to be modified for FMB specifics.
*   **`src/fmb/models/`** contains the **FMB-specific adaptations** of these models. This is where we implement custom training loops, specific data loading logic for FMB datasets, and architectural adjustments required for our benchmarks. For example, `src/fmb/models/astropt/retrain_spectra_images.py` imports core model classes from `external/astroPT` but implements a custom multimodal training strategy.

## Standard Workflow

1.  **Development** : Code is modified in `src/fmb`.
2.  **Local Testing** : Launch via `python -m fmb.cli ...` or directly the module `python -m fmb.models.astroclip.finetune_image_encoder ...`.
3.  **Deployment** : Use scripts in `slurm/` which configure the environment and launch the final Python commands on compute nodes.
