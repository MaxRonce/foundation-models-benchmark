# FMB Architecture

Technical documentation for the Foundation Models Benchmark codebase organization and design philosophy.

## Project Structure

```
foundation-models-benchmark/
├── src/fmb/              # Core Python package
│   ├── cli.py            # Unified CLI entry point
│   ├── paths.py          # Centralized path management
│   ├── models/           # Model-specific implementations
│   │   ├── aion/         # AION adapter training
│   │   ├── astropt/      # AstroPT fine-tuning
│   │   └── astroclip/    # AstroCLIP fine-tuning
│   ├── embeddings/       # Embedding extraction scripts
│   │   ├── generate_embeddings_aion.py
│   │   ├── generate_embeddings_astropt.py
│   │   └── generate_embeddings_astroclip.py
│   ├── detection/        # Anomaly detection methods
│   │   ├── run.py        # Normalizing Flows (main)
│   │   ├── cosine.py     # Cross-modal cosine similarity
│   │   └── multimodal.py # Fusion and ranking
│   ├── analysis/         # Post-detection analysis
│   │   ├── outliers.py   # Correlation, Jaccard, uplift
│   │   ├── similarity.py # Visual similarity search
│   │   ├── displacement.py # Cross-model retention
│   │   └── regression/   # Physical parameter prediction
│   ├── viz/              # Visualization scripts
│   │   ├── plot_paper_combined_umap.py
│   │   ├── similarity.py
│   │   ├── outliers/     # Anomaly-specific plots
│   │   │   ├── plot_paper_advanced_analysis.py
│   │   │   ├── plot_paper_outlier_grid.py
│   │   │   └── plot_paper_single_object.py
│   │   ├── utils.py      # Shared viz utilities
│   │   ├── spectrum.py   # Spectrum plotting helpers
│   │   └── style.py      # Unified matplotlib style
│   ├── data/             # Data loading and indexing
│   │   ├── load_display_data.py  # Main dataset loader
│   │   ├── index_dataset.py      # Index generation
│   │   ├── utils.py              # Shared utilities
│   │   └── astroclip_loader.py   # AstroCLIP-specific
│   └── configs/          # YAML configurations
│       ├── paths.template.yaml   # Path config template
│       └── viz_style.yaml        # Matplotlib style
├── configs/              # User-editable configs
│   ├── retrain/          # Model training configs
│   ├── embeddings/       # Embedding extraction configs
│   ├── analysis/         # Analysis task configs
│   └── detection/        # Detection method configs
├── external/             # Foundation model submodules
│   ├── AION/             # Original AION repository
│   ├── astroPT/          # Original AstroPT repository
│   └── AstroCLIP/        # Original AstroCLIP repository
├── slurm/                # HPC job submission scripts
│   ├── 01_retrain/
│   ├── 02_embeddings/
│   ├── 03_detection/
│   └── 04_analysis/
└── tests/                # Unit tests
```

## Design Philosophy

### 1. Centralized Path Management

All paths are managed via `src/fmb/paths.py` and configured in `src/fmb/configs/paths_local.yaml`:

```python
from fmb.paths import load_paths

paths = load_paths()
# Access standardized paths:
# paths.dataset, paths.embeddings, paths.outliers, etc.
```

This ensures consistency across all scripts and environments.

### 2. Unified CLI as Dispatcher

The CLI (`src/fmb/cli.py`) is a **thin wrapper** that dispatches to standalone scripts:

```python
# CLI command:
fmb retrain astropt --config configs/retrain/astropt.yaml

# Calls:
from fmb.models.astropt.retrain import main
main()
```

### 3. Foundation Models Integration

**`external/` vs `src/fmb/models/`:**

- **`external/`**: Original upstream repositories (submodules), unchanged
- **`src/fmb/models/`**: FMB-specific adaptations and training scripts

Example: `src/fmb/models/astropt/retrain.py` imports core AstroPT classes from `external/astroPT/` but implements custom training loops for Euclid+DESI data.

### 4. Modular Pipeline Stages

```
Stage 0: Setup          → data index, download weights
Stage 1: Retrain        → lightweight model adaptation
Stage 2: Embed          → extract embeddings
Stage 3: Detect         → anomaly detection (NFS, fusion)
Stage 4: Analyze        → correlations, similarity, regression
Stage 5: Visualize      → publication-ready plots
```

Each stage is independent and can be run separately.

### 5. Configuration Hierarchy

1. **Template configs** (`src/fmb/configs/*.template.yaml`) - Version controlled
2. **Local configs** (`src/fmb/configs/*_local.yaml`) - Gitignored, user-specific
3. **Task configs** (`configs/*/`) - Task-specific parameters, but pushed as template
4. **CLI arguments** - Runtime overrides

Priority: CLI args > Task configs > Local configs > Template defaults

## Key Modules

### `cli.py`
- Entry point: `fmb <command> <subcommand> [options]`
- Typer-based command structure
- Provides `--slurm` flag for cluster submission

### `paths.py`
- `FMBPaths` dataclass with all standardized paths
- `load_paths()` function with config resolution
- Ensures directories exist on first access

### `detection/run.py`
- Main anomaly detection via Normalizing Flows
- Processes embeddings per modality (image, spectrum)
- Outputs anomaly scores to CSV

### `detection/multimodal.py`
- Fuses per-modality scores with cross-modal alignment
- Implements geometric mean, minimum, average fusion
- Exports top-k anomalies per model

### `analysis/outliers.py`
- Spearman correlations between model rankings
- Jaccard indices for top-k overlap
- Disagreement analysis

### `viz/` modules
- `plot_paper_combined_umap.py`: Multi-model UMAP
- `outliers/plot_paper_advanced_analysis.py`: Correlation heatmaps
- `outliers/plot_paper_outlier_grid.py`: Anomaly gallery
- `similarity.py`: Nearest neighbor visualization

## Execution Modes

### Local Development
```bash
# Direct module execution
python -m fmb.models.astropt.retrain --config configs/retrain/astropt.yaml

# Or via CLI
fmb retrain astropt --config configs/retrain/astropt.yaml
```

### HPC/SLURM
```bash
# Via CLI
fmb retrain astropt --slurm

# Or direct sbatch
sbatch slurm/01_retrain/astropt.sbatch
```

## Testing

```bash
python -m unittest discover tests
```

Tests cover:
- Path resolution
- Data loading
- Config parsing
- Core utilities

## Adding New Components

### New Model
1. Add implementation to `src/fmb/models/<model_name>/`
2. Create config in `configs/retrain/<model_name>.yaml`
3. Register in `cli.py` under `retrain` command
4. Add tests in `tests/test_models.py`

### New Analysis
1. Add script to `src/fmb/analysis/<task_name>/`
2. Create config in `configs/analysis/<task_name>.yaml`
3. Register in `cli.py` under `analyze` command
4. Document in `CLI_COMMANDS.md`

### New Visualization
1. Add script to `src/fmb/viz/`
2. Use `fmb.viz.style.apply_style()` for consistency
3. Register in `cli.py` under `viz` command
4. Save to `paths.analysis` by default
