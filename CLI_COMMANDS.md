# FMB CLI Commands Reference

Complete command reference for the Foundation Models Benchmark pipeline.

---

## Setup & Configuration

### Display path configuration
```bash
fmb paths
```

### Display dataset samples
```bash
# View sample with all bands
fmb display --split all --show-bands --no-gui --save runs/images/test.png

# Interactive view
fmb display --split train --index 42 --show-bands
```

---

## Stage 0: Data Setup

### Download Pretrained Weights
```bash
python src/fmb/setup/download_weights_aion.py
python src/fmb/setup/download_weights_astroclip.py
```

### Environment Verification
```bash
python src/fmb/setup/check_environment_aion.py
python src/fmb/setup/check_environment_astroclip.py
```

### Create Dataset Index
```bash
# Index all splits (creates object_id → split/index mapping)
fmb data index --splits all

# Index specific splits
fmb data index --splits train,test --output custom_index.csv

# Overwrite existing index
fmb data index --splits all --overwrite
```

---

## Stage 1: Model Retraining (Lightweight Adaptation)

### AION
```bash
# Using config file
fmb retrain aion --config configs/retrain/aion.yaml

# With custom parameters
fmb retrain aion --epochs 10 --batch-size 32
```

### AstroPT
```bash
# Using config file
fmb retrain astropt --config configs/retrain/astropt.yaml

# Override parameters
fmb retrain astropt --max-iters 5000 --batch-size 8
```

### AstroCLIP
```bash
# Using config file
fmb retrain astroclip --config configs/retrain/astroclip.yaml

# With checkpoint
fmb retrain astroclip --checkpoint path/to/ckpt --epochs 5
```

---

## Stage 2: Embedding Extraction

### AION
```bash
# Full dataset
fmb embed aion --config configs/embeddings/aion.yaml

# Specific split with custom batch size
fmb embed aion --split test --batch-size 32
```

### AstroPT
```bash
fmb embed astropt --config configs/embeddings/astropt.yaml
```

### AstroCLIP
```bash
fmb embed astroclip --config configs/embeddings/astroclip.yaml
```

---

## Stage 3: Anomaly Detection

### Density Estimation (Normalizing Flows)
```bash
# Detect outliers using Normalizing Flows
fmb detect outliers

# With custom config
fmb detect outliers --config configs/detection/nfs.yaml
```

### Cosine Similarity (Optional)
```bash
# Compute cross-modal alignment scores
fmb detect cosine --aion-embeddings path/to/aion.pt
```

### Multimodal Fusion
```bash
# Combine per-modality and cross-modal scores
fmb detect multimodal --top-k 200 --fusion geo

# With custom thresholds
fmb detect multimodal --top-k 200 --fusion geo --t-img 0.99 --t-spec 0.99 --t-mis 0.99
```

**Fusion methods:**
- `geo`: Geometric mean (default)
- `min`: Minimum score
- `avg`: Average score

---

## Stage 4: Analysis

### Anomaly Correlation Analysis
```bash
# Analyze correlations, Jaccard indices, disagreements
fmb analyze outliers

# With custom input
fmb analyze outliers --input-csv runs/outliers/multimodal/all_scores.csv --top-k 200
```

### Visual Similarity Search
```bash
# Find similar objects for a single query
fmb analyze similarity --query 39633427976160866

# Multiple queries
fmb analyze similarity --query 123 --query 456 --query 789

# From CSV file
fmb analyze similarity --query-csv runs/outliers/top_anomalies.csv --n-similar 10

# Specify model
fmb analyze similarity --query 123 --model aion --save results/similarity_aion.png
```

### Neighbor Rank Analysis
```bash
# Analyze if neighbors of anomalies are also anomalous
fmb analyze neighbor-ranks --query 39633427976160866

# Multiple queries from CSV
fmb analyze neighbor-ranks --query-csv runs/outliers/top_anomalies.csv --n-similar 10
```

### Physical Parameter Regression
```bash
# Predict redshift, mass, SFR from embeddings
fmb analyze regression --config configs/analysis/regression.yaml

# Custom output directory
fmb analyze regression --config configs/analysis/regression.yaml --out-dir runs/regression_results
```

### Embedding Displacement Analysis
```bash
# Analyze retention across models and modalities
fmb analyze displacement

# With custom config
fmb analyze displacement --config configs/analysis/displacement.yaml --out-dir runs/displacement
```

---

## Stage 5: Visualization

### UMAP Embeddings
```bash
# Generate publication-ready combined UMAP
fmb viz paper-umap
```

### Advanced Analysis Figures
```bash
# Spearman correlations, Jaccard indices, disagreement scatter
fmb viz advanced-analysis

# With custom prefix
fmb viz advanced-analysis --save-prefix runs/analysis/paper_v1
```

### Outlier Grid
```bash
# Generate grid of outlier images + spectra
fmb viz outlier-grid --csv runs/outliers/anomaly_scores_aion.csv

# Custom layout
fmb viz outlier-grid --csv runs/outliers/top_200.csv --cols 4 --max 16

# Show interactively
fmb viz outlier-grid --csv runs/outliers/top_200.csv --show
```

**Defaults:**
- `--save`: `runs/analysis/outliers/outliers_grid.png`
- `--index`: `data/index.csv` (auto-detected)
- `--show`: `False` (save only)

### Single Object Detailed View
```bash
# Visualize spectrum + individual bands for one object
fmb viz single-object --object-id 39633427976160866

# Custom save path
fmb viz single-object --object-id 123456 --save runs/figures/object_123456.png

# With custom smoothing
fmb viz single-object --object-id 123456 --smooth 3.0 --dpi 300
```

**Defaults:**
- `--save`: `runs/analysis/objects/object_{id}.png`
- `--index`: `data/index.csv` (auto-detected)

---

## Testing

### Run Unit Tests
```bash
python -m unittest discover tests
```

---

## HPC/Slurm Integration

Add `--slurm` flag to submit jobs to a Slurm cluster:

```bash
# Submit retrain job
fmb retrain astropt --config configs/retrain/astropt.yaml --slurm

# Submit embedding extraction
fmb embed aion --slurm
```

**Note:** SLURM submission uses predefined scripts in `slurm/`. Extra arguments are not automatically forwarded.

---

## Configuration Files

### Path Configuration
- `src/fmb/configs/paths.template.yaml` - Template (versioned)
- `src/fmb/configs/paths_local.yaml` - Your local paths (gitignored)

### Model Configs
- `configs/retrain/` - Training/finetuning configurations
- `configs/embeddings/` - Embedding extraction configurations
- `configs/analysis/` - Analysis task configurations
- `configs/detection/` - Detection method configurations

### Visualization Style
- `src/fmb/configs/viz_style.yaml` - Publication-ready matplotlib style

---

## Example Workflows

### Full Pipeline (Single Model)
```bash
# 1. Setup
fmb data index --splits all

# 2. Retrain
fmb retrain astropt --config configs/retrain/astropt.yaml

# 3. Extract embeddings
fmb embed astropt --config configs/embeddings/astropt.yaml

# 4. Detect anomalies
fmb detect outliers

# 5. Visualize
fmb viz outlier-grid --csv runs/outliers/anomaly_scores_astropt.csv
```

### Multimodal Benchmark
```bash
# 1. Setup
fmb data index --splits all

# 2. Retrain all models
fmb retrain aion --config configs/retrain/aion.yaml
fmb retrain astropt --config configs/retrain/astropt.yaml
fmb retrain astroclip --config configs/retrain/astroclip.yaml

# 3. Extract all embeddings
fmb embed aion --config configs/embeddings/aion.yaml
fmb embed astropt --config configs/embeddings/astropt.yaml
fmb embed astroclip --config configs/embeddings/astroclip.yaml

# 4. Detect and fuse
fmb detect outliers
fmb detect multimodal --top-k 200 --fusion geo

# 5. Analyze
fmb analyze outliers
fmb analyze displacement
fmb analyze regression --config configs/analysis/regression.yaml

# 6. Visualize
fmb viz paper-umap
fmb viz advanced-analysis
fmb viz outlier-grid --csv runs/outliers/multimodal/top_200_geo.csv --cols 5
```

### Quick Test Run
```bash
# Fast iteration for debugging
fmb retrain astropt --max-iters 100 --eval-interval 10
fmb embed astropt --split test --batch-size 64
```
