# Analysis Module

Benchmark analyses for evaluating foundation model embeddings.

## Overview

This module provides tools to analyze and benchmark the quality of embeddings from foundation models through:
- Physical parameter regression
- Anomalies in Embedding space displacement analysis
- Embeddings similarity search

## Key Components

### Physical Parameter Regression

#### `regression/predict_physical_params.py`
Predict physical parameters (redshift, stellar mass, etc.) from embeddings.

```bash
fmb analyze regression --config configs/analysis/regression.yaml
```

**Methods**:
- **Ridge Regression** (linear baseline)
- **Light GBM** (non-linear)

**Metrics**:
- R², RMSE, MAE
- **PR** (Participation Ratio) - embedding dimensionality
- **EPD** (Effective Participation Dimension)
- **PES** (Performance-Efficiency Score)
- **CWP** (Combined Weighted Performance)

**Output**: `runs/analysis/regression/`
- `results_summary.csv` - All metrics
- `scatter_<param>.png` - Prediction vs ground truth
- `pareto.png` - R² vs PR plot

**Configuration**: `configs/analysis/regression.yaml`
```yaml
targets:
  - redshift
  - stellar_mass
models:
  - AION
  - AstroPT  
  - AstroCLIP
test_size: 0.2
run_shap: true
```

---

### Embedding Displacement

#### `displacement.py`
Analyze how models disagree about what is anomalous

---

### Visual Similarity

#### `similarity.py`
Find and visualize visually similar objects in embedding space.

```bash
fmb analyze similarity --query 39633427976160866 --top-k 9
```

**Process**:
1. Load embeddings for all models
2. Compute cosine similarity to query object
3. Generate grid visualization with top-K similar objects

**Output**: `runs/analysis/similarity/similarity_combined.png`

Shows query object + 9 most similar objects with:
- RGB images
- Spectra
- Similarity scores
- Physical parameters

---

## Metrics Explained

### Participation Ratio (PR)
Measures the effective dimensionality used by the model.
- PR ≈ 1.0: Uses all dimensions equally
- PR ≈ 0.0: Uses only a few dimensions

### Effective Participation Dimension (EPD)
`EPD = PR × D` where D is embedding dimension.

### Performance-Efficiency Score (PES)
`PES = R² / PR` - Balances prediction accuracy with embedding efficiency.

### Combined Weighted Performance (CWP)
`CWP = R² × PR` - Rewards models that achieve good performance using more embedding dimensions.

### Linear GAP 
Difference in prediction R² between linear and non linear model
