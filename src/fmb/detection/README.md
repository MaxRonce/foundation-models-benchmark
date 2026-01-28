# Detection Module

Anomaly detection methods for astronomical objects using foundation model embeddings.

## Overview

This module implements two complementary anomaly detection approaches:
1. **Normalizing Flows**: Density-based outlier detection in embedding space
2. **Cosine Mismatch**: Detect objects with inconsistent multimodal representations

## Key Components

### Normalizing Flow Detection

#### `models.py`
Neural network architectures for density estimation.

**Classes:**
- `CouplingFlow` - Coupling layer implementation
- `build_flow_model()` - Factory for creating NF models

**Supported Architectures:**
- Coupling flows (RealNVP-style)
- Autoregressive flows

---

#### `train.py`
Training script for Normalizing Flow models.

```python
from fmb.detection.train import train_flow

train_flow(
    embeddings_path="runs/embeddings/aion.pt",
    model_type="coupling",
    n_layers=8,
    hidden_dim=256
)
```

---


```python
from fmb.detection.run import detect_anomalies

scores = detect_anomalies(
    model_path="runs/nfs/aion/flow.pt",
    embeddings_path="runs/embeddings/aion.pt"
)
```

**Output**: CSV with columns:
- `object_id`
- `embedding_key` (e.g., "embedding_images")
- `anomaly_sigma` (z-score)
- `rank` (1 = most anomalous)

---

### Cosine Mismatch Detection

#### `cosine.py`
Detect objects where different modalities (images, spectra) produce inconsistent embeddings.

```python
from fmb.detection.cosine import compute_cosine_mismatch

mismatches = compute_cosine_mismatch(
    embeddings_path="runs/embeddings/astropt.pt"
)
```

**Method**:
1. Compute cosine similarity between image and spectrum embeddings
2. Rank objects by similarity (low = mismatch = anomalous)

**Output**: CSV with:
- `object_id`
- `cosine_similarity`
- `rank`

---

### Multimodal Fusion

#### `multimodal.py`
Combines Normalizing Flow and Cosine Mismatch scores for comprehensive anomaly detection.

```bash
fmb detect multimodal --top-k 200 --fusion geo
```

**Fusion Strategies**:
- **Geometric Mean**: `score = p_mis × √(p_img × p_spec)`
- **Minimum**: `score = p_mis × min(p_img, p_spec)`

**Output**:
- `runs/outliers/multimodal/all_scores.csv` - Full ranked list
- `runs/outliers/multimodal/ranked/` - Top-K per model
- `runs/outliers/multimodal/filtered/` - Threshold-filtered lists

---

### Utilities

#### `utils.py`
Helper functions for embedding extraction and preprocessing.

---

## Workflow

```bash
# 1. Train Normalizing Flow
fmb detect train --model aion --layers 8

# 2. Run NF inference
fmb detect outliers --model aion

# 3. Compute cosine mismatches
fmb detect cosine --model astropt

# 4. Fuse results
fmb detect multimodal --fusion geo --top-k 200
```

##Output Structure

```
runs/outliers/
├── anomaly_scores_aion.csv       # NF scores
├── anomaly_scores_astropt.csv
├── anomaly_scores_astroclip.csv
├── cosine_scores_aion.csv        # Cosine mismatch
├── cosine_scores_astropt.csv
├── cosine_scores_astroclip.csv
└── multimodal/                   # Fused scores
    ├── all_scores.csv
    ├── ranked/
    └── filtered/
```

## Configuration

Config files in `src/fmb/configs/detection/`:
- `anomalies.yaml` - NF training parameters
- Paths managed via `paths_local.yaml`

Example:
```yaml
# anomalies.yaml
model_type: coupling
n_layers: 8
hidden_dim: 256
learning_rate: 1e-4
```
