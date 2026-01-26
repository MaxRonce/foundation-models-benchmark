# FMB CLI Commands Reference

Complete reference for the Foundation Models Benchmark command-line interface.

## Setup

### Check paths configuration
```bash
python -m fmb.cli paths
```

### Display dataset samples
```bash
# View a single sample
python -m fmb.cli display --split train --index 0 --show-bands --save runs/images/sample.png

# View without GUI (save only)
python -m fmb.cli display --split all --index 5 --no-gui --save output.png
```

---

## 01. Model Retraining

### AION (Euclid ↔ HSC Adapter U-Net)
```bash
# Using config file
python -m fmb.cli retrain aion --config configs/retrain/aion.yaml

# Override specific parameters
python -m fmb.cli retrain aion --epochs 10 --batch-size 16
```

### AstroPT
```bash
# Using config file
python -m fmb.cli retrain astropt --config configs/retrain/astropt.yaml

# Override parameters
python -m fmb.cli retrain astropt --max-iters 5000 --batch-size 8
```

### AstroCLIP
```bash
# Using config file
python -m fmb.cli retrain astroclip --config configs/retrain/astroclip.yaml

# Override with checkpoint
python -m fmb.cli retrain astroclip --checkpoint path/to/ckpt --epochs 5
```

---

## 02. Embeddings Extraction

### AION
```bash
# Full dataset
python -m fmb.cli embed aion --split all

# Specific split with custom batch size
python -m fmb.cli embed aion --split test --batch-size 32
```

### AstroPT
```bash
python -m fmb.cli embed astropt --split all
```

### AstroCLIP
```bash
python -m fmb.cli embed astroclip
```

---

## 03. Anomaly Detection

### Cosine Similarity
```bash
python -m fmb.cli detect cosine --threshold-percent 1.0
```

### Normalizing Flows
```bash
python -m fmb.cli detect nfs
```

### Isolation Forest
```bash
python -m fmb.cli detect iforest
```

---

## 04. Analysis

### Physical Parameters Prediction
```bash
python -m fmb.cli analyze predict_params
```

---

## SLURM Integration

Add `--slurm` flag to submit any command as a cluster job:

```bash
# Submit retrain job to SLURM
python -m fmb.cli retrain astropt --slurm

# Submit embeddings extraction to SLURM
python -m fmb.cli embed aion --slurm
```

**Note:** SLURM submission uses predefined scripts in `slurm/` directory. Extra arguments are not automatically forwarded to sbatch.

---

## Configuration Files

Training configs are located in `configs/retrain/`:
- `aion.yaml` - AION training parameters
- `astropt.yaml` - AstroPT training parameters
- `astroclip.yaml` - AstroCLIP training parameters

Paths configuration is in `src/fmb/configs/`:
- `paths.template.yaml` - Template (versioned)
- `paths_local.yaml` - Your local paths (gitignored)

---

## Examples

### Full Training Pipeline
```bash
# 1. Retrain model
python -m fmb.cli retrain astropt --epochs 10

# 2. Extract embeddings
python -m fmb.cli embed astropt

# 3. Detect anomalies
python -m fmb.cli detect nfs

# 4. Analyze results
python -m fmb.cli analyze predict_params
```

### Quick Test Run
```bash
python -m fmb.cli retrain astropt --max-iters 100 --eval-interval 10
```
