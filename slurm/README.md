# SLURM Scripts

This directory contains SLURM batch scripts for submitting FMB pipeline tasks to the cluster.

## Structure

```
slurm/
├── 01_retrain/      # Model retraining and fine-tuning
├── 02_embeddings/   # Embedding extraction
├── 03_detection/    # Anomaly detection  
└── 04_analysis/     # Analysis and visualization
```

## Usage

### Submit a job
```bash
sbatch slurm/01_retrain/astropt.sbatch
```

### Monitor jobs
```bash
squeue -u $USER
```

### Check job output
```bash
tail -f slurm/logs/retrain_astropt_<JOBID>.out
```

## Convention

**All scripts call the unified CLI:**
```bash
python -m fmb.cli <command> <subcommand> [--options]
```

This ensures consistency between local execution and cluster runs.

## Available Scripts

### 01_retrain/
- `aion_codec.sbatch` : Retrain AION codec
- `aion_adapter.sbatch` : Retrain AION Euclid ↔ HSC adapter
- `astropt.sbatch` : Retrain AstroPT
- `astroclip.sbatch` : Fine-tune AstroCLIP (if exists)

### 02_embeddings/
- `aion.sbatch` : Extract AION embeddings
- `astropt.sbatch` : Extract AstroPT embeddings
- `astroclip.sbatch` : Extract AstroCLIP embeddings (if exists)

### 03_detection/
- `nfs.sbatch` : Anomaly detection using Normalizing Flows
- `cosine.sbatch` : Cosine similarity-based detection (if exists)

### 04_analysis/
- `physical_params.sbatch` : Predict physical parameters from embeddings

## Customization

To override config parameters, pass CLI arguments in the sbatch script:
```bash
python -m fmb.cli retrain astropt --epochs 15 --batch-size 16
```
