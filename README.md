# Foundation Models Benchmark (FMB)

**Official repository for:**
> **Benchmarking Foundation Models for Unsupervised Discovery in Large Multimodal Astrophysical Datasets**

A scalable pipeline for benchmarking pretrained multimodal foundation models (AION, AstroPT, AstroCLIP) on Euclid+DESI astronomical data.

## Quick Start

### Installation

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/MaxRonce/foundation-models-benchmark.git
cd foundation-models-benchmark

# 2. Create environment
conda env create -f environment.yml
conda activate fmb

# 3. Install package
pip install -e .

# 4. Configure paths
cp src/fmb/configs/paths.template.yaml src/fmb/configs/paths_local.yaml
# Edit paths_local.yaml with your data locations

# 5. Verify
fmb paths
```

### Run the Pipeline

```bash
# Setup
fmb data index --splits all

# Retrain (lightweight adaptation)
fmb retrain aion --config configs/retrain/aion.yaml
fmb retrain astropt --config configs/retrain/astropt.yaml
fmb retrain astroclip --config configs/retrain/astroclip.yaml

# Extract embeddings
fmb embed aion --config configs/embeddings/aion.yaml
fmb embed astropt --config configs/embeddings/astropt.yaml
fmb embed astroclip --config configs/embeddings/astroclip.yaml

# Anomaly detection
fmb detect outliers
fmb detect multimodal --top-k 200 --fusion geo

# Analysis & visualization
fmb analyze outliers
fmb viz paper-umap
fmb viz outlier-grid --csv runs/outliers/top_200.csv
```

## Documentation

- **[CLI_COMMANDS.md](CLI_COMMANDS.md)** - Complete command reference with examples
- **[architecture.md](architecture.md)** - Technical details on code organization

## What This Repository Does

- **Lightweight model adaptation** to Euclid+DESI data
- **Embedding extraction** from three foundation model paradigms
- **Scalable anomaly detection** via normalizing flows + multimodal fusion
- **Cross-model ranking analysis** to study representation-relative anomalies
- **Predictive probing** for decodability and dimensionality analysis
- **Visualizations** for embeddings exploration


## Acknowledgements

This work builds upon:
- **AstroPT**: [https://github.com/Smith42/astroPT](https://github.com/Smith42/astroPT) | [https://github.com/astroinfo-hacks/astroPT](https://github.com/astroinfo-hacks/astroPT)
- **AION**: [https://github.com/PolymathicAI/AION](https://github.com/PolymathicAI/AION)
- **AstroCLIP**: [https://github.com/PolymathicAI/AstroCLIP](https://github.com/PolymathicAI/AstroCLIP)

We thank the developers for making their foundation models publicly available.

## License

See LICENSE file for details.
