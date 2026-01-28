# Visualization Module

Publication-ready visualization tools and scripts.

## Overview

This module provides high-quality plotting utilities for:
- UMAP embeddings
- Outlier detection results
- Similarity search and neighbours analysis
- Spectra and multi-band images

All visualizations use consistent styling defined in `style.py`.

## Key Components

### Combined UMAP

#### `combined_umap.py`
Generate multi-panel UMAP visualizations comparing all foundation models.

```bash
fmb viz paper-umap
```

**Output**: `runs/analysis/umap/combined_umap.png`

Shows 2×3 grid:
- Top row: UMAPs colored by physical parameter (e.g., redshift)
- Mid row: UMAPs with thumbnail images
- Bottom row: cosine similarity distribution between images and spectra embeddings

---

### Similarity Visualization

#### `similarity.py`
Grid visualization of visually similar objects.

```bash
fmb analyze similarity --query <object_id> --top-k 9
```

**Output**: `runs/analysis/similarity/similarity_combined.png`

---

### Outlier Visualizations

#### `outliers/outlier_grid.py`
Generate grids of top anomalous objects.

```bash
fmb viz outlier-grid --csv runs/outliers/anomaly_scores_aion.csv --cols 4
```

**Output**: `runs/analysis/outliers/outliers_grid.png`

4× grid of most anomalous objects with:
- Images
- Spectra
- Anomaly sigma scores
- Object IDs

---

#### `outliers/plot_paper_single_object.py` (single_object.py)
Detailed visualization of a single object.

```bash
fmb viz single-object --object-id 39633427976160866
```

**Output**: `runs/analysis/objects/object_<id>.png`

Shows 5 panels:
1. Rest-frame spectrum with emission lines
2. VIS band image
3. NISP-Y band image
4. NISP-J band image
5. NISP-H band image

---

#### `outliers/advanced_analysis.py`
Cross-model comparison of outlier rankings.

```bash
fmb viz advanced-analysis --save-prefix runs/analysis/paper_v1
```

**Outputs**:
- `paper_v1_spearman_clustermap.png` - Rank correlation heatmap
- `paper_v1_jaccard_clustermap.png` - Jaccard similarity
- `paper_v1_disagreement_scatter.png` - Objects where models disagree
- `paper_v1_disagreement_objects.csv` - Controversial cases

---

### Utilities

#### `style.py`
Centralized visualization styling.

**Functions:**
- `set_style()` - Global matplotlib rc params
- `apply_style()` - Apply FMB style to current figure

Loads from `configs/viz_style.yaml`:
```yaml
colors:
  aion: "#FF6B6B"
  astropt: "#4ECDC4"
  astroclip: "#95E1D3"
font:
  family: sans-serif
  size: 10
```

---

#### `spectrum.py`
Spectrum extraction and visualization utilities.

**Functions:**
- `extract_spectrum(sample)` - Get wavelength and flux from sample
- `LATEX_REST_LINES` - Dictionary of emission line positions

---

#### `utils.py`
General visualization helpers.

**Functions:**
- `load_index(path)` - Load object_id → (split, idx) mapping
- `prepare_rgb_image(sample)` - Extract and normalize RGB
- `collect_samples(object_ids, ...)` - Batch load samples

---

## Configuration

Visualization settings in `configs/viz_style.yaml`:
- Colors for each model
- Font settings
- Figure dimensions
- DPI settings

Paths managed via `paths_local.yaml`:
```yaml
analysis: runs/analysis
dataset_index: data/index/euclid_index.csv
```