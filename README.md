# Foundation Models Hackathon - Embeddings Analysis

This directory contains the wrap-up work for the hackathon on using foundation models (AstroClip, astroPT, AION) with Euclid Q1 images and DESI spectra.

## Overview

We analyze embeddings from two different multimodal foundation models:
- **astroPT**: Transformer model trained on DESI spectra + Euclid images
- **AION**: Foundation model trained on Euclid images + DESI spectra

We explore:
1. How each model encodes spectral and imaging information
2. Correlation with physical galaxy properties (mass, SFR, age, colors)
3. Cross-model comparison to understand different architectures

## Hackathon Work Summary

This repository aggregates the hackathon outcomes comparing three multimodal model families on Euclid Q1 imaging + DESI spectra:
- **AION** (foundation multimodal): separate spectra-only, images-only and fused embeddings enabling complementarity analysis.
- **astroPT** (transformer multimodal): joint training producing a unified embedding space for spectra+images.
- **AstroCLIP** (contrastive): image–spectrum alignment; training issues identified and pending fixes for stable convergence.

### Delivered During Hackathon
- Extracted embeddings (spectra-only, images-only, fused) and performed UMAP projections across physical (mass, SFR, DN4000, colors) and morphological (ellipticity, concentration, asymmetry, smoothness, Gini, Sérsic) properties.
- Cross-model correlation benchmarking of encoded galaxy properties.
- Cosine similarity analyses between modality-specific and fused embeddings to quantify redundancy vs complementarity.
- Initial anomaly exploration (spectral class separation: GALAXY / QSO / STAR) informing downstream outlier strategies.

### Current Data Scope
- Working matched sample: ~15k Euclid cutouts + DESI spectra.
- Planned expansion (post-hackathon): larger Euclid sets (≈40k → DR1 scale) for improved statistical power.

### High-Level Findings
- Fused embeddings improve separation for age/mass-sensitive features (e.g., DN4000, stellar mass) relative to single modalities.
- Spectra emphasize stellar population/age indicators; images emphasize morphology—together yield complementary structure.
- Architectural differences (token fusion vs transformer joint encoding) produce distinct correlation profiles worth continued study.

Further actionable next steps sourced from team discussion are added in a separate TODO section below.
 
## TODO (from Discord message - Maxime 2025-11-11)

### Models
- Fine-tune AION encoder.
- Fix AstroCLIP.
- Consider using the same image encoder for AION and AstroCLIP.

### Data
- Use 40k images dataset (instead of current 15k).
- Consider full Euclid DR1 (~300k, membership + cutout generation + merge with DESI spectra).

### Embeddings
- Compare embedding types: spectrum-only, images-only, images + spectrum.
- Quantify whether different embeddings recover the same types of anomalies.
- Characterize anomalies and compare across embedding types.

### Algorithms / Methods
- Isolation Forest.
- Local Outlier Factor (LOF).
- Density-based methods.
- Bayesian MLP regression on physical parameters; use higher predictive sigma as anomaly score.
- Regression + SHAP to interpret what embeddings encode.

### Targeted Anomaly Recovery
- Lens catalog (images only).
- Dual AGN (≈50 in the ~15k sample).
- AGN.
- Identify anomalies recovered by multiple embedding types.

---
These items guide post-hackathon prioritization; keep this section synchronized with ongoing discussions. Do not edit `TODO.md` for these entries.

## Notebooks

### 1. `aion_embeddings_analysis.ipynb`
**AION multimodal model analysis**

This notebook analyzes embeddings from the AION foundation model:
- **Spectra-only embeddings**: DESI spectra processed by AION
- **Image-only embeddings**: Euclid images processed by AION
- **Multimodal embeddings**: Combined spectra + images from AION

Visualizations include both:
- **Spectral/Physical properties**: redshift, stellar mass, SFR, sSFR, DN4000, colors
- **Morphological properties**: ellipticity, size (Kron radius), concentration, asymmetry, smoothness, Gini coefficient, Sérsic index

**Key analyses:**
- UMAP dimensionality reduction and visualization
- Correlation with physical properties (stellar mass, SFR, DN4000, colors)
- Spectral type separation (GALAXY, QSO, STAR)
- Cosine similarity between image-only and multimodal embeddings

**Data sources:**
- AION embeddings: `/pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt`
- Catalog: `/pbs/throng/training/astroinfo2025/data/astroPT_euclid_desi_dataset/desi_euclid_catalog.fits`

### 2. `astropt_embeddings_analysis.ipynb`
**astroPT multimodal model analysis**

This notebook analyzes embeddings from the astroPT transformer model:
- **Multimodal embeddings**: DESI spectra + Euclid images combined

**Data sources:**
- astroPT embeddings: `/pbs/home/a/astroinfo09/logs/logs/astropt_multimodal_full_20251106_011934/embeddings_output_21000/`
- Model checkpoint: `/pbs/home/a/astroinfo09/logs/logs/astropt_multimodal_full_20251106_011934/ckpt_iter_21000.pt`
- Catalog: `/pbs/throng/training/astroinfo2025/data/astroPT_euclid_desi_dataset/desi_euclid_catalog.fits`

### 3. `model_comparison_astropt_aion.ipynb`
**Cross-model comparison: astroPT vs AION**

This notebook performs a comprehensive comparison between:
- **astroPT multimodal embeddings**: Transformer trained on spectra + images
- **AION multimodal embeddings**: Foundation model trained on images + spectra

**Key analyses:**
- Side-by-side UMAP visualizations colored by physical properties
- Correlation analysis comparing how well each model encodes physical features
- Direct comparison on common objects (if available)
- Cosine similarity between models

**Data sources:**
- astroPT embeddings: `/pbs/home/a/astroinfo09/logs/logs/astropt_multimodal_full_20251106_011934/embeddings_output_21000/`
- AION embeddings: Same as notebook #1
- Catalog: Same as notebook #1

## Data Files

### Required Files
- **AION embeddings**: `/pbs/throng/training/astroinfo2025/work/maxime/data_all_tokens_spectrums.pt`
  - Generated by Maxime during the hackathon
  - Contains embeddings for spectra, images, and multimodal (as separate tensors)
  - Keys: `embedding_desi` (spectra), `embedding_hsc` (images), `embedding_hsc_desi` (multimodal)

- **Catalog**: `/pbs/throng/training/astroinfo2025/data/astroPT_euclid_desi_dataset/desi_euclid_catalog.fits`
  - Combined DESI-Euclid catalog with physical and morphological properties
  - Spectral/Physical: TARGETID, redshift (Z), stellar mass (LOGM), SFR (LOGSFR), DN4000, colors (GR), spectral type (SPECTYPE)
  - Morphological: ellipticity, kron_radius, concentration, asymmetry, smoothness, gini, sersic_index, axis_ratio

- **astroPT embeddings**: `/pbs/home/a/astroinfo09/logs/logs/astropt_multimodal_full_20251106_011934/embeddings_output_21000/`
  - Multimodal embeddings from iteration 21000 checkpoint
  - Generated during hackathon training
  - Contains embeddings for combined spectra + images

## Running the Notebooks

### Prerequisites
```bash
pip install numpy matplotlib seaborn torch astropy umap-learn scipy scikit-learn
```

### Execution Order
1. **First**, run `aion_embeddings_analysis.ipynb` to analyze AION embeddings
2. **Then**, run `model_comparison_astropt_aion.ipynb` to compare models

### Expected Outputs

Each notebook generates multiple PNG files:

**From `aion_embeddings_analysis.ipynb`:**

*Spectral/Physical Properties:*
- `aion_embeddings_comparison_redshift.png`: 3-panel comparison (spectra/image/multimodal) colored by redshift
- `aion_embeddings_logm.png`: Colored by stellar mass
- `aion_embeddings_logsfr.png`: Colored by star formation rate
- `aion_embeddings_ssfr.png`: Colored by specific SFR
- `aion_embeddings_dn4000.png`: Colored by 4000Å break
- `aion_embeddings_gr_color.png`: Colored by g-r color
- `aion_embeddings_spectype.png`: Colored by spectral type

*Morphological Properties:*
- `aion_embeddings_ellipticity.png`: Colored by ellipticity
- `aion_embeddings_kron_radius.png`: Colored by size (Kron radius)
- `aion_embeddings_concentration.png`: Colored by concentration index
- `aion_embeddings_asymmetry.png`: Colored by asymmetry
- `aion_embeddings_smoothness.png`: Colored by smoothness/clumpiness
- `aion_embeddings_gini.png`: Colored by Gini coefficient
- `aion_embeddings_sersic_index.png`: Colored by Sérsic index

*Analysis:*
- `aion_cosine_similarity_analysis.png`: Similarity between all embedding type pairs

**From `astropt_embeddings_analysis.ipynb`:**
- `astropt_embeddings_redshift.png`: UMAP colored by redshift
- `astropt_embeddings_logm.png`: Colored by stellar mass
- `astropt_embeddings_logsfr.png`: Colored by SFR
- `astropt_embeddings_ssfr.png`: Colored by specific SFR
- `astropt_embeddings_dn4000.png`: Colored by 4000Å break
- `astropt_embeddings_gr_color.png`: Colored by g-r color
- `astropt_embeddings_spectype.png`: Colored by spectral type

**From `model_comparison_astropt_aion.ipynb`:**
- `model_comparison_redshift.png`: 2-model comparison colored by redshift
- `model_comparison_logm.png`: Colored by stellar mass
- `model_comparison_logsfr.png`: Colored by SFR
- `model_comparison_ssfr.png`: Colored by specific SFR
- `model_comparison_dn4000.png`: Colored by 4000Å break
- `model_comparison_gr_color.png`: Colored by g-r color
- `model_comparison_correlations.png`: Bar chart comparing correlations between models
- `model_comparison_cosine_similarity.png`: Direct similarity between astroPT and AION

## Key Scientific Questions

1. **Multimodal architectures comparison:**
   - How do astroPT (transformer) and AION (foundation model) compare as multimodal learners?
   - Which architecture better captures galaxy properties from combined data?

2. **Physical property encoding:**
   - Which properties are best captured by multimodal embeddings?
   - Do both models show similar correlation patterns with physical properties?

3. **Galaxy population separation:**
   - How well do multimodal models separate star-forming vs passive galaxies?
   - Can we identify quasars and stars effectively in both embedding spaces?

4. **Information from images vs spectra (AION analysis):**
   - How much information gain does adding spectra to images provide?
   - Are image and spectral information complementary or redundant in AION?

## References

- **astroPT**: Transformer model for astronomical spectra
  - Repository: `/home/zoubian/Workspace/AstroInfo/astroPT/`
  - Analysis: `astroPT/scripts/desi/visualisations/embeddings/visualize_embeddings.ipynb`

- **AION**: Multimodal foundation model
  - Repository: `/home/zoubian/Workspace/AstroInfo/AION/`
  - Analysis script: `AION/scratch/analyze_embeddings.py`

## Authors

- Analysis setup: Based on work in `astroPT/scripts/desi/visualisations/embeddings/`
- AION embeddings: Maxime (see `AION/scratch/analyze_embeddings.py`)
- Notebooks: Created for hackathon wrap-up

## Notes

- The notebooks are designed to match the visualization style from the astroPT analysis
- UMAP parameters are consistent across analyses for fair comparison
- Physical properties are extracted from the same catalog for consistency
- Both notebooks include extensive correlation analyses to quantify what information is encoded

## Troubleshooting

### File Not Found Errors
- Check that the data paths in the notebooks match your file locations
- Update paths to your saved astroPT embeddings in notebook #2

### No Common Objects
- If notebooks report no common objects, the datasets may be from different samples
- Analysis will still work, showing separate visualizations for each model

### Memory Issues
- UMAP can be memory-intensive for large datasets
- Consider reducing the dataset size or using a machine with more RAM
- Typical RAM requirement: 16GB+ for datasets with >100k objects
