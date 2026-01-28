"""
Multimodal Embedding Analysis for AstroPT

This notebook analyzes the learned embeddings from the multimodal AstroPT model 
trained on Euclid images and DESI spectra. We examine the embedding space structure 
and correlate it with astrophysical properties like stellar mass (LOGM) and redshift.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from astropy.io import fits
import seaborn as sns

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD AND CONFIGURE DATA PATHS
# ============================================================================

# Paths to embedding data and catalog
EMBEDDING_DIR = "/pbs/home/a/astroinfo09/logs/logs/astropt_multimodal_full_20251106_011934/embeddings_output_21000"
CATALOG_PATH = "/pbs/home/a/astroinfo09/data/astroPT_euclid_desi_dataset/desi_euclid_catalog.fits"
PREFIX = "multimodal_train"

print("=== MULTIMODAL EMBEDDING ANALYSIS ===")
print(f"Embedding directory: {EMBEDDING_DIR}")
print(f"Catalog path: {CATALOG_PATH}")

# Check if files exist
embedding_file = os.path.join(EMBEDDING_DIR, f"{PREFIX}_embeddings.npy")
if not os.path.exists(embedding_file):
    raise FileNotFoundError(f"Embeddings not found: {embedding_file}")
    
if not os.path.exists(CATALOG_PATH):
    raise FileNotFoundError(f"Catalog not found: {CATALOG_PATH}")

print("✓ All files found")

# ============================================================================
# 2. LOAD EMBEDDING DATA
# ============================================================================

print("\n=== LOADING EMBEDDING DATA ===")

# Load embeddings and metadata
embeddings = np.load(os.path.join(EMBEDDING_DIR, f"{PREFIX}_embeddings.npy"))
object_ids = np.load(os.path.join(EMBEDDING_DIR, f"{PREFIX}_object_ids.npy"))
target_ids = np.load(os.path.join(EMBEDDING_DIR, f"{PREFIX}_target_ids.npy"))
redshifts = np.load(os.path.join(EMBEDDING_DIR, f"{PREFIX}_redshifts.npy"))
has_image = np.load(os.path.join(EMBEDDING_DIR, f"{PREFIX}_has_image.npy"))
has_spectrum = np.load(os.path.join(EMBEDDING_DIR, f"{PREFIX}_has_spectrum.npy"))

# Load summary if available
summary_file = os.path.join(EMBEDDING_DIR, f"{PREFIX}_summary.json")
if os.path.exists(summary_file):
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    print(f"Summary loaded: {summary['n_samples']} samples, {summary['embedding_dim']}D")

print(f"Embeddings shape: {embeddings.shape}")
print(f"Object IDs shape: {object_ids.shape}")
print(f"Target IDs shape: {target_ids.shape}")
print(f"Redshifts range: {redshifts.min():.3f} - {redshifts.max():.3f}")
print(f"Mean redshift: {redshifts.mean():.3f} ± {redshifts.std():.3f}")

# Check modality coverage
n_with_images = np.sum(has_image)
n_with_spectra = np.sum(has_spectrum)
n_multimodal = np.sum(has_image & has_spectrum)
total = len(object_ids)

print(f"\nModality coverage:")
print(f"  With images: {n_with_images}/{total} ({n_with_images/total*100:.1f}%)")
print(f"  With spectra: {n_with_spectra}/{total} ({n_with_spectra/total*100:.1f}%)")
print(f"  Truly multimodal: {n_multimodal}/{total} ({n_multimodal/total*100:.1f}%)")

# ============================================================================
# 3. LOAD AND JOIN GALAXY CATALOG
# ============================================================================

print("\n=== LOADING GALAXY CATALOG ===")

# Load DESI catalog
with fits.open(CATALOG_PATH) as hdul:
    catalog_data = hdul[1].data
    
print(f"Catalog contains {len(catalog_data)} galaxies")
print(f"Catalog columns: {catalog_data.columns.names}")

# Convert to DataFrame for easier manipulation - explicit dtype conversion to fix pandas dtype error
catalog_df = pd.DataFrame({
    'TARGETID': np.array(catalog_data['TARGETID'], dtype=np.int64),
    'LOGM': np.array(catalog_data['LOGM'], dtype=np.float64),
    'Z': np.array(catalog_data['Z'], dtype=np.float64),
})

# Also include other potentially interesting columns if they exist
if 'LOGSFR' in catalog_data.columns.names:
    catalog_df['LOGSFR'] = np.array(catalog_data['LOGSFR'], dtype=np.float64)
if 'SERSIC_N' in catalog_data.columns.names:
    catalog_df['SERSIC_N'] = np.array(catalog_data['SERSIC_N'], dtype=np.float64)

print(f"Catalog DataFrame shape: {catalog_df.shape}")
print(f"LOGM range: {catalog_df['LOGM'].min():.3f} - {catalog_df['LOGM'].max():.3f}")
print(f"Mean LOGM: {catalog_df['LOGM'].mean():.3f} ± {catalog_df['LOGM'].std():.3f}")

# Create embedding DataFrame
embedding_df = pd.DataFrame({
    'TARGETID': target_ids,
    'OBJECT_ID': object_ids,
    'REDSHIFT': redshifts,
    'HAS_IMAGE': has_image.astype(bool),
    'HAS_SPECTRUM': has_spectrum.astype(bool),
})

# Add embedding components
for i in range(embeddings.shape[1]):
    embedding_df[f'EMB_{i:03d}'] = embeddings[:, i]

print(f"Embedding DataFrame shape: {embedding_df.shape}")

# Join embeddings with catalog
print("\nJoining embeddings with catalog...")
joined_df = embedding_df.merge(catalog_df, on='TARGETID', how='inner')
print(f"Joined DataFrame shape: {joined_df.shape}")
print(f"Successfully matched {len(joined_df)}/{len(embedding_df)} embeddings with catalog")

if len(joined_df) == 0:
    raise ValueError("No matches found between embeddings and catalog!")

# ============================================================================
# 4. BASIC EMBEDDING ANALYSIS
# ============================================================================

print("\n=== BASIC EMBEDDING ANALYSIS ===")

# Get matched embeddings and properties
matched_embeddings = embeddings[:len(joined_df)]
matched_logm = joined_df['LOGM'].values
matched_redshifts = joined_df['REDSHIFT'].values

# Check for and handle NaN values in astrophysical properties
logm_nan_mask = np.isnan(matched_logm)
redshift_nan_mask = np.isnan(matched_redshifts)

print(f"NaN values in LOGM: {logm_nan_mask.sum()}/{len(matched_logm)} ({logm_nan_mask.mean()*100:.1f}%)")
print(f"NaN values in redshifts: {redshift_nan_mask.sum()}/{len(matched_redshifts)} ({redshift_nan_mask.mean()*100:.1f}%)")

# Create valid mask for both properties
valid_mask = ~(logm_nan_mask | redshift_nan_mask)
print(f"Valid samples for analysis: {valid_mask.sum()}/{len(matched_embeddings)} ({valid_mask.mean()*100:.1f}%)")

if valid_mask.sum() == 0:
    raise ValueError("No valid samples found after removing NaN values!")

# Filter to valid samples only
matched_embeddings = matched_embeddings[valid_mask]
matched_logm = matched_logm[valid_mask]
matched_redshifts = matched_redshifts[valid_mask]

print(f"Analysis will proceed with {len(matched_embeddings)} valid samples")

# Basic embedding statistics
embedding_magnitudes = np.linalg.norm(matched_embeddings, axis=1)

print(f"Matched samples: {len(matched_embeddings)}")
print(f"Embedding dimensionality: {matched_embeddings.shape[1]}")
print(f"Mean embedding magnitude: {embedding_magnitudes.mean():.3f} ± {embedding_magnitudes.std():.3f}")
print(f"Embedding magnitude range: {embedding_magnitudes.min():.3f} - {embedding_magnitudes.max():.3f}")

# Check for NaN or infinite values
nan_count = np.isnan(matched_embeddings).sum()
inf_count = np.isinf(matched_embeddings).sum()
print(f"NaN values in embeddings: {nan_count}")
print(f"Infinite values in embeddings: {inf_count}")

# ============================================================================
# 5. PCA DIMENSIONALITY REDUCTION
# ============================================================================

print("\n=== PCA DIMENSIONALITY REDUCTION ===")

# Apply PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(matched_embeddings)

print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Also compute higher dimensional PCA for analysis
pca_10d = PCA(n_components=10)
embeddings_10d = pca_10d.fit_transform(matched_embeddings)
print(f"Top 10 PC explained variance: {pca_10d.explained_variance_ratio_.sum():.3f}")

# ============================================================================
# 6. VISUALIZATION: EMBEDDING SPACE
# ============================================================================

print("\n=== CREATING VISUALIZATIONS ===")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('AstroPT Multimodal Embedding Analysis', fontsize=16)

# 1. PCA projection (basic)
axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=20)
axes[0, 0].set_title('PCA Projection')
axes[0, 0].set_xlabel(f'PC1 (var: {pca.explained_variance_ratio_[0]:.1%})')
axes[0, 0].set_ylabel(f'PC2 (var: {pca.explained_variance_ratio_[1]:.1%})')
axes[0, 0].grid(True, alpha=0.3)

# 2. Colored by redshift
scatter_z = axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=matched_redshifts, alpha=0.7, s=20, cmap='viridis')
axes[0, 1].set_title('Colored by Redshift')
axes[0, 1].set_xlabel(f'PC1 (var: {pca.explained_variance_ratio_[0]:.1%})')
axes[0, 1].set_ylabel(f'PC2 (var: {pca.explained_variance_ratio_[1]:.1%})')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter_z, ax=axes[0, 1], label='Redshift')

# 3. Colored by LOGM
scatter_logm = axes[0, 2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                  c=matched_logm, alpha=0.7, s=20, cmap='plasma')
axes[0, 2].set_title('Colored by Log Stellar Mass')
axes[0, 2].set_xlabel(f'PC1 (var: {pca.explained_variance_ratio_[0]:.1%})')
axes[0, 2].set_ylabel(f'PC2 (var: {pca.explained_variance_ratio_[1]:.1%})')
axes[0, 2].grid(True, alpha=0.3)
plt.colorbar(scatter_logm, ax=axes[0, 2], label='Log M* [M☉]')

# 4. Embedding magnitude distribution
axes[1, 0].hist(embedding_magnitudes, bins=30, alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Embedding Magnitude Distribution')
axes[1, 0].set_xlabel('L2 Norm')
axes[1, 0].set_ylabel('Count')
axes[1, 0].grid(True, alpha=0.3)

# 5. Redshift distribution
axes[1, 1].hist(matched_redshifts, bins=30, alpha=0.7, edgecolor='black', color='green')
axes[1, 1].set_title('Redshift Distribution')
axes[1, 1].set_xlabel('Redshift')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(True, alpha=0.3)

# 6. LOGM distribution
axes[1, 2].hist(matched_logm, bins=30, alpha=0.7, edgecolor='black', color='orange')
axes[1, 2].set_title('Log Stellar Mass Distribution')
axes[1, 2].set_xlabel('Log M* [M☉]')
axes[1, 2].set_ylabel('Count')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(EMBEDDING_DIR, 'multimodal_embedding_analysis.png'), 
            dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================

print("\n=== CORRELATION ANALYSIS ===")

# Compute correlations between PCA components and physical properties
pc1_redshift_corr = pearsonr(embeddings_2d[:, 0], matched_redshifts)
pc2_redshift_corr = pearsonr(embeddings_2d[:, 1], matched_redshifts)
pc1_logm_corr = pearsonr(embeddings_2d[:, 0], matched_logm)
pc2_logm_corr = pearsonr(embeddings_2d[:, 1], matched_logm)

print(f"PC1 vs Redshift: r={pc1_redshift_corr[0]:.3f}, p={pc1_redshift_corr[1]:.3e}")
print(f"PC2 vs Redshift: r={pc2_redshift_corr[0]:.3f}, p={pc2_redshift_corr[1]:.3e}")
print(f"PC1 vs Log M*: r={pc1_logm_corr[0]:.3f}, p={pc1_logm_corr[1]:.3e}")
print(f"PC2 vs Log M*: r={pc2_logm_corr[0]:.3f}, p={pc2_logm_corr[1]:.3e}")

# Test correlation with embedding magnitude
mag_redshift_corr = pearsonr(embedding_magnitudes, matched_redshifts)
mag_logm_corr = pearsonr(embedding_magnitudes, matched_logm)

print(f"Embedding magnitude vs Redshift: r={mag_redshift_corr[0]:.3f}, p={mag_redshift_corr[1]:.3e}")
print(f"Embedding magnitude vs Log M*: r={mag_logm_corr[0]:.3f}, p={mag_logm_corr[1]:.3e}")

# Create correlation matrix for top 10 PCs
properties = np.column_stack([matched_redshifts, matched_logm, embedding_magnitudes])
property_names = ['Redshift', 'Log M*', 'Emb Magnitude']

pc_property_corrs = np.zeros((10, len(property_names)))
for i in range(10):
    for j, prop in enumerate(properties.T):
        pc_property_corrs[i, j] = pearsonr(embeddings_10d[:, i], prop)[0]

# Plot correlation heatmap
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
im = ax.imshow(pc_property_corrs, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(property_names)))
ax.set_xticklabels(property_names)
ax.set_yticks(range(10))
ax.set_yticklabels([f'PC{i+1}' for i in range(10)])
ax.set_title('Correlation: Principal Components vs Physical Properties')

# Add correlation values as text
for i in range(10):
    for j in range(len(property_names)):
        text = ax.text(j, i, f'{pc_property_corrs[i, j]:.2f}',
                      ha="center", va="center", color="black")

plt.colorbar(im, ax=ax, label='Pearson Correlation')
plt.tight_layout()
plt.savefig(os.path.join(EMBEDDING_DIR, 'correlation_heatmap.png'), 
            dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. DOWNSTREAM TASK EVALUATION
# ============================================================================

print("\n=== DOWNSTREAM TASK EVALUATION ===")

# Test linear regression for redshift prediction
print("Linear regression for redshift prediction:")
X_train = matched_embeddings
y_train = matched_redshifts

# Use cross-validation split
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Train linear regression
lr_z = LinearRegression()
lr_z.fit(X_tr, y_tr)
y_pred_z = lr_z.predict(X_te)
r2_z = r2_score(y_te, y_pred_z)

print(f"  Redshift R² score: {r2_z:.3f}")
print(f"  Redshift RMSE: {np.sqrt(np.mean((y_te - y_pred_z)**2)):.3f}")

# Test linear regression for stellar mass prediction
print("Linear regression for stellar mass prediction:")
y_train_logm = matched_logm
X_tr, X_te, y_tr_logm, y_te_logm = train_test_split(X_train, y_train_logm, test_size=0.3, random_state=42)

lr_logm = LinearRegression()
lr_logm.fit(X_tr, y_tr_logm)
y_pred_logm = lr_logm.predict(X_te)
r2_logm = r2_score(y_te_logm, y_pred_logm)

print(f"  Log M* R² score: {r2_logm:.3f}")
print(f"  Log M* RMSE: {np.sqrt(np.mean((y_te_logm - y_pred_logm)**2)):.3f}")

# Plot prediction vs true values
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Redshift predictions
axes[0].scatter(y_te, y_pred_z, alpha=0.7)
axes[0].plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--', lw=2)
axes[0].set_xlabel('True Redshift')
axes[0].set_ylabel('Predicted Redshift')
axes[0].set_title(f'Redshift Prediction (R² = {r2_z:.3f})')
axes[0].grid(True, alpha=0.3)

# Stellar mass predictions
axes[1].scatter(y_te_logm, y_pred_logm, alpha=0.7)
axes[1].plot([y_te_logm.min(), y_te_logm.max()], [y_te_logm.min(), y_te_logm.max()], 'r--', lw=2)
axes[1].set_xlabel('True Log M* [M☉]')
axes[1].set_ylabel('Predicted Log M* [M☉]')
axes[1].set_title(f'Stellar Mass Prediction (R² = {r2_logm:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(EMBEDDING_DIR, 'downstream_task_results.png'), 
            dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. SAVE ANALYSIS RESULTS
# ============================================================================

print("\n=== SAVING ANALYSIS RESULTS ===")

# Create results dictionary
results = {
    'embedding_stats': {
        'n_samples': len(matched_embeddings),
        'n_dimensions': matched_embeddings.shape[1],
        'mean_magnitude': float(embedding_magnitudes.mean()),
        'std_magnitude': float(embedding_magnitudes.std()),
    },
    'pca_analysis': {
        'pc1_variance': float(pca.explained_variance_ratio_[0]),
        'pc2_variance': float(pca.explained_variance_ratio_[1]),
        'total_variance_2d': float(pca.explained_variance_ratio_.sum()),
        'total_variance_10d': float(pca_10d.explained_variance_ratio_.sum()),
    },
    'correlations': {
        'pc1_redshift': {'r': float(pc1_redshift_corr[0]), 'p': float(pc1_redshift_corr[1])},
        'pc2_redshift': {'r': float(pc2_redshift_corr[0]), 'p': float(pc2_redshift_corr[1])},
        'pc1_logm': {'r': float(pc1_logm_corr[0]), 'p': float(pc1_logm_corr[1])},
        'pc2_logm': {'r': float(pc2_logm_corr[0]), 'p': float(pc2_logm_corr[1])},
        'magnitude_redshift': {'r': float(mag_redshift_corr[0]), 'p': float(mag_redshift_corr[1])},
        'magnitude_logm': {'r': float(mag_logm_corr[0]), 'p': float(mag_logm_corr[1])},
    },
    'downstream_tasks': {
        'redshift_prediction': {'r2': float(r2_z), 'rmse': float(np.sqrt(np.mean((y_te - y_pred_z)**2)))},
        'logm_prediction': {'r2': float(r2_logm), 'rmse': float(np.sqrt(np.mean((y_te_logm - y_pred_logm)**2)))},
    }
}

# Save results
results_file = os.path.join(EMBEDDING_DIR, 'embedding_analysis_results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

# Save processed data
np.save(os.path.join(EMBEDDING_DIR, 'embeddings_2d_pca.npy'), embeddings_2d)
np.save(os.path.join(EMBEDDING_DIR, 'embeddings_10d_pca.npy'), embeddings_10d)
joined_df.to_csv(os.path.join(EMBEDDING_DIR, 'joined_embeddings_catalog.csv'), index=False)

print(f"✓ Results saved to: {results_file}")
print(f"✓ PCA embeddings saved")
print(f"✓ Joined data saved to CSV")

# ============================================================================
# 10. SUMMARY
# ============================================================================

print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)

print(f"Analyzed {len(matched_embeddings)} multimodal embeddings (768D)")
print(f"Successfully matched with DESI catalog: {len(joined_df)}/{len(embedding_df)} samples")
print(f"Redshift range: {matched_redshifts.min():.3f} - {matched_redshifts.max():.3f}")
print(f"Stellar mass range: {matched_logm.min():.1f} - {matched_logm.max():.1f} log(M☉)")

print(f"\nKey findings:")
print(f"• PCA captures {pca.explained_variance_ratio_.sum():.1%} variance in 2D")
print(f"• Strongest correlation: {max(abs(pc1_redshift_corr[0]), abs(pc2_redshift_corr[0]), abs(pc1_logm_corr[0]), abs(pc2_logm_corr[0])):.3f}")
print(f"• Redshift prediction R²: {r2_z:.3f}")
print(f"• Stellar mass prediction R²: {r2_logm:.3f}")

if r2_z > 0.1 or r2_logm > 0.1:
    print("\n✅ Embeddings show good correlation with astrophysical properties!")
    print("   → Ready for full dataset training")
else:
    print("\n⚠️  Weak correlations - consider:")
    print("   → Longer training, different architectures, or more data")

print(f"\nVisualization files saved to: {EMBEDDING_DIR}")
print("="*60)