#!/usr/bin/env python3
"""
Script to compute dataset statistics and generate a compact redshift distribution plot.
for "dataset statistics" section of the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import seaborn as sns

# --- Publication Style Settings ---
# Matched to predict_physical_params.py / general paper guidelines
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
except Exception:
    print("Warning: LaTeX not available, falling back to STIX fonts.")
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    })

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "lines.linewidth": 1.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.figsize": (3.5, 2.5), # Compact size
})

CATALOG_PATH = Path("/home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_EM.fits")
OUTPUT_DIR = Path("paper_plots")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading catalog from {CATALOG_PATH}...")
    try:
        with fits.open(CATALOG_PATH) as hdul:
            data = hdul[1].data
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return

    print(f"Total rows in catalog: {len(data)}")
    
    # 1. Statistics
    n_objects = len(data)
    
    # Check for valid redshift
    # Usually 'Z' is the redshift.
    zs = data['Z']
    # Filter for finite and reasonably physical redshifts (e.g. >= -0.01 to allow for slight negatives due to noise, or just > -1)
    valid_z_mask = np.isfinite(zs) & (zs > -1)
    n_valid_z = np.sum(valid_z_mask)
    
    # Check for spectral data availability
    if 'SPECTYPE' in data.columns.names:
        spectypes = data['SPECTYPE']
        try:
            spectypes = [s.strip() for s in spectypes] 
        except:
            pass
        unique_spec, counts_spec = np.unique(spectypes, return_counts=True)
        print("Spectral Types distribution:")
        for u, c in zip(unique_spec, counts_spec):
            print(f"  {u}: {c}")
    else:
        print("SPECTYPE column not found.")
        unique_spec, counts_spec = [], []
        
    print("-" * 30)
    print(f"Total Objects: {n_objects}")
    print(f"Objects with Valid Redshift (Z): {n_valid_z} ({n_valid_z/n_objects:.1%})")
    print("-" * 30)
    
    # Write stats to file
    with open(OUTPUT_DIR / "dataset_statistics.txt", "w") as f:
        f.write(f"Total Objects: {n_objects}\n")
        f.write(f"Objects with Valid Redshift: {n_valid_z} ({n_valid_z/n_objects:.1%})\n")
        if 'SPECTYPE' in data.columns.names:
             f.write("Spectral Types:\n")
             for u, c in zip(unique_spec, counts_spec):
                f.write(f"  {u}: {c}\n")

    # 2. Redshift Distribution Plot
    # Compact, paper ready
    if n_valid_z > 0:
        z_valid = zs[valid_z_mask]
        # Fix endianness for pandas/seaborn
        z_valid = z_valid.astype(float)
        
        # Figure setup
        fig, ax = plt.subplots(figsize=(3.3, 2.0)) # Very compact
        
        # Histogram with step-filled style for clarity
        sns.histplot(
            z_valid, 
            bins=np.arange(0, 4.0, 0.1), # Fixed bins for consistent look
            element="step", 
            fill=True, 
            color="#3C4F76", # Darker, more professional blue
            alpha=0.3, 
            edgecolor="#3C4F76",
            linewidth=1.0, 
            ax=ax
        )
        
        # Median vertical line
        median_z = np.median(z_valid)
        ax.axvline(median_z, color="#D45B5B", linestyle='--', linewidth=1.2, label='Median')
        
        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel("Count")
        ax.grid(True, which='major', linestyle=':', color='gray', alpha=0.3)
        
        # Remove top and right spines
        sns.despine()
        
        # Compact stats legend/text
        stats_text = (
            f"\\textbf{{Euclid x DESI}}\n"
            f"$N = {n_valid_z:,}$\n"
            f"Median $z = {median_z:.2f}$"
        )
        # Position in top right, no box for cleaner look
        ax.text(0.95, 0.90, stats_text, transform=ax.transAxes, 
                ha='right', va='top', fontsize=9, linespacing=1.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        out_path = OUTPUT_DIR / "redshift_distribution_compact.pdf"
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot to {out_path}")
        
        # Also PNG
        plt.savefig(OUTPUT_DIR / "redshift_distribution_compact.png", dpi=300)
    else:
        print("No valid redshifts found to plot.")

if __name__ == "__main__":
    main()
