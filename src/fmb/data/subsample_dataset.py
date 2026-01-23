#!/usr/bin/env python3
"""
Script to subsample the Euclid/DESI dataset.
It subsamples the EM catalog, Physical Param catalog, and the index file.
"""

import os
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path

def subsample_fits(input_path, output_path, indices):
    """Subsample a FITS file using the given indices."""
    print(f"Subsampling {input_path} -> {output_path}...")
    with fits.open(input_path) as hdul:
        # Assuming data is in the first extension (index 1)
        data = hdul[1].data
        subsampled_data = data[indices]
        
        # Create a new HDU with the subsampled data
        new_hdu = fits.BinTableHDU(data=subsampled_data, header=hdul[1].header)
        
        # Keep the primary HDU (usually empty but contains metadata)
        primary_hdu = hdul[0]
        
        new_hdul = fits.HDUList([primary_hdu, new_hdu])
        new_hdul.writeto(output_path, overwrite=True)

def subsample_csv(input_path, output_path, indices):
    """Subsample a CSV file using the given indices."""
    print(f"Subsampling {input_path} -> {output_path}...")
    df = pd.read_csv(input_path)
    subsampled_df = df.iloc[indices].copy()
    
    # Update the 'index' column in euclid_index.csv if it exists
    if 'index' in subsampled_df.columns:
        subsampled_df['index'] = np.arange(len(subsampled_df))
        
    subsampled_df.to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Subsample Euclid/DESI dataset catalogs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the subsampled files.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory where the HF dataset is cached (e.g. /n03data/ronceray/datasets).")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to keep (default: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define input paths
    base_path = Path("/home/ronceray/AION")
    em_catalog = base_path / "DESI_DR1_Euclid_Q1_dataset_catalog_EM.fits"
    params_catalog = base_path / "DESI_DR1_Euclid_Q1_dataset_catalog_physical_param.fits"
    index_file = base_path / "euclid_index.csv"
    
    # Check if files exist
    for p in [em_catalog, params_catalog, index_file]:
        if not p.exists():
            print(f"Error: {p} not found.")
            return

    # Load one file to get total length
    with fits.open(em_catalog) as hdul:
        total_rows = len(hdul[1].data)
    
    print(f"Total rows in original dataset: {total_rows}")
    
    if args.n_samples > total_rows:
        print(f"Error: Requested {args.n_samples} samples, but only {total_rows} available.")
        return
        
    # Generate random indices
    np.random.seed(args.seed)
    indices = np.random.choice(total_rows, args.n_samples, replace=False)
    indices.sort() # Keep them sorted for better performance and consistency
    
    # Perform subsampling
    subsample_fits(em_catalog, output_dir / em_catalog.name, indices)
    subsample_fits(params_catalog, output_dir / params_catalog.name, indices)
    subsample_csv(index_file, output_dir / index_file.name, indices)
    
    # HF Dataset subsampling
    if args.cache_dir:
        from datasets import load_from_disk, concatenate_datasets
        print(f"\nSubsampling HF dataset from {args.cache_dir}...")
        
        train_name = "msiudek__astroPT_euclid_Q1_desi_dr1_dataset__train"
        test_name = "msiudek__astroPT_euclid_Q1_desi_dr1_dataset__test"
        
        train_path = Path(args.cache_dir) / train_name
        test_path = Path(args.cache_dir) / test_name
        
        if not train_path.exists():
            print(f"Warning: Train split not found at {train_path}")
            train_ds = None
        else:
            train_ds = load_from_disk(str(train_path))
            print(f"Loaded Train: {len(train_ds)} samples")
            
        if not test_path.exists():
            print(f"Warning: Test split not found at {test_path}")
            test_ds = None
        else:
            test_ds = load_from_disk(str(test_path))
            print(f"Loaded Test: {len(test_ds)} samples")
            
        if train_ds is not None and test_ds is not None:
            # Concatenate to match the FITS catalog ordering (assumed)
            full_ds = concatenate_datasets([train_ds, test_ds])
            print(f"Total HF samples: {len(full_ds)}")
            
            if len(full_ds) != total_rows:
                print(f"Warning: HF dataset size ({len(full_ds)}) differs from FITS rows ({total_rows}).")
            
            # Subsample
            # Note: Select returns a new dataset
            sub_ds = full_ds.select(indices)
            
            # Now split it back based on the indices
            # Train indices are [0, len(train_ds)), Test are [len(train_ds), len(full_ds))
            # But we want to keep it simple: if the user wants 100 samples, we can just save one split or both.
            # Usually users expect the same splits.
            
            train_cut = len(train_ds)
            sub_train_indices = [i for i, idx in enumerate(indices) if idx < train_cut]
            sub_test_indices = [i for i, idx in enumerate(indices) if idx >= train_cut]
            
            if sub_train_indices:
                mini_train = sub_ds.select(sub_train_indices)
                mini_train.save_to_disk(str(output_dir / train_name))
                print(f"Saved mini Train: {len(mini_train)} samples")
            
            if sub_test_indices:
                mini_test = sub_ds.select(sub_test_indices)
                mini_test.save_to_disk(str(output_dir / test_name))
                print(f"Saved mini Test: {len(mini_test)} samples")
        else:
            print("Could not load both splits for concatenation. Subsampling individually (if possible).")
            # Fallback or error
    
    print(f"\nSubsampling complete! Files saved to: {output_dir}")
    print(f"Saved {args.n_samples} samples.")

if __name__ == "__main__":
    main()
