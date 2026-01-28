#!/usr/bin/env python3
"""
Debug TARGETID mismatches in multimodal embeddings.
Check what TARGETIDs we're extracting vs what's in the catalog.
"""

import numpy as np
import torch
from astropy.io import fits
from scripts.euclid_desi_dataset.multimodal_dataloader import EuclidDESIMultimodalDataset

def debug_targetids():
    """Debug TARGETID mismatches between embeddings and catalog."""
    
    # Load the embeddings and separate metadata files
    embedding_dir = "/pbs/throng/training/astroinfo2025/work/msiudek/logs/astropt_multimodal_full_20251106_011934/embeddings_output_21000"
    
    try:
        embeddings = np.load(f"{embedding_dir}/multimodal_train_embeddings.npy")
        embedding_targetids = np.load(f"{embedding_dir}/multimodal_train_target_ids.npy")
        object_ids = np.load(f"{embedding_dir}/multimodal_train_object_ids.npy")
        redshifts = np.load(f"{embedding_dir}/multimodal_train_redshifts.npy")
        has_image = np.load(f"{embedding_dir}/multimodal_train_has_image.npy")
        has_spectrum = np.load(f"{embedding_dir}/multimodal_train_has_spectrum.npy")
        
        print(f"Loaded embeddings: {embeddings.shape}")
        print(f"Loaded target IDs: {len(embedding_targetids)} samples")
        print(f"Loaded object IDs: {len(object_ids)} samples")
        print(f"Loaded redshifts: {len(redshifts)} samples")
        print(f"Modality coverage: images={np.sum(has_image)}, spectra={np.sum(has_spectrum)}")
    except FileNotFoundError as e:
        print(f"Embedding files not found: {e}")
        return
    print(f"Embedding TARGETIDs: {len(embedding_targetids)} samples")
    print(f"First 10 embedding TARGETIDs: {embedding_targetids[:10]}")
    print(f"Unique embedding TARGETIDs: {len(np.unique(embedding_targetids))}")
    
    # Load Euclid-DESI catalog
    catalog_path = "/pbs/home/a/astroinfo09/data/astroPT_euclid_desi_dataset/desi_euclid_catalog.fits"
    try:
        with fits.open(catalog_path) as hdul:
            catalog_data = hdul[1].data
            catalog_targetids = catalog_data['TARGETID']
        print(f"Catalog TARGETIDs: {len(catalog_targetids)} samples")
        print(f"First 10 catalog TARGETIDs: {catalog_targetids[:10]}")
        print(f"Unique catalog TARGETIDs: {len(np.unique(catalog_targetids))}")
    except FileNotFoundError as e:
        print(f"Catalog file not found: {e}")
        return
    
    # Check overlap
    matches = np.isin(embedding_targetids, catalog_targetids)
    num_matches = np.sum(matches)
    print(f"\nTARGETID matches: {num_matches}/{len(embedding_targetids)} ({100*num_matches/len(embedding_targetids):.1f}%)")
    
    if num_matches == 0:
        print("No matches found! This suggests a fundamental indexing issue.")
        print(f"Embedding TARGETID range: {embedding_targetids.min()} - {embedding_targetids.max()}")
        print(f"Catalog TARGETID range: {catalog_targetids.min()} - {catalog_targetids.max()}")
        
        # Check if they're similar but offset
        if len(embedding_targetids) > 0 and len(catalog_targetids) > 0:
            print(f"Embedding TARGETIDs look like: {embedding_targetids[:5]}")
            print(f"Catalog TARGETIDs look like: {catalog_targetids[:5]}")
    
    # Also check what we get from the dataset directly
    print("\n--- Checking dataset directly ---")
    try:
        dataset = EuclidDESIMultimodalDataset(split="train_batch_1+train_batch_2+train_batch_3+train_batch_4+train_batch_5+train_batch_6+train_batch_7+train_batch_8+train_batch_9+train_batch_10+train_batch_11+train_batch_12+train_batch_13+train_batch_14+train_batch_15+train_batch_16+train_batch_17+train_batch_18+train_batch_19+train_batch_20+train_batch_21+train_batch_22+train_batch_23+train_batch_24+train_batch_25")  # Use correct split
        print(f"Dataset size: {len(dataset)}")
        
        # Check first 10 samples
        dataset_targetids = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            dataset_targetids.append(sample['targetid'])
        
        print(f"First 10 dataset TARGETIDs: {dataset_targetids}")
        
        # Compare with embedding TARGETIDs
        if len(embedding_targetids) >= 10:
            print(f"First 10 embedding TARGETIDs: {embedding_targetids[:10].tolist()}")
            matches = [dt == et for dt, et in zip(dataset_targetids, embedding_targetids[:10])]
            print(f"Direct dataset vs embedding matches: {sum(matches)}/10")
            
            # Check for duplication in embeddings
            unique_embedding_tids = np.unique(embedding_targetids)
            print(f"Unique vs total embedding TARGETIDs: {len(unique_embedding_tids)} / {len(embedding_targetids)}")
            if len(unique_embedding_tids) < len(embedding_targetids):
                print("⚠️  WARNING: Duplication detected in embedding TARGETIDs!")
                
                # Find most duplicated
                from collections import Counter
                tid_counts = Counter(embedding_targetids)
                most_common = tid_counts.most_common(5)
                print("Most duplicated TARGETIDs:")
                for tid, count in most_common:
                    print(f"  {tid}: {count} times")
    
    except Exception as e:
        print(f"Error checking dataset directly: {e}")

if __name__ == "__main__":
    debug_targetids()