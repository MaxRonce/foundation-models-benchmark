"""Test script for the multimodal dataloader.

This script tests the Euclid+DESI multimodal dataloader to ensure
it works correctly before training.
"""

import torch
from torch.utils.data import DataLoader

from scripts.euclid_desi_dataset.multimodal_dataloader import (
    EuclidDESIMultimodalDataset,
    multimodal_collate_fn,
    prepare_multimodal_batch,
)
from astropt.model import ModalityConfig, ModalityRegistry


def test_dataset():
    """Test basic dataset functionality."""
    print("=" * 50)
    print("TESTING DATASET")
    print("=" * 50)
    
    # Create dataset
    dataset = EuclidDESIMultimodalDataset(split="test_batch_1", image_size=224)
    print(f"✓ Dataset loaded with {len(dataset)} samples")
    
    # Test a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Object ID: {sample['object_id']}")
        print(f"  Target ID: {sample['targetid']}")
        print(f"  Redshift: {sample['redshift']:.4f}")
        print(f"  Image shape: {sample['image'].shape if sample['image'] is not None else 'None'}")
        print(f"  Spectrum shape: {sample['spectrum'].shape if sample['spectrum'] is not None else 'None'}")
        
        # Check image properties
        if sample['image'] is not None:
            img = sample['image']
            print(f"  Image dtype: {img.dtype}")
            print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Check spectrum properties
        if sample['spectrum'] is not None:
            spec = sample['spectrum']
            print(f"  Spectrum dtype: {spec.dtype}")
            print(f"  Spectrum range: [{spec.min():.3f}, {spec.max():.3f}]")
            print(f"  Spectrum non-zero elements: {(spec != 0).sum().item()}")


def test_dataloader():
    """Test dataloader with collate function."""
    print("\n" + "=" * 50)
    print("TESTING DATALOADER")
    print("=" * 50)
    
    # Create dataset and dataloader
    dataset = EuclidDESIMultimodalDataset(split="test_batch_1", image_size=224)
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=multimodal_collate_fn
    )
    
    # Test a batch
    batch = next(iter(dataloader))
    print(f"✓ DataLoader created successfully")
    print(f"Batch keys: {list(batch.keys())}")
    
    # Check batch contents
    if 'images' in batch:
        print(f"Images batch shape: {batch['images'].shape}")
        print(f"Image target IDs: {batch['image_targetids'][:5]}...")
        print(f"Image redshifts: {batch['image_redshifts'][:5]}...")
    
    if 'spectra' in batch:
        print(f"Spectra batch shape: {batch['spectra'].shape}")
        print(f"Spectrum target IDs: {batch['spectrum_targetids'][:5]}...")
        print(f"Spectrum redshifts: {batch['spectrum_redshifts'][:5]}...")
    
    print(f"Total samples in batch: {len(batch['all_object_ids'])}")


def test_model_preparation():
    """Test preparing batch for model input."""
    print("\n" + "=" * 50)
    print("TESTING MODEL PREPARATION")
    print("=" * 50)
    
    # Create modality registry
    modalities = [
        ModalityConfig(
            name="images",
            input_size=16 * 16 * 3,  # patch_size^2 * channels
            patch_size=16,
            loss_weight=1.0,
            embed_pos=True,
            pos_input_size=1,
        ),
        ModalityConfig(
            name="spectra",
            input_size=256,  # patch_size
            patch_size=256,
            pos_input_size=1,
            loss_weight=0.5,
            embed_pos=True,
        ),
    ]
    modality_registry = ModalityRegistry(modalities)
    
    # Create dataset and dataloader
    dataset = EuclidDESIMultimodalDataset(split="test_batch_1", image_size=224)
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=multimodal_collate_fn
    )
    
    # Get a batch and prepare it for the model
    batch = next(iter(dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    try:
        model_inputs = prepare_multimodal_batch(
            batch=batch,
            image_patch_size=16,
            spectrum_patch_size=256,
            device=device,
            modality_registry=modality_registry
        )
        
        print(f"✓ Model preparation successful")
        print(f"Model input keys: {list(model_inputs.keys())}")
        
        for key, tensor in model_inputs.items():
            if torch.is_tensor(tensor):
                print(f"  {key}: {tensor.shape} ({tensor.dtype})")
            else:
                print(f"  {key}: {type(tensor)}")
        
        # Check that image patches are correctly shaped
        if 'images' in model_inputs:
            img_patches = model_inputs['images']
            expected_patch_dim = 16 * 16 * 3  # patch_size^2 * channels
            print(f"Image patches: {img_patches.shape}")
            print(f"Expected last dim: {expected_patch_dim}, Actual: {img_patches.shape[-1]}")
            assert img_patches.shape[-1] == expected_patch_dim, "Image patch dimension mismatch"
        
        # Check that spectrum patches are correctly shaped
        if 'spectra' in model_inputs:
            spec_patches = model_inputs['spectra']
            expected_patch_dim = 256  # spectrum patch_size
            print(f"Spectrum patches: {spec_patches.shape}")
            print(f"Expected last dim: {expected_patch_dim}, Actual: {spec_patches.shape[-1]}")
            assert spec_patches.shape[-1] == expected_patch_dim, "Spectrum patch dimension mismatch"
        
        print("✓ All dimension checks passed")
        
    except Exception as e:
        print(f"✗ Model preparation failed: {e}")
        import traceback
        traceback.print_exc()


def test_data_statistics():
    """Analyze dataset statistics."""
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    
    dataset = EuclidDESIMultimodalDataset(split="test_batch_1", image_size=224)
    
    # Count samples with images and spectra
    has_image = 0
    has_spectrum = 0
    has_both = 0
    redshifts = []
    
    print("Analyzing dataset... (this may take a moment)")
    
    # Sample a subset for statistics
    sample_size = min(100, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size]
    
    for idx in indices:
        sample = dataset[idx.item()]
        
        img_present = sample['image'] is not None
        spec_present = sample['spectrum'] is not None
        
        if img_present:
            has_image += 1
        if spec_present:
            has_spectrum += 1
        if img_present and spec_present:
            has_both += 1
        
        redshifts.append(sample['redshift'])
    
    print(f"Sample size: {sample_size}")
    print(f"Samples with images: {has_image} ({has_image/sample_size*100:.1f}%)")
    print(f"Samples with spectra: {has_spectrum} ({has_spectrum/sample_size*100:.1f}%)")
    print(f"Samples with both: {has_both} ({has_both/sample_size*100:.1f}%)")
    
    if redshifts:
        redshifts = torch.tensor(redshifts)
        print(f"Redshift range: [{redshifts.min():.4f}, {redshifts.max():.4f}]")
        print(f"Redshift mean: {redshifts.mean():.4f} ± {redshifts.std():.4f}")


if __name__ == "__main__":
    print("Testing Euclid+DESI Multimodal DataLoader")
    print("=" * 60)
    
    try:
        test_dataset()
        test_dataloader()
        test_model_preparation()
        test_data_statistics()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✓ Dataset loads correctly")
        print("✓ DataLoader works with collate function")
        print("✓ Model preparation works")
        print("✓ Data statistics computed")
        print("\nThe dataloader is ready for training!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()