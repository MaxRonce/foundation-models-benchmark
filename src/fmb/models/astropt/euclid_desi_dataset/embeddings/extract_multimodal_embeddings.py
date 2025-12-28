"""Extract multimodal embeddings from trained AstroPT models.

This script extracts embeddings from the multimodal Euclid+DESI training,
incorporating lessons learned from the DESI-only extraction to avoid duplication
issues and ensure robust extraction.

Based on the clean extraction approach from extract_embeddings_clean.py.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import Counter
import json

# Add the scripts directory to path for imports
import sys
sys.path.insert(0, './scripts')

from euclid_desi_dataset.multimodal_dataloader import (
    EuclidDESIMultimodalDataset,
    multimodal_collate_fn,
    prepare_multimodal_batch,
)


def load_checkpoint(checkpoint_path: str, device: str):
    """Load multimodal model checkpoint from disk."""
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    if 'config' in checkpoint and hasattr(checkpoint['config'], '__dict__'):
        # Config is a dataclass/object
        config_dict = checkpoint['config'].__dict__ if hasattr(checkpoint['config'], '__dict__') else checkpoint['config']
    elif 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        # Config is already a dict
        config_dict = checkpoint['config']
    else:
        # Try model_args as fallback
        config_dict = checkpoint.get('model_args', {})
    
    # Extract relevant config parameters
    block_size = config_dict.get('block_size', 1024)
    n_layer = config_dict.get('n_layer', 12)
    n_head = config_dict.get('n_head', 12)
    n_embd = config_dict.get('n_embd', 768)
    bias = config_dict.get('bias', False)
    dropout = config_dict.get('dropout', 0.0)
    
    # Extract modality-specific parameters
    image_patch_size = config_dict.get('image_patch_size', 16)
    spectrum_patch_size = config_dict.get('spectrum_patch_size', 10)
    n_chan = config_dict.get('n_chan', 3)
    
    print(f"Model config: block_size={block_size}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
    print(f"Modality config: image_patch_size={image_patch_size}, spectrum_patch_size={spectrum_patch_size}")
    
    # Create GPT config
    gpt_config = GPTConfig(
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        bias=bias,
        dropout=dropout,
        attn_type="causal",
    )
    
    # Create modality registry (use balanced weights from observed frame training)
    modalities = [
        ModalityConfig(
            name="images",
            input_size=image_patch_size * image_patch_size * n_chan,
            patch_size=image_patch_size,
            loss_weight=779/196,  # From observed frame training
            embed_pos=True,
            pos_input_size=1,
        ),
        ModalityConfig(
            name="spectra",
            input_size=spectrum_patch_size,
            patch_size=spectrum_patch_size,
            pos_input_size=1,
            loss_weight=196/779,  # From observed frame training
            embed_pos=True,
        ),
    ]
    modality_registry = ModalityRegistry(modalities)
    
    # Create model
    model = GPT(gpt_config, modality_registry)
    
    # Load state dict, handling compiled model prefixes
    state_dict = checkpoint.get('model', checkpoint)
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model, gpt_config, (image_patch_size, spectrum_patch_size)


def find_final_norm(model):
    """Find the final normalization layer in the model."""
    candidates = []
    for name, module in model.named_modules():
        if 'ln_f' in name.lower() or 'final' in name.lower():
            candidates.append((name, module))
    
    if not candidates:
        raise ValueError("Could not find final normalization layer")
    
    # Return the last candidate (likely the final one)
    name, module = candidates[-1]
    print(f"Using final norm layer: {name}")
    return module


def extract_embeddings_with_diagnostics(
    model,
    dataloader,
    image_patch_size: int,
    spectrum_patch_size: int,
    device: str,
    max_batches: int = None,
):
    """Extract multimodal embeddings with duplication checks.
    
    Args:
        model: Trained multimodal model
        dataloader: DataLoader for multimodal data
        image_patch_size: Size of image patches
        spectrum_patch_size: Size of spectrum patches  
        device: Device to run on
        max_batches: Optional limit on number of batches
    
    Returns:
        Tuple of (embeddings, metadata_dict) or (None, None) if duplicates found
    """
    # Set up hook to capture final embeddings
    ln_final = find_final_norm(model)
    captured_hidden = {}
    
    def hook_fn(module, input, output):
        captured_hidden['last'] = output.detach()
    
    handle = ln_final.register_forward_hook(hook_fn)
    
    # Storage for results
    all_embeddings = []
    all_metadata = {
        'object_ids': [],
        'target_ids': [],
        'redshifts': [],
        'has_image': [],
        'has_spectrum': [],
    }
    
    # Track object IDs for duplication detection
    seen_object_ids = Counter()
    
    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Extracting multimodal embeddings")
            for batch_idx, batch in enumerate(pbar):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Prepare multimodal batch using the same function as training
                inputs = prepare_multimodal_batch(
                    batch, image_patch_size, spectrum_patch_size, device, model.modality_registry
                )
                
                if not inputs:
                    print(f"Warning: Empty inputs for batch {batch_idx}")
                    continue
                
                # Track object IDs for duplication detection
                object_ids = batch.get('all_object_ids', [])
                for oid in object_ids:
                    seen_object_ids[oid] += 1
                
                # Forward pass through model
                _ = model(inputs)
                
                # Extract embeddings from final hidden state
                hidden = captured_hidden['last']  # Shape: (batch_size, seq_len, n_embd)
                
                # Pool across sequence dimension to get per-sample embeddings
                embeddings = hidden.mean(dim=1)  # Shape: (batch_size, n_embd)
                
                # Store results
                all_embeddings.append(embeddings.cpu().numpy())
                
                # Store metadata
                all_metadata['object_ids'].extend(batch.get('all_object_ids', []))
                all_metadata['target_ids'].extend(batch.get('all_targetids', []))
                all_metadata['redshifts'].extend(batch.get('all_redshifts', []))
                
                # Track which modalities are present for each sample
                batch_size = embeddings.shape[0]
                has_images = 'images' in batch and len(batch['images']) == batch_size
                has_spectra = 'spectra' in batch and len(batch['spectra']) == batch_size
                
                all_metadata['has_image'].extend([has_images] * batch_size)
                all_metadata['has_spectrum'].extend([has_spectra] * batch_size)
                
                # Report progress every 100 batches
                if (batch_idx + 1) % 100 == 0:
                    unique = len(seen_object_ids)
                    total = sum(seen_object_ids.values())
                    dup_rate = (1 - unique/total) * 100 if total > 0 else 0
                    pbar.set_postfix({
                        'batch': batch_idx + 1,
                        'unique': unique,
                        'total': total,
                        'dup_rate': f'{dup_rate:.1f}%'
                    })
    
    finally:
        handle.remove()
    
    # Combine all embeddings
    if not all_embeddings:
        print("❌ No embeddings extracted!")
        return None, None
    
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    
    # Final diagnostics
    print(f"\n=== EXTRACTION DIAGNOSTICS ===")
    print(f"Total samples processed: {len(all_metadata['object_ids'])}")
    print(f"Unique object IDs: {len(seen_object_ids)}")
    print(f"Embedding shape: {embeddings_array.shape}")
    
    # Check for duplicates
    duplicates = {oid: count for oid, count in seen_object_ids.items() if count > 1}
    if duplicates:
        print(f"\n⚠️  WARNING: Found {len(duplicates)} duplicate object IDs!")
        top_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 duplicates:")
        for oid, count in top_dups:
            print(f"  {oid}: {count} times")
        print("\n❌ DUPLICATION DETECTED - DO NOT USE THESE EMBEDDINGS")
        return None, None
    
    # Check modality coverage
    n_with_images = sum(all_metadata['has_image'])
    n_with_spectra = sum(all_metadata['has_spectrum'])
    total = len(all_metadata['object_ids'])
    
    print(f"\nModality coverage:")
    print(f"  Samples with images: {n_with_images}/{total} ({n_with_images/total*100:.1f}%)")
    print(f"  Samples with spectra: {n_with_spectra}/{total} ({n_with_spectra/total*100:.1f}%)")
    print(f"  Truly multimodal: {min(n_with_images, n_with_spectra)}/{total}")
    
    print("\n✓ No duplicates found - embeddings are valid!")
    
    return embeddings_array, all_metadata


def save_embeddings(embeddings, metadata, output_dir, prefix="multimodal"):
    """Save embeddings and metadata to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    emb_path = os.path.join(output_dir, f"{prefix}_embeddings.npy")
    print(f"Saving embeddings to: {emb_path}")
    np.save(emb_path, embeddings)
    
    # Save metadata arrays
    for key, values in metadata.items():
        if key in ['object_ids', 'target_ids']:
            # Save as int64
            array = np.array(values, dtype=np.int64)
        elif key == 'redshifts':
            # Save as float32
            array = np.array(values, dtype=np.float32)
        else:
            # Save boolean arrays as uint8
            array = np.array(values, dtype=np.uint8)
        
        path = os.path.join(output_dir, f"{prefix}_{key}.npy")
        print(f"Saving {key} to: {path}")
        np.save(path, array)
    
    # Save metadata summary as JSON
    summary = {
        'n_samples': len(metadata['object_ids']),
        'n_unique_objects': len(set(metadata['object_ids'])),
        'n_unique_targets': len(set(metadata['target_ids'])),
        'embedding_dim': embeddings.shape[1],
        'modality_coverage': {
            'with_images': int(sum(metadata['has_image'])),
            'with_spectra': int(sum(metadata['has_spectrum'])),
            'total': len(metadata['object_ids'])
        },
        'redshift_stats': {
            'min': float(np.min(metadata['redshifts'])),
            'max': float(np.max(metadata['redshifts'])),
            'mean': float(np.mean(metadata['redshifts'])),
            'std': float(np.std(metadata['redshifts'])),
        }
    }
    
    json_path = os.path.join(output_dir, f"{prefix}_summary.json")
    print(f"Saving summary to: {json_path}")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract multimodal embeddings from trained AstroPT")
    
    # Model and data paths
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", help="Output directory (default: same as checkpoint dir)")
    
    # Data configuration
    parser.add_argument("--train-split", default="test_batch_1", help="Training split to use")
    parser.add_argument("--val-split", default="test_batch_2", help="Validation split to use")
    parser.add_argument("--use-val", action="store_true", help="Extract from validation split instead of training")
    
    # Processing configuration
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for extraction")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (0 recommended)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Limiting options (for testing)
    parser.add_argument("--max-batches", type=int, help="Maximum number of batches to process")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    print("="*70)
    print("MULTIMODAL EMBEDDING EXTRACTION")
    print("="*70)
    
    # Determine output directory
    if args.output_dir is None:
        # Place embeddings in same directory as checkpoint
        args.output_dir = os.path.dirname(args.checkpoint)
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using {'validation' if args.use_val else 'training'} split")
    
    if args.num_workers > 0:
        print("⚠️  WARNING: Using num_workers > 0 may cause duplication issues")
        print("   If you see duplicates, try running with --num-workers=0")
    
    # Load model
    model, config, (image_patch_size, spectrum_patch_size) = load_checkpoint(args.checkpoint, args.device)
    
    # Create dataset
    print(f"\nCreating multimodal dataset...")
    split = args.val_split if args.use_val else args.train_split
    
    dataset = EuclidDESIMultimodalDataset(
        split=split,
        image_size=224,
        spectrum_length=7781
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Apply max_samples limit if specified
    if args.max_samples is not None:
        n_samples = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, range(n_samples))
        print(f"Limiting to first {n_samples} samples ({n_samples/len(dataset.dataset)*100:.1f}% of full dataset)")
    
    # Create DataLoader
    print(f"\nCreating DataLoader (num_workers={args.num_workers})...")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for consistent extraction
        num_workers=args.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=(args.device == "cuda"),
        drop_last=False,  # Keep all samples
    )
    
    # Calculate max_batches from max_samples if needed
    max_batches = args.max_batches
    if args.max_samples is not None and max_batches is None:
        max_batches = (args.max_samples + args.batch_size - 1) // args.batch_size
    
    # Extract embeddings
    print(f"\nExtracting embeddings...")
    print(f"Image patch size: {image_patch_size}")
    print(f"Spectrum patch size: {spectrum_patch_size}")
    
    embeddings, metadata = extract_embeddings_with_diagnostics(
        model,
        dataloader,
        image_patch_size,
        spectrum_patch_size,
        args.device,
        max_batches,
    )
    
    # Save results if extraction was successful
    if embeddings is not None:
        split_suffix = "val" if args.use_val else "train"
        prefix = f"multimodal_{split_suffix}"
        
        save_embeddings(embeddings, metadata, args.output_dir, prefix)
        
        print(f"\n✓ Extraction complete!")
        print(f"Saved {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
        print(f"Files saved to: {args.output_dir}")
        
        return 0
    else:
        print("\n❌ Extraction FAILED due to duplicates - files NOT saved")
        print("Please investigate the dataset or try with --num-workers=0")
        return 1


if __name__ == "__main__":
    exit(main())