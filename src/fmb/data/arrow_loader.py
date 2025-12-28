"""Adapter module to load local Arrow dataset for AstroCLIP fine-tuning.

This module provides a loader that works with the local HuggingFace cache format (.arrow files)
instead of parquet files, using the same approach as EuclidDESIDataset.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk, Dataset

# Mapping of split names to local directory names (in cache_dir)
LOCAL_SPLIT_PATHS = {
    "train": "msiudek__astroPT_euclid_Q1_desi_dr1_dataset__train",
    "test": "msiudek__astroPT_euclid_Q1_desi_dr1_dataset__test",
}


def load_local_arrow_dataset(
    cache_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load dataset from local Arrow cache and convert to pandas DataFrame.
    
    Args:
        cache_dir: Directory containing the cached dataset (e.g., /n03data/ronceray/datasets)
        split: Which split to load ('train', 'test')
        max_samples: Optional limit on number of samples to load
        seed: Random seed for sampling
        
    Returns:
        pd.DataFrame with columns: spectrum, redshift, image, and other metadata
    """
    cache_path = Path(cache_dir)
    
    # Check if split exists locally
    local_split_name = LOCAL_SPLIT_PATHS.get(split)
    
    # Handle "all" split by loading train and test and concatenating
    if split == "all":
        print("Dataset split 'all' requested. Loading 'train' and 'test'...")
        df_train = load_local_arrow_dataset(cache_dir, "train", max_samples=None, seed=seed)
        df_test = load_local_arrow_dataset(cache_dir, "test", max_samples=None, seed=seed)
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        
        if max_samples is not None:
             # Re-sample from the combined dataframe if needed
             if len(df_all) > max_samples:
                print(f"Sampling {max_samples} from {len(df_all)} combined samples")
                df_all = df_all.sample(n=max_samples, random_state=seed).reset_index(drop=True)
        return df_all

    if not local_split_name:
        raise ValueError(f"Unknown split '{split}'. Available: {list(LOCAL_SPLIT_PATHS.keys()) + ['all']}")
    
    local_path = cache_path / local_split_name
    
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local Arrow dataset not found at {local_path}\n"
            f"Expected directory structure: {cache_dir}/{local_split_name}/"
        )
    
    print(f"📂 Loading dataset from {local_path}")
    
    # Load using HuggingFace's load_from_disk
    dataset: Dataset = load_from_disk(str(local_path))
    
    print(f"✓ Loaded {len(dataset)} samples from '{split}' split")
    
    # Sample if requested
    if max_samples is not None and max_samples < len(dataset):
        print(f"📊 Sampling {max_samples} from {len(dataset)} samples")
        indices = list(range(len(dataset)))
        import random
        random.seed(seed)
        random.shuffle(indices)
        indices = indices[:max_samples]
        dataset = dataset.select(indices)
    
    # Convert to pandas DataFrame
    # Note: HuggingFace Dataset.to_pandas() works well for most cases
    df = dataset.to_pandas()
    
    print(f"✓ Converted to DataFrame with {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # Verify required columns
    required_cols = ["spectrum", "redshift"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for image columns (might be rgb_image or separate bands)
    image_cols = [col for col in df.columns if 'image' in col.lower()]
    print(f"  Image columns found: {image_cols}")
    
    return df


def prepare_image_from_euclid_bands(
    sample: dict,
    image_size: int = 224,
    target_key: str = "image"
) -> torch.Tensor:
    """Prepare image tensor from Euclid bands or RGB image.
    
    This function handles different image formats in the dataset:
    - If 'RGB_image' or 'rgb_image' exists, use it directly
    - Otherwise, stack individual bands (VIS_image, NISP_Y_image, etc.)
    
    Args:
        sample: Dictionary from the dataset row
        image_size: Target size for the image
        target_key: Key to look for in the sample (e.g., 'rgb_image')
        
    Returns:
        torch.Tensor of shape (C, H, W)
    """
    # Try to get RGB image first (case-insensitive)
    rgb_key = None
    for key in ["RGB_image", "rgb_image"]:
        if key in sample:
            rgb_key = key
            break
    
    if rgb_key:
        image = sample[rgb_key]
        if isinstance(image, torch.Tensor):
            img = image.float()
        else:
            img = torch.as_tensor(image, dtype=torch.float32)
    else:
        # Try to stack individual bands (uppercase first, then lowercase)
        band_keys_upper = ["VIS_image", "NISP_Y_image", "NISP_J_image", "NISP_H_image"]
        band_keys_lower = ["vis_image", "nisp_y_image", "nisp_j_image", "nisp_h_image"]
        
        bands = []
        # Try uppercase first
        for key in band_keys_upper:
            if key in sample and sample[key] is not None:
                band = torch.as_tensor(sample[key], dtype=torch.float32)
                if band.ndim == 3 and band.shape[0] == 1:
                    band = band.squeeze(0)
                bands.append(band)
        
        # If no uppercase bands found, try lowercase
        if not bands:
            for key in band_keys_lower:
                if key in sample and sample[key] is not None:
                    band = torch.as_tensor(sample[key], dtype=torch.float32)
                    if band.ndim == 3 and band.shape[0] == 1:
                        band = band.squeeze(0)
                    bands.append(band)
        
        if not bands:
            raise ValueError(
                f"No image data found in sample. "
                f"Checked: RGB_image, rgb_image, {band_keys_upper}, {band_keys_lower}"
            )
        
        # Stack to create multi-channel image
        img = torch.stack(bands, dim=0)
    
    # Normalize and clean
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure proper channel ordering (C, H, W)
    if img.ndim == 3 and img.shape[0] not in (1, 3, 4):
        img = img.permute(2, 0, 1).contiguous()
    
    # Resize if needed
    if image_size and (img.shape[-1] != image_size or img.shape[-2] != image_size):
        img = F.interpolate(
            img.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    
    # For RGB (3 channels), clamp to [0, 1]
    if img.shape[0] == 3:
        img = img.clamp(0.0, 1.0)
    
    return img


def convert_dataset_to_astroclip_format(
    df: pd.DataFrame,
    image_size: int = 144,
) -> pd.DataFrame:
    """Convert DataFrame from EuclidDESI format to AstroCLIP expected format.
    
    AstroCLIP expects:
    - 'image': torch.Tensor (C, H, W)
    - 'spectrum': dict with 'flux' and optionally 'wavelength'
    - 'redshift': float
    
    Args:
        df: Input DataFrame from load_local_arrow_dataset
        image_size: Target image size
        
    Returns:
        DataFrame with processed columns
    """
    print(f"🔄 Converting {len(df)} samples to AstroCLIP format...")
    
    try:
        from PIL import Image
    except ImportError:
        Image = None
    
    processed_rows = []
    
    for idx, row in df.iterrows():
        try:
            # Get RGB image (uppercase key)
            rgb_image = row.get('RGB_image')
            if rgb_image is None:
                continue
                        
            # Convert PIL Image to numpy array
            if Image is not None and isinstance(rgb_image, Image.Image):
                rgb_image = np.array(rgb_image)
            
            # SPECIAL CASE: If it's a dict (Arrow struct), try to convert
            if isinstance(rgb_image, dict):
                # Sometimes images are stored as dicts in Arrow (e.g. {'bytes': ..., 'path': ...})
                if 'bytes' in rgb_image:
                     import io
                     if Image is None:
                         from PIL import Image
                     rgb_image = np.array(Image.open(io.BytesIO(rgb_image['bytes'])))
                elif 'array' in rgb_image: # hypothetical
                     rgb_image = np.array(rgb_image['array'])
            
            # Convert to tensor format (C, H, W)
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.ndim == 3:
                    image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
                else:
                    image_tensor = torch.from_numpy(rgb_image).unsqueeze(0).float() / 255.0
            else:
                # Fallback: if already tensor-like
                image_tensor = torch.as_tensor(rgb_image).float()
                if image_tensor.ndim == 3 and image_tensor.shape[0] not in (1, 3):
                    # Try to convert to (C, H, W)
                    image_tensor = image_tensor.permute(2, 0, 1).contiguous()
            
            # Resize if needed
            if image_size and (image_tensor.shape[-1] != image_size or image_tensor.shape[-2] != image_size):
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            
            # Process spectrum (should already be a dict)
            spectrum = row.get('spectrum')
            if spectrum is None:
                continue  # Skip samples without spectrum
            
            # Ensure spectrum is properly formatted
            if not isinstance(spectrum, dict):
                # If it's an array, convert to dict
                if isinstance(spectrum, (list, np.ndarray)):
                    spectrum = {"flux": spectrum}
            
            # Get redshift
            redshift = row.get('redshift', 0.0)
            
            processed_rows.append({
                "image": image_tensor,
                "spectrum": spectrum,
                "redshift": redshift,
                "object_id": row.get('object_id'),
                "targetid": row.get('targetid'),
            })
            
        except Exception as e:
            print(f"Warning: Failed to process row {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    result_df = pd.DataFrame(processed_rows)
    print(f"✓ Successfully converted {len(result_df)} samples")
    
    return result_df


if __name__ == "__main__":
    # Test loading
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/n03data/ronceray/datasets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()
    
    df = load_local_arrow_dataset(
        cache_dir=args.cache_dir,
        split=args.split,
        max_samples=args.max_samples,
    )
    
    print(f"\n📊 Dataset info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  First row preview:")
    print(f"    Redshift: {df.iloc[0]['redshift']}")
    print(f"    Spectrum type: {type(df.iloc[0].get('spectrum'))}")
    
    # Test conversion
    converted = convert_dataset_to_astroclip_format(df, image_size=144)
    print(f"\n✓ Conversion test successful!")
    print(f"  Converted shape: {converted.shape}")
    if len(converted) > 0:
        print(f"  Image shape: {converted.iloc[0]['image'].shape}")
