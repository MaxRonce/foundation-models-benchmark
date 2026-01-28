"""
Foundation Models Benchmark (FMB)

Module: fmb.models.astropt.euclid_desi_dataset.multimodal_dataloader
Description: Custom collate functions for multimodal data
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


class EuclidDESIMultimodalDataset(Dataset):
    """PyTorch Dataset for Euclid images + DESI spectra multimodal training."""

    def __init__(
        self,
        split: str = "test_batch_1",
        image_size: int = 224,
        spectrum_length: int = 7781,  # Standard DESI spectrum length
        image_transform=None,
        data_dir: str = "/pbs/home/a/astroinfo09/data/astroPT_euclid_desi_dataset",
    ):
        """
        Initialize the multimodal dataset.

        Args:
            split: Which split to load from local data
            image_size: Target size for images (will be resized)
            spectrum_length: Expected length of spectra (for padding/truncating)
            image_transform: Optional transform to apply to images
            data_dir: Local directory containing the downloaded dataset
        """
        self.dataset = load_dataset(data_dir, split=split)
        self.image_size = image_size
        self.spectrum_length = spectrum_length
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single sample with both image and spectrum data."""
        sample = self.dataset[idx]

        # Process RGB image
        rgb_image = sample["RGB_image"]
        if isinstance(rgb_image, Image.Image):
            # Resize to target size
            rgb_image = rgb_image.resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            rgb_image = np.array(rgb_image)

        # Convert to tensor format (C, H, W) and normalize
        if rgb_image.ndim == 3:
            rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        else:
            rgb_image = torch.from_numpy(rgb_image).unsqueeze(0).float() / 255.0

        # Apply transforms if provided
        if self.image_transform is not None:
            rgb_image = self.image_transform(rgb_image)

        # Process spectrum data
        spectrum_flux = None
        if sample["spectrum"] is not None and sample["spectrum"]["flux"] is not None:
            flux = np.array(sample["spectrum"]["flux"], dtype=np.float32)

            # Handle spectrum length - pad or truncate to standard length
            if len(flux) < self.spectrum_length:
                # Pad with zeros
                padded_flux = np.zeros(self.spectrum_length, dtype=np.float32)
                padded_flux[: len(flux)] = flux
                spectrum_flux = torch.from_numpy(padded_flux)
            elif len(flux) > self.spectrum_length:
                # Truncate
                spectrum_flux = torch.from_numpy(flux[: self.spectrum_length])
            else:
                spectrum_flux = torch.from_numpy(flux)

            # Normalize spectrum (basic normalization - could be improved)
            if spectrum_flux.sum() > 0:
                spectrum_flux = spectrum_flux / (spectrum_flux.std() + 1e-8)

        return {
            "object_id": sample["object_id"],
            "targetid": sample["targetid"],
            "redshift": sample["redshift"],
            "image": rgb_image,  # Shape: (C, H, W)
            "spectrum": spectrum_flux,  # Shape: (spectrum_length,) or None
        }


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for multimodal training batches.

    Handles cases where some samples might not have spectra.
    """
    # Separate samples with and without spectra
    image_samples = []
    spectrum_samples = []

    for sample in batch:
        if sample["image"] is not None:
            image_samples.append(sample)
        if sample["spectrum"] is not None:
            spectrum_samples.append(sample)

    # Collate images
    collated = {}
    if image_samples:
        collated["images"] = torch.stack([s["image"] for s in image_samples])
        collated["image_object_ids"] = [s["object_id"] for s in image_samples]
        collated["image_targetids"] = torch.tensor(
            [s["targetid"] for s in image_samples]
        )
        collated["image_redshifts"] = torch.tensor(
            [s["redshift"] for s in image_samples]
        )

    # Collate spectra
    if spectrum_samples:
        collated["spectra"] = torch.stack([s["spectrum"] for s in spectrum_samples])
        collated["spectrum_object_ids"] = [s["object_id"] for s in spectrum_samples]
        collated["spectrum_targetids"] = torch.tensor(
            [s["targetid"] for s in spectrum_samples]
        )
        collated["spectrum_redshifts"] = torch.tensor(
            [s["redshift"] for s in spectrum_samples]
        )

    # Also include all metadata for reference
    collated["all_object_ids"] = [s["object_id"] for s in batch]
    collated["all_targetids"] = torch.tensor([s["targetid"] for s in batch])
    collated["all_redshifts"] = torch.tensor([s["redshift"] for s in batch])

    return collated


def prepare_multimodal_batch(
    batch: Dict[str, Any],
    image_patch_size: int,
    spectrum_patch_size: int,
    device: torch.device,
    modality_registry,
) -> Dict[str, torch.Tensor]:
    """
    Prepare a multimodal batch for AstroPT model input.

    Args:
        batch: Collated batch from multimodal_collate_fn
        image_patch_size: Size of image patches (e.g., 16)
        spectrum_patch_size: Size of spectrum patches (e.g., 256)
        device: Target device
        modality_registry: AstroPT modality registry

    Returns:
        Dictionary with model inputs
    """
    inputs = {}

    # Process images if present
    if "images" in batch:
        images = batch["images"].to(device)  # (B, C, H, W)
        B, C, H, W = images.shape

        # Ensure image dimensions are divisible by patch size
        H_pad = (image_patch_size - (H % image_patch_size)) % image_patch_size
        W_pad = (image_patch_size - (W % image_patch_size)) % image_patch_size
        if H_pad or W_pad:
            images = F.pad(images, (0, W_pad, 0, H_pad))
            H, W = images.shape[2], images.shape[3]

        # Create image patches
        patches_h = H // image_patch_size
        patches_w = W // image_patch_size
        num_patches = patches_h * patches_w

        # Reshape to patches: (B, C, H, W) -> (B, num_patches, patch_size*patch_size*C)
        image_patches = (
            images.unfold(2, image_patch_size, image_patch_size)
            .unfold(3, image_patch_size, image_patch_size)
            .contiguous()
            .view(B, C, patches_h, patches_w, image_patch_size, image_patch_size)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .view(B, num_patches, -1)
        )

        # Create position indices for image patches
        image_positions = torch.arange(num_patches, device=device, dtype=torch.long)
        image_positions = image_positions.unsqueeze(0).expand(B, -1)

        inputs["images"] = image_patches
        inputs["images_positions"] = image_positions

    # Process spectra if present
    if "spectra" in batch:
        spectra = batch["spectra"].to(device)  # (B, L)
        B, L = spectra.shape

        # Pad spectra to be divisible by patch size
        pad = (spectrum_patch_size - (L % spectrum_patch_size)) % spectrum_patch_size
        if pad:
            spectra = F.pad(spectra, (0, pad))

        # Reshape into patches: (B, L) -> (B, num_patches, patch_size)
        spectrum_patches = spectra.view(B, -1, spectrum_patch_size)
        num_spectrum_patches = spectrum_patches.size(1)

        # Create position indices for spectrum patches
        spectrum_positions = torch.arange(
            num_spectrum_patches, device=device, dtype=torch.long
        )
        spectrum_positions = spectrum_positions.unsqueeze(0).expand(B, -1)

        inputs["spectra"] = spectrum_patches
        inputs["spectra_positions"] = spectrum_positions

    return inputs


if __name__ == "__main__":
    # Test the dataset and dataloader
    print("Testing Euclid+DESI multimodal dataset...")

    dataset = EuclidDESIMultimodalDataset(split="train_batch_1")
    print(f"Dataset loaded with {len(dataset)} samples")

    # Test a single sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(
        f"Image shape: {sample['image'].shape if sample['image'] is not None else 'None'}"
    )
    print(
        f"Spectrum shape: {sample['spectrum'].shape if sample['spectrum'] is not None else 'None'}"
    )
    print(f"Object ID: {sample['object_id']}")
    print(f"Redshift: {sample['redshift']}")

    # Test dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=multimodal_collate_fn)
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    if "images" in batch:
        print(f"Batch images shape: {batch['images'].shape}")
    if "spectra" in batch:
        print(f"Batch spectra shape: {batch['spectra'].shape}")
