import numpy as np 
from tqdm import tqdm  
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset, load_from_disk 

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

import os

"""
Script to download the Euclid+DESI dataset from Hugging Face.
It sets up the environment variables for cache directories and defines a PyTorch Dataset wrapper
to verify that the data can be loaded and processed correctly.

Usage:
    python -m scratch.download_data
"""

os.environ["HF_HOME"] = "/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home"
os.environ["HF_HUB_CACHE"] = "/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/hub"
os.environ["HF_DATASETS_CACHE"] = "/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" 



class EuclidDESIDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for the Euclid+DESI HuggingFace dataset."""
    
    def __init__(self, split="train_batch_1", transform=None,
                 cache_dir="/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets"):
        import os
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Loading dataset (split={split}) into cache_dir={cache_dir}")
        self.dataset = load_dataset(
            "msiudek/astroPT_euclid_desi_dataset",
            split=split,
            cache_dir=cache_dir
        )
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample = self.dataset[idx]
        
        # Convert PIL image to tensor
        rgb_image = sample['RGB_image']
        if isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)
        
        # Convert to tensor format (C, H, W)
        if rgb_image.ndim == 3:
            rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        else:
            rgb_image = torch.from_numpy(rgb_image).unsqueeze(0).float() / 255.0
        
        # Process spectrum data
        spectrum_data = None
        if sample['spectrum'] is not None:
            flux = sample['spectrum']['flux']
            wavelength = sample['spectrum']['wavelength']
            if flux is not None:
                spectrum_data = {
                    'flux': torch.from_numpy(np.array(flux)).float(),
                    'wavelength': torch.from_numpy(np.array(wavelength)).float() if wavelength is not None else None
                }
        
        # Process SED data
        sed_fluxes = None
        if sample['sed_data'] is not None:
            flux_keys = [k for k in sample['sed_data'].keys() if k.startswith('flux_')]
            if flux_keys:
                sed_fluxes = torch.tensor([sample['sed_data'][k] for k in flux_keys]).float()
        
        # Process individual band images
        vis_image = None
        nisp_y_image = None
        nisp_j_image = None
        nisp_h_image = None
        
        if 'VIS_image' in sample and sample['VIS_image'] is not None:
            vis_image = torch.from_numpy(np.array(sample['VIS_image'])).float()
        if 'NISP_Y_image' in sample and sample['NISP_Y_image'] is not None:
            nisp_y_image = torch.from_numpy(np.array(sample['NISP_Y_image'])).float()
        if 'NISP_J_image' in sample and sample['NISP_J_image'] is not None:
            nisp_j_image = torch.from_numpy(np.array(sample['NISP_J_image'])).float()
        if 'NISP_H_image' in sample and sample['NISP_H_image'] is not None:
            nisp_h_image = torch.from_numpy(np.array(sample['NISP_H_image'])).float()
        
        return {
            'object_id': sample['object_id'],
            'targetid': sample['targetid'],
            'redshift': sample['redshift'],
            'rgb_image': rgb_image,
            'vis_image': vis_image,
            'nisp_y_image': nisp_y_image,
            'nisp_j_image': nisp_j_image,
            'nisp_h_image': nisp_h_image,
            'spectrum': spectrum_data,
            'sed_fluxes': sed_fluxes,
        }

def test_dataloader():
    """Test the PyTorch DataLoader functionality."""
    print("Creating PyTorch dataset...")
    
    try:
        # Create dataset
        dataset = EuclidDESIDataset(split="train_batch_1")
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print(f"DataLoader created with batch size 4")
        
        # Test loading a batch
        batch = next(iter(dataloader))
        
        print(f"\nBatch info:")
        print(f"RGB images shape: {batch['rgb_image'].shape}")
        print(f"Object IDs: {batch['object_id']}")
        print(f"Redshifts: {batch['redshift']}")
        
        if batch['sed_fluxes'] is not None and len(batch['sed_fluxes']) > 0:
            print(f"SED fluxes shape: {batch['sed_fluxes'][0].shape if batch['sed_fluxes'][0] is not None else 'None'}")
        
        # Check individual band availability
        has_vis = batch['vis_image'][0] is not None
        has_nisp_y = batch['nisp_y_image'][0] is not None
        has_nisp_j = batch['nisp_j_image'][0] is not None
        has_nisp_h = batch['nisp_h_image'][0] is not None
        print(f"Individual bands available: VIS={has_vis}, Y={has_nisp_y}, J={has_nisp_j}, H={has_nisp_h}")
        
        # Create visualization with RGB, spectrum, and individual bands
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Object {batch['object_id'][0]} (z={batch['redshift'][0]:.4f})", fontsize=16)
        
        # Show RGB image
        rgb = batch['rgb_image'][0].permute(1, 2, 0).numpy()
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("RGB Composite")
        axes[0, 0].axis('off')
        
        # Show spectrum if available
        if batch['spectrum'][0] is not None and batch['spectrum'][0]['flux'] is not None:
            flux = batch['spectrum'][0]['flux'].numpy()
            wavelength = batch['spectrum'][0]['wavelength'].numpy() if batch['spectrum'][0]['wavelength'] is not None else np.arange(len(flux))
            
            axes[0, 1].plot(wavelength, flux, 'b-', linewidth=0.8)
            axes[0, 1].set_xlabel('Wavelength (Å)')
            axes[0, 1].set_ylabel('Flux')
            axes[0, 1].set_title('DESI Spectrum')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No spectrum data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Spectrum (Not Available)')
        
        # Show SED photometry
        if batch['sed_fluxes'][0] is not None:
            sed_fluxes = batch['sed_fluxes'][0].numpy()
            axes[0, 2].bar(range(len(sed_fluxes)), sed_fluxes)
            axes[0, 2].set_xlabel('Filter Index')
            axes[0, 2].set_ylabel('Flux')
            axes[0, 2].set_title(f'SED Photometry ({len(sed_fluxes)} bands)')
        else:
            axes[0, 2].text(0.5, 0.5, 'No SED data', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('SED (Not Available)')
        
        # Show individual band images
        band_data = [
            (batch['vis_image'][0], 'VIS Band'),
            (batch['nisp_y_image'][0], 'NIR-Y Band'),
            (batch['nisp_j_image'][0], 'NIR-J Band')
        ]
        
        for i, (band_image, title) in enumerate(band_data):
            if band_image is not None:
                image_data = band_image.numpy()
                im = axes[1, i].imshow(image_data, cmap='viridis')
                axes[1, i].set_title(title)
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            else:
                axes[1, i].text(0.5, 0.5, f'No {title}', ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title(f'{title} (Missing)')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()

        # Save the figure explicitly
        output_file = 'dataloader_test_batch.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved as '{output_file}'")
        
        # Close the figure to free memory
        plt.close(fig)
        
        print("DataLoader test successful!")
        
    except Exception as e:
        print(f"Error in dataloader test: {e}")

if __name__ == "__main__":
    print("=== PyTorch DataLoader Test ===")
    test_dataloader()