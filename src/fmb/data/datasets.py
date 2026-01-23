
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import sys

# Attempt to import AION types for AionDataset
try:
    from aion.modalities import EuclidImage
    AION_AVAILABLE = True
except ImportError:
    AION_AVAILABLE = False
    EuclidImage = Any

from fmb.data.load_display_data import EuclidDESIDataset

# Constants for AION
EUCLID_BANDS = ["EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H"]
EUCLID_ZP_NU = {
    "vis_image": 2835.34,
    "nisp_y_image": 1916.10,
    "nisp_j_image": 1370.25,
    "nisp_h_image": 918.35,
}

@dataclass
class FMBDataConfig:
    """Shared configuration for FMB datasets."""
    split: str = "train"
    cache_dir: Optional[str] = None
    image_size: int = 96
    max_entries: Optional[int] = None
    # For AION/AstroCLIP
    crop_size: int = 96 
    # For AstroCLIP/AstroPT
    spectrum_length: int = 7781
    # For AstroCLIP
    spectrum_norm: str = "zscore" # none, zscore, minmax
    include_wavelength: bool = False
    slice_length: Optional[int] = None # Alias for spectrum_length in AstroCLIP config
    
    def __post_init__(self):
        # Unify slice_length and spectrum_length
        if self.slice_length is not None:
            self.spectrum_length = self.slice_length

class FMBBaseDataset(Dataset):
    """Base class for FMB datasets wrapping EuclidDESIDataset."""
    
    def __init__(self, config: FMBDataConfig, verbose: bool = False):
        self.config = config
        self.base = EuclidDESIDataset(
            split=config.split, 
            cache_dir=config.cache_dir, 
            verbose=verbose
        )
        
        self._indices = list(range(len(self.base)))
        if config.max_entries and config.max_entries > 0:
            self._indices = self._indices[:config.max_entries]

    def __len__(self) -> int:
        return len(self._indices)

    def _get_base_sample(self, idx: int) -> Dict[str, Any]:
        base_idx = self._indices[idx]
        return self.base[base_idx]

    def _process_image(self, image_data: Any, resize_to: Optional[int] = None) -> torch.Tensor:
        """Standard image tensor conversion and optional resizing."""
        if image_data is None:
             # Return zero tensor or handle as needed by subclass? 
             # Usually subclasses check for None. Here we assume valid input or handle logic outside.
             return torch.zeros((3, resize_to or 64, resize_to or 64), dtype=torch.float32)

        img = torch.as_tensor(image_data, dtype=torch.float32)
        img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure (C, H, W)
        if img.ndim == 3 and img.shape[0] not in (1, 3, 4): 
            # If shape is like (H, W, C) -> permute
            img = img.permute(2, 0, 1).contiguous()
        elif img.ndim == 2:
            img = img.unsqueeze(0)
            
        # Clamp usually to [0, 1] for safety in many models, but specific models might differ.
        # AstroCLIP clamps to [0, 1]. AstroPT clamps to [0, 1]. AION converts units.
        # We'll leave clamping to specific implementations or define a default here.
        # img = img.clamp(0.0, 1.0) 

        if resize_to and (img.shape[-1] != resize_to or img.shape[-2] != resize_to):
            img = F.interpolate(
                img.unsqueeze(0),
                size=(resize_to, resize_to),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            
        return img

    def _pad_or_trim_spectrum(self, flux: torch.Tensor, target_len: int) -> torch.Tensor:
        if flux.numel() < target_len:
            pad_len = target_len - flux.numel()
            flux = torch.cat([flux, torch.zeros(pad_len, dtype=flux.dtype, device=flux.device)])
        elif flux.numel() > target_len:
            flux = flux[:target_len]
        return flux


class AstroClipDataset(FMBBaseDataset):
    """Dataset for AstroCLIP training (Image + Spectrum)."""
    
    def _normalise_spectrum(self, tensor: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "none": return tensor
        if mode == "zscore":
            std = tensor.std(unbiased=False).clamp(min=1e-6)
            return (tensor - tensor.mean()) / std
        if mode == "minmax":
            scale = (tensor.max() - tensor.min()).clamp(min=1e-6)
            return (tensor - tensor.min()) / scale
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._get_base_sample(idx)
        
        # Image
        img = self._process_image(sample.get("rgb_image"), resize_to=self.config.image_size)
        img = img.clamp(0.0, 1.0) # AstroCLIP specific
        
        # Spectrum
        spec_dict = sample.get("spectrum") or {}
        flux = np.asarray(spec_dict.get("flux", []))
        
        # Handle wavelength if needed (defaults to linear dummy if empty/not requested)
        wavelength = np.asarray(spec_dict.get("wavelength", []))
        if len(wavelength) == 0 and len(flux) > 0:
             wavelength = np.linspace(0, 1, len(flux), dtype=np.float32)

        if len(flux) == 0:
             # Handle empty spectrum case gracefully? 
             flux = np.zeros(self.config.spectrum_length, dtype=np.float32)
             wavelength = np.zeros(self.config.spectrum_length, dtype=np.float32)

        flux_tensor = torch.as_tensor(flux, dtype=torch.float32)
        flux_tensor = self._pad_or_trim_spectrum(flux_tensor, self.config.spectrum_length)
        flux_tensor = self._normalise_spectrum(flux_tensor, self.config.spectrum_norm)
        
        if self.config.include_wavelength:
            wave_tensor = torch.as_tensor(wavelength, dtype=torch.float32)
            wave_tensor = self._pad_or_trim_spectrum(wave_tensor, self.config.spectrum_length)
            spectrum = torch.stack([flux_tensor, wave_tensor], dim=-1)
        else:
            spectrum = flux_tensor.unsqueeze(-1)
            
        return {
            "image": img,
            "spectrum": spectrum,
            "object_id": sample.get("object_id") or sample.get("targetid"),
            "targetid": sample.get("targetid"),
            "redshift": sample.get("redshift"),
        }


class AstroPTDataset(FMBBaseDataset):
    """Dataset for AstroPT training (Multimodal)."""
    
    def _prepare_spectrum_astropt(self, spectrum_dict: Optional[dict]) -> Optional[torch.Tensor]:
        if spectrum_dict is None or spectrum_dict.get("flux") is None:
            return None
        flux = torch.as_tensor(spectrum_dict["flux"], dtype=torch.float32)
        flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

        flux = self._pad_or_trim_spectrum(flux, self.config.spectrum_length)

        if flux.std() > 0:
            flux = flux / (flux.std() + 1e-8)
        return flux

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._get_base_sample(idx)
        
        image = self._process_image(sample.get("rgb_image"), resize_to=self.config.image_size)
        image = image.clamp(0.0, 1.0)
        
        spectrum = self._prepare_spectrum_astropt(sample.get("spectrum"))

        targetid = sample.get("targetid")
        try:
            targetid_val = int(targetid) if targetid is not None else -1
        except Exception:
            targetid_val = -1
        
        redshift = sample.get("redshift")
        redshift_val = float(redshift) if redshift is not None else 0.0

        return {
            "object_id": sample.get("object_id") or targetid_val,
            "targetid": targetid_val,
            "redshift": redshift_val,
            "image": image,
            "spectrum": spectrum,
        }


class AionDataset(FMBBaseDataset):
    """Dataset for AION training (Euclid 4-band images)."""
    
    def __getitem__(self, idx: int) -> Union[EuclidImage, Any]:
        sample = self._get_base_sample(idx)

        keys = ["vis_image", "nisp_y_image", "nisp_j_image", "nisp_h_image"]
        bands = []
        for key in keys:
            t = sample.get(key)
            if t is None:
                raise ValueError(f"Missing band '{key}' at index {idx}")
            t = torch.as_tensor(t, dtype=torch.float32)
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

            # ADU -> nanomaggies
            zp_nu = EUCLID_ZP_NU[key]
            scale_factor = zp_nu / 3631.0
            t = t * scale_factor

            if t.ndim == 3 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.ndim != 2:
                # Some samples might be 3D, take first channel or squeeze
                if t.ndim == 3: t = t[0]
                else: raise ValueError(f"Expected 2D band, got {tuple(t.shape)}")
            bands.append(t)

        flux = torch.stack(bands, dim=0)  # (4,H,W)

        # AION typically expects resizing too, but uses crop often?
        # retrain_aion.py had:
        # if self.config.resize and ... interpolation
        
        if self.config.image_size and (flux.shape[-1] != self.config.image_size or flux.shape[-2] != self.config.image_size):
            flux = F.interpolate(
                flux.unsqueeze(0),
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            
        if AION_AVAILABLE:
            return EuclidImage(flux=flux, bands=EUCLID_BANDS)
        else:
            return {"flux": flux, "bands": EUCLID_BANDS} # Fallback for testing without AION


class AionMultimodalDataset(AionDataset):
    """Dataset for AION embeddings (Euclid Image + DESI Spectrum)."""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get base sample
        sample = self._get_base_sample(idx)
        
        # 1. Process Image using AionDataset logic (but we need to call it manually or reuse logic)
        # Inheriting from AionDataset allows reuse if we factor out image processing?
        # AionDataset.__getitem__ returns just the image.
        # Let's call super().__getitem__ to get the image object
        euclid_image = super().__getitem__(idx)
        
        # 2. Process Spectrum
        spec = sample.get("spectrum")
        # Return dict with both
        return {
            "object_id": sample.get("object_id") or sample.get("targetid"),
            "redshift": sample.get("redshift"),
            "image": euclid_image,
            "spectrum": spec # Raw dict, processed by collator/script usually
        }
