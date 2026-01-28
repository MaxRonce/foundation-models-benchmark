"""
Foundation Models Benchmark (FMB)

Module: fmb.models.aion.trainer
Description: AION training loop implementation
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils
from torchvision.transforms import RandomCrop

from fmb.models.aion.config import AIONTrainingConfig
from fmb.models.base.trainer import BaseTrainer


class AIONTrainer(BaseTrainer):
    """
    Trainer for AION Euclid ↔ HSC adapters.

    Trains two U-Net adapters to translate between Euclid and HSC image spaces,
    using a frozen AION codec as a bridge.

    Parameters
    ----------
    euclid_to_hsc : nn.Module
        Euclid → HSC adapter network.
    hsc_to_euclid : nn.Module
        HSC → Euclid adapter network.
    codec : nn.Module
        Frozen AION ImageCodec.
    config : AIONTrainingConfig
        Training configuration.
    train_loader : DataLoader
        Training data loader (yields EuclidImage objects).
    val_loader : Optional[DataLoader]
        Validation data loader.
    """

    def __init__(
        self,
        euclid_to_hsc: nn.Module,
        hsc_to_euclid: nn.Module,
        codec: nn.Module,
        config: AIONTrainingConfig,
        train_loader,
        val_loader=None,
    ):
        self.euclid_to_hsc = euclid_to_hsc
        self.hsc_to_euclid = hsc_to_euclid
        self.codec = codec
        self.aion_config = config

        # Combine adapters into a single model for BaseTrainer
        model = nn.ModuleDict(
            {
                "euclid_to_hsc": euclid_to_hsc,
                "hsc_to_euclid": hsc_to_euclid,
            }
        )

        super().__init__(model, config, train_loader, val_loader)

        # Setup criterion
        self.criterion = nn.MSELoss(reduction="mean")

        # Setup cropping
        self.crop = RandomCrop(size=config.crop_size)

        # Setup codec gradient mode
        self.use_codec_ckpt = not config.disable_codec_checkpointing

        if config.codec_grad == "ste":
            self.codec_bridge = self._codec_roundtrip_ste
            print("🔧 Codec gradient mode: STE (straight-through estimator)")
        else:
            self.codec_bridge = self._codec_roundtrip_full
            print("🔧 Codec gradient mode: Full backprop")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for both adapters."""
        params = list(self.euclid_to_hsc.parameters()) + list(
            self.hsc_to_euclid.parameters()
        )
        return torch.optim.Adam(
            params,
            lr=self.config.learning_rate,
        )

    def _codec_roundtrip_flux(self, hsc_flux: torch.Tensor) -> torch.Tensor:
        """
        Pass HSC flux through codec roundtrip.

        Parameters
        ----------
        hsc_flux : torch.Tensor
            HSC flux tensor (B, 5, H, W).

        Returns
        -------
        torch.Tensor
            Reconstructed HSC flux.
        """
        try:
            from aion.modalities import HSCImage
        except ImportError:
            raise ImportError(
                "AION not found. Initialize submodules: git submodule update --init"
            )

        HSC_BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]

        hsc_obj = HSCImage(flux=hsc_flux, bands=HSC_BANDS)
        toks = self.codec.encode(hsc_obj)
        hsc_rec = self.codec.decode(toks, bands=HSC_BANDS)
        return hsc_rec.flux

    def _codec_roundtrip_ste(self, hsc_flux: torch.Tensor) -> torch.Tensor:
        """Codec roundtrip with straight-through estimator (no gradient)."""
        with torch.no_grad():
            y = self._codec_roundtrip_flux(hsc_flux)
        return hsc_flux + (y - hsc_flux).detach()

    def _codec_roundtrip_full(self, hsc_flux: torch.Tensor) -> torch.Tensor:
        """Codec roundtrip with full gradient (optionally checkpointed)."""
        if self.use_codec_ckpt and hsc_flux.requires_grad:
            return checkpoint_utils.checkpoint(
                self._codec_roundtrip_flux,
                hsc_flux,
                use_reentrant=False,
            )
        return self._codec_roundtrip_flux(hsc_flux)

    def _preprocess_and_crop(self, euclid_flux_cpu: torch.Tensor) -> torch.Tensor:
        """
        Preprocess and crop Euclid flux.

        Parameters
        ----------
        euclid_flux_cpu : torch.Tensor
            Euclid flux on CPU.

        Returns
        -------
        torch.Tensor
            Preprocessed and cropped flux on device.
        """
        if self.aion_config.cpu_crop:
            # Preprocess on CPU
            x = torch.nan_to_num(euclid_flux_cpu, nan=0.0, posinf=0.0, neginf=0.0)
            if self.aion_config.max_abs > 0:
                x = torch.clamp(x, -self.aion_config.max_abs, self.aion_config.max_abs)
            x = self.crop(x)
            return x.to(self.device, non_blocking=True)

        # Preprocess on GPU
        x = euclid_flux_cpu.to(self.device, non_blocking=False)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.aion_config.max_abs > 0:
            x = torch.clamp(x, -self.aion_config.max_abs, self.aion_config.max_abs)
        return self.crop(x)

    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute one AION training step.

        Parameters
        ----------
        batch : EuclidImage
            Batch of Euclid images.

        Returns
        -------
        Dict[str, float]
            Dictionary with 'loss' key.
        """
        # Preprocess
        x = self._preprocess_and_crop(batch.flux)

        # Forward: Euclid → HSC → Codec → HSC → Euclid
        hsc_like = self.euclid_to_hsc(x)
        hsc_dec = self.codec_bridge(hsc_like)
        euclid_rec = self.hsc_to_euclid(hsc_dec)

        # Compute loss
        loss = self.criterion(euclid_rec, x)

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss encountered")

        return {"loss": loss}

    def val_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute one AION validation step.

        Parameters
        ----------
        batch : EuclidImage
            Batch of Euclid images.

        Returns
        -------
        Dict[str, float]
            Dictionary with 'loss' key.
        """
        # Same as train step but without gradient
        x = self._preprocess_and_crop(batch.flux)

        hsc_like = self.euclid_to_hsc(x)
        hsc_dec = self.codec_bridge(hsc_like)
        euclid_rec = self.hsc_to_euclid(hsc_dec)

        loss = self.criterion(euclid_rec, x)

        return {"loss": loss}
