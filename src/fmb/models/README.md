# Models Module

Foundation model implementations and training scripts.

## Overview

This module contains three foundation models adapted for astronomical data:
- **AION**: Multimodal foundation model
- **AstroPT**: GPT-based transformer for images and spectra
- **AstroCLIP**: Vision-language contrastive model

## Structure

```
models/
├── aion/           # AION multimodal model
├── astropt/        # AstroPT transformer
├── astroclip/      # AstroCLIP vision-language
└── base/           # Shared utilities
```

---

## AION

**Path**: `models/aion/`

Multimodal foundation model supporting images, spectra, and metadata.

### Key Files

- `model.py` - Main AION model wrapper
- `retrain.py` - CLI entry point for retraining
- `retrain_euclid_hsc_adapter_unet.py` - Fine-tuning script with U-Net adapter
- `trainer.py` - Training loop implementation
- `codec_manager.py` - Local codec loading (offline-compatible for Candide/Jean-Zay Use)
- `modalities.py` - Modality definitions (new EuclidImage, DESISpectrum)
- `config.py` - Model configuration
- `load_weights.py` - Weight loading utilities

---

## AstroPT

**Path**: `models/astropt/`

GPT-based transformer trained on patches of images and spectra.

### Key Files

- `retrain.py` - CLI wrapper
- `retrain_spectra_images.py` - Main training script (supports DDP)
- `config.py` - Model configuration, from LEGACY code
- `euclid_desi_dataset/multimodal_dataloader.py` - Custom collate functions from LEGACY code

**Architecture:**
- Modality-aware GPT with separate patch embeddings
- Image patches: 16×16 pixels
- Spectrum patches: 10 wavelength bins
- Joint embedding space

---

## AstroCLIP

**Path**: `models/astroclip/`

Vision-language contrastive model using CLIP architecture.

### Key Files

- `finetune.py` - Fine-tuning script
- `config.py` - Model and training config
- `core/astroclip.py` - Main AstroCLIP model adapter for finetuning during AstroINFO2025 hackathon
- `core/modules.py` - Vision and spectrum encoders
- `core/specformer.py` - Spectrum transformer


## Base Utilities

**Path**: `models/base/`

Shared utilities for all models.

### Files

- `config.py` - Base configuration classes
- `trainer.py` - Abstract trainer interface
- `utils.py` - Common model utilities

---

## External Imports

**File**: `external_imports.py`

Handles imports from external submodules (astroPT, AION, AstroCLIP repos in `external/`).

---

## Weight Management

### Base Weights (Pretrained)
Loaded from `data/weights/base/`:
- `aion_checkpoint/`
- `astropt_checkpoint.pt`
- `astroclip_checkpoint/`

### Retrained Weights
Saved to `runs/weights/`:
- `aion/ckpt.pt`
- `astropt/ckpt.pt`
- `astroclip/best.pt`

Paths configured in `paths_local.yaml`.

---

## Configuration

All model training uses YAML configs in `src/fmb/configs/retrain/`:
- `aion.yaml`
- `astropt.yaml`
- `astroclip.yaml`

Example:
```yaml
# astropt.yaml
batch_size: 8
learning_rate: 6e-4
max_iters: 3000
gradient_accumulation_steps: 4
```

---

## CLI Integration

All models accessible via:
```bash
fmb retrain <model> [OPTIONS]
```

Where `<model>` ∈ {aion, astropt, astroclip}
