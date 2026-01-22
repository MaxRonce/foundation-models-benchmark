# generate_embeddings.py
"""
Script to generate embeddings for the Euclid+DESI dataset using a pre-trained AION model.
It processes the dataset, projects Euclid images to HSC-like images, and runs the model
to produce embeddings for images, spectra, and both combined.

Usage:
    python -m scratch.generate_embeddings --output /path/to/embeddings.pt --batch-size 20 --split all --keep-tokens
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional
from collections import Counter

import torch
import torch.nn.functional as F
from tqdm import tqdm

from scratch.load_display_data import EuclidDESIDataset
from utils.load_weights import load_model_and_codec

from aion.modalities import DESISpectrum, HSCImage

# New adapter and constants
from scratch.retrain_euclid_hsc_adapter import EUCLID_ZP_NU, HSC_BANDS


@torch.inference_mode()
def generate_embeddings(
    split: str = "train",
    cache_dir: str = "/n03data/ronceray/datasets",
    model_dir: Path = Path("/n03data/ronceray/huggingface/aion"),
    codec_dir: Path | None = None,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
    output_path: str | Path | None = None,
    verbose: bool = False,
    batch_size: int = 1,
    keep_tokens: bool = False,
    # Adapter args
    adapter_checkpoint: str | Path | None = None,
    adapter_hidden: int = 64,
    max_abs: float = 100.0,
    resize: int = 96,
) -> list[dict]:
    """Encode multiple Euclid+DESI samples using a trained Euclid->HSC adapter."""

    if output_path is None:
        raise ValueError("output_path must be provided to save embeddings.")
    
    if adapter_checkpoint is None:
        raise ValueError("adapter_checkpoint is required for this version of the script.")

    work_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Load AION model and codec
    model, codec_manager = load_model_and_codec(model_dir=model_dir, device=work_device, codec_dir=codec_dir)

    # Load Adapter Checkpoint
    print(f"Loading EuclidToHSC adapter from {adapter_checkpoint}...")
    adapter_ckpt = torch.load(adapter_checkpoint, map_location=work_device)
    state_dict = adapter_ckpt["euclid_to_hsc"]
    
    # Auto-detect architecture based on keys
    # First, try to read hidden dim from saved args if available
    if "args" in adapter_ckpt and "hidden" in adapter_ckpt["args"]:
        start_hidden = adapter_ckpt["args"]["hidden"]
        if start_hidden != adapter_hidden:
            print(f"Auto-detected hidden dimension {start_hidden} from checkpoint (overriding arg {adapter_hidden}).")
            adapter_hidden = start_hidden

    # U-Net has "inc.double_conv..." while CNN has "net.0..."
    first_key = next(iter(state_dict.keys()))
    if "inc." in first_key or "down1." in first_key:
        print(f"Detected U-Net architecture (hidden={adapter_hidden}).")
        from scratch.retrain_euclid_hsc_adapter_unet import EuclidToHSC as EuclidToHSCUnet
        euclid_to_hsc = EuclidToHSCUnet(hidden=adapter_hidden, use_checkpointing=False).to(work_device)
    else:
        print(f"Detected CNN architecture (hidden={adapter_hidden}).")
        from scratch.retrain_euclid_hsc_adapter import EuclidToHSC
        euclid_to_hsc = EuclidToHSC(in_ch=4, out_ch=5, hidden=adapter_hidden).to(work_device)
        
    euclid_to_hsc.load_state_dict(state_dict)
    euclid_to_hsc.eval()

    # PATCH: The frozen codec checkpoint has 9 channels (HSC+DES), but current BAND_TO_INDEX has 13 (including Euclid).
    # We must restrict BAND_TO_INDEX to match the checkpoint, otherwise ImageCodec init shapes won't match weights.
    from aion.codecs.preprocessing.band_to_index import BAND_TO_INDEX
    keys_to_remove = [k for k in BAND_TO_INDEX if "EUCLID" in k]
    if keys_to_remove:
        print(f"Temporarily removing {len(keys_to_remove)} EUCLID bands from BAND_TO_INDEX to match codec checkpoint...")
        for k in keys_to_remove:
            del BAND_TO_INDEX[k]
    
    # Check manual cache logic (optional, but good for verification)
    # The codec_manager will load ImageCodec on first usage.

    dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
    total_dataset = len(dataset)
    limit = total_dataset if max_samples is None else min(total_dataset, max_samples)
    indices: Iterable[int] = range(limit)

    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    results: list[dict] = []
    skipped = 0
    skip_reasons = Counter()

    # Accumulators for the current batch
    batch_euclid_tensors: list[torch.Tensor] = [] # List of (4, H, W)
    batch_desi_flux: list[torch.Tensor] = []
    batch_desi_ivar: list[torch.Tensor] = []
    batch_desi_mask: list[torch.Tensor] = []
    batch_desi_wave: list[torch.Tensor] = []
    batch_metadata: list[tuple[str, float]] = []

    def _process_batch() -> None:
        nonlocal batch_euclid_tensors, batch_desi_flux, batch_desi_ivar, batch_desi_mask, batch_desi_wave, batch_metadata
        if not batch_metadata:
            return

        current_batch_size = len(batch_metadata)
        
        try:
            # 1. Prepare Batched Inputs
            # Euclid stack: (B, 4, H, W)
            euclid_batch = torch.stack(batch_euclid_tensors).to(work_device)
            
            # DESI stacks: (B, L)
            desi_flux_batch = torch.stack(batch_desi_flux).to(work_device)
            desi_ivar_batch = torch.stack(batch_desi_ivar).to(work_device)
            desi_mask_batch = torch.stack(batch_desi_mask).to(work_device)
            desi_wave_batch = torch.stack(batch_desi_wave).to(work_device)

            # 2. Run Adapter (Euclid -> HSC) on Batch
            # Output: (B, 5, H, W)
            with torch.inference_mode():
                hsc_flux_batch = euclid_to_hsc(euclid_batch)
            
            # 3. Create Batched Modalities
            hsc_img_batch = HSCImage(flux=hsc_flux_batch, bands=HSC_BANDS)
            
            desi_spec_batch = DESISpectrum(
                flux=desi_flux_batch,
                wavelength=desi_wave_batch,
                ivar=desi_ivar_batch,
                mask=desi_mask_batch,
            )

            # 4. Batch Codec Encoding (VQ-VAE)
            # This is the heavy lifting we want to batch
            tokens_spec_image = codec_manager.encode(hsc_img_batch, desi_spec_batch)
            tokens_image = codec_manager.encode(hsc_img_batch)
            tokens_spec_only = codec_manager.encode(desi_spec_batch)

            # 5. Batch Transformer Embedding
            # model.encode expects a dict of tokens, and processes them in batch.
            # Output: (B, D)
            embedded_spec = model.encode(tokens_spec_image).mean(dim=1).cpu()
            embedded_img = model.encode(tokens_image).mean(dim=1).cpu()
            embedded_spec_only = model.encode(tokens_spec_only).mean(dim=1).cpu()

            # 6. Unpack and Store Results
            for i in range(current_batch_size):
                object_id, redshift = batch_metadata[i]
                
                record = {
                    "object_id": object_id,
                    "redshift": redshift,
                    "embedding_hsc_desi": embedded_spec[i],
                    "embedding_hsc": embedded_img[i],
                    "embedding_spectrum": embedded_spec_only[i],
                }

                if keep_tokens:
                    # Helper to extract the i-th batch item for each token key
                    def extract_tokens(token_dict, idx):
                        return {k: v[idx].detach().cpu() for k, v in token_dict.items()}

                    record["tokens_hsc_desi"] = extract_tokens(tokens_spec_image, i)
                    record["tokens_hsc"] = extract_tokens(tokens_image, i)
                    record["tokens_spectrum"] = extract_tokens(tokens_spec_only, i)
                
                results.append(record)

        except Exception as e:
            if verbose:
                print(f"Error processing batch starting with object {batch_metadata[0][0]}: {e}")
            
            nonlocal skipped
            count = len(batch_metadata)
            skipped += count
            skip_reasons["batch_error"] += count
            if len(skip_reasons["batch_error_examples"]) < 5:
                 if "batch_error_examples" not in skip_reasons: skip_reasons["batch_error_examples"] = []
                 skip_reasons["batch_error_examples"].append(str(e))
        
        # Clear accumulators
        batch_euclid_tensors.clear()
        batch_desi_flux.clear()
        batch_desi_ivar.clear()
        batch_desi_mask.clear()
        batch_desi_wave.clear()
        batch_metadata.clear()

    progress = tqdm(indices, total=limit, desc="Encoding", unit="obj", leave=False)
    
    # Keys for input Euclid bands
    euclid_keys = ["vis_image", "nisp_y_image", "nisp_j_image", "nisp_h_image"]

    for idx in progress:
        sample = dataset[idx]
        object_id = sample["object_id"]
        redshift = sample["redshift"]

        # --- Euclid Preprocessing (Per Sample) ---
        try:
            bands = []
            for key in euclid_keys:
                tensor = sample.get(key)
                if tensor is None:
                    # Specific error tracking
                    skip_reasons[f"missing_band_{key}"] += 1
                    raise ValueError(f"Missing band '{key}'")
                
                # To float, handle NaNs
                tensor = tensor.float() # Keep on CPU for accumulation if memory is tight, or move to GPU later
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

                # ADU -> nanomaggies
                zp_nu = EUCLID_ZP_NU[key]
                scale_factor = zp_nu / 3631.0
                tensor = tensor * scale_factor

                if tensor.ndim == 3 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                bands.append(tensor)
            
            # Stack: (4, H, W)
            euclid_flux = torch.stack(bands, dim=0)

            # Resize if needed
            if resize and (euclid_flux.shape[-1] != resize or euclid_flux.shape[-2] != resize):
                euclid_flux = F.interpolate(
                    euclid_flux.unsqueeze(0),
                    size=(resize, resize),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            
            # Clamp
            if max_abs > 0:
                euclid_flux = torch.clamp(euclid_flux, min=-max_abs, max=max_abs)
            
            # Check DESI spectrum
            spec = sample.get("spectrum")
            if spec is None or spec.get("flux") is None:
                if verbose:
                    progress.write(f"Skipping index {idx}: missing DESI spectrum")
                skipped += 1
                skip_reasons["missing_spectrum"] += 1
                continue

            # Add to batch accumulators
            batch_euclid_tensors.append(euclid_flux)
            batch_desi_flux.append(spec["flux"].float())
            batch_desi_ivar.append(spec["ivar"].float())
            batch_desi_mask.append(spec["mask"].bool())
            batch_desi_wave.append(spec["wavelength"].float())
            batch_metadata.append((object_id, redshift))

            # Trigger batch processing if full
            if len(batch_metadata) >= batch_size:
                _process_batch()

        except Exception as e:
            if verbose:
                progress.write(f"Skipping index {idx} (prep error): {e}")
            skipped += 1
            if "Missing band" not in str(e):
                 skip_reasons["prep_error"] += 1
            continue

    # Flush remaining items
    _process_batch()
    progress.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)
    summary = f"Saved {len(results)} embeddings to {output_path}"
    if skipped:
        summary += f" (skipped {skipped})"
        summary += "\nSkip reasons:"
        for reason, count in skip_reasons.items():
            if reason == "batch_error_examples": continue
            summary += f"\n  - {reason}: {count}"
        if "batch_error_examples" in skip_reasons:
             summary += f"\n  (First few batch errors: {skip_reasons['batch_error_examples']})"

    if keep_tokens:
        summary += " with tokens"
    print(summary)

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate AION embeddings for multiple Euclid+DESI objects using a trained Adapter."
    )
    # Standard args
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--cache-dir", type=str, default="/n03data/ronceray/datasets", help="Dataset cache contents")
    parser.add_argument("--model-dir", type=str, default="/n03data/ronceray/huggingface/aion")
    parser.add_argument("--codec-dir", type=str, default=None, help="Optional codec dir")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output", type=str, required=True, help="Path to save embeddings (.pt)")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples per batch")
    parser.add_argument("--keep-tokens", action="store_true", default=False, help="Save tokens")
    
    # Adapter args
    parser.add_argument("--adapter-checkpoint", type=str, required=True, help="Path to adapter .pt file")
    parser.add_argument("--adapter-hidden", type=int, default=64, help="Adapter hidden dim")
    parser.add_argument("--max-abs", type=float, default=100.0, help="Clamp absolute value (ADU/flux)")
    parser.add_argument("--resize", type=int, default=96, help="Resize Euclid images to this size before adapter")

    args = parser.parse_args(argv)
    
    generate_embeddings(
        split=args.split,
        cache_dir=args.cache_dir,
        model_dir=Path(args.model_dir),
        codec_dir=Path(args.codec_dir) if args.codec_dir else None,
        device=args.device,
        max_samples=args.max_samples,
        output_path=args.output,
        verbose=args.verbose,
        batch_size=args.batch_size,
        keep_tokens=args.keep_tokens,
        adapter_checkpoint=args.adapter_checkpoint,
        adapter_hidden=args.adapter_hidden,
        max_abs=args.max_abs,
        resize=args.resize,
    )


if __name__ == "__main__":
    main()
