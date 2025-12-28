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

import torch
from tqdm import tqdm

from fmb.data.load_display_data_hsc import EuclidDESIDataset
# Assuming utils/load_weights.py is still accessible or will be moved. 
# For now, let's assume it's moved to fmb.utils or we keep it as it is if it's in src.
# Looking at the file tree, I don't see a 'utils' dir in src/fmb yet.
# User said: "external/ contains des submodules ; certains scripts originaux vivent encore dans external/* pour le moment."
# and "Je veux éviter de casser des scripts : prioriser une migration “safe” (wrappers + subprocess si nécessaire), puis modulariser ensuite."

from fmb.paths import load_paths
from fmb.viz.show_object_detail import project_euclid_to_hsc

# AION modalities might need to be imported from external for now if not refactored
try:
    from aion.modalities import DESISpectrum
except ImportError:
    # Fallback or sys.path adjustment might be needed in Phase 1
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[3] / "external" / "AION"))
    from aion.modalities import DESISpectrum


@torch.inference_mode()
def generate_embeddings(
    split: str = "train",
    paths: Optional[FMBPaths] = None,
    codec_dir: Path | None = None,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
    output_path: str | Path | None = None,
    verbose: bool = False,
    batch_size: int = 1,
    keep_tokens: bool = False,
) -> list[dict]:
    """Encode multiple Euclid+DESI samples and optionally persist their embeddings (and tokens)."""

    if paths is None:
        paths = load_paths()

    if output_path is None:
        output_path = paths.embeddings_dir("aion") / f"embeddings_{split}.pt"

    work_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Needs modularization of load_model_and_codec
    from fmb.models.aion.retrain_euclid_codec import load_model_and_codec # Dummy anchor for now or fix path
    
    model_dir = paths.checkpoints / "aion"
    model, codec_manager = load_model_and_codec(model_dir=model_dir, device=work_device, codec_dir=codec_dir)

    dataset = EuclidDESIDataset(split=split, cache_dir=str(paths.data), verbose=verbose)
    total_dataset = len(dataset)
    limit = total_dataset if max_samples is None else min(total_dataset, max_samples)
    indices: Iterable[int] = range(limit)

    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")

    results: list[dict] = []
    skipped = 0

    batch_tokens_spec_image: list[dict[str, torch.Tensor]] = []
    batch_tokens_image: list[dict[str, torch.Tensor]] = []
    batch_tokens_spec_only: list[dict[str, torch.Tensor]] = []
    batch_metadata: list[tuple[str, float]] = []

    def _concat_token_dicts(token_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = list(token_dicts[0].keys())
        return {k: torch.cat([d[k] for d in token_dicts], dim=0) for k in keys}

    def _flush_batches() -> None:
        nonlocal batch_tokens_spec_image, batch_tokens_image, batch_tokens_spec_only, batch_metadata
        if not batch_metadata:
            return

        tokens_spec = _concat_token_dicts(batch_tokens_spec_image)
        tokens_img = _concat_token_dicts(batch_tokens_image)
        tokens_spec_only = _concat_token_dicts(batch_tokens_spec_only)

        embedded_spec = model.encode(tokens_spec).mean(dim=1).cpu()
        embedded_img = model.encode(tokens_img).mean(dim=1).cpu()
        embedded_spec_only = model.encode(tokens_spec_only).mean(dim=1).cpu()

        for idx_entry, ((object_id, redshift), emb_spec, emb_img, emb_spec_only) in enumerate(
            zip(batch_metadata, embedded_spec, embedded_img, embedded_spec_only)
        ):
            results.append(
                {
                    "object_id": object_id,
                    "redshift": redshift,
                    "embedding_hsc_desi": emb_spec,
                    "embedding_hsc": emb_img,
                    "embedding_spectrum": emb_spec_only,
                }
            )

            if keep_tokens:
                record = results[-1]
                record["tokens_hsc_desi"] = {
                    key: batch_tokens_spec_image[idx_entry][key].squeeze(0).detach().cpu()
                    for key in batch_tokens_spec_image[idx_entry]
                }
                record["tokens_hsc"] = {
                    key: batch_tokens_image[idx_entry][key].squeeze(0).detach().cpu()
                    for key in batch_tokens_image[idx_entry]
                }
                record["tokens_spectrum"] = {
                    key: batch_tokens_spec_only[idx_entry][key].squeeze(0).detach().cpu()
                    for key in batch_tokens_spec_only[idx_entry]
                }

        batch_tokens_spec_image.clear()
        batch_tokens_image.clear()
        batch_tokens_spec_only.clear()
        batch_metadata.clear()

    progress = tqdm(indices, total=limit, desc="Encoding", unit="obj", leave=False)
    for idx in progress:
        sample = dataset[idx]
        object_id = sample["object_id"]
        redshift = sample["redshift"]

        # Build HSC-like image from Euclid bands.
        hsc_img = project_euclid_to_hsc(
            vis_image=sample["vis_image"],
            y_image=sample["nisp_y_image"],
            j_image=sample["nisp_j_image"],
            h_image=sample["nisp_h_image"],
            device=work_device,
            verbose=verbose,
        )

        spec = sample.get("spectrum")
        if spec is None or spec.get("flux") is None:
            skipped += 1
            if verbose:
                progress.write(f"Skipping index {idx}: missing DESI spectrum")
            continue

        desi_spec = DESISpectrum(
            flux=spec["flux"].unsqueeze(0).float().to(work_device),
            wavelength=spec["wavelength"].unsqueeze(0).float().to(work_device),
            ivar=spec["ivar"].unsqueeze(0).float().to(work_device),
            mask=spec["mask"].unsqueeze(0).bool().to(work_device),
        )

        tokens_spec_image = codec_manager.encode(hsc_img, desi_spec)
        tokens_image = codec_manager.encode(hsc_img)
        tokens_spec_only = codec_manager.encode(desi_spec)

        batch_tokens_spec_image.append(tokens_spec_image)
        batch_tokens_image.append(tokens_image)
        batch_tokens_spec_only.append(tokens_spec_only)

        batch_metadata.append((object_id, redshift))

        if len(batch_metadata) >= batch_size:
            _flush_batches()

    _flush_batches()
    progress.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)
    summary = f"Saved {len(results)} embeddings to {output_path}"
    if skipped:
        summary += f" (skipped {skipped})"
    if keep_tokens:
        summary += " with tokens"
    print(summary)

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate AION embeddings for multiple Euclid+DESI objects."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (comma-separated list or 'all' for every available split)",
    )
    # Removing hardcoded defaults in favor of paths_local
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    parser.add_argument(
        "--codec-dir",
        type=str,
        default=None,
        help="Optional directory containing codec weights (e.g., retrained image codec). Defaults to model-dir.",
    )
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output", type=str, default=None, help="Path to save embeddings (.pt)")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples per model batch")
    parser.add_argument(
        "--keep-tokens",
        action="store_true",
        default=False,
        help="Also store HSC-only and HSC+DESI token tensors for each object.",
    )

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
    )


if __name__ == "__main__":
    main()
