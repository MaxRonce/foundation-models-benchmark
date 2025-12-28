# encode_one_object.py
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scratch.load_display_data import EuclidDESIDataset
from utils.load_weights import load_model_and_codec  

# AION modality wrappers
from aion.modalities import HSCImage, DESISpectrum

@torch.inference_mode()
def project_euclid_to_hsc(
    vis_image,
    y_image,
    j_image,
    h_image,
    target_size: int = 120,
    device: torch.device | str | None = None,
    verbose: bool = False,
    return_debug: bool = False,
) -> HSCImage | tuple[HSCImage, dict]:
    """Map Euclid VIS/Y/J/H bands into HSC-like g/r/i/z bands."""

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    euclid_images = {
        "VIS": vis_image,
        "Y": y_image,
        "J": j_image,
        "H": h_image,
    }

    # Scaling coefficients to approximate HSC stats.
    band_mappings = {
        "g": ("VIS", 0.1312, 0.01147),
        "r": ("Y", 0.008414, 0.01346),
        "i": ("J", 0.03052, -0.0003554),
        "z": ("H", 0.02096, 0.005976),
    }

    fluxes = []
    debug_original = [] if return_debug else None
    debug_resized = [] if return_debug else None
    debug_normalized = [] if return_debug else None
    debug_labels = [] if return_debug else None

    for hsc_band, (euclid_key, a, b) in band_mappings.items():
        data = euclid_images.get(euclid_key)
        if data is None:
            raise ValueError(f"Missing Euclid {euclid_key} image for HSC-{hsc_band} projection")

        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = data

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif not (tensor.ndim == 3 and tensor.shape[0] == 1):
            raise ValueError(
                f"Euclid {euclid_key} image must be (H,W) or (1,H,W); got {tuple(tensor.shape)}"
            )

        tensor = tensor.to(torch.float32)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)

        if return_debug:
            debug_original.append(tensor.squeeze(0).detach().cpu())

        resized = F.interpolate(
            tensor.unsqueeze(0),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        resized = resized.squeeze(0)

        if return_debug:
            debug_resized.append(resized.detach().cpu())

        normalized = resized * a + b

        if return_debug:
            debug_normalized.append(normalized.detach().cpu())
            debug_labels.append(f"{euclid_key}→HSC-{hsc_band.upper()}")

        fluxes.append(normalized)

    flux = torch.stack(fluxes, dim=0).unsqueeze(0).to(device)
    hsc_bands = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z"]
    img_mod = HSCImage(flux=flux, bands=hsc_bands)
    if verbose:
        print(f"HSCImage flux shape={tuple(img_mod.flux.shape)} bands={img_mod.bands}")
        print(img_mod)

    if not return_debug:
        return img_mod

    debug_data = {
        "original": debug_original,
        "resized": debug_resized,
        "normalized": debug_normalized,
        "labels": debug_labels,
    }
    return img_mod, debug_data


def visualize_projection_debug(
    original_bands: list[torch.Tensor],
    resized_bands: list[torch.Tensor],
    normalized_bands: list[torch.Tensor],
    labels: list[str],
    save_path: str | None = None,
) -> None:
    """Display (and optionally save) the Euclid→HSC projection diagnostics."""

    if not (len(original_bands) == len(resized_bands) == len(normalized_bands) == len(labels)):
        raise ValueError("Debug arrays must all have the same length")

    num_cols = len(labels)
    fig, axes = plt.subplots(3, num_cols, figsize=(4 * num_cols, 9))
    if num_cols == 1:
        axes = axes.reshape(3, 1)

    row_specs = [
        ("Original", original_bands),
        ("Resized 120x120", resized_bands),
        ("Normalized", normalized_bands),
    ]

    for col_idx, label in enumerate(labels):
        axes[0, col_idx].set_title(label)

    for row_idx, (row_title, tensors) in enumerate(row_specs):
        for col_idx, tensor in enumerate(tensors):
            ax = axes[row_idx, col_idx]
            array = tensor.detach().cpu().numpy()
            ax.imshow(array, cmap="gray")
            if col_idx == 0:
                ax.set_ylabel(row_title)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Projection verification figure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

def find_mask(tokens: dict, candidates: list[str]) -> torch.Tensor | None:
    for k in candidates:
        if k in tokens and tokens[k] is not None:
            return tokens[k]
    return None

@torch.inference_mode()
def encode_one_object(
    index: int = 0,
    split: str = "train_batch_1",
    cache_dir: str = "/n03data/ronceray/datasets",
    model_dir: Path = Path("/n03data/ronceray/huggingface/aion"),
    device: str | torch.device = None,
    save_path: str | None = None,
    see_image: str | None = None,
    verbose: bool = True,
):
    """
    Load one Euclid+DESI sample, project Euclid VIS/Y/J/H -> HSC-like g/r/i/z,
    add DESI spectra, and return (and optionally save) the encoder embeddings.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, codec_manager = load_model_and_codec(model_dir=model_dir, device=device)

    dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=verbose)
    sample = dataset[index]
    object_id = sample["object_id"]
    redshift = sample["redshift"]
    if verbose:
        print(f"Loaded sample {index} with object_id={object_id} z={redshift}")

    # --- Project Euclid bands to HSC-like image ---
    if see_image is not None:
        hsc_img, debug_data = project_euclid_to_hsc(
            vis_image=sample["vis_image"],
            y_image=sample["nisp_y_image"],
            j_image=sample["nisp_j_image"],
            h_image=sample["nisp_h_image"],
            device=device,
            verbose=verbose,
            return_debug=True,
        )
        visualize_projection_debug(
            debug_data["original"],
            debug_data["resized"],
            debug_data["normalized"],
            debug_data["labels"],
            save_path=see_image,
        )
    else:
        hsc_img = project_euclid_to_hsc(
            vis_image=sample["vis_image"],
            y_image=sample["nisp_y_image"],
            j_image=sample["nisp_j_image"],
            h_image=sample["nisp_h_image"],
            device=device,
            verbose=verbose,
        )

    if verbose:
        print(f"Projected Euclid bands to HSC-like modality on device {device}")

    # --- Build DESI spectrum modality ---
    spec = sample.get("spectrum")
    if verbose:
        print(f"Loaded DESI spectrum: {spec is not None and spec.get('flux') is not None}")
        print(spec.keys() if spec is not None else "No spectrum keys")
    if spec is None or spec.get("flux") is None:
        raise ValueError("Sample has no DESI spectrum.")
    desi_spec = DESISpectrum(
        flux=spec["flux"].unsqueeze(0).float().to(device),
        wavelength=spec["wavelength"].unsqueeze(0).float().to(device),
        ivar=spec["ivar"].unsqueeze(0).float().to(device),
        mask=spec["mask"].unsqueeze(0).bool().to(device),

    )

    print(DESISpectrum)
    print(hsc_img)

    if verbose:
        print(f"Created DESISpectrum modality on device {device}")

    # --- Tokenize and encode ---
    tokens_spec_image = codec_manager.encode(hsc_img, desi_spec)
    tokens_image = codec_manager.encode(hsc_img)

    if verbose:
        print(f"Tokenized modalities on device {device}")
        print(f"Tokenized modalities shapes: {tokens_image['tok_image_hsc'].shape}")

    # Generate embeddings
    embeddings_spec_image = model.encode(tokens_spec_image)
    embeddings_image = model.encode(tokens_image)

    # print embeddings shape
    if verbose:
        print(f"Generated embeddings on device {device}")
        print(f"Embeddings shape (spec_image): {embeddings_spec_image.shape}")
        print(f"Embeddings shape (image): {embeddings_image.shape}")

    # Average pooling 
    embeddings_spec_image = embeddings_spec_image.mean(dim=1)
    embeddings_image = embeddings_image.mean(dim=1)
    if verbose:
        print(f"Pooled embeddings shape (spec_image): {embeddings_spec_image.shape}")
        print(f"Pooled embeddings shape (image): {embeddings_image.shape}")

    representation = {
        "object_id": object_id,
        "redshift": redshift,
        "embedding_hsc_desi": embeddings_spec_image.squeeze(0).cpu(),
        "embedding_hsc": embeddings_image.squeeze(0).cpu(),
    }

    # Optionally save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(representation, save_path)
        if verbose:
            print(f"Representation saved to {save_path}")

    return representation


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Encode one Euclid+DESI object with AION (Euclid→HSC + DESI spectra)."
    )
    parser.add_argument("--index", type=int, default=0, help="Dataset index")
    parser.add_argument(
        "--split",
        type=str,
        default="train_batch_1",
        help="Dataset split (comma-separated list or 'all' for every available split)",
    )
    parser.add_argument("--cache-dir", type=str, default="/n03data/ronceray/datasets")
    parser.add_argument("--model-dir", type=str, default="/n03data/ronceray/huggingface/aion")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save embeddings (.pt)")
    parser.add_argument(
        "--seeimage",
        nargs="?",
        const="projection_verification.png",
        default=None,
        help="Optional path to save Euclid→HSC projection figure (defaults to projection_verification.png).",
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")

    args = parser.parse_args(argv)
    encode_one_object(
        index=args.index,
        split=args.split,
        cache_dir=args.cache_dir,
        model_dir=Path(args.model_dir),
        device=args.device,
        save_path=args.save,
        see_image=args.seeimage,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
