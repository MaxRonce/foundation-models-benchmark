"""
Foundation Models Benchmark (FMB)

Module: fmb.detection.cosine
Description: Cosine mismatch detection between modalities
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from fmb.detection import utils
from fmb.paths import load_paths

# Reusing the known key pairs logic but adapted
KNOWN_KEY_PAIRS = {
    "astropt": ("embedding_images", "embedding_spectra"),
    "aion": (
        "embedding_hsc",
        "embedding_spectrum",
    ),  # heuristic, might be embedding_hsc_desi
    "astroclip": ("embedding_images", "embedding_spectra"),
}


def detect_keys(record: Dict, model_name: str) -> Tuple[str, str]:
    """Detect image/spectrum keys for a record."""
    keys = set(record.keys())

    # Check explicit known pairs first
    if model_name in KNOWN_KEY_PAIRS:
        img, spec = KNOWN_KEY_PAIRS[model_name]
        # Check if they exist (AION keys vary)
        if img in keys and spec in keys:
            return img, spec
        # Fallback for AION variants
        if model_name == "aion":
            if "embedding_hsc_desi" in keys and "embedding_spectrum" in keys:
                return "embedding_hsc_desi", "embedding_spectrum"

    # Fallback heuristics
    img_candidates = [k for k in keys if "image" in k.lower() or "hsc" in k.lower()]
    spec_candidates = [k for k in keys if "spectr" in k.lower()]

    if len(img_candidates) >= 1 and len(spec_candidates) >= 1:
        # Pick shortest or first?
        return img_candidates[0], spec_candidates[0]

    raise ValueError(f"Could not key pair in {list(keys)}")


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1 / norm1, vec2 / norm2))


def process_file(input_path: Path, output_path: Path, model_name: str) -> None:
    print(f"\n---> Processing Cosine for {model_name} in {input_path}")

    try:
        records = utils.load_records(input_path)
    except Exception as e:
        print(f"[error] Failed to load {input_path}: {e}")
        return

    if not records:
        print("[warn] Empty records.")
        return

    try:
        img_key, spec_key = detect_keys(records[0], model_name)
        print(f"      Keys: Image='{img_key}', Spec='{spec_key}'")
    except ValueError as e:
        print(f"[error] {e}")
        return

    # Extract
    # We need aligned lists. extract_embeddings returns arrays and IDs.
    # But we need to ensure we match image and spec for the SAME object.
    # The utils.extract_embeddings extracts one key.

    # Let's iterate manually to ensure pairing
    rows = []
    skipped = 0

    for i, rec in enumerate(records):
        obj_id = rec.get("object_id", str(i))

        # Helper to get numpy
        def get_np(k):
            v = rec.get(k)
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().flatten()
            return np.asarray(v).flatten()

        v_img = get_np(img_key)
        v_spec = get_np(spec_key)

        if v_img is None or v_spec is None:
            skipped += 1
            continue

        if np.any(np.isnan(v_img)) or np.any(np.isnan(v_spec)):
            skipped += 1
            continue

        sim = compute_cosine_similarity(v_img, v_spec)
        rows.append({"object_id": str(obj_id), "cosine_similarity": sim})

    if skipped:
        print(f"      Skipped {skipped} records.")

    if not rows:
        print("[warn] No valid pairs found.")
        return

    # Rank (1 = lowest similarity = most anomalous)
    # Sort ascending score
    rows.sort(key=lambda x: x["cosine_similarity"])
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    # Save
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["object_id", "cosine_similarity", "rank"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[success] Saved {len(rows)} cosine scores to {output_path}")


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser(
        description="Compute Cosine Similarity (Image vs Spectrum)"
    )
    # Optional overrides
    parser.add_argument("--aion-embeddings", type=str)
    parser.add_argument("--astropt-embeddings", type=str)
    parser.add_argument("--astroclip-embeddings", type=str)

    args = parser.parse_args(argv)
    paths = load_paths()
    out_dir = paths.outliers
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve Inputs
    # Similar logic to run.py
    inputs = []

    # AION
    p = (
        Path(args.aion_embeddings)
        if args.aion_embeddings
        else (paths.embeddings / "aions_embeddings.pt")
    )
    if p.exists():
        inputs.append((p, "aion"))

    # AstroPT
    p = (
        Path(args.astropt_embeddings)
        if args.astropt_embeddings
        else (paths.embeddings / "astropt_embeddings.pt")
    )
    if p.exists():
        inputs.append((p, "astropt"))

    # AstroCLIP
    p = (
        Path(args.astroclip_embeddings)
        if args.astroclip_embeddings
        else (paths.embeddings / "embeddings_astroclip.pt")
    )
    if p.exists():
        inputs.append((p, "astroclip"))

    if not inputs:
        print("[error] No embedding files found.")
        return

    for p, name in inputs:
        out_file = out_dir / f"cosine_scores_{name}.csv"
        process_file(p, out_file, name)


if __name__ == "__main__":
    main()
