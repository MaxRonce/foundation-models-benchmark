#!/usr/bin/env python3
"""
Compare the flux normalization (order of magnitude) between the Euclid VIS/Y/J/H
dataset on disk and the HSC GRIZY dataset streamed from HF
(`MultimodalUniverse/hsc`). We sample a handful of objects and report basic
statistics per band (median, 95th percentile of |flux|, max |flux|).

Usage example:
  python -m scratch.compare_hsc_euclid_normalization \\
    --euclid-cache-dir /scratch/ronceray/datasets \\
    --euclid-split train \\
    --euclid-max 200 \\
    --hsc-split train \\
    --hsc-max 200
"""
import argparse
import os
from typing import Dict, Iterable, List, Optional

import torch

from scratch.load_display_data import EuclidDESIDataset
from scratch.load_display_data_hsc import HSCDataset


def _to_tensor(x) -> Optional[torch.Tensor]:
    if x is None:
        return None
    t = torch.as_tensor(x).float()
    return t


def _summaries(values: List[torch.Tensor]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    flat = torch.cat([v.reshape(-1) for v in values])
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return None
    # Torch quantile can choke on very large tensors; subsample for a robust estimate.
    max_samples = 1_000_000
    if flat.numel() > max_samples:
        idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device)
        flat = flat[idx]
    abs_flat = flat.abs()
    return {
        "median": float(torch.median(flat)),
        "mean": float(torch.mean(flat)),
        "std": float(torch.std(flat)),
        "p95_abs": float(torch.quantile(abs_flat, 0.95)),
        "max_abs": float(torch.max(abs_flat)),
    }


def collect_euclid_stats(dataset: Iterable, max_items: int) -> Dict[str, Optional[Dict[str, float]]]:
    buckets: Dict[str, List[torch.Tensor]] = {k: [] for k in ("EUCLID-VIS", "EUCLID-Y", "EUCLID-J", "EUCLID-H")}
    for idx, sample in enumerate(dataset):
        if idx >= max_items:
            break
        for key, name in [
            ("vis_image", "EUCLID-VIS"),
            ("nisp_y_image", "EUCLID-Y"),
            ("nisp_j_image", "EUCLID-J"),
            ("nisp_h_image", "EUCLID-H"),
        ]:
            t = _to_tensor(sample.get(key))
            if t is None:
                continue
            buckets[name].append(t)
    return {k: _summaries(v) for k, v in buckets.items()}


def collect_hsc_stats(dataset: Iterable, max_items: int) -> Dict[str, Optional[Dict[str, float]]]:
    buckets: Dict[str, List[torch.Tensor]] = {k: [] for k in ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")}
    for idx, sample in enumerate(dataset):
        if idx >= max_items:
            break
        for key, name in [
            ("hsc_g", "HSC-G"),
            ("hsc_r", "HSC-R"),
            ("hsc_i", "HSC-I"),
            ("hsc_z", "HSC-Z"),
            ("hsc_y", "HSC-Y"),
        ]:
            t = _to_tensor(sample.get(key))
            if t is None:
                continue
            buckets[name].append(t)
    return {k: _summaries(v) for k, v in buckets.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Euclid vs HSC flux normalization (order of magnitude).")
    p.add_argument("--euclid-cache-dir", type=str, default="/scratch", help="Local HF cache/dataset root for Euclid.")
    p.add_argument("--euclid-split", type=str, default="train", help="Euclid split(s), e.g. train,test,all.")
    p.add_argument("--euclid-max", type=int, default=200, help="Number of Euclid samples to scan.")
    p.add_argument("--hsc-split", type=str, default="train", help="HSC split (HF).")
    p.add_argument("--hsc-max", type=int, default=200, help="Number of HSC samples to scan (streaming).")
    p.add_argument(
        "--hsc-cache-dir",
        type=str,
        default="./.cache/hsc",
        help="Writable cache dir for HSC HF metadata (avoids /pbs path).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[Euclid] Loading split='{args.euclid_split}' from {args.euclid_cache_dir}")
    euclid_ds = EuclidDESIDataset(split=args.euclid_split, cache_dir=args.euclid_cache_dir, verbose=False)
    euclid_stats = collect_euclid_stats(euclid_ds, max_items=max(args.euclid_max, 0))

    print(f"[HSC] Streaming split='{args.hsc_split}' from HF (MultimodalUniverse/hsc)")
    os.makedirs(args.hsc_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = os.path.abspath(args.hsc_cache_dir)
    os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    hsc_ds = HSCDataset(
        split=args.hsc_split,
        streaming=True,
        max_items=max(args.hsc_max, 0),
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )
    hsc_stats = collect_hsc_stats(hsc_ds, max_items=max(args.hsc_max, 0))

    def pretty_print(title: str, stats: Dict[str, Optional[Dict[str, float]]]) -> None:
        print(f"\n{title}")
        for band, s in stats.items():
            if s is None:
                print(f"  {band:<8} : missing or empty")
                continue
            print(
                f"  {band:<8} median={s['median']:.3g}  mean={s['mean']:.3g}  std={s['std']:.3g}  "
                f"p95|.|={s['p95_abs']:.3g}  max|.|={s['max_abs']:.3g}"
            )

    pretty_print("Euclid bands (raw flux)", euclid_stats)
    pretty_print("HSC bands (raw flux)", hsc_stats)

    print("\nReminder: HSC images are rescaled to zeropoint 27 in AION before encoding; Euclid is not.")
    print("If Euclid values are orders of magnitude off from HSC, consider adding a rescale factor.")


if __name__ == "__main__":
    main()
