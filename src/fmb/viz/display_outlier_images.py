"""
Script to display a grid of Euclid RGB images for a list of object IDs.
Useful for visual inspection of outliers or specific samples.

Usage:
    python -m scratch.display_outlier_images \
        --csv outliers.csv \
        --save outliers_grid.png
"""
import argparse
import csv
import math
from pathlib import Path
from typing import Sequence, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from scratch.load_display_data import EuclidDESIDataset


def read_object_ids(
    csv_paths: Sequence[Path],
    limit: int | None = None,
    verbose: bool = False,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for path in csv_paths:
        if verbose:
            print(f"Reading object IDs from {path}")
        with path.open() as handle:
            reader = csv.DictReader(handle)
            if "object_id" not in reader.fieldnames:
                raise ValueError(f"CSV {path} is missing 'object_id' column")
            for row in reader:
                oid = str(row["object_id"]).strip()
                if not oid or oid in seen:
                    continue
                ids.append(oid)
                seen.add(oid)
                if limit is not None and len(ids) >= limit:
                    return ids
    return ids


def collect_samples(
    dataset: EuclidDESIDataset,
    target_ids: Sequence[str],
    verbose: bool = False,
) -> list[dict]:
    wanted = {str(oid): None for oid in target_ids}
    collected: list[dict] = []
    remaining = set(wanted.keys())
    iterator = dataset
    if verbose:
        iterator = tqdm(dataset, desc="Scanning dataset", unit="sample")
    for sample in iterator:
        oid = str(sample.get("object_id"))
        if oid in remaining:
            collected.append(sample)
            remaining.remove(oid)
            if not remaining:
                break
    missing = set(wanted.keys()) - {str(s.get("object_id")) for s in collected}
    if missing:
        print(f"Warning: {len(missing)} object IDs not found: {sorted(list(missing))[:5]}...")
    return collected


def load_index(index_path: Path) -> Dict[str, Tuple[str, int]]:
    mapping: Dict[str, Tuple[str, int]] = {}
    with index_path.open() as handle:
        reader = csv.DictReader(handle)
        required = {"object_id", "split", "index"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Index file must contain columns {required}")
        for row in reader:
            oid = str(row["object_id"]).strip()
            split = row["split"].strip()
            try:
                idx = int(row["index"])
            except ValueError as exc:
                raise ValueError(f"Invalid index for object_id={oid}: {row['index']}") from exc
            if oid:
                mapping[oid] = (split, idx)
    return mapping


def collect_samples_with_index(
    cache_dir: str,
    object_ids: Sequence[str],
    index_map: Dict[str, Tuple[str, int]],
    verbose: bool = False,
) -> list[dict]:
    from collections import defaultdict

    grouped: Dict[str, list[Tuple[str, int]]] = defaultdict(list)
    for oid in object_ids:
        if oid not in index_map:
            continue
        grouped[index_map[oid][0]].append((oid, index_map[oid][1]))

    samples_by_id: Dict[str, dict] = {}
    for split, entries in grouped.items():
        if verbose:
            print(f"Loading split '{split}' to fetch {len(entries)} samples")
        dataset = EuclidDESIDataset(split=split, cache_dir=cache_dir, verbose=False)
        for oid, idx in entries:
            if idx < 0 or idx >= len(dataset):
                print(f"Warning: index {idx} out of range for split '{split}' (object {oid})")
                continue
            sample = dataset[idx]
            samples_by_id[oid] = sample

    missing = [oid for oid in object_ids if oid not in samples_by_id]
    if missing:
        print(f"Warning: {len(missing)} object IDs missing in index/dataset: {missing[:5]}...")

    return [samples_by_id[oid] for oid in object_ids if oid in samples_by_id]


def prepare_rgb_image(sample: dict) -> np.ndarray:
    rgb = sample.get("rgb_image")
    if rgb is None:
        raise ValueError("Sample missing 'rgb_image'")
    if isinstance(rgb, torch.Tensor):
        tensor = rgb.detach().cpu()
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.squeeze(0) if tensor.shape[0] == 1 else tensor
            array = tensor.permute(1, 2, 0).numpy()
        elif tensor.dim() == 2:
            array = tensor.numpy()
        else:
            raise ValueError(f"Unexpected tensor shape for rgb_image: {tuple(tensor.shape)}")
    else:  # assume numpy array
        array = np.asarray(rgb)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.moveaxis(array, 0, -1)
    array = np.clip(array, 0.0, 1.0)
    return array


def plot_grid(samples: Sequence[dict], cols: int, save_path: Path | None, show: bool) -> None:
    count = len(samples)
    if count == 0:
        print("No samples to display.")
        return
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < count:
                sample = samples[idx]
                image = prepare_rgb_image(sample)
                if image.ndim == 3 and image.shape[2] == 1:
                    ax.imshow(image[..., 0], cmap="gray")
                else:
                    ax.imshow(image, cmap="gray")
                title = f"{sample.get('object_id', 'N/A')}"
                ax.set_title(title)
            ax.axis("off")
            idx += 1
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
        print(f"Saved grid to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Display Euclid RGB images for anomaly-detected object IDs",
    )
    parser.add_argument("--csv", nargs="+", required=True, help="CSV file(s) with object_id column")
    parser.add_argument("--split", type=str, default="all", help="Dataset split(s) for EuclidDESIDataset")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/n03data/ronceray/datasets",
    )
    parser.add_argument("--max", type=int, default=12, help="Maximum number of images to display")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid")
    parser.add_argument("--index", type=str, default=None, help="Optional CSV mapping object_id -> split/index")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive display")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)

    csv_paths = [Path(p) for p in args.csv]
    object_ids = read_object_ids(csv_paths, limit=args.max, verbose=args.verbose)
    if not object_ids:
        raise SystemExit("No object IDs found in provided CSV files")

    if args.index:
        index_map = load_index(Path(args.index))
        samples = collect_samples_with_index(
            cache_dir=args.cache_dir,
            object_ids=object_ids,
            index_map=index_map,
            verbose=args.verbose,
        )
    else:
        dataset = EuclidDESIDataset(split=args.split, cache_dir=args.cache_dir, verbose=args.verbose)
        samples = collect_samples(dataset, object_ids, verbose=args.verbose)

    if not samples:
        raise SystemExit("None of the requested object IDs were found in the dataset")

    plot_grid(
        samples,
        cols=max(1, args.cols),
        save_path=Path(args.save) if args.save else None,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
