"""
Script to create a CSV index of the dataset.
It maps object IDs to their split (train/test) and index within the dataset.
This is useful for quick lookups and ensuring consistent ordering.

Usage:
    python -m scratch.index_dataset --output euclid_index.csv --splits all
"""
import argparse
import csv
from pathlib import Path
from typing import Sequence

from datasets import get_dataset_split_names, load_dataset, load_from_disk
from tqdm import tqdm

# Chemin local où se trouvent les deux dossiers train/test téléchargés
DEFAULT_CACHE = "/n03data/ronceray/datasets"
HF_DATASET_ID = "msiudek/astroPT_euclid_Q1_desi_dr1_dataset"
LOCAL_SPLITS = {
    "train": "msiudek__astroPT_euclid_Q1_desi_dr1_dataset__train",
    "test": "msiudek__astroPT_euclid_Q1_desi_dr1_dataset__test",
}


def index_dataset(cache_dir: str, splits: Sequence[str], output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise SystemExit(f"Output file {output} already exists. Use --overwrite to replace it.")

    def _load_split(split_name: str):
        local_dir = LOCAL_SPLITS.get(split_name)
        if local_dir:
            path = Path(cache_dir) / local_dir
            if path.is_dir():
                print(f"  Loading split '{split_name}' from local directory: {path}")
                return load_from_disk(str(path))
        print(f"  Loading split '{split_name}' from HF dataset {HF_DATASET_ID}")
        return load_dataset(HF_DATASET_ID, split=split_name, cache_dir=cache_dir)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["object_id", "split", "index"])

        for split in splits:
            print(f"Indexing split '{split}'...")
            ds = _load_split(split)
            progress = tqdm(ds, desc=f"{split}", unit="sample")
            for idx, sample in enumerate(progress):
                oid = sample.get("object_id") or sample.get("targetid")
                if oid is None:
                    continue
                writer.writerow([oid, split, idx])
            progress.close()
            print(f"  Recorded {len(ds)} entries for split '{split}'.")

    print(f"Index written to {output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Precompute object_id to split/index mapping for Euclid dataset")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE,
        help="Dataset cache directory",
    )
    parser.add_argument(
        "--splits",
        default="all",
        help="Comma-separated list of splits or 'all' (default)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )

    args = parser.parse_args(argv)

    if args.splits.strip().lower() == "all":
        # Priorité aux splits disponibles localement, sinon on récupère ceux du Hub
        local_splits = [s for s, p in LOCAL_SPLITS.items() if (Path(args.cache_dir) / p).is_dir()]
        if local_splits:
            splits = local_splits
        else:
            splits = get_dataset_split_names(HF_DATASET_ID)
    else:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        if not splits:
            raise SystemExit("No valid splits provided")

    index_dataset(
        cache_dir=args.cache_dir,
        splits=splits,
        output=Path(args.output),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
