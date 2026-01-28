"""
Foundation Models Benchmark (FMB)

Module: fmb.data.index_dataset
Description: Indexing utilities for fast object lookups
"""

import argparse
import csv
from pathlib import Path
from typing import Sequence, Optional, List

from datasets import get_dataset_split_names, load_dataset, load_from_disk
from tqdm import tqdm

from fmb.paths import load_paths

# Mapping for local directories if available
LOCAL_SPLITS = {
    "train": "msiudek__astroPT_euclid_Q1_desi_dr1_dataset__train",
    "test": "msiudek__astroPT_euclid_Q1_desi_dr1_dataset__test",
}
# Default HF ID if not in config (though paths.py handles this)
DEFAULT_HF_ID = "msiudek/astroPT_euclid_Q1_desi_dr1_dataset"


def run_indexing(
    cache_dir: Optional[str] = None,
    splits: Sequence[str] = ("all",),
    output: Optional[Path] = None,
    overwrite: bool = False,
    hf_dataset_id: Optional[str] = None
) -> None:
    """
    Main entry point for indexing the dataset.
    """
    paths = load_paths()
    
    # Defaults
    if cache_dir is None:
        cache_dir = str(paths.dataset)
    if output is None:
        output = paths.dataset_index
    if hf_dataset_id is None:
        hf_dataset_id = paths.dataset_hf_id

    if output.exists() and not overwrite:
        print(f"Index file {output} already exists. Use --overwrite to replace it.")
        return

    # Determine splits
    final_splits: List[str] = []
    
    # Check if "all" is requested
    if "all" in [s.lower() for s in splits]:
        # Priority to local directories in cache_dir
        local_found = []
        for s_name, local_name in LOCAL_SPLITS.items():
            if (Path(cache_dir) / local_name).is_dir():
                local_found.append(s_name)
        
        if local_found:
            final_splits = local_found
            print(f"Found local splits: {final_splits}")
        else:
            try:
                final_splits = get_dataset_split_names(hf_dataset_id)
                print(f"Found remote splits: {final_splits}")
            except Exception as e:
                print(f"Error fetching splits from HF: {e}")
                raise
    else:
        final_splits = list(splits)

    if not final_splits:
        raise ValueError("No splits found or provided.")

    # Helper to load
    def _load_split(split_name: str):
        # Try local first
        local_dir = LOCAL_SPLITS.get(split_name)
        if local_dir:
            path = Path(cache_dir) / local_dir
            if path.is_dir():
                print(f"  Loading split '{split_name}' from local directory: {path}")
                return load_from_disk(str(path))
        
        print(f"  Loading split '{split_name}' from HF dataset {hf_dataset_id}")
        return load_dataset(hf_dataset_id, split=split_name, cache_dir=cache_dir)

    # Write Index
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["object_id", "split", "index"])

        for split in final_splits:
            print(f"Indexing split '{split}'...")
            ds = _load_split(split)
            # Use 'targetid' or 'object_id'
            # Euclid dataset often uses 'object_id', older versions might use 'targetid'
            # We check first sample to be efficient? No, iterate all.
            
            count = 0
            for idx, sample in enumerate(tqdm(ds, desc=f"{split}", unit="sample")):
                oid = sample.get("object_id") or sample.get("targetid")
                if oid is not None:
                    writer.writerow([oid, split, idx])
                    count += 1
            
            print(f"  Recorded {count} entries for split '{split}'.")

    print(f"Index written to {output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Precompute object_id mapping")
    parser.add_argument("--cache-dir", default=None, help="Dataset cache directory")
    parser.add_argument("--splits", default="all", help="Comma-separated splits")
    parser.add_argument("--output", default=None, help="Path to output CSV")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing file")
    
    args = parser.parse_args(argv)

    splits_list = [s.strip() for s in args.splits.split(",") if s.strip()]
    
    run_indexing(
        cache_dir=args.cache_dir,
        splits=splits_list,
        output=Path(args.output) if args.output else None,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()
