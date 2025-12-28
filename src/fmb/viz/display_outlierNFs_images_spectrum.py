"""
Script to display image grids with spectra for top-k Normalizing Flow anomalies.
It loads the anomaly scores, selects the top anomalies, and generates a visualization
showing the Euclid RGB image and DESI spectrum for each.

Usage:
    python -m scratch.display_outlierNFs_images_spectrum \
        --scores-csv scratch/outputs/anomaly_scores.csv \
        --output-dir scratch/outputs/nf_anomaly_grids
"""
import argparse
import csv
from pathlib import Path
from typing import Sequence

from scratch.load_display_data import EuclidDESIDataset
from scratch.display_outlier_images import (
    collect_samples,
    collect_samples_with_index,
    load_index,
)
from scratch.display_outlier_images_spectrum import plot_vertical_panels  # reuse plotting


EMBEDDING_KEYS = [
    "embedding_hsc_desi",
    "embedding_hsc",
    "embedding_spectrum",
]


def load_topk_from_scores(
    path: Path,
    max_per_key: int,
    verbose: bool = False,
) -> dict[str, list[str]]:
    by_key: dict[str, list[tuple[int, str]]] = {k: [] for k in EMBEDDING_KEYS}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"object_id", "embedding_key", "rank"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Scores CSV missing columns: {sorted(missing)}")
        for row in reader:
            key = row["embedding_key"]
            if key not in by_key:
                continue
            try:
                rank = int(row["rank"])  # 1 = most anomalous
            except ValueError:
                continue
            oid = str(row["object_id"])
            by_key[key].append((rank, oid))
    topk: dict[str, list[str]] = {}
    for key, pairs in by_key.items():
        pairs.sort(key=lambda x: x[0])
        selected = [oid for _, oid in pairs[: max(0, int(max_per_key))]]
        if verbose:
            print(f"[{key}] selected {len(selected)} objects (top {max_per_key} by rank)")
        if selected:
            topk[key] = selected
    return topk


def sanitize_key(key: str) -> str:
    return key.replace("embedding_", "")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Display Euclid RGB images and DESI spectra for top-K NF anomalies "
            "for each embedding key (HSC, HSC+DESI, spectrum)."
        ),
    )
    parser.add_argument("--scores-csv", required=True, help="CSV from scratch.detect_outliers_NFs")
    parser.add_argument("--split", type=str, default="all", help="Dataset split(s) for EuclidDESIDataset")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/n03data/ronceray/datasets",
    )
    parser.add_argument("--max", type=int, default=12, help="Max number of anomalies per embedding key")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in each grid")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated figures")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive display")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--index", type=str, default=None, help="Optional CSV mapping object_id -> split/index")
    args = parser.parse_args(argv)

    scores_path = Path(args.scores_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topk = load_topk_from_scores(scores_path, args.max, verbose=args.verbose)
    if not topk:
        raise SystemExit("No anomalies found in the provided scores CSV.")

    index_map = None
    dataset = None
    if args.index:
        index_map = load_index(Path(args.index))
    else:
        dataset = EuclidDESIDataset(split=args.split, cache_dir=args.cache_dir, verbose=args.verbose)

    for key in EMBEDDING_KEYS:
        object_ids = topk.get(key)
        if not object_ids:
            if args.verbose:
                print(f"[skip] No selected anomalies for {key}")
            continue
        if index_map is not None:
            samples = collect_samples_with_index(
                cache_dir=args.cache_dir,
                object_ids=object_ids,
                index_map=index_map,
                verbose=args.verbose,
            )
        else:
            samples = collect_samples(dataset, object_ids, verbose=args.verbose)  # type: ignore[arg-type]
        if not samples:
            if args.verbose:
                print(f"[skip] None of the selected IDs found for {key}")
            continue
        save_path = out_dir / f"{sanitize_key(key)}_nf_anomalies_with_spectra.png"
        print(f"[{key}] plotting {len(samples)} objects → {save_path}")
        plot_vertical_panels(
            samples,
            cols=max(1, args.cols),
            save_path=save_path,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()

