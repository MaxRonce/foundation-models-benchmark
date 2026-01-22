"""
Script to select top-k anomalies from Normalizing Flow scores.
It can optionally intersect these anomalies with those found by Isolation Forest.
Produces a CSV file containing the selected outlier IDs.

Adapted for AstroPT.

Usage:
    python -m scratch.select_nf_outliers_astropt \
        --scores-csv scratch/outputs/anomaly_scores_astropt.csv \
        --output scratch/outputs/outlier_NFS_intersection_astropt.csv \
        --top-k 150
"""
import argparse
import csv
from pathlib import Path
from typing import Sequence

# Import from AstroPT version
from scratch.detect_outliers_astropt import EMBEDDING_KEYS
from scratch.display_outlier_images import read_object_ids


def parse_scores(
    scores_path: Path,
    embedding_keys: Sequence[str],
    top_k: int,
) -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, list[dict[str, float]]] = {key: [] for key in embedding_keys}
    with scores_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"object_id", "embedding_key", "log_prob", "neg_log_prob", "anomaly_sigma", "rank"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Scores CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            key = row["embedding_key"].strip()
            if key not in data:
                continue
            try:
                rank = int(row["rank"])
                log_prob = float(row["log_prob"])
                neg_log_prob = float(row["neg_log_prob"])
                sigma = float(row["anomaly_sigma"])
            except ValueError:
                continue
            record = {
                "object_id": row["object_id"].strip(),
                "rank": rank,
                "log_prob": log_prob,
                "neg_log_prob": neg_log_prob,
                "anomaly_sigma": sigma,
            }
            data[key].append(record)

    selected: dict[str, dict[str, dict[str, float]]] = {}
    for key, records in data.items():
        records.sort(key=lambda r: r["rank"])
        if top_k > 0:
            trimmed = records[:top_k]
        else:
            trimmed = records
        per_id: dict[str, dict[str, float]] = {}
        for record in trimmed:
            oid = record["object_id"]
            if not oid:
                continue
            per_id[oid] = {
                "rank": record["rank"],
                "log_prob": record["log_prob"],
                "neg_log_prob": record["neg_log_prob"],
                "anomaly_sigma": record["anomaly_sigma"],
            }
        selected[key] = per_id
    return selected


def intersect_ids(selected: dict[str, dict[str, dict[str, float]]]) -> set[str]:
    sets = [set(per_key.keys()) for per_key in selected.values() if per_key]
    if not sets:
        return set()
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return common


def build_rows(
    common_ids: set[str],
    selected: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for oid in common_ids:
        entry: dict[str, float | str] = {"object_id": oid}
        ranks: list[int] = []
        sigmas: list[float] = []
        for key, per_key in selected.items():
            metrics = per_key.get(oid)
            if metrics is None:
                continue
            entry[f"{key}_rank"] = metrics["rank"]
            entry[f"{key}_log_prob"] = metrics["log_prob"]
            entry[f"{key}_neg_log_prob"] = metrics["neg_log_prob"]
            entry[f"{key}_anomaly_sigma"] = metrics["anomaly_sigma"]
            ranks.append(int(metrics["rank"]))
            sigmas.append(float(metrics["anomaly_sigma"]))
        if ranks:
            entry["mean_rank"] = sum(ranks) / len(ranks)
            entry["max_rank"] = max(ranks)
        if sigmas:
            entry["mean_anomaly_sigma"] = sum(sigmas) / len(sigmas)
            entry["max_anomaly_sigma"] = max(sigmas)
        rows.append(entry)
    rows.sort(key=lambda r: (r.get("mean_rank", float("inf")), r.get("max_rank", float("inf"))))
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    if not rows:
        raise SystemExit("No rows to write; intersection is empty.")
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = ["object_id"] + sorted(k for k in keys if k != "object_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the top-k NF anomalies per embedding key and compute their intersection, "
            "optionally intersecting with Isolation Forest outliers (AstroPT version)."
        ),
    )
    parser.add_argument("--scores-csv", required=True, help="CSV file from scratch.detect_outliers_NFs_astropt")
    parser.add_argument("--output", required=True, help="Path to write the intersection CSV")
    parser.add_argument("--top-k", type=int, default=150, help="Number of top-ranked anomalies to keep per key")
    parser.add_argument(
        "--embedding-key",
        choices=EMBEDDING_KEYS,
        nargs="+",
        default=EMBEDDING_KEYS,
        help="Embedding key(s) to include when computing the intersection",
    )
    parser.add_argument(
        "--intersect-with-isf",
        nargs="+",
        default=None,
        metavar="CSV",
        help="Optional Isolation Forest outlier CSVs (object_id column) to intersect with",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)

    scores_path = Path(args.scores_csv)
    output_path = Path(args.output)
    if args.top_k < 0:
        raise SystemExit("--top-k must be non-negative")

    selected = parse_scores(scores_path, args.embedding_key, args.top_k)
    common_ids = intersect_ids(selected)
    if args.verbose:
        print(f"Intersection across embeddings: {len(common_ids)} objects")

    if args.intersect_with_isf:
        isf_paths = [Path(p) for p in args.intersect_with_isf]
        isf_ids = set(read_object_ids(isf_paths, limit=None, verbose=args.verbose))
        if args.verbose:
            print(f"Isolation Forest candidate set size: {len(isf_ids)}")
        common_ids &= isf_ids
        if args.verbose:
            print(f"After ISF intersection: {len(common_ids)} objects")

    if not common_ids:
        raise SystemExit("No objects remain after intersections; nothing to write.")

    rows = build_rows(common_ids, selected)
    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
