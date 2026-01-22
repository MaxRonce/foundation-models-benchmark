"""
Script to find and display similar objects for a list of query anomalies.
It uses cosine similarity on embeddings to find neighbors and visualizes them
in a grid (Query + N Neighbors) with images and spectra.

Usage:
    python -m scratch.find_similar_anomalies \
        --input /path/to/embeddings.pt \
        --object_ids 39633430668904665 39633461148912540 \
        --n-similar 3 \
        --save outputs/similar_anomalies.png
"""
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scratch.load_display_data import EuclidDESIDataset
from scratch.display_outlier_images_spectrum import plot_vertical_panels
from scratch.display_outlier_images import collect_samples, read_object_ids, collect_samples_with_index, load_index


def load_records(path: Path) -> list[dict]:
    """Loads embedding records from a .pt file."""
    print(f"Loading embeddings from {path}...")
    data = torch.load(path, map_location="cpu")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported embeddings format: {type(data)}")


def get_embedding_matrices(records: list[dict]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """
    Extracts all available embedding matrices and corresponding object IDs.
    Auto-detects AION vs AstroPT keys.
    Returns: ({key: matrix}, object_ids)
    """
    sample = records[0]
    available_keys = []
    
    # Check for AstroPT style
    has_astro_img = "embedding_images" in sample
    has_astro_spec = "embedding_spectra" in sample
    
    # Check for AION style
    has_aion_img = "embedding_hsc" in sample
    has_aion_spec = "embedding_spectrum" in sample or "embedding_hsc_desi" in sample # hsc_desi might be pre-fused
    
    # Define keys to extract based on presence
    keys_to_extract = []
    
    if has_astro_img and has_astro_spec:
        # AstroPT
        keys_to_extract = ["embedding_images", "embedding_spectra", "embedding_joint"]
    elif has_aion_img or has_aion_spec:
        # AION
        # Try to find standard AION keys
        possible = ["embedding_hsc", "embedding_spectrum", "embedding_hsc_desi"]
        keys_to_extract = [k for k in possible if k in sample]
        # If we have hsc and spectrum but no hsc_desi, maybe we want to create joint?
        # AION code usually calls it embedding_hsc_desi
        if "embedding_hsc" in sample and "embedding_spectrum" in sample and "embedding_hsc_desi" not in sample:
            keys_to_extract.append("embedding_hsc_desi_computed") # Custom tag to trigger compute
            
    if not keys_to_extract:
         # Fallback: grab everything starting with embedding_
         keys_to_extract = [k for k in sample.keys() if k.startswith("embedding_")]
         
    print(f"Detected embedding types: {keys_to_extract}")
    
    vectors_map = {k: [] for k in keys_to_extract}
    object_ids = []
    
    for rec in records:
        oid = str(rec.get("object_id", ""))
        if not oid:
            continue
            
        valid_rec = True
        
        # Temp buffers to ensure we only add if all requested keys succeed? 
        # Or individual? 
        # Usually we want the intersection of valid objects.
        # Let's try to get all.
        
        current_vecs = {}
        
        for key in keys_to_extract:
            vec = None
            if key == "embedding_joint":
                # AstroPT construct
                img = rec.get("embedding_images")
                spec = rec.get("embedding_spectra")
                if img is not None and spec is not None:
                     if isinstance(img, torch.Tensor): img = img.detach().cpu()
                     else: img = torch.tensor(img)
                     if isinstance(spec, torch.Tensor): spec = spec.detach().cpu()
                     else: spec = torch.tensor(spec)
                     vec = torch.cat([img, spec])
            elif key == "embedding_hsc_desi_computed":
                # AION construct if needed
                img = rec.get("embedding_hsc")
                spec = rec.get("embedding_spectrum")
                if img is not None and spec is not None:
                     if isinstance(img, torch.Tensor): img = img.detach().cpu()
                     else: img = torch.tensor(img)
                     if isinstance(spec, torch.Tensor): spec = spec.detach().cpu()
                     else: spec = torch.tensor(spec)
                     vec = torch.cat([img, spec])
            else:
                vec = rec.get(key)
                if vec is not None:
                    if not isinstance(vec, torch.Tensor):
                        vec = torch.tensor(vec)
                    vec = vec.detach().cpu()
            
            if vec is None:
                valid_rec = False
                break
            current_vecs[key] = vec
            
        if valid_rec:
            object_ids.append(oid)
            for k, v in current_vecs.items():
                vectors_map[k].append(v)
                
    if not object_ids:
        raise ValueError("No valid records found containing all required embeddings.")
        
    matrices = {}
    for k, v_list in vectors_map.items():
        # Rename computed key back to standard if preferred, or keep descriptive
        # Let's map "embedding_hsc_desi_computed" to "embedding_joint" for output clarity
        out_key = "embedding_joint" if k == "embedding_hsc_desi_computed" else k
        
        mat = torch.stack(v_list)
        mat = F.normalize(mat, p=2, dim=1)
        matrices[out_key] = mat
        
    return matrices, object_ids


def find_neighbors(
    query_ids: list[str],
    all_ids: list[str],
    embeddings: torch.Tensor,
    n_similar: int
) -> list[str]:
    """
    Finds nearest neighbors for each query ID.
    Returns a flat list of object IDs: [Q1, N1_1, N1_2, ..., Q2, N2_1, ...]
    """
    
    id_to_idx = {oid: i for i, oid in enumerate(all_ids)}
    ordered_results = []
    
    # Pre-compute similarity if needed, but doing it per query is fine for small N
    # Actually for many queries, matrix multiplication is faster
    # limiting strictly to the queries present in the file
    
    valid_queries = []
    query_indices = []
    
    for qid in query_ids:
        if qid not in id_to_idx:
            print(f"Warning: Query ID {qid} not found in embeddings file.")
            continue
        valid_queries.append(qid)
        query_indices.append(id_to_idx[qid])
        
    if not valid_queries:
        return []
        
    query_vecs = embeddings[query_indices] # (N_queries, D)
    
    # Similarity: (N_queries, N_total)
    print("Computing similarities...")
    sim_matrix = torch.mm(query_vecs, embeddings.t())
    
    # We want top (n_similar + 1) because the query itself will likely be #1
    # But wait, if duplicates exist or numerical issues, checking equality is safer.
    # Let's just get top N+5 and filter.
    
    # Get top k values
    search_k = n_similar + 1 + 5 # buffer
    if search_k > len(all_ids):
        search_k = len(all_ids)
        
    topk_vals, topk_inds = torch.topk(sim_matrix, k=search_k, dim=1)
    
    final_ordered_ids = []
    
    for i, qid in enumerate(valid_queries):
        # Start with the query itself
        row_ids = [qid] 
        
        indices = topk_inds[i].tolist()
        # Filter neighbors
        found_neighbors = 0
        for idx in indices:
            neighbor_id = all_ids[idx]
            if neighbor_id == qid:
                continue
            row_ids.append(neighbor_id)
            found_neighbors += 1
            if found_neighbors == n_similar:
                break
        
        # Ensure we fill up to n_similar even if not enough found (unlikely)
        # But wait, we want a fixed grid.
        final_ordered_ids.extend(row_ids)
        
    return final_ordered_ids


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Find and display similar anomalies using embedding cosine similarity."
    )
    parser.add_argument("--input", help="Path to simple embeddings .pt file (legacy)")
    parser.add_argument("--aion-embeddings", help="Path to AION embeddings .pt file")
    parser.add_argument("--astropt-embeddings", help="Path to AstroPT embeddings .pt file")
    parser.add_argument("--astroclip-embeddings", help="Path to AstroCLIP embeddings .pt file")
    
    parser.add_argument("--object_ids", nargs="+", help="List of object IDs to query")
    parser.add_argument("--csv", nargs="+", help="CSV files containing object_id column")
    
    parser.add_argument("--n-similar", type=int, default=3, help="Number of similar objects to find per query")
    parser.add_argument("--save", type=str, default="similar_anomalies.png", help="Path to save output image")
    parser.add_argument("--split", type=str, default="all", help="Dataset split")
    parser.add_argument("--smooth", type=float, default=3.0, help="Sigma for Gaussian smoothing of spectrum (default: 3.0)")
    parser.add_argument("--cache-dir", type=str, default="/n03data/ronceray/datasets")
    parser.add_argument("--index", type=str, default=None, help="Optional CSV mapping object_id -> split/index")

    args = parser.parse_args(argv)
    
    # 1. Collect Query IDs (Logic Unchanged)
    query_ids = []
    if args.object_ids:
        query_ids.extend(args.object_ids)
    
    if args.csv:
        csv_paths = [Path(p) for p in args.csv]
        from scratch.display_outlier_images import read_object_ids
        file_ids = read_object_ids(csv_paths, limit=None)
        query_ids.extend(file_ids)
        
    seen = set()
    unique_query_ids = []
    for q in query_ids:
        if q not in seen:
            unique_query_ids.append(q)
            seen.add(q)
    query_ids = unique_query_ids
    
    if not query_ids:
        raise SystemExit("No object IDs provided via --object_ids or --csv")
        
    print(f"Querying for {len(query_ids)} objects...")

    # Define tasks: (Model Name, Path)
    tasks = []
    if args.input:
        tasks.append(("Generic", Path(args.input)))
    if args.aion_embeddings:
        tasks.append(("AION", Path(args.aion_embeddings)))
    if args.astropt_embeddings:
        tasks.append(("AstroPT", Path(args.astropt_embeddings)))
    if args.astroclip_embeddings:
        tasks.append(("AstroCLIP", Path(args.astroclip_embeddings)))
        
    if not tasks:
        raise SystemExit("No embedding files provided! Use --aion-embeddings, --astropt-embeddings, etc.")

    # Shared Dataset
    dataset = None 
    all_annotated_samples = []
    collected_labels = []

    for model_name, path in tasks:
        print(f"\n=== Processing Model: {model_name} ===")
        print(f"Loading from {path}...")
        
        try:
            records = load_records(path)
            matrices, all_ids = get_embedding_matrices(records)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

        # Force specific order if known? 
        # For AION: Image, Spectrum, Joint
        # For AstroPT/Clip: Image, Spectrum, Joint
        
        # Sort keys to ensure consistent order: Image -> Spec -> Joint
        # Custom sort function
        def key_sort(k):
             k = k.lower()
             if "image" in k or "hsc" in k and "desi" not in k: return 0
             if "spectr" in k: return 1
             if "joint" in k or "hsc_desi" in k: return 2
             return 99

        sorted_keys = sorted(matrices.keys(), key=key_sort)

        for key in sorted_keys:
            embedding_matrix = matrices[key]
            
            # Label Handling
            suffix = key.replace("embedding_", "")
            
            # Map suffix to pretty name
            type_label_map = {
                "images": "Image",
                "hsc": "Image",
                "spectra": "Spectrum",
                "spectrum": "Spectrum",
                "joint": "Joint",
                "hsc_desi": "Joint",
                "hsc_desi_computed": "Joint"
            }
            type_pretty = type_label_map.get(suffix, suffix.capitalize())
            full_label = f"{model_name}\n{type_pretty}"
            
            print(f"  -> Modality: {type_pretty} ({key})")
            
            # Determine individual save path (optional, maybe skip individual for multi-mode?)
            # Let's keep individual saves, but namespaced by model
            base_save = Path(args.save)
            stem = base_save.stem
            model_slug = model_name.lower().replace(" ", "")
            new_filename = f"{stem}_{model_slug}_{suffix}{base_save.suffix}"
            save_path = base_save.parent / new_filename
            
            # Find Neighbors
            ordered_ids = find_neighbors(query_ids, all_ids, embedding_matrix, args.n_similar)
            
            if not ordered_ids:
                print(f"    No neighbors found, skipping.")
                continue
            
            if args.index:
                index_map = load_index(Path(args.index))
                samples = collect_samples_with_index(
                    cache_dir=args.cache_dir,
                    object_ids=ordered_ids,
                    index_map=index_map,
                    verbose=False # Reduce spam
                )
            else:
                if dataset is None:
                     dataset = EuclidDESIDataset(split=args.split, cache_dir=args.cache_dir)
                samples = collect_samples(dataset, ordered_ids, verbose=False)
                
            # Alignment & Placeholder Logic
            sample_map = {str(s["object_id"]): s for s in samples}
            final_samples = []
            for oid in ordered_ids:
                if oid in sample_map:
                    final_samples.append(sample_map[oid])
                else:
                     final_samples.append({
                        "object_id": oid, 
                        "image": np.zeros((64, 64, 3), dtype=np.uint8),
                        "redshift": None 
                    })

            # Annotate
            cols = args.n_similar + 1
            annotated_samples = []
            for i, s in enumerate(final_samples):
                new_s = s.copy()
                original_id = str(new_s.get("object_id", ""))
                if i % cols == 0:
                    label_prefix = f"[QUERY]"
                else:
                    rank = i % cols
                    label_prefix = f"[NEIGHBOR {rank}]"
                new_s["object_id"] = f"{label_prefix} {original_id}"
                annotated_samples.append(new_s)

            # Plot Individual (Silent)
            plot_vertical_panels(
                annotated_samples,
                cols=cols,
                save_path=save_path,
                show=False,
                smooth_sigma=args.smooth
            )
            
            # Collect for Combined
            all_annotated_samples.extend(annotated_samples)
            collected_labels.append(full_label)

    # 4. Generate Combined Plot
    if all_annotated_samples:
        print(f"\nGenerating combined grid across {len(tasks)} models...")
        base_save = Path(args.save)
        combined_filename = f"{base_save.stem}_combined{base_save.suffix}"
        combined_path = base_save.parent / combined_filename
        
        cols = args.n_similar + 1
        
        plot_vertical_panels(
            all_annotated_samples,
            cols=cols,
            save_path=combined_path,
            show=False,
            row_labels=collected_labels,
            smooth_sigma=args.smooth
        )

if __name__ == "__main__":
    main()
