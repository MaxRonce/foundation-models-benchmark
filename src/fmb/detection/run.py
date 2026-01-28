"""
Foundation Models Benchmark (FMB)

Module: fmb.detection.run
Description: Inference and anomaly scoring
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import List

import torch

from fmb.paths import load_paths
from fmb.detection import utils, models, train

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_single_embedding(
    path: Path,
    possible_keys: List[str],
    config: dict,
    model_name: str,
    output_dir: Path
) -> None:
    
    # 1. Load Data
    try:
        records = utils.load_records(path)
    except Exception as e:
        print(f"[error] Failed to load {path}: {e}")
        return

    # 2. Key Discovery
    if not records:
        print(f"[warn] Empty records in {path}")
        return

    valid_keys = [k for k in possible_keys if k in records[0]]
    
    # AstroPT Synthesis Fallback
    if model_name == "astropt" and "embedding_joint" in possible_keys and "embedding_joint" not in valid_keys:
        if "embedding_images" in records[0] and "embedding_spectra" in records[0]:
            print("      [info] AstroPT: 'embedding_joint' not found. Synthesizing...")
            valid_keys.append("embedding_joint")
            # We synthesize on the fly during extraction below if needed, 
            # OR we update records here. Let's update records here for simplicity.
            for rec in records:
                img = rec.get("embedding_images")
                spec = rec.get("embedding_spectra")
                if img is not None and spec is not None:
                     # Ensure numpy
                    if isinstance(img, torch.Tensor): img = img.cpu().numpy()
                    if isinstance(spec, torch.Tensor): spec = spec.cpu().numpy()
                    rec["embedding_joint"] = import_numpy().concatenate([img.flatten(), spec.flatten()])
    
    if not valid_keys:
        print(f"[warn] No valid keys found in {path}. Expected: {possible_keys}")
        return

    # 3. Process Each Key
    all_rows = []
    
    for key in valid_keys:
        print(f"\n---> Processing {model_name} key: '{key}'")
        
        # Extract
        embeddings, ids = utils.extract_embeddings(records, key)
        if len(embeddings) == 0:
            print(f"[warn] Extraction failed for {key}")
            continue

        # Clean
        embeddings_tensor = torch.from_numpy(embeddings).float()
        embeddings_tensor, ids = utils.filter_nonfinite_rows(embeddings_tensor, ids)
        
        if len(embeddings_tensor) < 2:
            print(f"[warn] Not enough data for {key}")
            continue

        # PCA (if configured)
        pca_comps = config.get("pca_components", 0)
        if pca_comps > 0:
            if pca_comps < embeddings_tensor.shape[1]:
                embeddings_tensor, _ = utils.apply_pca(embeddings_tensor, pca_comps)
            else:
                 print(f"[info] Skipping PCA (requested {pca_comps} >= dim {embeddings_tensor.shape[1]})")

        # Standardize
        if config.get("standardize", True):
            embeddings_tensor, _, _ = utils.standardize_tensor(embeddings_tensor)

        # Clip
        sigma = config.get("clip_sigma", 0.0)
        embeddings_tensor = utils.clip_embeddings_by_sigma(embeddings_tensor, sigma)

        # Build Flow
        dim = embeddings_tensor.shape[1]
        flow = models.build_flow(
            flow_type=config.get("flow_type", "autoregressive"),
            dim=dim,
            hidden_features=config.get("hidden_features", 256),
            num_transforms=config.get("num_transforms", 8),
        )
        
        print(f"[{key}] Training {config.get('flow_type')} flow (dim={dim})...")
        
        # Train
        train.train_flow_model(
            flow=flow,
            data=embeddings_tensor,
            epochs=config.get("epochs", 100),
            batch_size=config.get("batch_size", 512),
            lr=config.get("lr", 1e-4),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            log_every=config.get("log_every", 25),
            grad_clip=config.get("grad_clip", 5.0),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        
        # Score
        log_probs = train.compute_log_probs(
            flow, 
            embeddings_tensor, 
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Collate
        rows = utils.collate_rows(ids, key, log_probs)
        all_rows.extend(rows)

    # 4. Save
    if all_rows:
        out_csv = output_dir / f"anomaly_scores_{model_name}.csv"
        print(f"[success] Saving {len(all_rows)} scores to {out_csv}")
        utils.save_scores_csv(out_csv, all_rows)
    else:
        print(f"[warn] No results generated for {model_name}")

    # Cleanup
    del records
    import gc
    gc.collect() 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def import_numpy():
    import numpy as np
    return np

def main(argv: List[str] = None):
    paths = load_paths()
    
    parser = argparse.ArgumentParser(description="Run Normalizing Flow Outlier Detection")
    parser.add_argument("--config", type=str, default=str(paths.repo_root / "src/fmb/configs/detection/anomalies.yaml"), help="Path to config YAML")
    
    # Embedding inputs (defaults from paths.py)
    # If not provided in CLI, we check paths.py properties
    # But paths.py properties usually return Paths, which might not exist if not downloaded.
    # We will use reasonable defaults or check existence.
    
    # We allow explicit CLI overrides, otherwise we check default locations
    parser.add_argument("--aion-embeddings", type=str, help="Path to AION embeddings")
    parser.add_argument("--astropt-embeddings", type=str, help="Path to AstroPT embeddings")
    parser.add_argument("--astroclip-embeddings", type=str, help="Path to AstroCLIP embeddings")
    
    args = parser.parse_args(argv)
    
    # Load config
    config = load_config(Path(args.config))
    utils.set_random_seed(config.get("random_seed", 42))
    
    # Determine Output Directory
    # Use paths.outliers (from paths_local.yaml)
    output_dir = paths.outliers
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine Inputs
    # If arg is present, use it. If not, try default from paths.embeddings directory?
    # Or just require them?
    # The previous script used absolute paths.
    # We can try to guess defaults based on known filenames if not provided.
    
    tasks = []
    
    # AION
    aion_path = Path(args.aion_embeddings) if args.aion_embeddings else (paths.embeddings / "aions_embeddings.pt")
    if aion_path.exists():
        tasks.append((aion_path, utils.KEYS_AION, "aion"))
    elif args.aion_embeddings:
        print(f"[error] AION file not found: {aion_path}")

    # AstroPT
    astropt_path = Path(args.astropt_embeddings) if args.astropt_embeddings else (paths.embeddings / "astropt_embeddings.pt")
    if astropt_path.exists():
        tasks.append((astropt_path, utils.KEYS_ASTROPT, "astropt"))
    elif args.astropt_embeddings:
        print(f"[error] AstroPT file not found: {astropt_path}")

    # AstroCLIP
    astroclip_path = Path(args.astroclip_embeddings) if args.astroclip_embeddings else (paths.embeddings / "embeddings_astroclip.pt")
    if astroclip_path.exists():
        tasks.append((astroclip_path, utils.KEYS_ASTROCLIP, "astroclip"))
    elif args.astroclip_embeddings:
        print(f"[error] AstroCLIP file not found: {astroclip_path}")

    if not tasks:
        print("[error] No embedding files found. Please provide paths via --aion-embeddings etc. or ensure they exist in configured 'embeddings_path'.")
        return

    for fpath, keys, name in tasks:
        print(f"\n=== Processing {name.upper()} ===")
        try:
            process_single_embedding(fpath, keys, config, name, output_dir)
        except KeyboardInterrupt:
            print("\n[info] Interrupted by user.")
            sys.exit(1)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[error] Failed processing {name}: {e}")

if __name__ == "__main__":
    main()
