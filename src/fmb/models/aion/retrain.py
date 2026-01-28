"""
2026-01-24
aion/retrain.py


Description
-----------
Entry point for AION (Euclid ↔ HSC adapter) retraining.
Delegates to retrain_euclid_hsc_adapter_unet.py.

Usage
-----
python -m fmb.models.aion.retrain --epochs 15 --batch-size 8
python -m fmb.models.aion.retrain --config configs/aion.yaml
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parents[3]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import and delegate to the full implementation
from fmb.models.aion.retrain_euclid_hsc_adapter_unet import main

if __name__ == "__main__":
    # Inject default config if not present
    if "--config" not in sys.argv:
        root_dir = Path(__file__).resolve().parents[4]
        config_path = root_dir / "configs" / "retrain" / "aion.yaml"
        if config_path.exists():
            print(f"Injecting default config: {config_path}")
            sys.argv.extend(["--config", str(config_path)])

    main()
