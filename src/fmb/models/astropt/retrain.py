"""
2026-01-23
astropt/retrain.py


Description
-----------
Simplified entry point for AstroPT multimodal training.
Delegates heavy lifting to the original retrain_spectra_images.py.

Usage
-----
python -m fmb.models.astropt.retrain --epochs 30 --batch-size 16
python -m fmb.models.astropt.retrain --config configs/astropt.yaml
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parents[3]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import and delegate to the full implementation
from fmb.models.astropt.retrain_spectra_images import main

if __name__ == "__main__":
    # Inject default config if not present
    if "--config" not in sys.argv:
        root_dir = Path(__file__).resolve().parents[4]
        config_path = root_dir / "configs" / "retrain" / "astropt.yaml"
        if config_path.exists():
            print(f"Injecting default config: {config_path}")
            sys.argv.extend(["--config", str(config_path)])

    main()
