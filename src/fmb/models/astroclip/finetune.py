"""
2026-01-23
astroclip/finetune.py


Description
-----------
Simplified entry point for AstroCLIP image encoder fine-tuning.
Delegates to the full implementation in finetune_image_encoder.py.

Usage
-----
python -m fmb.models.astroclip.finetune --checkpoint path/to/ckpt --epochs 5
python -m fmb.models.astroclip.finetune --config configs/astroclip.yaml
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parents[3]
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import and delegate to the full implementation
from fmb.models.astroclip.finetune_image_encoder import main

if __name__ == "__main__":
    # Inject default config if not present
    if "--config" not in sys.argv:
        root_dir = Path(__file__).resolve().parents[4]
        config_path = root_dir / "configs" / "retrain" / "astroclip.yaml"
        if config_path.exists():
            print(f"Injecting default config: {config_path}")
            sys.argv.extend(["--config", str(config_path)])

    main()
