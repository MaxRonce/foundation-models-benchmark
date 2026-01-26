"""
2026-01-26
astropt/retrain.py

Simplified entry point for AstroPT multimodal training.
Delegates to the main script in retrain_spectra_images.py.
"""

from fmb.models.astropt.retrain_spectra_images import main

if __name__ == "__main__":
    main()
