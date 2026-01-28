"""
Foundation Models Benchmark (FMB)

Module: fmb.viz.spectrum
Description: Spectrum extraction and visualization utilities
"""

from typing import Optional, Tuple

import numpy as np
import torch

REST_LINES = {
    r"Ly$\alpha$": 1216.0,
    r"C IV": 1549.0,
    r"C III]": 1909.0,
    r"Mg II": 2798.0,
    r"[O II]": 3727.0,
    r"[Ne III]": 3869.0,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0,
    r"[O III]": 4959.0,
    r"[O III]": 5007.0,
    r"[N II]": 6548.0,
    r"H$\alpha$": 6563.0,
    r"[N II]": 6584.0,
    r"[S II]": 6717.0,
    r"[S II]": 6731.0,
}

LATEX_REST_LINES = {
    r"Ly$\alpha$": 1216.0,
    "C IV": 1549.0,
    "C III]": 1909.0,
    "Mg II": 2798.0,
    "[O II]": 3727.0,
    "[Ne III]": 3869.0,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4341.0,
    r"H$\beta$": 4861.0,
    "[O III]": 4959.0,
    "[O III]": 5007.0,
    "[N II]": 6548.0,
    r"H$\alpha$": 6563.0,
    "[N II]": 6584.0,
    "[S II]": 6717.0,
    "[S II]": 6731.0,
}


def extract_spectrum(sample: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract wavelength and flux from a sample dictionary.
    Returns: (wavelength, flux) as numpy arrays.
    """
    spec = sample.get("spectrum")
    if spec is None:
        return None, None

    flux = spec.get("flux")
    if flux is None:
        return None, None

    if isinstance(flux, torch.Tensor):
        flux_np = flux.detach().cpu().numpy()
    else:
        flux_np = np.asarray(flux)
    flux_np = np.squeeze(flux_np)

    wavelength = spec.get("wavelength")
    if wavelength is None:
        wavelength_np = np.arange(len(flux_np))
    else:
        if isinstance(wavelength, torch.Tensor):
            wavelength_np = wavelength.detach().cpu().numpy()
        else:
            wavelength_np = np.asarray(wavelength)
        wavelength_np = np.squeeze(wavelength_np)

    return wavelength_np, flux_np
