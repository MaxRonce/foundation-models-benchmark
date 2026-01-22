"""
Script to compute and compare flux distributions between HSC and Euclid datasets.
It can also estimate affine rescaling parameters to match Euclid flux statistics to HSC.

Usage:
    python -m scratch.compute_flux_history --both --rescale --nsample 500 \
        --hsc-cache-dir /path/to/hsc \
        --euclid-cache-dir /path/to/euclid \
        --save hsc_vs_euclid_flux_hist.png --no-gui
"""
import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------- Backend helper (independent of loaders) ----------
def _maybe_switch_to_agg(no_gui: bool):
    if no_gui:
        matplotlib.use("Agg")
    else:
        try:
            import tkinter  # noqa: F401
        except Exception:
            matplotlib.use("Agg")

# ---------- Conditional imports of loaders ----------
HSCDataset = None
EuclidDESIDataset = None

# Adjust these import paths to your repo layout if needed:
#  - HSC:     scratch.load_display_data_hsc.HSCDataset
#  - Euclid:  load_display_data.EuclidDESIDataset (or scratch.load_display_data)
try:
    from scratch.load_display_data_hsc import HSCDataset as _HSCDataset
    HSCDataset = _HSCDataset
except Exception:
    pass

try:
    from load_display_data import EuclidDESIDataset as _EuclidDESIDataset
    EuclidDESIDataset = _EuclidDESIDataset
except Exception:
    try:
        from scratch.load_display_data import EuclidDESIDataset as _EuclidDESIDataset
        EuclidDESIDataset = _EuclidDESIDataset
    except Exception:
        pass


# ---------- Utility ----------
def _as_numpy_1d(x):
    """Convert a tensor/array-like to a clean 1D numpy array without NaN/Inf."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x)
    x = x.ravel()
    x = x[np.isfinite(x)]
    return x


def _iter_hsc_samples(nsample, cache_dir, split="train"):
    """Generator over HSC samples."""
    if HSCDataset is None:
        raise ImportError(
            "HSCDataset not found. Make sure 'scratch.load_display_data_hsc' is importable."
        )
    ds = HSCDataset(
        split=split,
        cache_dir=cache_dir,
        streaming=True,
        max_items=nsample,
    )
    for sample in ds:
        yield sample


def _iter_euclid_samples(nsample, cache_dir, split="train_batch_1"):
    """Generator over Euclid samples."""
    if EuclidDESIDataset is None:
        raise ImportError(
            "EuclidDESIDataset not found. Make sure 'load_display_data.py' (or scratch.load_display_data) is importable."
        )
    ds = EuclidDESIDataset(split=split, cache_dir=cache_dir)
    total = len(ds)
    ns = total if (nsample is None or nsample <= 0 or nsample > total) else nsample
    for i in range(ns):
        yield ds[i]


# Keys for each dataset/band
HSC_KEYS = {
    "g": "hsc_g",
    "r": "hsc_r",
    "i": "hsc_i",
    "y": "hsc_y",
    "z": "hsc_z",
}
EUC_KEYS = {
    "vis": "vis_image",
    "y": "nisp_y_image",
    "j": "nisp_j_image",
    "h": "nisp_h_image",
}

def _title(dataset: str, band: str):
    if dataset == "hsc":
        return f"HSC {band}-band"
    if dataset == "euclid":
        band_name = {"vis": "VIS", "y": "NISP-Y", "j": "NISP-J", "h": "NISP-H"}[band]
        return f"Euclid {band_name}"
    return f"{dataset}:{band}"

def _key_for(dataset: str, band: str):
    if dataset == "hsc":
        if band not in HSC_KEYS:
            raise ValueError(f"HSC band '{band}' not available in HSC_KEYS.")
        return HSC_KEYS[band]
    if dataset == "euclid":
        if band not in EUC_KEYS:
            raise ValueError(f"Euclid band '{band}' not available in EUC_KEYS.")
        return EUC_KEYS[band]
    raise ValueError("dataset must be 'hsc' or 'euclid'.")

def _collect_fluxes(dataset: str, band: str, nsample: int, cache_dir: str, split: str):
    """
    Return (title_prefix, key, flux_values[np.ndarray], count_int, split_used)
    """
    dataset = dataset.lower()
    band = band.lower() if band else band
    if dataset not in {"hsc", "euclid"}:
        raise ValueError("--dataset must be 'hsc' or 'euclid'")

    title_prefix = _title(dataset, band)
    key = _key_for(dataset, band)

    if dataset == "hsc":
        split = split or "train"
        cache_dir = cache_dir or "/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets"
        iterator = _iter_hsc_samples(nsample=nsample, cache_dir=cache_dir, split=split)
    else:
        split = split or "train_batch_1"
        cache_dir = cache_dir or "/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets"
        iterator = _iter_euclid_samples(nsample=nsample, cache_dir=cache_dir, split=split)

    print(f"[{title_prefix}] Computing flux distribution for {nsample} samples...")
    print(f"[{title_prefix}] Using cache_dir = {cache_dir}")
    print(f"[{title_prefix}] Using split = {split}")

    all_fluxes = []
    count = 0
    for i, sample in enumerate(iterator):
        img = sample.get(key)
        if img is None:
            print(f"[{title_prefix}] Warning: missing '{key}' for sample {i}, skipping.")
            continue
        flux = _as_numpy_1d(img)
        if flux.size == 0:
            print(f"[{title_prefix}] Warning: empty/invalid '{key}' for sample {i}, skipping.")
            continue
        all_fluxes.append(flux)
        count += 1
        if i < 5 or (i + 1) % 50 == 0:
            print(f"[{title_prefix}] Collected fluxes from sample {i}: {flux.size} pixels")

    if not all_fluxes:
        raise RuntimeError(f"[{title_prefix}] No valid '{key}' data found in the given samples.")

    all_fluxes = np.concatenate(all_fluxes)
    print(f"[{title_prefix}] Combined {len(all_fluxes):,} pixel flux values from {count} samples.")
    return title_prefix, key, all_fluxes, count, split


def _stats(arr: np.ndarray):
    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))


def _common_log_bins(*arrays, nbins=200):
    """Build common positive log-spaced bins covering all given arrays."""
    concat = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size > 0])
    concat = concat[concat > 0]
    if concat.size == 0:
        # fallback (rare)
        return np.linspace(0.0, 1.0, nbins)
    mn = np.min(concat)
    mx = np.max(concat)
    if not np.isfinite(mn) or not np.isfinite(mx) or mn <= 0 or mx <= mn:
        mn, mx = 1e-12, 1.0
    return np.logspace(np.log10(mn), np.log10(mx), nbins)


def _learn_affine_match(source_flux: np.ndarray, target_flux: np.ndarray):
    """
    Learn a,b so that a*target + b matches source in mean/std.
    Returns (a, b, stats_dict)
    """
    s_mean, s_median, s_std = _stats(source_flux)
    t_mean, t_median, t_std = _stats(target_flux)
    if t_std == 0:
        raise RuntimeError("Target std is zero; cannot learn affine scaling.")
    a = s_std / t_std
    b = s_mean - a * t_mean
    t_after = a * target_flux + b
    a_mean, a_median, a_std = _stats(t_after)
    return a, b, dict(source=(s_mean, s_median, s_std),
                      target=(t_mean, t_median, t_std),
                      after=(a_mean, a_median, a_std))


def _save_path_for(base_or_dir: str, pair_tag: str, rescaled: bool = False):
    """
    Build a path for saving figures.
    - If base_or_dir ends with an image extension, use it as base name.
    - Else treat as directory and build '<dir>/<pair_tag>.png'
    - If rescaled=True, add '_rescaled' suffix.
    """
    default_name = f"{pair_tag}.png"
    if not base_or_dir:
        return None
    # Directory?
    if os.path.isdir(base_or_dir) or not os.path.splitext(base_or_dir)[1]:
        path = os.path.join(base_or_dir, default_name)
    else:
        base, ext = os.path.splitext(base_or_dir)
        path = f"{base}_{pair_tag}{ext or '.png'}"
    if rescaled:
        b, ext = os.path.splitext(path)
        path = f"{b}_rescaled{ext or '.png'}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


# ---------- Single-dataset (kept for completeness) ----------
def compute_flux_distribution(
    dataset: str = "euclid",
    band: str = "vis",
    nsample: int = 20,
    cache_dir: str = None,
    split: str = None,
    save_path: str = None,
):
    title_prefix, key, all_fluxes, count, split = _collect_fluxes(dataset, band, nsample, cache_dir, split)
    mean, median, std = _stats(all_fluxes)
    print(f"[{title_prefix}] Mean: {mean:.6g}, Median: {median:.6g}, Std: {std:.6g}")

    plt.figure(figsize=(7, 5))
    plt.hist(all_fluxes, bins=200, alpha=0.7)
    plt.xlabel("Flux value")
    plt.ylabel("Pixel count")
    # keep y log as in your latest version
    plt.yscale("log")
    plt.title(f"{title_prefix} flux distribution (n={count} samples; split={split})")
    plt.grid(True, linestyle="--", alpha=0.5)

    ax = plt.gca()
    ax.text(0.97, 0.97, f"mean: {mean:.4g}\nmedian: {median:.4g}\nstd: {std:.4g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[{title_prefix}] Histogram saved to {save_path}")
    plt.show()
    plt.close()


# ---------- New: multi-pair comparison & optional rescaling ----------
def compute_flux_distribution_pairs(
    nsample: int = 20,
    hsc_cache_dir: str = "/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets",
    euclid_cache_dir: str = "/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets",
    hsc_split: str = "train",
    euclid_split: str = "train_batch_1",
    save_base: str = None,
    rescale: bool = False,
):
    """
    Do four mappings and save four diagrams:
      VIS -> g, Y -> r, J -> y, H -> z
    For each pair: make a raw-overlap figure, and if rescale=True, a second '_rescaled' figure.
    """
    pairs = [
        ("vis", "g"),  # Euclid VIS -> HSC g
        ("y",   "r"),  # NISP-Y   -> HSC r
        ("j",   "i"),  # NISP-J   -> HSC y
        ("h",   "z"),  # NISP-H   -> HSC z
    ]

    for e_band, h_band in pairs:
        e_title, _, e_flux, e_n, e_split_used = _collect_fluxes(
            dataset="euclid", band=e_band, nsample=nsample,
            cache_dir=euclid_cache_dir, split=euclid_split
        )
        h_title, _, h_flux, h_n, h_split_used = _collect_fluxes(
            dataset="hsc", band=h_band, nsample=nsample,
            cache_dir=hsc_cache_dir, split=hsc_split
        )

        # Stats
        h_mean, h_median, h_std = _stats(h_flux)
        e_mean, e_median, e_std = _stats(e_flux)

        print(f"[{h_title}] Mean={h_mean:.6g}, Median={h_median:.6g}, Std={h_std:.6g}")
        print(f"[{e_title}] Mean={e_mean:.6g}, Median={e_median:.6g}, Std={e_std:.6g}")

        # Raw overlap
        bins = _common_log_bins(h_flux, e_flux, nbins=200)
        plt.figure(figsize=(8, 5.5))
        plt.hist(h_flux, bins=bins, histtype="step", linewidth=1.5, label=f"{h_title} (n={h_n})", alpha=0.95)
        plt.hist(e_flux, bins=bins, histtype="step", linewidth=1.5, label=f"{e_title} (n={e_n})", alpha=0.95)
        plt.xlabel("Flux value")
        plt.ylabel("Pixel count")
        plt.yscale("log")
        plt.title(f"Overlap — {e_title} (split={e_split_used}) vs {h_title} (split={h_split_used})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower left")

        ax = plt.gca()
        ax.text(0.03, 0.97, f"{h_title}\nmean: {h_mean:.4g}\nmedian: {h_median:.4g}\nstd: {h_std:.4g}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"))
        ax.text(0.97, 0.97, f"{e_title}\nmean: {e_mean:.4g}\nmedian: {e_median:.4g}\nstd: {e_std:.4g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"))

        pair_tag = f"{e_band}2{h_band}"
        raw_path = _save_path_for(save_base or "", pair_tag, rescaled=False)
        if raw_path:
            plt.tight_layout(); plt.savefig(raw_path, dpi=150)
            print(f"[BOTH] Overlapped histogram saved to {raw_path}")
        plt.show(); plt.close()

        if rescale:
            a, b, info = _learn_affine_match(source_flux=h_flux, target_flux=e_flux)
            (s_mean, s_median, s_std) = info["source"]
            (t_mean, t_median, t_std) = info["target"]
            (a_mean, a_median, a_std) = info["after"]

            print(f"[RESCALE {e_band.upper()}→{h_band}] Learned affine parameters:")
            print(f"  a = {a:.8g}, b = {b:.8g}")
            print(f"  target/raw  mean={t_mean:.6g}, median={t_median:.6g}, std={t_std:.6g}")
            print(f"  source/ref  mean={s_mean:.6g}, median={s_median:.6g}, std={s_std:.6g}")
            print(f"  after/fit   mean={a_mean:.6g}, median={a_median:.6g}, std={a_std:.6g}")

            e_flux_rescaled = a * e_flux + b
            bins2 = _common_log_bins(h_flux, e_flux, e_flux_rescaled, nbins=200)

            plt.figure(figsize=(8, 5.5))
            plt.hist(h_flux, bins=bins2, histtype="step", linewidth=1.6, label=f"{h_title} (ref)", alpha=0.95)
            plt.hist(e_flux, bins=bins2, histtype="step", linewidth=1.2, label=f"{e_title} (raw)", alpha=0.85)
            plt.hist(e_flux_rescaled, bins=bins2, histtype="step", linewidth=1.6,
                     label=f"{e_title} → {h_title} (rescaled)", alpha=0.95)
            plt.xlabel("Flux value")
            plt.ylabel("Pixel count")
            plt.yscale("log")
            plt.title(f"{e_title} → {h_title} affine rescaling (mean/std match)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="lower left")

            ax = plt.gca()
            ax.text(0.97, 0.97, f"a = {a:.4g}\nb = {b:.4g}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"))

            rescaled_path = _save_path_for(save_base or "", pair_tag, rescaled=True)
            if rescaled_path:
                plt.tight_layout(); plt.savefig(rescaled_path, dpi=150)
                print(f"[RESCALE] Overlapped histogram saved to {rescaled_path}")
            plt.show(); plt.close()


# ---------- CLI ----------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute and plot pixel flux histograms for HSC and Euclid bands; "
                    "supports multi-pair comparisons and optional Euclid→HSC rescaling."
    )
    # Single-dataset (unchanged)
    p.add_argument("--dataset", type=str, choices=["hsc", "euclid"], default="euclid",
                   help="Dataset to use when not using --both (default: euclid)")
    p.add_argument("--band", type=str, default="vis",
                   help="Band to use for single-dataset mode. HSC: g/r/i/y/z. Euclid: vis/y/j/h.")
    p.add_argument("--nsample", type=int, default=20,
                   help="Number of samples to use per dataset (default: 20)")
    p.add_argument("--split", type=str, default=None,
                   help="Dataset split when not using --both "
                        "(HSC: 'train'; Euclid: 'train_batch_1' by default)")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="HF cache directory when not using --both")

    # Multi-pair mode
    p.add_argument("--both", action="store_true",
                   help="If set, run four mappings: VIS→g, Y→r, J→y, H→z and save four diagrams.")
    p.add_argument("--rescale", action="store_true",
                   help="With --both: learn affine (Euclid→HSC) per pair, and save extra *_rescaled figures.")
    p.add_argument("--hsc-cache-dir", type=str,
                   default="/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets",
                   help="HSC HF cache directory (used with --both).")
    p.add_argument("--euclid-cache-dir", type=str,
                   default="/pbs/throng/training/astroinfo2025/model/euclid_desi/hf_home/datasets",
                   help="Euclid HF cache directory (used with --both).")
    p.add_argument("--hsc-split", type=str, default="train",
                   help="HSC split for --both (default: train).")
    p.add_argument("--euclid-split", type=str, default="train_batch_1",
                   help="Euclid split for --both (default: train_batch_1).")

    p.add_argument("--save", type=str, default=None,
                   help=("If a file, used as base name and suffixed per pair "
                         "(e.g., base_vis2g.png). If a directory, figures are saved there."))
    p.add_argument("--no-gui", action="store_true",
                   help="Use non-interactive Agg backend (clusters/headless)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _maybe_switch_to_agg(args.no_gui)

    if args.both:
        compute_flux_distribution_pairs(
            nsample=args.nsample,
            hsc_cache_dir=args.hsc_cache_dir,
            euclid_cache_dir=args.euclid_cache_dir,
            hsc_split=args.hsc_split,
            euclid_split=args.euclid_split,
            save_base=args.save,
            rescale=args.rescale,
        )
        return

    # Single-dataset mode
    dataset = args.dataset.lower()
    band = args.band.lower() if args.band else None
    if dataset not in {"hsc", "euclid"}:
        raise ValueError("--dataset must be 'hsc' or 'euclid'.")
    if dataset == "hsc" and band not in HSC_KEYS:
        raise ValueError(f"HSC band must be one of {sorted(HSC_KEYS.keys())}")
    if dataset == "euclid" and band not in EUC_KEYS:
        raise ValueError(f"Euclid band must be one of {sorted(EUC_KEYS.keys())}")

    compute_flux_distribution(
        dataset=dataset,
        band=band,
        nsample=args.nsample,
        cache_dir=args.cache_dir,
        split=args.split,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
