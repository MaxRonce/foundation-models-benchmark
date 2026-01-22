import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scratch.load_display_data_hsc import HSCDataset, _maybe_switch_to_agg


def compute_flux_distribution(nsample=20, cache_dir=None, save_path=None):
    print(f"[HSC] Computing flux distribution for {nsample} samples (i-band only)...")
    ds = HSCDataset(
        split="train",
        cache_dir=cache_dir,
        streaming=True,
        max_items=nsample,
    )

    all_fluxes = []

    for i, sample in enumerate(ds):
        img_i = sample.get("hsc_i")
        if img_i is None:
            print(f"[HSC] Warning: missing i-band for sample {i}, skipping.")
            continue

        flux = img_i.numpy().flatten()
        flux = flux[np.isfinite(flux)]  # remove NaN or inf
        all_fluxes.append(flux)

        print(f"[HSC] Collected fluxes from sample {i}: {flux.size} pixels")

    if not all_fluxes:
        raise RuntimeError("No valid i-band data found in the given samples.")

    all_fluxes = np.concatenate(all_fluxes)
    print(f"[HSC] Combined {len(all_fluxes):,} pixel flux values.")

    # Plot histogram
    plt.figure(figsize=(7, 5))
    plt.hist(all_fluxes, bins=200, color="steelblue", alpha=0.7)
    plt.xlabel("Flux value")
    plt.ylabel("Pixel count")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"HSC i-band flux distribution ({nsample} samples)")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[HSC] Histogram saved to {save_path}")

    mean = np.mean(all_fluxes)
    median = np.median(all_fluxes)
    std = np.std(all_fluxes)
    print(f"Mean: {mean:.4f}, Median: {median:.4f}, Std: {std:.4f}")

    plt.show()
    plt.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute and plot HSC i-band flux histogram."
    )
    p.add_argument("--nsample", type=int, default=20,
                   help="Number of samples to use (default: 20)")
    p.add_argument("--cache-dir", type=str,
                   default="/pbs/throng/training/astroinfo2025/model/hsc/hf_home/datasets",
                   help="Cache directory for the HSC dataset")
    p.add_argument("--save", type=str, default=None,
                   help="Optional path to save the histogram (e.g. histo.png)")
    p.add_argument("--no-gui", action="store_true",
                   help="Use non-interactive Agg backend (for clusters)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _maybe_switch_to_agg(args.no_gui)

    compute_flux_distribution(
        nsample=args.nsample,
        cache_dir=args.cache_dir,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
