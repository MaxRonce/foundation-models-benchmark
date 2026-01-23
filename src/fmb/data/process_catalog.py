
import numpy as np
from astropy.io import fits
from astropy.table import Table, join
import os

# Define paths
fits_dir = "/home/ronceray/AION/fits"
files = {
    "mer": os.path.join(fits_dir, "mer.fits"),
    "morph": os.path.join(fits_dir, "morph.fits"),
    "phys": os.path.join(fits_dir, "phys.fits"),
    "phz": os.path.join(fits_dir, "phz.fits")
}

def load_fits_as_table(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    return Table.read(path, format='fits')

print("Loading files...")
mer = load_fits_as_table(files["mer"])
morph = load_fits_as_table(files["morph"])
phys = load_fits_as_table(files["phys"])
phz = load_fits_as_table(files["phz"])

# Merge tables on object_id (or similar unique ID)
# First check what ID column is common.
common_id = "object_id" # or TARGETID?
# Inspect columns
if mer: print(f"MER ID: {'object_id' in mer.colnames} {'TARGETID' in mer.colnames}")
if morph: print(f"MORPH ID: {'object_id' in morph.colnames} {'TARGETID' in morph.colnames}")

# Helper to merge and handle duplicates
def merge_and_clean(left, right, suffixes, key=common_id):
    # Identify common columns that are NOT the key
    common_cols = [c for c in right.colnames if c in left.colnames and c != key]
    # Remove them from right
    right_clean = right.copy()
    right_clean.remove_columns(common_cols)
    return join(left, right_clean, keys=key, join_type='outer')

print("Merging tables...")
# Start with mer
main_table = mer
if morph:
    main_table = merge_and_clean(main_table, morph, ['_mer', '_morph'])
if phys:
    main_table = merge_and_clean(main_table, phys, ['', '_phys'])
if phz:
    main_table = merge_and_clean(main_table, phz, ['', '_phz'])


print(f"Merged table size: {len(main_table)}")

# --- Calculations ---
print("Calculating helper columns...")

# Probabilities
def calc_probs(t, cols):
    # Sum columns
    total = np.zeros(len(t))
    for c in cols:
        if c in t.colnames:
            total += t[c]
    
    # Avoid division by zero
    total[total == 0] = 1.0 # Or nan
    
    res = {}
    for c in cols:
        if c in t.colnames:
            res[f"prob_{c}"] = t[c] / total
    return res

if morph:
    # Bar probabilities
    bar_cols = ["bar_no", "bar_weak", "bar_strong"]
    bar_probs = calc_probs(main_table, bar_cols)
    for k, v in bar_probs.items():
        main_table[k] = v

    # Disk edge on
    edge_cols = ["disk_edge_on_yes", "disk_edge_on_no"]
    edge_probs = calc_probs(main_table, edge_cols)
    for k, v in edge_probs.items():
        main_table[k] = v

    # Spiral arms
    spiral_cols = ["has_spiral_arms_yes", "has_spiral_arms_no"]
    spiral_probs = calc_probs(main_table, spiral_cols)
    for k, v in spiral_probs.items():
        main_table[k] = v

    # Smooth or featured 
    # Use 'smooth_or_featured_smooth' directly (it's likely a probability or score)
    # The snippet divides by 100? "properties_vis["smooth_or_featured_smooth"]/100"
    # Inspect values first? Let's assume raw values are 0-1 or 0-100.
    pass

# Flux to Mag
# flux_detection_total -> mag
if "flux_detection_total" in main_table.colnames:
    flux = main_table["flux_detection_total"]
    # Handle <= 0 or NaNs
    valid = (flux > 0)
    mag = np.full(len(main_table), np.nan)
    mag[valid] = 23.9 - 2.5 * np.log10(flux[valid])
    main_table["mag_detection_total"] = mag

# --- Check for DESI columns ---
desi_phys = ["LOGM", "LOGSFR", "Z", "AGNFRAC", "NUVR", "RK", "UV", "VJ"]
desi_spec = ["HBETA_FLUX", "HBETA_EW", "HBETA_BROAD_FLUX", "HALPHA_FLUX", "DN4000"]
# Note: Check for variations like 'HBETA_FLUX' vs 'flux_hbeta' etc.

missing = []
for col in desi_phys + desi_spec:
    if col not in main_table.colnames:
        missing.append(col)

print(f"Missing columns: {missing}")

# --- Load and Merge DESI Catalog ---
desi_path = "/home/ronceray/AION/DESI_DR1_Euclid_Q1_dataset_catalog_physical_param.fits"
if os.path.exists(desi_path):
    print(f"Loading DESI catalog from {desi_path}...")
    desi_table = Table.read(desi_path, format='fits')
    
    # Ensure TARGETID is in main_table for joining
    if "TARGETID" not in main_table.colnames:
        print("Warning: TARGETID not found in main table. attempting to find it in source tables...")
        # (It should be there from the joins if we didn't drop it. We haven't dropped it yet.)
    
    print("Merging DESI data on TARGETID...")
    # Clean duplicates again just in case
    main_table = merge_and_clean(main_table, desi_table, ['', '_desi'], key='TARGETID')
else:
    print(f"Warning: DESI file {desi_path} not found.")

# --- Select columns to keep ---
# User requested specific parameters.
keep_cols = [
    "object_id", "TARGETID",
    "bar_no", "bar_weak", "bar_strong",
    "disk_edge_on_yes", "disk_edge_on_no",
    "has_spiral_arms_yes", "has_spiral_arms_no",
    "sersic_sersic_vis_index",
    "smooth_or_featured_smooth",
    "flux_detection_total", "mag_detection_total",
    "det_quality_flag", "spurious_flag",
    "phz_pp_median_stellarmass", "phz_pp_median_redshift", "phz_pp_median_sfr",
    "flux_h_sersic",
    "prob_bar_no", "prob_bar_weak", "prob_bar_strong",
    "prob_disk_edge_on_yes", "prob_disk_edge_on_no",
    "prob_has_spiral_arms_yes", "prob_has_spiral_arms_no"
]

# Add DESI columns
desi_cols = [
    "LOGM", "LOGSFR", "Z", "AGNFRAC", "NUVR", "RK", "UV", "VJ",
    "HBETA_FLUX", "HBETA_EW", "HBETA_BROAD_FLUX", "HALPHA_FLUX", "DN4000"
]
keep_cols.extend(desi_cols)
# Also add any columns strictly detected in the new file just in case of naming diffs
# (The user asked for specific ones, I verified they exist in step 55)

# Filter columns
final_cols = [c for c in keep_cols if c in main_table.colnames]
final_table = main_table[final_cols] 

print(f"Final table shape: {len(final_table)} rows, {len(final_table.colnames)} columns")

# Save the curated table
output_path = "/home/ronceray/AION/scratch/curated_catalog.fits"
try:
    final_table.write(output_path, format='fits', overwrite=True)
    print(f"Saved to {output_path}")
except Exception as e:
    print(f"Error saving file: {e}")


