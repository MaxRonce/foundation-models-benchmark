
import torch
import numpy as np
import subprocess
import os
from pathlib import Path

def create_dummy_pt(filename, key_pair):
    img_key, spec_key = key_pair
    
    # Create 10 objects
    # 5 perfectly aligned (sim = 1.0)
    # 5 perfectly misaligned (sim = 0.0 or -1.0)
    
    records = []
    for i in range(5):
        # Aligned
        vec = np.random.randn(10).astype(np.float32)
        records.append({
            "object_id": f"aligned_{i}",
            img_key: vec,
            spec_key: vec
        })
        
    for i in range(5):
        # Orthogonal (Misaligned)
        v1 = np.array([1, 0, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0, 0], dtype=np.float32)
        records.append({
            "object_id": f"misaligned_{i}",
            img_key: v1,
            spec_key: v2
        })
        
    torch.save(records, filename)
    print(f"Created {filename} with keys {key_pair}")

def run_test(filename, threshold_percent, expected_anomalies):
    print(f"\nRunning test on {filename}...")
    output_csv = "test_outliers.csv"
    
    cmd = [
        "python", "-m", "scratch.detect_cosine_anomalies",
        "--input", filename,
        "--output", output_csv,
        "--threshold-percent", str(threshold_percent)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Test failed!")
        print(result.stderr)
        return False
        
    # Check output
    with open(output_csv, 'r') as f:
        lines = f.readlines()
        # Header + anomalies
        count = len(lines) - 1
        
    print(f"Found {count} anomalies (Expected ~{expected_anomalies})")
    
    # Check if misaligned are in the top
    content = "".join(lines)
    if "misaligned_0" in content:
        print(" Correctly identified misaligned objects")
    else:
        print("❌ Failed to identify misaligned objects")
        return False

    return True

def main():
    # Test AstroPT keys
    create_dummy_pt("test_astropt.pt", ("embedding_images", "embedding_spectra"))
    if not run_test("test_astropt.pt", 50, 5):
        exit(1)
        
    # Test AION keys
    create_dummy_pt("test_aion.pt", ("embedding_hsc", "embedding_spectrum"))
    if not run_test("test_aion.pt", 50, 5):
        exit(1)
        
    # Clean up
    for f in ["test_astropt.pt", "test_aion.pt", "test_outliers.csv"]:
        if os.path.exists(f):
            os.remove(f)
            
    print("\n All tests passed!")

if __name__ == "__main__":
    main()
