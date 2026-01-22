
import pytest
import subprocess
import tempfile
import shutil
import os
import sys

class TestAstroPTRetraining:
    
    @pytest.fixture
    def mock_args(self):
        # Create a temporary directory for output
        self.test_dir = tempfile.mkdtemp()
        yield self.test_dir
        # Cleanup
        shutil.rmtree(self.test_dir)

    def test_run_training_minimal(self, mock_args):
        """Test that main() runs with minimal steps using subprocess."""
        
        output_dir = os.path.join(mock_args, "astropt_test")
        script_path = os.path.join("src", "fmb", "models", "astropt", "retrain_spectra_images.py")
        
        # Ensure script exists
        assert os.path.exists(script_path), f"Script not found at {script_path}"
        
        # Arguments to pass
        cmd = [
            sys.executable, script_path,
            "--out-dir", output_dir,
            "--batch-size", "2",
            "--grad-accum", "1", 
            "--max-iters", "2", # Only 2 steps
            "--warmup-iters", "0",
            "--lr-decay-iters", "2",
            "--eval-interval", "1",
            "--log-interval", "1",
            "--image-size", "64",
            "--device", "cpu", 
            "--no-compile",
        ]
        
        # Run via subprocess
        # We need to ensure PYTHONPATH includes the current root so it can find things if needed?
        # The script does its own path setup, so it should be fine.
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # print output for debugging if it fails
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
        assert result.returncode == 0, f"Script failed with code {result.returncode}"
        
        # Verify output
        assert os.path.exists(output_dir)
        # Check for checkpoint
        # Note: script saves ckpt_final.pt at the end
        assert os.path.exists(os.path.join(output_dir, "ckpt_final.pt"))
        
        # Check if log file exists (starts with training_log_)
        log_files = [f for f in os.listdir(output_dir) if f.startswith("training_log_")]
        assert len(log_files) > 0
        
        # Check content of log file for summary
        with open(os.path.join(output_dir, log_files[0]), 'r') as f:
            content = f.read()
            assert "TRAINING COMPLETE SUMMARY" in content
            assert "Total Iterations: 2" in content

