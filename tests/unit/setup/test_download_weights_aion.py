import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

from tests.base import FMBTestCase
from fmb.setup.download_weights_aion import download_aion_model

class TestDownloadWeightsAIONUnit(FMBTestCase):
    """Unit tests for AION weight downloading logic (mocked)."""
    
    @patch("fmb.setup.download_weights_aion.snapshot_download")
    def test_download_aion_calls_snapshot_download(self, mock_snapshot_download):
        """Test that download_aion_model calls snapshot_download with correct args."""
        repo_id = "test/repo"
        revision = "main"
        dest_dir = self.test_dir / "aion"
        
        download_aion_model(repo_id, revision, dest_dir)
        
        mock_snapshot_download.assert_called_once_with(
            repo_id=repo_id,
            revision=revision,
            local_dir=dest_dir,
            local_dir_use_symlinks=False
        )
        self.assertTrue(dest_dir.exists())

    @patch("fmb.setup.download_weights_aion.load_paths")
    @patch("fmb.setup.download_weights_aion.download_aion_model")
    @patch("fmb.setup.download_weights_aion.prime_aion_codecs")
    def test_main_calls_download_when_empty(self, mock_prime, mock_download, mock_load_paths):
        """Test main triggers download if directory is empty."""
        # Setup mock paths
        mock_paths = MagicMock()
        target_dir = self.test_dir / "target_weights"
        target_dir.mkdir() # Empty dir
        mock_paths.base_weights_aion = target_dir
        mock_load_paths.return_value = mock_paths
        
        from fmb.setup.download_weights_aion import main
        
        # Patch argparse
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = MagicMock(
                repo="aion/repo", 
                revision=None, 
                force_codecs=False
            )
            main()
            
            mock_download.assert_called_once()
            mock_prime.assert_called_once()

    @patch("fmb.setup.download_weights_aion.load_paths")
    @patch("fmb.setup.download_weights_aion.download_aion_model")
    @patch("fmb.setup.download_weights_aion.prime_aion_codecs")
    def test_main_skips_download_when_exists(self, mock_prime, mock_download, mock_load_paths):
        """Test main skips download if directory has content."""
        # Setup mock paths
        mock_paths = MagicMock()
        target_dir = self.test_dir / "target_weights"
        self.create_dummy_file(target_dir / "model.safetensors") # Not empty
        mock_paths.base_weights_aion = target_dir
        mock_load_paths.return_value = mock_paths
        
        from fmb.setup.download_weights_aion import main
        
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = MagicMock(
                repo="aion/repo", 
                revision=None, 
                force_codecs=False
            )
            main()
            
            mock_download.assert_not_called()
            # Prime shouldn't be called unless forced
            mock_prime.assert_not_called()

if __name__ == "__main__":
    unittest.main()
