import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

from tests.base import FMBTestCase
from fmb.setup.download_weights_astroclip import download_astroclip_weights

class TestDownloadWeightsAstroCLIPUnit(FMBTestCase):
    """Unit tests for AstroCLIP weight downloading logic (mocked)."""

    @patch("fmb.setup.download_weights_astroclip.hf_hub_download")
    def test_download_astroclip_calls_hf_hub_download(self, mock_hf_download):
        """Test that download_astroclip_weights calls hf_hub_download for each file."""
        dest_dir = self.test_dir / "astroclip"
        
        download_astroclip_weights(dest_dir)
        
        # Should be called 3 times
        self.assertEqual(mock_hf_download.call_count, 3)
        self.assertTrue(dest_dir.exists())

    @patch("fmb.setup.download_weights_astroclip.load_paths")
    @patch("fmb.setup.download_weights_astroclip.download_astroclip_weights")
    def test_main_calls_download(self, mock_download, mock_load_paths):
        """Test main triggers download function."""
        mock_paths = MagicMock()
        mock_paths.base_weights_astroclip = self.test_dir / "astroclip_weights"
        mock_load_paths.return_value = mock_paths
        
        from fmb.setup.download_weights_astroclip import main
        
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            # Setting default dest in mock
            mock_parse_args.return_value = MagicMock(dest=mock_paths.base_weights_astroclip)
            main()
            
            mock_download.assert_called_once_with(mock_paths.base_weights_astroclip)

if __name__ == "__main__":
    unittest.main()
