import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

from tests.base import FMBTestCase
from fmb.setup.check_environment_astroclip import check_astroclip

class TestCheckEnvironmentAstroCLIPUnit(FMBTestCase):
    """Unit tests for AstroCLIP environment checks."""

    @patch("fmb.setup.check_environment_astroclip.torch.load")
    @patch("sys.exit")
    def test_check_astroclip_success(self, mock_exit, mock_torch_load):
        """Test successful weight verification."""
        model_dir = self.test_dir / "astroclip_weights"
        model_dir.mkdir()
        
        # Create dummy ckpt files
        for f in ["astrodino.ckpt", "specformer.ckpt", "astroclip.ckpt"]:
            self.create_dummy_file(model_dir / f)
            
        with patch("builtins.print"):
            check_astroclip(model_dir, "cpu")
            
        self.assertEqual(mock_torch_load.call_count, 3)
        mock_exit.assert_not_called()

    @patch("sys.exit")
    def test_check_astroclip_missing_files(self, mock_exit):
        """Test failure when files are missing."""
        model_dir = self.test_dir / "astroclip_weights"
        model_dir.mkdir()
        
        # Create only one file
        self.create_dummy_file(model_dir / "astrodino.ckpt")
        
        with patch("builtins.print"):
            check_astroclip(model_dir, "cpu")
            
        mock_exit.assert_called_with(1)

if __name__ == "__main__":
    unittest.main()
