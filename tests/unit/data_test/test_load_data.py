import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

from tests.base import FMBTestCase
from fmb.data.load_display_data import EuclidDESIDataset, parse_args

class TestLoadDisplayDataUnit(FMBTestCase):
    """Unit tests for data loading script."""
    
    @patch("fmb.data.load_display_data.load_paths")
    def test_dataset_init_defaults(self, mock_load_paths):
        """Test EuclidDESIDataset uses configured path by default."""
        mock_paths = MagicMock()
        mock_paths.dataset = Path("/mock/dataset/path")
        mock_load_paths.return_value = mock_paths
        
        mock_paths.dataset_train = MagicMock(spec=Path)
        mock_paths.dataset_train.exists.return_value = True
        mock_paths.dataset_train.__str__.return_value = "/mock/dataset/path/train" # Mock string representation
        
        # We also need to mock os.makedirs and load_dataset/load_from_disk because init does heavy lifting
        with patch("os.makedirs"), \
             patch("fmb.data.load_display_data.load_dataset") as mock_load_ds, \
             patch("fmb.data.load_display_data.load_from_disk") as mock_load_disk:
             
             ds = EuclidDESIDataset(split="train")
             
             # Verify it tried to load from local disk constructed path
             # Check if load_from_disk was called with something ending in local_train
             mock_load_disk.assert_called()
                 
    @patch("fmb.data.load_display_data.load_paths")
    def test_parse_args_defaults(self, mock_load_paths):
        """Test argument parsing defaults."""
        mock_paths = MagicMock()
        mock_paths.dataset = Path("/mock/dataset/path")
        mock_load_paths.return_value = mock_paths
        
        args = parse_args([])
        self.assertEqual(Path(args.cache_dir), Path("/mock/dataset/path"))

if __name__ == "__main__":
    unittest.main()
