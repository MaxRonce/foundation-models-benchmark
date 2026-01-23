
import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml
import shutil
import tempfile
import argparse
from unittest.mock import MagicMock, patch

# Ensure src is in pythonpath
src_path = Path(__file__).resolve().parents[4] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    
# Mock external/AION/aion imports BEFORE importing module under test
sys.modules["aion"] = MagicMock()
sys.modules["aion.codecs"] = MagicMock()
sys.modules["aion.codecs.config"] = MagicMock()
sys.modules["aion.codecs.preprocessing"] = MagicMock()
sys.modules["aion.codecs.preprocessing.image"] = MagicMock()
sys.modules["aion.modalities"] = MagicMock()

# Mock specific attributes used
# We need these to be actual classes or mocks that can be instantiated
class MockEuclidImage:
    def __init__(self, flux, bands): self.flux = flux; self.bands = bands
class MockHSCImage:
    def __init__(self, flux, bands): self.flux = flux; self.bands = bands
sys.modules["aion.modalities"].EuclidImage = MockEuclidImage
sys.modules["aion.modalities"].HSCImage = MockHSCImage

# Mock ImageCodec
class MockCodec(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(5, 10) # Fake encoder
        self.decoder = nn.Linear(10, 5) # Fake decoder
    def encode(self, x): return torch.randn(x.flux.shape[0], 10)
    def decode(self, x, bands): return MockHSCImage(torch.randn(x.shape[0], 5, 32, 32), bands)
    def load_state_dict(self, state, strict=False): return [], []

sys.modules["aion.codecs"].ImageCodec = MockCodec
sys.modules["aion.codecs.config"].HF_REPO_ID = "fake/repo"

# Import the module under test
# We use a try-except block here in case the import fails due to other side effects
try:

    from fmb.models.aion.retrain_aion import (
        TrainingConfig, 
        parse_args, 
        EuclidToHSC, 
        HSCToEuclid, 
        load_frozen_codec,
        EUCLID_BANDS
    )
    from fmb.data.datasets import AionDataset, FMBDataConfig
except ImportError:
    TrainingConfig = MagicMock()
    EuclidToHSC = MagicMock()
    HSCToEuclid = MagicMock()
    AionDataset = MagicMock()
    FMBDataConfig = MagicMock()

class TestRetrainAion(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / "test_config.yaml"
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_training_config_defaults(self):
        config = TrainingConfig()
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.epochs, 15)
        self.assertEqual(config.resize, 96)
        self.assertEqual(config.device, "cuda")
        self.assertAlmostEqual(config.learning_rate, 1e-4)

    def test_config_overrides(self):
        # Direct instantiation test
        config = TrainingConfig(batch_size=32, device="cpu")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.device, "cpu")

    def test_unet_shapes(self):
        # Test EuclidToHSC (4 -> 5 channels)
        model = EuclidToHSC(hidden=8, use_checkpointing=False)
        x = torch.randn(2, 4, 64, 64)
        y = model(x)
        self.assertEqual(y.shape, (2, 5, 64, 64))
        
        # Test HSCToEuclid (5 -> 4 channels)
        model2 = HSCToEuclid(hidden=8, use_checkpointing=False)
        x2 = torch.randn(2, 5, 64, 64)
        y2 = model2(x2)
        self.assertEqual(y2.shape, (2, 4, 64, 64))

    @patch("fmb.data.datasets.EuclidDESIDataset")
    def test_dataset_preprocessing(self, mock_dataset_cls):
        # Mock the underlying dataset
        mock_base = MagicMock()
        mock_base.__len__.return_value = 10
        
        # Mock a sample
        # 4 bands, random data
        sample = {
            "vis_image": torch.randn(100, 100),
            "nisp_y_image": torch.randn(100, 100),
            "nisp_j_image": torch.randn(100, 100),
            "nisp_h_image": torch.randn(100, 100)
        }
        mock_base.__getitem__.return_value = sample
        mock_dataset_cls.return_value = mock_base
        
        config = FMBDataConfig(split="train", image_size=32)
        dataset = AionDataset(config=config)
        
        self.assertEqual(len(dataset), 10)
        
        item = dataset[0]
        # Check return type is mocked EuclidImage
        self.assertIsInstance(item, MockEuclidImage)
        # Check shape after resize
        self.assertEqual(item.flux.shape, (4, 32, 32))
        # Check bands
        self.assertEqual(item.bands, EUCLID_BANDS)

    @patch("fmb.models.aion.retrain_aion.hf_hub_download")
    @patch("fmb.models.aion.retrain_aion.st.load_file")
    @patch("builtins.open")
    @patch("json.load")
    def test_training_step_integration(self, mock_json, mock_open, mock_st_load, mock_hf):
        # Setup mocks for codec loading
        mock_json.return_value = {
            "quantizer_levels": 1,
            "hidden_dims": 1, 
            "multisurvey_projection_dims": 1,
            "n_compressions": 1,
            "num_consecutive": 1,
            "embedding_dim": 1,
            "range_compression_factor": 1,
            "mult_factor": 1
        }
        mock_st_load.return_value = {} # Empty state dict
        
        # Setup simple models
        device = torch.device("cpu")
        euclid_to_hsc = EuclidToHSC(hidden=2, use_checkpointing=False).to(device)
        hsc_to_euclid = HSCToEuclid(hidden=2, use_checkpointing=False).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(list(euclid_to_hsc.parameters()), lr=1e-3)
        
        # Fake batch
        x_mb = torch.randn(2, 4, 32, 32).to(device)
        
        # Fake Codec Bridge (identity for test)
        def codec_bridge(hsc_flux): return hsc_flux
        
        # Forward pass logic from script
        hsc_like = euclid_to_hsc(x_mb)
        self.assertEqual(hsc_like.shape, (2, 5, 32, 32))
        
        hsc_dec = codec_bridge(hsc_like)
        euclid_rec = hsc_to_euclid(hsc_dec)
        self.assertEqual(euclid_rec.shape, (2, 4, 32, 32))
        
        loss = criterion(euclid_rec, x_mb)
        loss.backward()
        optimizer.step()
        
        self.assertFalse(torch.isnan(loss))

if __name__ == "__main__":
    unittest.main()
