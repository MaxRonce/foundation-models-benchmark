import unittest
from unittest.mock import patch, MagicMock
import sys

from tests.base import FMBTestCase
from fmb.setup.check_environment_aion import check_hf_auth, check_torch, check_aion

class TestCheckEnvironmentUnit(FMBTestCase):
    """Unit tests for environment checks (mocked)."""

    @patch("fmb.setup.check_environment_aion.torch")
    def test_check_torch(self, mock_torch):
        """Test check_torch prints version and checks cuda."""
        mock_torch.__version__ = "2.0.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        
        check_torch("cuda")
        
        mock_torch.cuda.is_available.assert_called()
        mock_torch.randn.assert_called()

    @patch("fmb.setup.check_environment_aion.HfApi")
    def test_check_hf_auth_success(self, mock_hf_api_cls):
        """Test successful HF auth check."""
        mock_api = MagicMock()
        mock_api.whoami.return_value = {"name": "TestUser", "orgs": []}
        mock_hf_api_cls.return_value = mock_api
        
        check_hf_auth()
        
        mock_api.whoami.assert_called_once()

    def test_check_aion_loads_model(self):
        """Test check_aion attempts to load model."""
        # Mock the aion module and its AION class
        mock_aion_module = MagicMock()
        mock_aion_cls = MagicMock()
        mock_aion_module.AION = mock_aion_cls
        
        mock_model = MagicMock()
        mock_aion_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        # Patch import system AND hf_hub_download
        with patch.dict(sys.modules, {"aion": mock_aion_module}), \
             patch("huggingface_hub.hf_hub_download") as mock_hf_download, \
             patch("builtins.open", unittest.mock.mock_open(read_data='{"test": 1}')):
            
            from fmb.setup.check_environment_aion import check_aion
            
            mock_hf_download.return_value = "dummy_config.json"
            model_name = "test/aion"
            device = "cpu"
            
            with patch("builtins.print"): # Suppress print
                check_aion(model_name, None, device, skip_codecs=True)
            
            mock_hf_download.assert_called_with(unittest.mock.ANY, "config.json")
            mock_aion_cls.from_pretrained.assert_called()
            mock_model.to.assert_called_with(device)
            mock_model.eval.assert_called()

if __name__ == "__main__":
    unittest.main()
