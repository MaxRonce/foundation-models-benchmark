import unittest
import shutil
import tempfile
from pathlib import Path
import sys

# Ensure src is in path for all tests
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

class FMBTestCase(unittest.TestCase):
    """Base class for FMB tests with common setup/teardown."""
    
    def setUp(self):
        # Create a temporary directory for each test
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        # Cleanup temporary directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
    def create_dummy_file(self, path: Path, content: str = ""):
        """Helper to create dummy files."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
