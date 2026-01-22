#!/usr/bin/env python3
"""
Entry point script to run the FMB CLI without installing the package.
Usage: python fmb.py [command]
"""
import sys
import os
from pathlib import Path

# Add src to pythonpath so we can import 'fmb' package
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

try:
    from fmb.cli import app
except ImportError as e:
    print(f"Error importing fmb.cli: {e}")
    print("Ensure you are running this script from the project root and 'src/fmb' exists.")
    sys.exit(1)

if __name__ == "__main__":
    app()
