"""
Foundation Models Benchmark (FMB)

Module: fmb.models.external_imports
Description: External submodule import handler
"""

import sys
from pathlib import Path
from typing import List


def get_repo_root() -> Path:
    """
    Get repository root directory.

    Returns
    -------
    Path
        Absolute path to repository root.
    """
    # This file is in src/fmb/models/, so go up 3 levels
    return Path(__file__).resolve().parents[3]


def setup_external_paths(libraries: List[str] = None) -> None:
    """
    Add external library paths to sys.path.

    Parameters
    ----------
    libraries : List[str], optional
        List of library names to add. Defaults to all: ['AION', 'astroPT', 'AstroCLIP'].

    Raises
    ------
    FileNotFoundError
        If external directory doesn't exist.

    Notes
    -----
    This function is idempotent: calling it multiple times is safe.
    """
    if libraries is None:
        libraries = ["AION", "astroPT", "AstroCLIP"]

    repo_root = get_repo_root()
    external_dir = repo_root / "external"

    if not external_dir.exists():
        raise FileNotFoundError(
            f"External directory not found: {external_dir}\n"
            "Initialize submodules with: git submodule update --init --recursive"
        )

    for lib in libraries:
        lib_path = external_dir / lib

        # Special handling for astroPT which has src/ subdirectory
        if lib == "astroPT":
            lib_path = lib_path / "src"

        if lib_path.exists() and str(lib_path) not in sys.path:
            sys.path.insert(0, str(lib_path))
            print(f" Added to sys.path: {lib_path}")
        elif not lib_path.exists():
            print(f"  Warning: {lib} not found at {lib_path}")
            print("   Run: git submodule update --init --recursive")


def check_external_available(library: str) -> bool:
    """
    Check if an external library is available for import.

    Parameters
    ----------
    library : str
        Library name ('AION', 'astroPT', or 'AstroCLIP').

    Returns
    -------
    bool
        True if library can be imported, False otherwise.
    """
    try:
        if library == "AION":
            import aion
        elif library == "astroPT":
            import astropt
        elif library == "AstroCLIP":
            import astroclip
        else:
            return False
        return True
    except ImportError:
        return False


# Auto-setup on module import
try:
    setup_external_paths()
except FileNotFoundError as e:
    print(f"  {e}")
    print("Some models may not work without external dependencies.")
