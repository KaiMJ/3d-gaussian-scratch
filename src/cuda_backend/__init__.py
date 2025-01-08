import os
import sys
from pathlib import Path

# Add the build directory to the Python path
build_dir = str(Path(__file__).parent / "build" / "Debug")
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

from .cuda_backend import add
__all__ = ['add']
