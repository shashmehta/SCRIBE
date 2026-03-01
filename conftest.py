"""Root conftest.py — adds the project root to sys.path so tests can import cellclassifier."""

import sys
import os

# Make sure `cellclassifier/` is importable when running `pytest` from the project root.
# Without this, pytest won't find the package unless you set PYTHONPATH manually.
sys.path.insert(0, os.path.dirname(__file__))
