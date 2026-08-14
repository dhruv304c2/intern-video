"""Adds the repo root to sys.path so tests can import its top-level modules.

Each test file is run directly (`python tests/test_x.py`), which already
puts tests/ on sys.path - so a plain `import _path` here works with no
sys.path setup of its own, and does the one-time insert for everyone else.
"""

import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, ROOT)
