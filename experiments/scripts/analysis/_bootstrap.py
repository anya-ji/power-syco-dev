"""Path + .env setup for scripts in this subfolder.

Delegates to ``scripts/_common.py``; the distinct module name keeps it from
shadowing itself on import.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _common  # noqa: E402,F401

ROOT = _common.ROOT
