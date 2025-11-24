# utils/__init__.py
"""
Central place to re-export helper functions used across the project.
"""

from lib.db import make_engine          # lib/db.py
from tools.atr import compute_atr       # tools/atr.py


def get_engine():
    """
    Thin wrapper around lib.db.make_engine() so old code can use utils.get_engine().
    """
    return make_engine()


__all__ = ["get_engine", "compute_atr"]
