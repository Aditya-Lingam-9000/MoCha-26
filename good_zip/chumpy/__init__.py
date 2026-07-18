"""Tiny compatibility shim for legacy SMPL pickles.

Some official SMPL `.pkl` files contain objects pickled as `chumpy.ch.Ch`.
This baseline only needs those objects as NumPy arrays, so this local
shim avoids requiring the old external `chumpy` package at inference time.
"""

from .ch import Ch

__all__ = ["Ch"]
