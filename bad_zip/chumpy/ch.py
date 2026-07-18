from __future__ import annotations

import numpy as np


class Ch:
    """Minimal array-like stand-in for `chumpy.ch.Ch` in SMPL pickle loading."""

    @property
    def r(self):
        return np.asarray(getattr(self, "x"))

    @property
    def shape(self):
        return self.r.shape

    @property
    def dtype(self):
        return self.r.dtype

    @property
    def ndim(self):
        return self.r.ndim

    def __array__(self, dtype=None):
        array = self.r
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def __getitem__(self, item):
        return self.r[item]
