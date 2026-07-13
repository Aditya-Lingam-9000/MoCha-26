"""
Optional helper utilities for participants.

The official entry point is run.py::predict(data). This file is not called by
CodaBench directly, but you can use it from your own run.py/model.py.
"""

import numpy as np


def get_motion_arrays(sample):
    pose = np.asarray(sample.get('pose', []), dtype=np.float32)
    trans = np.asarray(sample.get('trans', []), dtype=np.float32)
    beta = np.asarray(sample.get('beta', []), dtype=np.float32)
    fps = int(sample.get('fps', 30))
    return pose, trans, beta, fps
