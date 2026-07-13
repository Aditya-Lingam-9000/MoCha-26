"""
Minimal baseline model for MoCha2026.

Participants may replace this file with any model implementation. The only
requirement is that run.py returns predictions in the required nested dict format.
"""

import numpy as np


class Model:
    def __init__(self):
        pass

    def predict(self, sample):
        """
        Args:
            sample: dict with keys pose, trans, beta, fps.

        Returns:
            Integer-like prediction in {0, 1, 2, 3}.
        """
        pose = np.asarray(sample.get('pose', []), dtype=np.float32)
        trans = np.asarray(sample.get('trans', []), dtype=np.float32)

        if pose.size == 0 or trans.size == 0:
            return 0

        # Tiny baseline: use trajectory length as a weak hand-written feature.
        if len(trans) > 1:
            step_lengths = np.linalg.norm(np.diff(trans[:, [0, 2]], axis=0), axis=1)
            trajectory_length = float(np.sum(step_lengths))
        else:
            trajectory_length = 0.0

        # This is intentionally simple; participants should train a real model.
        if trajectory_length < 1.0:
            return 0
        if trajectory_length < 2.0:
            return 1
        if trajectory_length < 3.0:
            return 2
        return 3
