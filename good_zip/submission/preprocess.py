"""Preprocessing helpers for this MoCha baseline submission."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data.preprocessing.humanml3d import HumanML3DConfig, HumanML3DConverter


def load_normalization(norm_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mean = np.load(norm_dir / "mean.npy").astype(np.float32)
    std = np.load(norm_dir / "std.npy").astype(np.float32)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def resample_array(array: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Linearly resample a time sequence while preserving duration."""
    array = np.asarray(array, dtype=np.float32)
    if array.shape[0] <= 1 or abs(source_fps - target_fps) < 1e-6:
        return array.copy()

    duration = (array.shape[0] - 1) / source_fps
    old_t = np.linspace(0.0, duration, array.shape[0], dtype=np.float64)
    new_len = int(round(duration * target_fps)) + 1
    new_t = np.linspace(0.0, duration, new_len, dtype=np.float64)

    flat = array.reshape(array.shape[0], -1)
    out = np.empty((new_len, flat.shape[1]), dtype=np.float32)
    for dim in range(flat.shape[1]):
        out[:, dim] = np.interp(new_t, old_t, flat[:, dim])
    return out.reshape((new_len,) + array.shape[1:]).astype(np.float32)


def maybe_resample_sample(sample: Mapping[str, Any], target_fps: float | None) -> Mapping[str, Any]:
    if target_fps is None:
        return sample

    source_fps = float(sample.get("fps", target_fps))
    if source_fps <= 0 or abs(source_fps - target_fps) < 1e-6:
        return sample

    resampled = dict(sample)
    resampled["pose"] = resample_array(np.asarray(sample["pose"], dtype=np.float32), source_fps, target_fps)
    resampled["trans"] = resample_array(np.asarray(sample["trans"], dtype=np.float32), source_fps, target_fps)
    resampled["fps"] = target_fps
    return resampled


class MotionPreprocessor:
    """Convert one SMPL sample into normalized HumanML3D features."""

    def __init__(
        self,
        *,
        smpl_model_path: Path,
        normalization_dir: Path,
        device: torch.device,
        sequence_len: int,
        target_fps: float | None,
        apply_slope_correction: bool = False,
    ):
        self.sequence_len = int(sequence_len)
        self.target_fps = target_fps
        self.mean, self.std = load_normalization(normalization_dir)
        self.converter = HumanML3DConverter(
            HumanML3DConfig(
                smpl_model_path=smpl_model_path,
                device=device,
                apply_slope_correction=apply_slope_correction,
            )
        )

    def __call__(self, sample: Mapping[str, Any]) -> tuple[np.ndarray, int]:
        sample = maybe_resample_sample(sample, self.target_fps)
        humanml = self.converter(sample)
        humanml = humanml[:, :263]

        motion = ((humanml - self.mean[:263]) / self.std[:263]).astype(np.float32)
        motion_len = min(int(motion.shape[0]), self.sequence_len)

        if motion.shape[0] < self.sequence_len:
            pad = np.zeros((self.sequence_len - motion.shape[0], motion.shape[1]), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        elif motion.shape[0] > self.sequence_len:
            motion = motion[: self.sequence_len]

        return motion.astype(np.float32), max(motion_len, 1)
