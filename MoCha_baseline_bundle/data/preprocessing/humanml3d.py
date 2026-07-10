from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from data.preprocessing.create_redundant_representation import process_file
from data.preprocessing.trajectory_correction_amass import transform_seq_so_it_has_no_slope_AMASS
from data.preprocessing.transforms.paramUtil import t2m_kinematic_chain, t2m_raw_offsets
from data.preprocessing.transforms.skeleton import Skeleton
from human_body_prior.body_model.body_model import BodyModel


@dataclass
class HumanML3DConfig:
    smpl_model_path: Path
    device: torch.device
    apply_slope_correction: bool = False
    feet_thre: float = 0.002


class HumanML3DConverter:
    """Convert one challenge SMPL sample into the 263-D HumanML3D representation."""

    def __init__(self, cfg: HumanML3DConfig):
        self.cfg = cfg
        if not cfg.smpl_model_path.exists():
            raise FileNotFoundError(
                f"SMPL model not found: {cfg.smpl_model_path}. "
                "Set SMPL_MODEL_PATH or place SMPL_NEUTRAL.pkl under weights/smpl/."
            )

        self.body_model = BodyModel(
            bm_fname=str(cfg.smpl_model_path),
            num_betas=10,
        ).to(cfg.device)
        self.body_model.eval()

        this_dir = Path(__file__).resolve().parent
        example_path = this_dir / "transforms" / "000021.npy"
        example_data = np.load(example_path)
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)

        self.joints_num = 22
        self.l_idx1, self.l_idx2 = 5, 8
        self.fid_r, self.fid_l = [8, 11], [7, 10]
        self.face_joint_indx = [2, 1, 17, 16]
        self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        self.kinematic_chain = t2m_kinematic_chain
        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
        self.tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    def __call__(self, sample: Mapping[str, object]) -> np.ndarray:
        joints = self.smpl_to_joints(sample)
        if self.cfg.apply_slope_correction:
            joints = transform_seq_so_it_has_no_slope_AMASS(joints)

        data, _, _, _ = process_file(
            joints,
            self.cfg.feet_thre,
            self.tgt_offsets,
            self.face_joint_indx,
            self.fid_l,
            self.fid_r,
            self.l_idx1,
            self.l_idx2,
            self.n_raw_offsets,
            self.kinematic_chain,
        )
        return data.astype(np.float32)

    def smpl_to_joints(self, sample: Mapping[str, object]) -> np.ndarray:
        pose = np.asarray(sample["pose"], dtype=np.float32)
        trans = np.asarray(sample["trans"], dtype=np.float32)
        beta_key = "beta" if "beta" in sample else "betas"
        betas = np.asarray(sample[beta_key], dtype=np.float32)

        if pose.ndim == 3:
            pose = pose.reshape(pose.shape[0], -1)
        if pose.shape[-1] != 72:
            raise ValueError(f"Expected pose shape (T, 72) or (T, 24, 3), got {pose.shape}")
        if trans.shape[-1] != 3:
            raise ValueError(f"Expected trans shape (T, 3), got {trans.shape}")

        num_frames = pose.shape[0]
        betas = betas.reshape(-1, betas.shape[-1])
        if betas.shape[0] == 1:
            betas = np.repeat(betas, num_frames, axis=0)
        elif betas.shape[0] != num_frames:
            betas = np.repeat(betas[:1], num_frames, axis=0)
        betas = betas[:, :10]

        pose_world = pose.reshape(num_frames, 24, 3)
        body_parms = {
            "root_orient": torch.as_tensor(pose_world[:, 0, :], dtype=torch.float32, device=self.cfg.device),
            "pose_body": torch.as_tensor(pose_world[:, 1:24, :].reshape(num_frames, -1), dtype=torch.float32, device=self.cfg.device),
            "trans": torch.as_tensor(trans, dtype=torch.float32, device=self.cfg.device),
            "betas": torch.as_tensor(betas, dtype=torch.float32, device=self.cfg.device),
            "pose_hand": None,
        }

        with torch.no_grad():
            body = self.body_model(**body_parms)
        return body.Jtr.detach().cpu().numpy()[:, :22].astype(np.float32)
