import json
from pathlib import Path
from typing import Any, Mapping
import collections

import numpy as np
import torch
import torch.nn as nn

from model.momask.model import RVQVAE
import model.momask.get_opt as momask_get_opt
from submission.preprocess import MotionPreprocessor
from model.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.get_opt import get_opt as baseline_get_opt

ROOT = Path(__file__).resolve().parents[1]


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained_weights(model, checkpoint):
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model_dict = model.state_dict()
    model_first_key = next(iter(model_dict))
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        if 'module.' not in model_first_key and k.startswith('module.'):
            k = k[7:]
        if k in model_dict:
            new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=False)


def extract_time_series_stats(tensor):
    mean = tensor.mean(dim=1).squeeze(0)
    std  = torch.nan_to_num(tensor.std(dim=1).squeeze(0), 0.0)
    max_v = tensor.max(dim=1).values.squeeze(0)
    min_v = tensor.min(dim=1).values.squeeze(0)
    return torch.cat([mean, std, max_v, min_v])


# MLP architecture must match exactly what was trained in kaggle_notebook_cells.py
class SeverityMLP(nn.Module):
    def __init__(self, D: int, C: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.45),
            nn.Linear(1024, 256), nn.LayerNorm(256),  nn.GELU(), nn.Dropout(0.35),
            nn.Linear(256,  64), nn.LayerNorm(64),   nn.GELU(), nn.Dropout(0.25),
            nn.Linear(64, C),
        )
    def forward(self, x): return self.net(x)


class Model:
    def __init__(self):
        self.device = choose_device()
        self.preprocess = MotionPreprocessor(
            smpl_model_path=ROOT / "weights" / "smpl" / "SMPL_NEUTRAL.pkl",
            normalization_dir=ROOT / "weights" / "stats" / "pdgam",
            device=self.device,
            sequence_len=200,
            target_fps=25.0,
            apply_slope_correction=False,
        )
        self.baseline_wrapper = self._load_baseline()
        self.momask_model     = self._load_momask()

        # Preprocessing parameters
        self.valid_features = torch.from_numpy(
            np.load(ROOT / "weights" / "valid_features.npy")
        ).long().to(self.device)

        self.scaler_mean = torch.from_numpy(
            np.load(ROOT / "weights" / "scaler_mean.npy")
        ).float().to(self.device)

        self.scaler_std = torch.from_numpy(
            np.load(ROOT / "weights" / "scaler_std.npy")
        ).float().to(self.device)

        # MLP classifier
        n_feat = int(self.valid_features.shape[0])
        self.mlp = SeverityMLP(n_feat).to(self.device)
        state = torch.load(
            ROOT / "weights" / "mlp_classifier.pth",
            map_location=self.device,
            weights_only=True,
        )
        self.mlp.load_state_dict(state)
        self.mlp.eval()

    # ------------------------------------------------------------------ #
    def _load_baseline(self):
        opt_path = ROOT / "weights" / "backbone" / "Comp_v6_KLD005" / "opt.txt"
        chk_path = ROOT / "weights" / "backbone" / "motion_encoder_finetuned.pth"
        opt = baseline_get_opt(opt_path, self.device)
        opt.checkpoints_dir = str(ROOT / "weights" / "backbone")
        wrapper = EvaluatorModelWrapper(opt)
        state = torch.load(chk_path, map_location=self.device, weights_only=True)
        wrapper.motion_encoder.load_state_dict(state)
        wrapper.motion_encoder.eval()
        wrapper.movement_encoder.eval()
        return wrapper

    def _load_momask(self):
        opt_path = str(ROOT / "weights" / "momask" / "opt.txt")
        chk_path = str(ROOT / "weights" / "momask" / "net_best_fid.tar")
        vq_opt = momask_get_opt.get_opt(opt_path, device=self.device)
        model = RVQVAE(
            args=vq_opt, input_width=263,
            nb_code=vq_opt.nb_code, code_dim=vq_opt.code_dim,
            output_emb_width=vq_opt.code_dim,
            down_t=vq_opt.down_t, stride_t=vq_opt.stride_t,
            width=vq_opt.width, depth=vq_opt.depth,
            dilation_growth_rate=vq_opt.dilation_growth_rate,
            activation=vq_opt.vq_act, norm=vq_opt.vq_norm,
        )
        checkpoint = torch.load(chk_path, map_location=self.device)['net']
        load_pretrained_weights(model, checkpoint)
        model.eval()
        model.to(self.device)
        return model

    # ------------------------------------------------------------------ #
    def _extract_features(self, motion_tensor: torch.Tensor, length: int) -> torch.Tensor:
        with torch.no_grad():
            raw_stats = extract_time_series_stats(motion_tensor)
            length_t  = torch.as_tensor([length], dtype=torch.long, device=self.device)
            base_emb  = self.baseline_wrapper.get_motion_embeddings_ordered(
                            motion_tensor, length_t).squeeze(0)
            mo_out = self.momask_model(motion_tensor)
            if isinstance(mo_out, tuple):
                mo_out = mo_out[0]
            if mo_out.shape[-1] == 512:
                mo_stats = extract_time_series_stats(mo_out)
            else:
                mo_stats = extract_time_series_stats(mo_out.permute(0, 2, 1))
        return torch.cat([raw_stats, base_emb, mo_stats])

    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, D_raw) → predictions: (N,) long tensor."""
        filtered = features[:, self.valid_features]
        scaled   = (filtered - self.scaler_mean) / self.scaler_std
        with torch.no_grad():
            logits = self.mlp(scaled)
        return logits.argmax(dim=1)

    # ------------------------------------------------------------------ #
    def predict_dataset(
        self,
        dataset: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> dict[str, dict[str, int]]:
        predictions: dict[str, dict[str, int]] = {}
        features_list, keys = [], []

        print("Extracting features from test dataset...")
        for subject_id, walks in dataset.items():
            subject_key = str(subject_id)
            predictions[subject_key] = {}
            for walk_id, sample in walks.items():
                motion, length = self.preprocess(sample)
                motion_tensor  = torch.as_tensor(
                    motion[None], dtype=torch.float32, device=self.device)
                feat = self._extract_features(motion_tensor, length)
                features_list.append(feat)
                keys.append((subject_key, str(walk_id)))

        if features_list:
            features = torch.stack(features_list)      # (N, D)
            preds    = self._classify(features).cpu().numpy()
            for idx, (sk, wk) in enumerate(keys):
                predictions[sk][wk] = int(preds[idx])

        return predictions

    def predict(self, sample: Mapping[str, Any]) -> int:
        motion, length = self.preprocess(sample)
        motion_tensor  = torch.as_tensor(
            motion[None], dtype=torch.float32, device=self.device)
        feat = self._extract_features(motion_tensor, length).unsqueeze(0)
        return int(self._classify(feat).item())
