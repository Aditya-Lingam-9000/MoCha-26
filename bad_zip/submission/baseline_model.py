import json
from pathlib import Path
from typing import Any, Mapping
import collections

import numpy as np
import torch
import joblib

from model.momask.model import RVQVAE
import model.momask.get_opt as momask_get_opt
from submission.preprocess import MotionPreprocessor
from submission.clinical_gait_features import extract_clinical_gait_features
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
    mean = torch.nan_to_num(tensor.mean(dim=1).squeeze(0), 0.0)
    std = torch.nan_to_num(tensor.std(dim=1).squeeze(0), 0.0)
    
    # Handle potentially empty tensors for max/min to avoid inf/-inf
    if tensor.shape[1] == 0:
        max_v = torch.zeros_like(mean)
        min_v = torch.zeros_like(mean)
    else:
        max_v = torch.nan_to_num(tensor.max(dim=1).values.squeeze(0), 0.0, posinf=0.0, neginf=0.0)
        min_v = torch.nan_to_num(tensor.min(dim=1).values.squeeze(0), 0.0, posinf=0.0, neginf=0.0)
        
    return torch.cat([mean, std, max_v, min_v])


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
        self.momask_model = self._load_momask()

        # Feature selection & scaler
        self.valid_features = np.load(ROOT / "weights" / "valid_features.npy")
        
        # Load fitted scaler, sub_idx, and classifier
        scaler_path = ROOT / "weights" / "scaler.joblib"
        sub_idx_path = ROOT / "weights" / "sub_idx.joblib"
        clf_path = ROOT / "weights" / "classifier.joblib"

        if scaler_path.exists() and clf_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.sub_idx = joblib.load(sub_idx_path) if sub_idx_path.exists() else None
            self.clf = joblib.load(clf_path)
            self.use_joblib = True
        else:
            self.use_joblib = False
            self.scaler_mean = torch.from_numpy(np.load(ROOT / "weights" / "scaler_mean.npy")).float().to(self.device)
            self.scaler_std = torch.from_numpy(np.load(ROOT / "weights" / "scaler_std.npy")).float().to(self.device)
            self.coef = torch.from_numpy(np.load(ROOT / "weights" / "fusion_coef.npy")).float().to(self.device)
            self.intercept = torch.from_numpy(np.load(ROOT / "weights" / "fusion_intercept.npy")).float().to(self.device)
            self.model_type = int(np.load(ROOT / "weights" / "model_type.npy")[0])
            self.thresholds = np.load(ROOT / "weights" / "thresholds.npy")

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

    def _extract_features(self, motion_tensor: torch.Tensor, joints: np.ndarray, length: int) -> torch.Tensor:
        with torch.no_grad():
            raw_stats = extract_time_series_stats(motion_tensor)
            length_t = torch.as_tensor([length], dtype=torch.long, device=self.device)
            base_emb = self.baseline_wrapper.get_motion_embeddings_ordered(
                motion_tensor, length_t
            ).squeeze(0)

            mo_out = self.momask_model(motion_tensor)
            if isinstance(mo_out, tuple):
                mo_out = mo_out[0]
            if mo_out.shape[-1] == 512:
                mo_stats = extract_time_series_stats(mo_out)
            else:
                mo_stats = extract_time_series_stats(mo_out.permute(0, 2, 1))

            clinical_feat = extract_clinical_gait_features(joints, fps=25.0)
            clinical_feat = np.nan_to_num(clinical_feat, nan=0.0, posinf=0.0, neginf=0.0)
            clinical_tensor = torch.from_numpy(clinical_feat).float().to(self.device)

        return torch.cat([raw_stats, base_emb, mo_stats, clinical_tensor])

    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, D_raw) -> predictions: (N,) int tensor."""
        feat_np = features.cpu().numpy()
        
        if getattr(self, 'use_joblib', False):
            expected_raw = int(np.max(self.valid_features)) + 1
            actual_raw = feat_np.shape[1]
            if actual_raw < expected_raw:
                raise ValueError(
                    f"Feature dimension mismatch: inference has {actual_raw} features "
                    f"but valid_features expects at least {expected_raw}. "
                    f"Check that _extract_features() matches training pipeline."
                )
            filtered = feat_np[:, self.valid_features]
            scaled = self.scaler.transform(filtered)
            preds = self.clf.predict(scaled)
            return torch.from_numpy(np.array(preds, dtype=np.int64)).to(self.device)

        valid_torch = torch.from_numpy(self.valid_features).long().to(self.device)
        filtered = features[:, valid_torch]
        scaled = (filtered - self.scaler_mean) / self.scaler_std

        if self.model_type == 1:
            pred_cont = torch.matmul(scaled, self.coef.t()) + self.intercept
            pred_cont_np = pred_cont.squeeze(-1).cpu().numpy()
            t0, t1, t2 = self.thresholds
            preds = np.zeros_like(pred_cont_np, dtype=int)
            preds[pred_cont_np >= t0] = 1
            preds[pred_cont_np >= t1] = 2
            preds[pred_cont_np >= t2] = 3
            return torch.from_numpy(preds).to(self.device)
        else:
            logits = torch.matmul(scaled, self.coef.t()) + self.intercept
            return torch.argmax(logits, dim=1)

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
                motion, joints, length = self.preprocess.extract_with_joints(sample)
                motion_tensor = torch.as_tensor(
                    motion[None], dtype=torch.float32, device=self.device
                )
                feat = self._extract_features(motion_tensor, joints, length)
                features_list.append(feat)
                keys.append((subject_key, str(walk_id)))

        if features_list:
            features = torch.stack(features_list)
            preds = self._classify(features).cpu().numpy()
            for idx, (sk, wk) in enumerate(keys):
                predictions[sk][wk] = int(preds[idx])

        return predictions

    def predict(self, sample: Mapping[str, Any]) -> int:
        motion, joints, length = self.preprocess.extract_with_joints(sample)
        motion_tensor = torch.as_tensor(
            motion[None], dtype=torch.float32, device=self.device
        )
        feat = self._extract_features(motion_tensor, joints, length).unsqueeze(0)
        return int(self._classify(feat).item())
