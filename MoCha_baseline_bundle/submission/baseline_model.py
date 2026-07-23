import json
from pathlib import Path
from typing import Any, Mapping
import collections

import numpy as np
import torch
import torch.nn as nn
import joblib

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


class OrdinalDANNModel(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.severity_predictor = nn.Linear(128, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        severity = self.severity_predictor(features)
        return severity

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        res = x
        x = self.fc1(self.drop1(self.act1(self.norm1(x))))
        x = self.fc2(self.drop2(self.act2(self.norm2(x))))
        return res + x

class ResMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.block1 = ResBlock(hidden_dim, dropout=0.35)
        self.down = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU()
        )
        self.block2 = ResBlock(hidden_dim // 2, dropout=0.25)
        self.head = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        return self.head(x)

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
        valid_feat_path = ROOT / "weights" / "valid_features.npy"
        if valid_feat_path.exists():
            self.valid_features = np.load(valid_feat_path)
        else:
            self.valid_features = np.arange(3612)
        
        model_type_path = ROOT / "weights" / "model_type.npy"
        if model_type_path.exists():
            self.model_type = int(np.load(model_type_path)[0])
        else:
            self.model_type = 5
        
        if self.model_type == 5:
            # Ordinal DANN + PCA Model
            self.pca_components = torch.load(ROOT / "weights" / "pca_components.pt", map_location=self.device, weights_only=False)
            self.pca_mean = torch.load(ROOT / "weights" / "pca_mean.pt", map_location=self.device, weights_only=False)
            self.scaler_mean = torch.load(ROOT / "weights" / "scaler_mean.pt", map_location=self.device, weights_only=False)
            self.scaler_std = torch.load(ROOT / "weights" / "scaler_std.pt", map_location=self.device, weights_only=False)
            
            self.dann = OrdinalDANNModel(input_dim=256).to(self.device)
            dann_state = torch.load(ROOT / "weights" / "classifier_ordinal_dann.pth", map_location=self.device, weights_only=False)
            if 'optimized_thresholds' in dann_state:
                self.thresholds = dann_state.pop('optimized_thresholds').cpu().numpy()
            else:
                self.thresholds = np.array([0.40, 1.30, 2.10])
                
            self.dann.load_state_dict(dann_state, strict=False)
            self.dann.eval()
            
        elif self.model_type == 4:
            # Heterogeneous Tri-Model Ensemble (ResMLP + Logistic + Ridge)
            self.fold_scalers = []
            self.resmlp_models = []
            self.logreg_coefs = []
            self.logreg_intercepts = []
            self.ridge_coefs = []
            self.ridge_intercepts = []

            input_dim = len(self.valid_features)
            for fold in range(5):
                mean = torch.from_numpy(np.load(ROOT / "weights" / f"scaler_mean_fold{fold}.npy")).float().to(self.device)
                std = torch.from_numpy(np.load(ROOT / "weights" / f"scaler_std_fold{fold}.npy")).float().to(self.device)
                self.fold_scalers.append((mean, std))

                # 1. ResMLP Model
                resmlp = ResMLPClassifier(input_dim=input_dim).to(self.device)
                resmlp_state = torch.load(ROOT / "weights" / f"resmlp_fold{fold}.pt", map_location=self.device, weights_only=False)
                resmlp.load_state_dict(resmlp_state)
                resmlp.eval()
                self.resmlp_models.append(resmlp)

                # 2. Logistic Regression
                lr_c = torch.from_numpy(np.load(ROOT / "weights" / f"logreg_coef_fold{fold}.npy")).float().to(self.device)
                lr_i = torch.from_numpy(np.load(ROOT / "weights" / f"logreg_intercept_fold{fold}.npy")).float().to(self.device)
                self.logreg_coefs.append(lr_c)
                self.logreg_intercepts.append(lr_i)

                # 3. Ridge Classifier
                rg_c = torch.from_numpy(np.load(ROOT / "weights" / f"ridge_coef_fold{fold}.npy")).float().to(self.device)
                rg_i = torch.from_numpy(np.load(ROOT / "weights" / f"ridge_intercept_fold{fold}.npy")).float().to(self.device)
                self.ridge_coefs.append(rg_c)
                self.ridge_intercepts.append(rg_i)

        elif self.model_type == 3:
            # Load 5 MLP fold models and 5 scalers
            self.fold_scalers = []
            self.fold_w1 = []
            self.fold_b1 = []
            self.fold_w2 = []
            self.fold_b2 = []
            self.fold_w3 = []
            self.fold_b3 = []
            self.fold_w4 = []
            self.fold_b4 = []
            
            for fold in range(5):
                mean = torch.from_numpy(np.load(ROOT / "weights" / f"scaler_mean_fold{fold}.npy")).float().to(self.device)
                std = torch.from_numpy(np.load(ROOT / "weights" / f"scaler_std_fold{fold}.npy")).float().to(self.device)
                self.fold_scalers.append((mean, std))
                
                w1 = torch.from_numpy(np.load(ROOT / "weights" / f"w1_fold{fold}.npy")).float().to(self.device)
                b1 = torch.from_numpy(np.load(ROOT / "weights" / f"b1_fold{fold}.npy")).float().to(self.device)
                w2 = torch.from_numpy(np.load(ROOT / "weights" / f"w2_fold{fold}.npy")).float().to(self.device)
                b2 = torch.from_numpy(np.load(ROOT / "weights" / f"b2_fold{fold}.npy")).float().to(self.device)
                w3 = torch.from_numpy(np.load(ROOT / "weights" / f"w3_fold{fold}.npy")).float().to(self.device)
                b3 = torch.from_numpy(np.load(ROOT / "weights" / f"b3_fold{fold}.npy")).float().to(self.device)
                w4 = torch.from_numpy(np.load(ROOT / "weights" / f"w4_fold{fold}.npy")).float().to(self.device)
                b4 = torch.from_numpy(np.load(ROOT / "weights" / f"b4_fold{fold}.npy")).float().to(self.device)
                
                self.fold_w1.append(w1)
                self.fold_b1.append(b1)
                self.fold_w2.append(w2)
                self.fold_b2.append(b2)
                self.fold_w3.append(w3)
                self.fold_b3.append(b3)
                self.fold_w4.append(w4)
                self.fold_b4.append(b4)
        else:
            self.scaler_mean = torch.from_numpy(np.load(ROOT / "weights" / "scaler_mean.npy")).float().to(self.device)
            self.scaler_std = torch.from_numpy(np.load(ROOT / "weights" / "scaler_std.npy")).float().to(self.device)
            self.coef = torch.from_numpy(np.load(ROOT / "weights" / "fusion_coef.npy")).float().to(self.device)
            self.intercept = torch.from_numpy(np.load(ROOT / "weights" / "fusion_intercept.npy")).float().to(self.device)
            
            # Load thresholds only if regression model type
            threshold_path = ROOT / "weights" / "thresholds.npy"
            if threshold_path.exists():
                self.thresholds = np.load(threshold_path)

    def _load_baseline(self):
        opt_path = ROOT / "weights" / "backbone" / "Comp_v6_KLD005" / "opt.txt"
        chk_path = ROOT / "weights" / "backbone" / "motion_encoder_finetuned.pth"
        opt = baseline_get_opt(opt_path, self.device)
        opt.checkpoints_dir = str(ROOT / "weights" / "backbone")
        wrapper = EvaluatorModelWrapper(opt)
        state = torch.load(chk_path, map_location=self.device, weights_only=False)
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
        checkpoint = torch.load(chk_path, map_location=self.device, weights_only=False)['net']
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

        return torch.cat([raw_stats, base_emb, mo_stats])
    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, D_raw) -> predictions: (N,) int tensor."""
        valid_torch = torch.from_numpy(self.valid_features).long().to(self.device)
        
        expected_raw = int(torch.max(valid_torch).item()) + 1
        actual_raw = features.shape[1]
        if actual_raw < expected_raw:
            raise ValueError(
                f"Feature dimension mismatch: inference has {actual_raw} features "
                f"but valid_features expects at least {expected_raw}. "
                f"Check that _extract_features() matches training pipeline."
            )
            
        filtered = features[:, valid_torch]
        if self.model_type == 5:
            # Ordinal DANN + 256-dim PCA Projection
            x_scaled = (features - self.scaler_mean) / self.scaler_std
            x_pca = torch.matmul(x_scaled - self.pca_mean, self.pca_components.T)
            
            with torch.no_grad():
                severity = self.dann(x_pca).squeeze(-1).cpu().numpy()
                
            t0, t1, t2 = self.thresholds
            preds = np.zeros_like(severity, dtype=int)
            preds[severity >= t0] = 1
            preds[severity >= t1] = 2
            preds[severity >= t2] = 3
            return torch.from_numpy(preds).to(self.device)

        elif self.model_type == 4:
            probs_sum = torch.zeros((features.shape[0], 4), device=self.device)
            for fold in range(5):
                mean, std = self.fold_scalers[fold]
                scaled = (filtered - mean) / std

                # 1. ResMLP probabilities
                with torch.no_grad():
                    logits_resmlp = self.resmlp_models[fold](scaled)
                    probs_resmlp = torch.softmax(logits_resmlp, dim=1)

                # 2. Logistic Regression probabilities
                logits_logreg = torch.matmul(scaled, self.logreg_coefs[fold].t()) + self.logreg_intercepts[fold]
                probs_logreg = torch.softmax(logits_logreg, dim=1)

                # 3. Ridge Classifier probabilities
                logits_ridge = torch.matmul(scaled, self.ridge_coefs[fold].t()) + self.ridge_intercepts[fold]
                probs_ridge = torch.softmax(logits_ridge, dim=1)

                # Tri-model weighted fusion (0.6 ResMLP, 0.2 Logistic, 0.2 Ridge)
                fold_probs = 0.6 * probs_resmlp + 0.2 * probs_logreg + 0.2 * probs_ridge
                probs_sum += fold_probs

            return torch.argmax(probs_sum, dim=1)

        elif self.model_type == 3:
            probs_sum = torch.zeros((features.shape[0], 4), device=self.device)
            for fold in range(5):
                mean, std = self.fold_scalers[fold]
                scaled = (filtered - mean) / std
                
                # Layer 1
                x = torch.matmul(scaled, self.fold_w1[fold].t()) + self.fold_b1[fold]
                x = torch.relu(x)
                # Layer 2
                x = torch.matmul(x, self.fold_w2[fold].t()) + self.fold_b2[fold]
                x = torch.relu(x)
                # Layer 3
                x = torch.matmul(x, self.fold_w3[fold].t()) + self.fold_b3[fold]
                x = torch.relu(x)
                # Output Layer
                logits = torch.matmul(x, self.fold_w4[fold].t()) + self.fold_b4[fold]
                
                probs = torch.softmax(logits, dim=1)
                probs_sum += probs
                
            return torch.argmax(probs_sum, dim=1)
            
        else:
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
