import json
from pathlib import Path
from typing import Any, Mapping
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if not 'module.' in model_first_key:
            if k.startswith('module.'):
                k = k[7:]
        if k in model_dict:
            new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=False)

def extract_time_series_stats(tensor):
    mean = tensor.mean(dim=1).squeeze(0)
    std = tensor.std(dim=1).squeeze(0)
    max_v, _ = tensor.max(dim=1)
    max_v = max_v.squeeze(0)
    min_v, _ = tensor.min(dim=1)
    min_v = min_v.squeeze(0)
    std = torch.nan_to_num(std, 0.0)
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
        
        # Load weights
        self.valid_features = torch.from_numpy(np.load(ROOT / "weights" / "valid_features.npy")).long().to(self.device)
        self.model_type = int(np.load(ROOT / "weights" / "model_type.npy")[0])
        self.pca_components = torch.from_numpy(np.load(ROOT / "weights" / "pca_components.npy")).float().to(self.device)
        self.pca_mean = torch.from_numpy(np.load(ROOT / "weights" / "pca_mean.npy")).float().to(self.device)
        self.coef = torch.from_numpy(np.load(ROOT / "weights" / "fusion_coef.npy")).float().to(self.device)
        self.intercept = torch.from_numpy(np.load(ROOT / "weights" / "fusion_intercept.npy")).float().to(self.device)
        self.thresholds = torch.from_numpy(np.load(ROOT / "weights" / "thresholds.npy")).float().to(self.device)

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
        model = RVQVAE(args=vq_opt, input_width=263, nb_code=vq_opt.nb_code, code_dim=vq_opt.code_dim, 
                       output_emb_width=vq_opt.code_dim, down_t=vq_opt.down_t, stride_t=vq_opt.stride_t, 
                       width=vq_opt.width, depth=vq_opt.depth, dilation_growth_rate=vq_opt.dilation_growth_rate, 
                       activation=vq_opt.vq_act, norm=vq_opt.vq_norm)
        
        checkpoint = torch.load(chk_path, map_location=self.device)['net']
        load_pretrained_weights(model, checkpoint)
        model.eval()
        model.to(self.device)
        return model

    def predict_dataset(self, dataset: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, int]]:
        predictions = {}
        features_list = []
        keys = []
        
        print("Extracting features from dataset...")
        for subject_id, walks in dataset.items():
            subject_key = str(subject_id)
            predictions[subject_key] = {}
            for walk_id, sample in walks.items():
                walk_key = str(walk_id)
                motion, length = self.preprocess(sample)
                motion_tensor = torch.as_tensor(motion[None], dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    raw_stats = extract_time_series_stats(motion_tensor)
                    
                    length_tensor = torch.as_tensor([length], dtype=torch.long, device=self.device)
                    baseline_emb = self.baseline_wrapper.get_motion_embeddings_ordered(motion_tensor, length_tensor)
                    baseline_emb = baseline_emb.squeeze(0)
                    
                    momask_out = self.momask_model(motion_tensor)
                    if isinstance(momask_out, tuple):
                        momask_out = momask_out[0]
                    if momask_out.shape[-1] == 512:
                        momask_stats = extract_time_series_stats(momask_out)
                    else:
                        momask_stats = extract_time_series_stats(momask_out.permute(0, 2, 1))
                        
                    combined = torch.cat([raw_stats, baseline_emb, momask_stats])
                    features_list.append(combined)
                    keys.append((subject_key, walk_key))
                    
        # Batch predict
        if len(features_list) > 0:
            features = torch.stack(features_list)
            
            with torch.no_grad():
                # 1. Filter features
                features_filtered = features[:, self.valid_features]
                
                # 2. Site Normalization (Compute mean/std of this test dataset)
                mean = features_filtered.mean(dim=0, keepdim=True)
                std = features_filtered.std(dim=0, keepdim=True) + 1e-2
                features_scaled = (features_filtered - mean) / std
                
                # 3. PCA Projection
                features_pca = torch.matmul(features_scaled - self.pca_mean, self.pca_components.t())
                
                # 4. Predict
                if self.model_type == 1:
                    scores = torch.matmul(features_pca, self.coef.t()) + self.intercept
                    scores = scores.squeeze(1)
                    
                    t0, t1, t2 = self.thresholds[0], self.thresholds[1], self.thresholds[2]
                    preds = torch.zeros_like(scores, dtype=torch.long)
                    preds[scores >= t0] = 1
                    preds[scores >= t1] = 2
                    preds[scores >= t2] = 3
                else:
                    logits = torch.matmul(features_pca, self.coef.t()) + self.intercept
                    preds = torch.argmax(logits, dim=1)
                
                preds_list = preds.cpu().numpy()
                
            for idx, (sub_key, walk_key) in enumerate(keys):
                predictions[sub_key][walk_key] = int(preds_list[idx])
                
        return predictions

    def predict(self, sample: Mapping[str, Any]) -> int:
        motion, length = self.preprocess(sample)
        motion_tensor = torch.as_tensor(motion[None], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            raw_stats = extract_time_series_stats(motion_tensor)
            length_tensor = torch.as_tensor([length], dtype=torch.long, device=self.device)
            baseline_emb = self.baseline_wrapper.get_motion_embeddings_ordered(motion_tensor, length_tensor).squeeze(0)
            
            momask_out = self.momask_model(motion_tensor)
            if isinstance(momask_out, tuple): momask_out = momask_out[0]
            if momask_out.shape[-1] == 512:
                momask_stats = extract_time_series_stats(momask_out)
            else:
                momask_stats = extract_time_series_stats(momask_out.permute(0, 2, 1))
                
            combined = torch.cat([raw_stats, baseline_emb, momask_stats]).unsqueeze(0)
            combined_filtered = combined[:, self.valid_features]
            
            combined_scaled = (combined_filtered - combined_filtered.mean(dim=0, keepdim=True)) / 1.0
            combined_pca = torch.matmul(combined_scaled - self.pca_mean, self.pca_components.t())
            
            if self.model_type == 1:
                scores = torch.matmul(combined_pca, self.coef.t()) + self.intercept
                score = scores.item()
                
                t0, t1, t2 = self.thresholds[0].item(), self.thresholds[1].item(), self.thresholds[2].item()
                if score < t0:
                    return 0
                elif score >= t0 and score < t1:
                    return 1
                elif score >= t1 and score < t2:
                    return 2
                else:
                    return 3
            else:
                logits = torch.matmul(combined_pca, self.coef.t()) + self.intercept
                prediction = torch.argmax(logits, dim=1).item()
                return int(prediction)
