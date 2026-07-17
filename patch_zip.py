import zipfile
import shutil
import os
from pathlib import Path

print("Patching submission.zip...")

# Define paths
zip_path = Path("submission.zip")
fixed_zip_path = Path("submission_fixed.zip")
temp_dir = Path("temp_patch")

if temp_dir.exists():
    shutil.rmtree(temp_dir)
temp_dir.mkdir()

# Extract zip
print("Extracting current submission.zip...")
with zipfile.ZipFile(zip_path, 'r') as zipf:
    zipf.extractall(temp_dir)

# 1. Write correct baseline_model.py
correct_baseline_model = """import json
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

class FusionClassifier(nn.Module):
    def __init__(self, input_dim=3612, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.fc(x)

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
        
        # Load Fusion Model
        self.fusion_model = FusionClassifier(input_dim=3612, num_classes=4).to(self.device)
        fusion_state = torch.load(ROOT / "weights" / "classifier_fusion.pth", map_location=self.device, weights_only=True)
        self.fusion_model.load_state_dict(fusion_state, strict=True)
        self.fusion_model.eval()
        
        # Load Numpy Scaler
        self.scaler_mean = torch.from_numpy(np.load(ROOT / "weights" / "fusion_scaler_mean.npy")).float().to(self.device)
        self.scaler_std = torch.from_numpy(np.load(ROOT / "weights" / "fusion_scaler_std.npy")).float().to(self.device)

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

    def predict(self, sample: Mapping[str, Any]) -> int:
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
                
            combined = torch.cat([raw_stats, baseline_emb, momask_stats]).unsqueeze(0)
            
            # Predict
            combined_scaled = (combined - self.scaler_mean) / self.scaler_std
            logits = self.fusion_model(combined_scaled)
            prediction = torch.argmax(logits, dim=1).item()
            
        return int(prediction)
"""

with open(temp_dir / "submission" / "baseline_model.py", "w") as f:
    f.write(correct_baseline_model)
    
print("Updated baseline_model.py logic!")

# 2. Inject MoMask weights
momask_dest = temp_dir / "weights" / "momask"
momask_dest.mkdir(parents=True, exist_ok=True)
momask_src = Path("CARE-PD_github/assets/Pretrained_checkpoints/momask")
shutil.copy2(momask_src / "opt.txt", momask_dest / "opt.txt")
shutil.copy2(momask_src / "net_best_fid.tar", momask_dest / "net_best_fid.tar")
print("Injected missing MoMask weights!")

# 3. Create the fixed zip
print("Zipping fixed payload...")
with zipfile.ZipFile(fixed_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(temp_dir)
            zipf.write(file_path, arcname)

print("SUCCESS! Created submission_fixed.zip")
