# ==============================================================================
# MoCha 2026 - Kaggle Semi-Supervised Pipeline (Target: 0.65+)
# ==============================================================================
# INSTRUCTIONS FOR KAGGLE:
# 1. Create a new Notebook on Kaggle.
# 2. Select Accelerator: GPU T4 x2 (or P100).
# 3. Turn ON "Internet" in the Notebook settings.
# 4. Copy-paste this entire script into a cell and run it!
# ==============================================================================

import os
import subprocess
import sys
import gc
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import shutil
import zipfile
import collections
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def merge_dirs(src, dst):
    src = Path(src)
    dst = Path(dst)
    for item in src.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src)
            target = dst / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(item, target)

print("--- 1. Setting up Environment & Downloading Weights/Datasets ---")
run_cmd("pip install huggingface_hub pandas scikit-learn lightgbm numpy torch tqdm")

# Define Kaggle paths
KAGGLE_WORKING = Path("/kaggle/working").resolve()
REPO_DIR = KAGGLE_WORKING / "MoCha-26"
DATASET_DIR = KAGGLE_WORKING / "CARE-PD"

# 1. Clone your main repository
if not REPO_DIR.exists():
    run_cmd(f"git clone https://github.com/Aditya-Lingam-9000/MoCha-26.git {REPO_DIR}")

# Define paths relative to the cloned repo
BASELINE_DIR = REPO_DIR / "MoCha_baseline_bundle"
CAREPD_DIR = REPO_DIR / "CARE-PD_github"

# 2. Download Baseline weights by cloning the original baseline bundle
TEMP_BASELINE = KAGGLE_WORKING / "temp_baseline"
if not TEMP_BASELINE.exists():
    run_cmd(f"git clone https://github.com/TaatiTeam/MoCha_baseline_bundle {TEMP_BASELINE}")
    run_cmd(f"cd {TEMP_BASELINE} && git lfs pull")
merge_dirs(TEMP_BASELINE / "weights", BASELINE_DIR / "weights")

# 3. Download MoMask assets by cloning the original CARE-PD repo
TEMP_CAREPD = KAGGLE_WORKING / "temp_carepd"
if not TEMP_CAREPD.exists():
    run_cmd(f"git clone https://github.com/TaatiTeam/CARE-PD.git {TEMP_CAREPD}")
merge_dirs(TEMP_CAREPD / "assets", CAREPD_DIR / "assets")

# 4. Download CARE-PD Dataset Pickles from HuggingFace
if not DATASET_DIR.exists():
    print("Downloading CARE-PD dataset from HuggingFace...")
    # Clear HF cache locks in case a previous interrupted run left them hanging
    run_cmd("rm -rf ~/.cache/huggingface/hub/.locks")
    
    from huggingface_hub import snapshot_download
    # Using max_workers=1 is CRITICAL to prevent unauthenticated rate-limit hangs
    snapshot_download(
        repo_id="vida-adl/CARE-PD", 
        repo_type="dataset", 
        local_dir=DATASET_DIR, 
        max_workers=1, 
        resume_download=True
    )

# Change directory to the repository root so imports work naturally
os.chdir(str(REPO_DIR))

print("--- 2. Extracting Features (Supervised & Unsupervised) ---")
# Import baseline dependencies using temporary path insertion to avoid namespace conflicts
sys.path.insert(0, str(BASELINE_DIR))
from submission.preprocess import MotionPreprocessor
from utils.get_opt import get_opt as baseline_get_opt
from model.t2m_eval_wrapper import EvaluatorModelWrapper
sys.path.remove(str(BASELINE_DIR))

# Import MoMask dependencies using temporary path insertion to avoid namespace conflicts
sys.path.insert(0, str(CAREPD_DIR))
from model.momask.model import RVQVAE
from model.momask.get_opt import get_opt as momask_get_opt
sys.path.remove(str(CAREPD_DIR))

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
    model.load_state_dict(model_dict, strict=True)

def extract_time_series_stats(tensor):
    mean = tensor.mean(dim=1).squeeze(0)
    std = tensor.std(dim=1).squeeze(0)
    max_v, _ = tensor.max(dim=1)
    max_v = max_v.squeeze(0)
    min_v, _ = tensor.min(dim=1)
    min_v = min_v.squeeze(0)
    std = torch.nan_to_num(std, 0.0)
    return torch.cat([mean, std, max_v, min_v])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Extractors (we temporarily append paths during initialization to resolve internal sub-imports)
sys.path.insert(0, str(BASELINE_DIR))
opt_path = BASELINE_DIR / "weights" / "backbone" / "Comp_v6_KLD005" / "opt.txt"
chk_path = BASELINE_DIR / "weights" / "backbone" / "motion_encoder_finetuned.pth"
opt = baseline_get_opt(opt_path, device)
opt.checkpoints_dir = str(BASELINE_DIR / "weights" / "backbone")
baseline_wrapper = EvaluatorModelWrapper(opt)
state = torch.load(chk_path, map_location=device, weights_only=True)
baseline_wrapper.motion_encoder.load_state_dict(state)
baseline_wrapper.motion_encoder.eval()
baseline_wrapper.movement_encoder.eval()
sys.path.remove(str(BASELINE_DIR))

sys.path.insert(0, str(CAREPD_DIR))
opt_path_m = str(CAREPD_DIR / "assets" / "Pretrained_checkpoints" / "momask" / "opt.txt")
chk_path_m = str(CAREPD_DIR / "assets" / "Pretrained_checkpoints" / "momask" / "net_best_fid.tar")
vq_opt = momask_get_opt(opt_path_m, device=device)
momask_model = RVQVAE(args=vq_opt, input_width=263, nb_code=vq_opt.nb_code, code_dim=vq_opt.code_dim, 
               output_emb_width=vq_opt.code_dim, down_t=vq_opt.down_t, stride_t=vq_opt.stride_t, 
               width=vq_opt.width, depth=vq_opt.depth, dilation_growth_rate=vq_opt.dilation_growth_rate, 
               activation=vq_opt.vq_act, norm=vq_opt.vq_norm)
checkpoint = torch.load(chk_path_m, map_location=device)['net']
load_pretrained_weights(momask_model, checkpoint)
momask_model.eval()
momask_model.to(device)
sys.path.remove(str(CAREPD_DIR))

# Temporarily insert BASELINE_DIR to initialize preprocess model which pulls from data.preprocessing
sys.path.insert(0, str(BASELINE_DIR))
preprocess = MotionPreprocessor(
    smpl_model_path=BASELINE_DIR / "weights" / "smpl" / "SMPL_NEUTRAL.pkl",
    normalization_dir=BASELINE_DIR / "weights" / "stats" / "pdgam",
    device=device, sequence_len=200, target_fps=25.0, apply_slope_correction=False,
)
sys.path.remove(str(BASELINE_DIR))

pkl_files = list((DATASET_DIR / "Canonicalized_SMPL_pickles").glob("*.pkl"))
records = []

# We keep both BASELINE_DIR and CAREPD_DIR in path during loop to allow dynamically loaded models to access their dependencies
sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(CAREPD_DIR))

for pkl_file in pkl_files:
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    for subject_id, walks in tqdm(data.items(), desc=pkl_file.name):
        for walk_id, sample in walks.items():
            label = sample.get("UPDRS_GAIT", -1) # Default to -1 for unsupervised
            if label is None:
                label = -1
                
            motion, length = preprocess(sample)
            motion_tensor = torch.as_tensor(motion[None], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                raw_stats = extract_time_series_stats(motion_tensor)
                length_tensor = torch.as_tensor([length], dtype=torch.long, device=device)
                baseline_emb = baseline_wrapper.get_motion_embeddings_ordered(motion_tensor, length_tensor).squeeze(0)
                
                momask_out = momask_model(motion_tensor)
                if isinstance(momask_out, tuple): momask_out = momask_out[0]
                momask_stats = extract_time_series_stats(momask_out) if momask_out.shape[-1] == 512 else extract_time_series_stats(momask_out.permute(0, 2, 1))
                    
            combined = torch.cat([raw_stats, baseline_emb, momask_stats]).cpu().numpy()
            row = {'subject_id': subject_id, 'walk_id': walk_id, 'label': label}
            for i, v in enumerate(combined): row[f'f_{i}'] = v
            records.append(row)

# Clean up path after loop
sys.path.remove(str(CAREPD_DIR))
sys.path.remove(str(BASELINE_DIR))
            
df = pd.DataFrame(records)
print(f"Extraction Complete. Fused Shape: {df.shape}")
df.to_csv(REPO_DIR / "features_fused.csv", index=False)
print("Saved features to features_fused.csv so you never have to extract again!")

print("--- 3. Semi-Supervised Domain Adaptation ---")
class FusionClassifier(nn.Module):
    def __init__(self, input_dim=3612, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    def forward(self, x): return self.fc(x)

df_sup = df[df['label'] != -1].copy()
df_unsup = df[df['label'] == -1].copy()

X_sup = df_sup.drop(columns=['subject_id', 'walk_id', 'label']).values
y_sup = df_sup['label'].values
X_unsup = df_unsup.drop(columns=['subject_id', 'walk_id', 'label']).values

# Fit scaler on supervised, apply to both
X_mean = X_sup.mean(axis=0, keepdims=True)
X_std = X_sup.std(axis=0, keepdims=True) + 1e-8
X_sup = (X_sup - X_mean) / X_std
X_unsup = (X_unsup - X_mean) / X_std

np.save(REPO_DIR / "fusion_scaler_mean.npy", X_mean)
np.save(REPO_DIR / "fusion_scaler_std.npy", X_std)

# Train initial model
print("Training initial PyTorch model on supervised data...")
model = FusionClassifier().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

from sklearn.utils.class_weight import compute_class_weight
weights_sup = compute_class_weight('balanced', classes=np.unique(y_sup), y=y_sup)
criterion_sup = nn.CrossEntropyLoss(weight=torch.tensor(weights_sup, dtype=torch.float32, device=device))

train_loader = DataLoader(TensorDataset(torch.tensor(X_sup, dtype=torch.float32, device=device), 
                                      torch.tensor(y_sup, dtype=torch.long, device=device)), 
                          batch_size=32, shuffle=True)

for epoch in range(15):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion_sup(model(bx), by)
        loss.backward()
        optimizer.step()

# Pseudo-Labeling
print("Generating Pseudo-Labels for Unsupervised Data...")
model.eval()
with torch.no_grad():
    unsup_tensor = torch.tensor(X_unsup, dtype=torch.float32, device=device)
    logits = model(unsup_tensor)
    probs = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = torch.max(probs, dim=1)
    
confident_idx = max_probs > 0.90
X_pseudo = X_unsup[confident_idx.cpu().numpy()]
y_pseudo = pseudo_labels[confident_idx].cpu().numpy()

print(f"Kept {len(y_pseudo)} highly confident unsupervised samples!")

X_combined = np.vstack([X_sup, X_pseudo])
y_combined = np.concatenate([y_sup, y_pseudo])

print("Retraining on Combined Dataset...")
model_final = FusionClassifier().to(device)
optimizer = optim.AdamW(model_final.parameters(), lr=1e-3, weight_decay=1e-2)

weights_comb = compute_class_weight('balanced', classes=np.unique(y_combined), y=y_combined)
criterion_comb = nn.CrossEntropyLoss(weight=torch.tensor(weights_comb, dtype=torch.float32, device=device))

train_loader_final = DataLoader(TensorDataset(torch.tensor(X_combined, dtype=torch.float32, device=device), 
                                            torch.tensor(y_combined, dtype=torch.long, device=device)), 
                                batch_size=32, shuffle=True)

for epoch in range(20):
    model_final.train()
    for bx, by in train_loader_final:
        optimizer.zero_grad()
        loss = criterion_comb(model_final(bx), by)
        loss.backward()
        optimizer.step()
        
torch.save(model_final.state_dict(), REPO_DIR / "classifier_fusion.pth")
print("Saved final domain-adapted model!")

print("--- 4. Packaging Submission ---")
shutil.copy2(REPO_DIR / "classifier_fusion.pth", BASELINE_DIR / "weights" / "classifier_fusion.pth")
shutil.copy2(REPO_DIR / "fusion_scaler_mean.npy", BASELINE_DIR / "weights" / "fusion_scaler_mean.npy")
shutil.copy2(REPO_DIR / "fusion_scaler_std.npy", BASELINE_DIR / "weights" / "fusion_scaler_std.npy")

momask_dest = BASELINE_DIR / "weights" / "momask"
momask_dest.mkdir(parents=True, exist_ok=True)
momask_src = CAREPD_DIR / "assets" / "Pretrained_checkpoints" / "momask"
shutil.copy2(momask_src / "opt.txt", momask_dest / "opt.txt")
shutil.copy2(momask_src / "net_best_fid.tar", momask_dest / "net_best_fid.tar")

zip_path = KAGGLE_WORKING / "submission.zip"
exclude_dirs = [".git", "classifier"]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(BASELINE_DIR):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
        for file in files:
            if file.endswith('.pyc'): continue
            file_path = Path(root) / file
            arcname = file_path.relative_to(BASELINE_DIR)
            zipf.write(file_path, arcname)

print(f"Kaggle Pipeline Complete! Download {zip_path} from the Kaggle Output section and submit to CodaBench!")
