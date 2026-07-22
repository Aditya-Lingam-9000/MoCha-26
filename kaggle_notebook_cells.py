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

# 2. Load Baseline weights from Kaggle Dataset (Bypassing broken GitHub LFS)
WEIGHTS_DATASET = Path("/kaggle/input/datasets/jyothiradithyalingam/mocha-baseline-weights")
if not WEIGHTS_DATASET.exists():
    raise FileNotFoundError(
        "You must upload 'mocha_baseline_weights.zip' as a Kaggle dataset "
        "and name it 'mocha-baseline-weights'. The original GitHub repository exceeded its LFS bandwidth."
    )
merge_dirs(WEIGHTS_DATASET, BASELINE_DIR / "weights")

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
    
    import os
    hf_token = None
    # Try to load HF_TOKEN from Kaggle Secrets if the user has it configured
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            print("Successfully loaded HF_TOKEN from Kaggle Secrets. Downloading from official Hugging Face servers...")
    except Exception:
        pass

    # Use the official HF mirror ONLY if unauthenticated to bypass bandwidth throttling
    if not hf_token:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("No HF_TOKEN found. Redirecting HuggingFace downloads to hf-mirror.com...")
    
    # Disable progress bars to prevent Kaggle's HTML/JS notebook UI from freezing
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
    from huggingface_hub import list_repo_files, hf_hub_download
    print("Listing files in CARE-PD dataset repository...")
    try:
        all_files = list_repo_files(repo_id="vida-adl/CARE-PD", repo_type="dataset")
        # CRITICAL: Filter to ONLY download the preprocessed pickles we actually use!
        # The root folder contains raw 3D mesh parameters which are 20GB+ and completely unused.
        files = [f for f in all_files if f.startswith("Canonicalized_SMPL_pickles/")]
        print(f"Found {len(files)} canonical files to download sequentially.")
        for idx, file in enumerate(files):
            print(f"[{idx+1}/{len(files)}] Downloading: {file} ...")
            hf_hub_download(
                repo_id="vida-adl/CARE-PD", 
                filename=file,
                repo_type="dataset", 
                local_dir=DATASET_DIR, 
                resume_download=True
            )
        print("CARE-PD dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection or Hugging Face credentials.")

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
    mean = torch.nan_to_num(tensor.mean(dim=1).squeeze(0), 0.0)
    std = torch.nan_to_num(tensor.std(dim=1).squeeze(0), 0.0)
    
    if tensor.shape[1] == 0:
        max_v = torch.zeros_like(mean)
        min_v = torch.zeros_like(mean)
    else:
        max_v = torch.nan_to_num(tensor.max(dim=1).values.squeeze(0), 0.0, posinf=0.0, neginf=0.0)
        min_v = torch.nan_to_num(tensor.min(dim=1).values.squeeze(0), 0.0, posinf=0.0, neginf=0.0)
        
    return torch.cat([mean, std, max_v, min_v])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Extractors (we temporarily append paths during initialization to resolve internal sub-imports)
sys.path.insert(0, str(BASELINE_DIR))
opt_path = BASELINE_DIR / "weights" / "backbone" / "Comp_v6_KLD005" / "opt.txt"
chk_path = BASELINE_DIR / "weights" / "backbone" / "motion_encoder_finetuned.pth"
opt = baseline_get_opt(opt_path, device)
opt.checkpoints_dir = str(BASELINE_DIR / "weights" / "backbone")
baseline_wrapper = EvaluatorModelWrapper(opt)
state = torch.load(chk_path, map_location=device, weights_only=False)
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
checkpoint = torch.load(chk_path_m, map_location=device, weights_only=False)['net']
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
    site_name = pkl_file.name.split('_')[0]
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    for subject_id, walks in tqdm(data.items(), desc=pkl_file.name):
        for walk_id, sample in walks.items():
            label = sample.get("UPDRS_GAIT", -1) # Default to -1 for unsupervised
            if label is None:
                label = -1
                
            motion, joints, length = preprocess.extract_with_joints(sample)
            motion_tensor = torch.as_tensor(motion[None], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                raw_stats = extract_time_series_stats(motion_tensor)
                length_tensor = torch.as_tensor([length], dtype=torch.long, device=device)
                baseline_emb = baseline_wrapper.get_motion_embeddings_ordered(motion_tensor, length_tensor).squeeze(0)
                
                momask_out = momask_model(motion_tensor)
                if isinstance(momask_out, tuple): momask_out = momask_out[0]
                momask_stats = extract_time_series_stats(momask_out) if momask_out.shape[-1] == 512 else extract_time_series_stats(momask_out.permute(0, 2, 1))

            combined = np.concatenate([
                raw_stats.cpu().numpy(),
                baseline_emb.cpu().numpy(),
                momask_stats.cpu().numpy()
            ])
            
            row = {'subject_id': subject_id, 'walk_id': walk_id, 'label': label, 'site': site_name}
            for i, v in enumerate(combined): row[f'f_{i}'] = v
            records.append(row)

# Clean up path after loop
sys.path.remove(str(CAREPD_DIR))
sys.path.remove(str(BASELINE_DIR))
            
df = pd.DataFrame(records)
print(f"Extraction Complete. Fused Shape: {df.shape}")
df.to_csv(REPO_DIR / "features_fused.csv", index=False)
print("Saved features to features_fused.csv so you never have to extract again!")

print("--- 3. Advanced Feature Selection & Ensemble Search ---")

# ── PATH SAFETY: Ensure all directories are writable ────────────────────────
# This block runs whether Section 1 was executed or not in this session.
import subprocess
from pathlib import Path
KAGGLE_WORKING = Path("/kaggle/working")
REPO_DIR       = KAGGLE_WORKING / "MoCha-26"
BASELINE_DIR   = REPO_DIR / "MoCha_baseline_bundle"
CAREPD_DIR     = REPO_DIR / "CARE-PD_github"

# Clone repo if not already present at writable path
if not REPO_DIR.exists():
    print("Cloning repo to writable path...")
    subprocess.run(f"git clone https://github.com/Aditya-Lingam-9000/MoCha-26.git {REPO_DIR}", shell=True, check=True)
    # Restore baseline weights from the read-only input dataset if available
    import glob, shutil
    for src in glob.glob("/kaggle/input/**/weights/**/*", recursive=True):
        src_p = Path(src)
        if src_p.is_file() and "MoCha_baseline_bundle" in str(src_p):
            rel = src_p.relative_to(Path(src_p.parts[0]) / src_p.parts[1] / src_p.parts[2] / "MoCha-26")
            dst = REPO_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src_p, dst)

# --- FORCE RE-EXTRACTION ---
# The previous features_fused.csv contains deleted clinical features. We must extract fresh.
FORCE_REEXTRACT = True

# Load features_fused.csv ? try writable dir first, then read-only input
import os
csv_candidates = [
    REPO_DIR / "features_fused.csv",
    *list(Path("/kaggle/input").glob("**/features_fused.csv")),
]
csv_path = next((p for p in csv_candidates if p.exists()), None)

if FORCE_REEXTRACT or csv_path is None:
    print("Forcing feature re-extraction (or no cached features found). Proceed to Section 2.")
else:
    print(f"Loading features from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded features: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────

import joblib
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge, LogisticRegression

from sklearn.svm import SVC, SVR
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.metrics import f1_score

# Split into supervised data
df_sup = df[df['label'] != -1].copy()
X_sup = df_sup.drop(columns=['subject_id', 'walk_id', 'label', 'site']).values
y_sup = df_sup['label'].values.astype(int)
sites_sup = df_sup['site'].values
subjects_sup = df_sup['subject_id'].astype(str).values

num_total_features = X_sup.shape[1]
num_clinical = 36
clinical_indices = list(range(num_total_features - num_clinical, num_total_features))

# 1. Variance Filter
variances = np.var(X_sup, axis=0)
valid_features_idx = np.where(variances > 1e-4)[0]
print(f"Filtering features: kept {len(valid_features_idx)} out of {num_total_features} features.")

# Map clinical indices in filtered space
clinical_indices_filtered = [i for i, orig_idx in enumerate(valid_features_idx) if orig_idx in clinical_indices]
X_sup_filtered = X_sup[:, valid_features_idx]

# NOTE: Site-wise mean centering is intentionally REMOVED.
# It cannot be replicated at CodaBench inference time (test site labels are unknown).
# Fitting the scaler on site-centered data caused 0.00 CodaBench scores.
# ==============================================================================
# 3. PyTorch ResMLP + Tri-Model Ensemble (ResMLP + Logistic + Ridge)
# ==============================================================================
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

def oversample_data_with_jitter(X, y, noise_std=0.08):
    unique, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    X_resampled, y_resampled = [], []
    for label in unique:
        idx = np.where(y == label)[0]
        sampled_idx = np.random.choice(idx, size=max_count, replace=True)
        X_samples = X[sampled_idx].copy()
        if len(idx) < max_count:
            feature_stds = np.std(X, axis=0, keepdims=True)
            noise = np.random.normal(0.0, noise_std, size=X_samples.shape) * (feature_stds + 1e-6)
            X_samples += noise
        X_resampled.append(X_samples)
        y_resampled.append(y[sampled_idx])
    X_res = np.concatenate(X_resampled, axis=0)
    y_res = np.concatenate(y_resampled, axis=0)
    shuf_idx = np.random.permutation(len(y_res))
    return X_res[shuf_idx], y_res[shuf_idx]

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

from sklearn.linear_model import RidgeClassifier

print("\n--- Training Heterogeneous 15-Model Tri-Ensemble (ResMLP + Logistic + Ridge) ---")
gkf = GroupKFold(n_splits=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

fold_scalers = []
fold_resmlps = []
fold_logregs = []
fold_ridges = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_sup_filtered, y_sup, groups=subjects_sup)):
    print(f"\n--- Training Fold {fold+1}/5 ---")
    X_tr_raw, y_tr_raw = X_sup_filtered[tr_idx], y_sup[tr_idx]
    X_va_raw, y_va_raw = X_sup_filtered[va_idx], y_sup[va_idx]
    
    # 1. StandardScaler
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_va_scaled = scaler.transform(X_va_raw)
    
    # 2. Oversample with Gaussian Noise Jittering
    X_tr_bal, y_tr_bal = oversample_data_with_jitter(X_tr_scaled, y_tr_raw, noise_std=0.08)
    
    # --- Model A: PyTorch ResMLP Classifier ---
    X_tr_t = torch.tensor(X_tr_bal, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_bal, dtype=torch.long)
    X_va_t = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
    y_va_t = torch.tensor(y_va_raw, dtype=torch.long).to(device)
    
    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    input_dim = X_tr_bal.shape[1]
    resmlp = ResMLPClassifier(input_dim=input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(resmlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    best_val_f1 = -1.0
    best_resmlp_weights = None
    
    for epoch in range(80):
        resmlp.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            inputs, targets_a, targets_b, lam = mixup_data(batch_x, batch_y, alpha=0.2)
            optimizer.zero_grad()
            outputs = resmlp(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        resmlp.eval()
        with torch.no_grad():
            val_outputs = resmlp(X_va_t)
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            val_f1 = f1_score(y_va_raw, val_preds, average='macro', zero_division=0)
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_resmlp_weights = copy.deepcopy(resmlp.state_dict())
            
    print(f"Fold {fold+1} ResMLP Best Val F1 = {best_val_f1:.4f}")
    
    # --- Model B: Multinomial Logistic Regression ---
    clf_logreg = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=3000, class_weight='balanced', random_state=42
    )
    clf_logreg.fit(X_tr_bal, y_tr_bal)
    
    # --- Model C: Ridge Classifier ---
    clf_ridge = RidgeClassifier(class_weight='balanced', alpha=10.0, random_state=42)
    clf_ridge.fit(X_tr_bal, y_tr_bal)
    
    fold_scalers.append((scaler.mean_, scaler.scale_))
    fold_resmlps.append(best_resmlp_weights)
    fold_logregs.append((clf_logreg.coef_, clf_logreg.intercept_))
    fold_ridges.append((clf_ridge.coef_, clf_ridge.intercept_))

# Save exact model weights and scalers into the staging weights folder
import shutil
import zipfile

# ── GUARANTEED WRITABLE STAGING DIRECTORY ──
KAGGLE_WORKING = Path("/kaggle/working")
STAGING_DIR = KAGGLE_WORKING / "submission_package"
if STAGING_DIR.exists():
    shutil.rmtree(STAGING_DIR)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# 1. Copy the baseline code to staging
baseline_candidates = [
    KAGGLE_WORKING / "MoCha-26" / "MoCha_baseline_bundle",
    *list(Path("/kaggle/input").glob("**/MoCha_baseline_bundle"))
]
true_baseline_dir = next((p for p in baseline_candidates if p.exists()), None)
if true_baseline_dir is None:
    raise FileNotFoundError("Could not find MoCha_baseline_bundle in working or input dirs!")

def ignore_patterns(path, names):
    return [n for n in names if n == '.git' or n == '__pycache__' or n.endswith('.pyc')]
shutil.copytree(true_baseline_dir, STAGING_DIR, dirs_exist_ok=True, ignore=ignore_patterns)

# 2. Save the trained weights directly into the staging weights folder
weights_dir = STAGING_DIR / "weights"
weights_dir.mkdir(parents=True, exist_ok=True)

np.save(weights_dir / "valid_features.npy", valid_features_idx)

for fold in range(5):
    mean, scale = fold_scalers[fold]
    resmlp_weights = fold_resmlps[fold]
    lr_c, lr_i = fold_logregs[fold]
    rg_c, rg_i = fold_ridges[fold]
    
    np.save(weights_dir / f"scaler_mean_fold{fold}.npy", mean)
    np.save(weights_dir / f"scaler_std_fold{fold}.npy", scale)
    
    torch.save(resmlp_weights, weights_dir / f"resmlp_fold{fold}.pt")
    
    np.save(weights_dir / f"logreg_coef_fold{fold}.npy", lr_c)
    np.save(weights_dir / f"logreg_intercept_fold{fold}.npy", lr_i)
    
    np.save(weights_dir / f"ridge_coef_fold{fold}.npy", rg_c)
    np.save(weights_dir / f"ridge_intercept_fold{fold}.npy", rg_i)

np.save(weights_dir / "model_type.npy", np.array([4])) # 4 = Heterogeneous 15-Model Tri-Ensemble

print(f"Saved Heterogeneous Tri-Model Ensemble weights! Total features used = {len(valid_features_idx)}")

# 3. Copy momask pretrained weights
momask_dest = weights_dir / "momask"
momask_dest.mkdir(parents=True, exist_ok=True)
momask_candidates = [
    KAGGLE_WORKING / "MoCha-26" / "CARE-PD_github" / "assets" / "Pretrained_checkpoints" / "momask",
    *list(Path("/kaggle/input").glob("**/Pretrained_checkpoints/momask"))
]
true_momask_dir = next((p for p in momask_candidates if p.exists()), None)
if true_momask_dir:
    shutil.copy2(true_momask_dir / "opt.txt", momask_dest / "opt.txt")
    shutil.copy2(true_momask_dir / "net_best_fid.tar", momask_dest / "net_best_fid.tar")

print("--- 4. Packaging Submission ---")
zip_path = KAGGLE_WORKING / "submission.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(STAGING_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(STAGING_DIR)
            zipf.write(file_path, arcname)

print(f"Kaggle Pipeline Complete! Download {zip_path} from the Kaggle Output section and submit to CodaBench!")
