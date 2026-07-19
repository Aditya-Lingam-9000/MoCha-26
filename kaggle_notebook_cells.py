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
from submission.clinical_gait_features import extract_clinical_gait_features

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

# Load features_fused.csv — try writable dir first, then read-only input
import os
csv_candidates = [
    REPO_DIR / "features_fused.csv",
    *list(Path("/kaggle/input").glob("**/features_fused.csv")),
]
csv_path = next((p for p in csv_candidates if p.exists()), None)
if csv_path is None:
    raise FileNotFoundError(
        "features_fused.csv not found! Please run Section 2 first to extract features."
    )
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
# We use X_sup_filtered directly for a consistent train/inference pipeline.
X_site_centered = X_sup_filtered  # alias kept so rest of code is unchanged

# Ensemble Soft-Voting Classifier Helper (4 Diverse Classifiers)
def build_ensemble(weights=[2.0, 1.5, 1.5, 1.0]):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    m1 = SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    m2 = lgb.LGBMClassifier(max_depth=5, n_estimators=300, learning_rate=0.03, class_weight='balanced', random_state=42, verbosity=-1)
    m3 = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, random_state=42)
    m4 = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
    return VotingClassifier(estimators=[('svc', m1), ('lgb', m2), ('hgb', m3), ('rf', m4)], voting='soft', weights=weights)

# 3. Model Search under GroupKFold & LOGO-CV
print("\n--- Running Multi-Metric Feature & Model Search ---")
unique_sites = np.unique(sites_sup)
gkf = GroupKFold(n_splits=5)

configs = [
    ("Quantile + MutualInfo 256 + 4-Model Soft Ensemble", "quantile", "mi", 256, build_ensemble()),
    ("Quantile + MutualInfo 384 + 4-Model Soft Ensemble", "quantile", "mi", 384, build_ensemble()),
    ("Quantile + MutualInfo 512 + 4-Model Soft Ensemble", "quantile", "mi", 512, build_ensemble()),
    ("Quantile + MutualInfo 768 + 4-Model Soft Ensemble", "quantile", "mi", 768, build_ensemble()),
    ("StandardScaler + MutualInfo 384 + 4-Model Soft Ensemble", "standard", "mi", 384, build_ensemble()),
    ("StandardScaler + MutualInfo 512 + 4-Model Soft Ensemble", "standard", "mi", 512, build_ensemble()),
]

best_score = -1.0
best_config = None

for name, stype, feat_mode, k_feat, model_template in configs:
    logo_scores = []
    gkf_scores = []
    
    # 1. LOGO-CV Evaluation
    for val_site in unique_sites:
        tr = sites_sup != val_site
        va = sites_sup == val_site
        
        X_tr_raw = X_site_centered[tr]
        X_va_raw = X_site_centered[va]
        y_tr, y_va = y_sup[tr], y_sup[va]
        
        if stype == "quantile":
            scaler = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(len(y_tr), 1000))
        else:
            scaler = StandardScaler()
            
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_va_scaled = scaler.transform(X_va_raw)
        
        if feat_mode == "anova":
            selector = SelectKBest(f_classif, k=k_feat)
            selector.fit(X_tr_scaled, y_tr)
            sub_idx = selector.get_support(indices=True)
        elif feat_mode == "mi":
            selector = SelectKBest(mutual_info_classif, k=k_feat)
            selector.fit(X_tr_scaled, y_tr)
            sub_idx = selector.get_support(indices=True)
        elif feat_mode == "et":
            et_sel = ExtraTreesClassifier(n_estimators=100, random_state=42)
            et_sel.fit(X_tr_scaled, y_tr)
            importances = et_sel.feature_importances_
            sub_idx = np.argsort(importances)[-k_feat:]
        elif feat_mode == "clinical_mi":
            non_clinical = [i for i in range(X_tr_scaled.shape[1]) if i not in clinical_indices_filtered]
            selector = SelectKBest(mutual_info_classif, k=k_feat)
            selector.fit(X_tr_scaled[:, non_clinical], y_tr)
            selected_non_clinical = [non_clinical[i] for i in selector.get_support(indices=True)]
            sub_idx = list(set(clinical_indices_filtered + selected_non_clinical))
        else:
            sub_idx = list(range(X_tr_scaled.shape[1]))
            
        X_tr = X_tr_scaled[:, sub_idx]
        X_va = X_va_scaled[:, sub_idx]
        
        import copy
        clf = copy.deepcopy(model_template)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        s = f1_score(y_va, preds, average='macro', zero_division=0)
        logo_scores.append(s)
        
    # 2. GroupKFold (by Subject) Evaluation
    for tr, va in gkf.split(X_site_centered, y_sup, groups=subjects_sup):
        X_tr_raw, X_va_raw = X_site_centered[tr], X_site_centered[va]
        y_tr, y_va = y_sup[tr], y_sup[va]
        
        if stype == "quantile":
            scaler = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(len(y_tr), 1000))
        else:
            scaler = StandardScaler()
            
        X_tr_scaled = scaler.fit_transform(X_tr_raw)
        X_va_scaled = scaler.transform(X_va_raw)
        
        if feat_mode == "anova":
            selector = SelectKBest(f_classif, k=k_feat)
            selector.fit(X_tr_scaled, y_tr)
            sub_idx = selector.get_support(indices=True)
        elif feat_mode == "mi":
            selector = SelectKBest(mutual_info_classif, k=k_feat)
            selector.fit(X_tr_scaled, y_tr)
            sub_idx = selector.get_support(indices=True)
        elif feat_mode == "et":
            et_sel = ExtraTreesClassifier(n_estimators=100, random_state=42)
            et_sel.fit(X_tr_scaled, y_tr)
            importances = et_sel.feature_importances_
            sub_idx = np.argsort(importances)[-k_feat:]
        elif feat_mode == "clinical_mi":
            non_clinical = [i for i in range(X_tr_scaled.shape[1]) if i not in clinical_indices_filtered]
            selector = SelectKBest(mutual_info_classif, k=k_feat)
            selector.fit(X_tr_scaled[:, non_clinical], y_tr)
            selected_non_clinical = [non_clinical[i] for i in selector.get_support(indices=True)]
            sub_idx = list(set(clinical_indices_filtered + selected_non_clinical))
        else:
            sub_idx = list(range(X_tr_scaled.shape[1]))
            
        X_tr = X_tr_scaled[:, sub_idx]
        X_va = X_va_scaled[:, sub_idx]
        
        clf = copy.deepcopy(model_template)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        s = f1_score(y_va, preds, average='macro', zero_division=0)
        gkf_scores.append(s)
        
    mean_logo = np.mean(logo_scores)
    mean_gkf = np.mean(gkf_scores)
    print(f"Config: {name:58s} | LOGO F1 = {mean_logo:.4f} | Subject GroupKFold F1 = {mean_gkf:.4f}")
    
    # Select based on Subject GroupKFold F1 since CodaBench evaluates all subjects together
    if mean_gkf > best_score:
        best_score = mean_gkf
        best_config = (name, stype, feat_mode, k_feat, model_template)

print(f"\nWinning Config: {best_config[0]} with Subject GroupKFold F1 = {best_score:.4f}")

# 4. Train final model on ALL supervised data using winning config
name, stype, feat_mode, k_feat, model_template = best_config

if feat_mode == "anova":
    selector = SelectKBest(f_classif, k=k_feat)
    selector.fit(X_site_centered, y_sup)
    final_sub_idx = selector.get_support(indices=True)
elif feat_mode == "mi":
    selector = SelectKBest(mutual_info_classif, k=k_feat)
    selector.fit(X_site_centered, y_sup)
    final_sub_idx = selector.get_support(indices=True)
elif feat_mode == "et":
    et_sel = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_sel.fit(X_site_centered, y_sup)
    importances = et_sel.feature_importances_
    final_sub_idx = np.argsort(importances)[-k_feat:]
elif feat_mode == "clinical_mi":
    non_clinical = [i for i in range(X_site_centered.shape[1]) if i not in clinical_indices_filtered]
    selector = SelectKBest(mutual_info_classif, k=k_feat)
    selector.fit(X_site_centered[:, non_clinical], y_sup)
    selected_non_clinical = [non_clinical[i] for i in selector.get_support(indices=True)]
    final_sub_idx = list(set(clinical_indices_filtered + selected_non_clinical))
else:
    final_sub_idx = list(range(X_site_centered.shape[1]))

final_valid_features_idx = valid_features_idx[final_sub_idx]
X_selected_unscaled = X_site_centered[:, final_sub_idx]

if stype == "quantile":
    final_scaler = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=min(len(y_sup), 1000))
else:
    final_scaler = StandardScaler()

X_final_input = final_scaler.fit_transform(X_selected_unscaled)

final_clf = copy.deepcopy(model_template)
final_clf.fit(X_final_input, y_sup)
train_preds = final_clf.predict(X_final_input)

train_f1 = f1_score(y_sup, train_preds, average='macro', zero_division=0)
print(f"Final Train Macro F1 = {train_f1:.4f}")

# Save exact winning model, scaler, and selected indices using joblib
import shutil
import zipfile

# ── GUARANTEED WRITABLE STAGING DIRECTORY ──
KAGGLE_WORKING = Path("/kaggle/working")
STAGING_DIR = KAGGLE_WORKING / "submission_package"
if STAGING_DIR.exists():
    shutil.rmtree(STAGING_DIR)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# 1. Copy the baseline code to staging
# We find the baseline dir either in /kaggle/working or in /kaggle/input
baseline_candidates = [
    KAGGLE_WORKING / "MoCha-26" / "MoCha_baseline_bundle",
    *list(Path("/kaggle/input").glob("**/MoCha_baseline_bundle"))
]
true_baseline_dir = next((p for p in baseline_candidates if p.exists()), None)
if true_baseline_dir is None:
    raise FileNotFoundError("Could not find MoCha_baseline_bundle in working or input dirs!")

# Copy everything from baseline to staging (excluding git and cache)
def ignore_patterns(path, names):
    return [n for n in names if n == '.git' or n == '__pycache__' or n.endswith('.pyc')]
shutil.copytree(true_baseline_dir, STAGING_DIR, dirs_exist_ok=True, ignore=ignore_patterns)

# 2. Save the trained weights directly into the staging weights folder
weights_dir = STAGING_DIR / "weights"
weights_dir.mkdir(parents=True, exist_ok=True)

np.save(weights_dir / "valid_features.npy", final_valid_features_idx)
joblib.dump(final_scaler, weights_dir / "scaler.joblib")
joblib.dump(final_sub_idx, weights_dir / "sub_idx.joblib")
joblib.dump(final_clf, weights_dir / "classifier.joblib")
print(f"Saved final winning model ({name}) to classifier.joblib! Total selected features = {len(final_valid_features_idx)}")

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
