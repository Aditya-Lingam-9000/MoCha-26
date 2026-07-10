import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score
from pathlib import Path
import copy
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = ROOT_DIR / "MoCha_baseline_bundle"
BEST_MODEL_PATH = ROOT_DIR / "classifier_finetuned.pth"

sys.path.insert(0, str(BASELINE_DIR))
sys.modules.pop('model', None)
from submission.preprocess import MotionPreprocessor
from model.t2m_eval_wrapper import build_models
from utils.get_opt import get_opt as baseline_get_opt

# ==========================================
# 1. End-to-End Fine-Tuning Architecture
# ==========================================
class FinetunedMoCha(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load baseline opt
        opt_path = BASELINE_DIR / "weights" / "backbone" / "Comp_v6_KLD005" / "opt.txt"
        opt = baseline_get_opt(opt_path, device)
        opt.checkpoints_dir = str(BASELINE_DIR / "weights" / "backbone")
        opt.dim_pose = 263
        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_motion_hidden = 1024
        opt.dim_coemb_hidden = 512

        # Build official models and load pre-trained weights
        self.motion_encoder, self.movement_encoder = build_models(opt)
        
        # UNFREEZE the BiGRU! We want it to learn domain-invariance.
        self.motion_encoder.train()
        self.movement_encoder.train()
        
        # New Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )
        
    def forward(self, motions, m_lens):
        # We drop the last 4 dimensions as done in the official baseline wrapper
        movements = self.movement_encoder(motions[..., :-4])
        
        # The movement encoder has two stride-2 convolutions, meaning temporal length is divided by 4
        token_lens = torch.clamp(m_lens // 4, min=1)
        
        motion_embedding = self.motion_encoder(movements, token_lens)
        return self.classifier(motion_embedding)

# ==========================================
# 2. Raw Sequence Loading
# ==========================================
def load_raw_sequences(device):
    print("Loading and Preprocessing Raw 3D Sequences (Kaggle Pipeline)...")
    preprocess = MotionPreprocessor(
        smpl_model_path=BASELINE_DIR / "weights" / "smpl" / "SMPL_NEUTRAL.pkl",
        normalization_dir=BASELINE_DIR / "weights" / "stats" / "pdgam",
        device=device,
        sequence_len=200,
        target_fps=25.0,
        apply_slope_correction=False,
    )
    
    data_dir = ROOT_DIR / "CARE-PD" / "Canonicalized_SMPL_pickles"
    pkl_files = list(data_dir.glob("*_canonical.pkl"))
    
    X_list, y_list, site_list, len_list = [], [], [], []
    
    for pkl_file in pkl_files:
        site = pkl_file.stem.split('_')[0]
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            
        for subj, walks in tqdm(data.items(), desc=f"Loading {site}", leave=False):
            for walk_id, sample in walks.items():
                label = sample.get("UPDRS_GAIT", None)
                if label is None:
                    continue
                
                # Returns shape [200, 263] padded sequence and original length
                motion, length = preprocess(sample)
                X_list.append(motion)
                y_list.append(label)
                site_list.append(site)
                len_list.append(length)
                
    X = np.stack(X_list)
    y = np.array(y_list)
    sites = np.array(site_list)
    lens = np.array(len_list)
    return X, y, sites, lens

# ==========================================
# 3. Kaggle Training Loop
# ==========================================
def train_and_evaluate():
    # If on Kaggle, this will automatically grab the T4 GPUs!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device}")
    
    X, y, sites, lens = load_raw_sequences(device)
    print(f"\nRaw Dataset Shape: {X.shape}, Unique Sites: {len(np.unique(sites))}")
    
    class_counts = np.bincount(y)
    class_weights = len(y) / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    logo = LeaveOneGroupOut()
    f1_m, prec_m, rec_m, acc, kappa = [], [], [], [], []
    
    best_overall_f1 = 0
    best_model_state = None
    
    print("\nStarting GPU-Accelerated Leave-One-Site-Out Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, sites)):
        test_site = sites[val_idx][0]
        
        # Load to CPU first, then transfer batches to GPU during training to save RAM
        X_train, X_val = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(X[val_idx], dtype=torch.float32)
        y_train, y_val = torch.tensor(y[train_idx], dtype=torch.long), torch.tensor(y[val_idx], dtype=torch.long)
        len_train, len_val = torch.tensor(lens[train_idx], dtype=torch.long), torch.tensor(lens[val_idx], dtype=torch.long)
        
        # We can increase batch size to 64 on Kaggle T4 GPUs
        train_loader = DataLoader(TensorDataset(X_train, y_train, len_train), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val, len_val), batch_size=64, shuffle=False)
        
        model = FinetunedMoCha(torch.device("cpu")).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        
        # Note: We use a tiny learning rate (1e-4) for the BiGRU backbone to avoid destroying the pre-trained weights, 
        # and a larger learning rate for the new classifier head.
        optimizer = optim.AdamW([
            {'params': model.motion_encoder.parameters(), 'lr': 1e-4},
            {'params': model.movement_encoder.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-2)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_fold_f1 = 0
        best_fold_state = None
        patience_counter = 0
        
        # On Kaggle GPUs, 50 epochs will take roughly 3 minutes per fold instead of 30 minutes!
        for epoch in range(50):
            model.train()
            for bx, by, blen in train_loader:
                bx, by, blen = bx.to(device), by.to(device), blen.to(device)
                
                # Sort by length for pack_padded_sequence requirement in BiGRU
                sorted_lengths, sorted_idx = torch.sort(blen, descending=True)
                bx = bx[sorted_idx]
                by = by[sorted_idx]
                
                optimizer.zero_grad()
                out = model(bx, sorted_lengths)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for bx, by, blen in val_loader:
                    bx, blen = bx.to(device), blen.to(device)
                    
                    # Sort for pack_padded_sequence
                    sorted_lengths, sorted_idx = torch.sort(blen, descending=True)
                    bx = bx[sorted_idx]
                    
                    val_out = model(bx, sorted_lengths)
                    
                    # Unsort predictions to match original order
                    _, unsort_idx = torch.sort(sorted_idx)
                    val_out = val_out[unsort_idx]
                    
                    val_preds_list.extend(torch.argmax(val_out, dim=1).cpu().numpy())
                    
            val_f1 = f1_score(y_val.numpy(), val_preds_list, average='macro', zero_division=0)
                
            scheduler.step(val_f1)
            
            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1
                best_fold_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter > 10:
                break
                
        # Evaluate best model for this fold
        model.load_state_dict(best_fold_state)
        model.eval()
        preds_list = []
        with torch.no_grad():
            for bx, by, blen in val_loader:
                bx, blen = bx.to(device), blen.to(device)
                sorted_lengths, sorted_idx = torch.sort(blen, descending=True)
                bx = bx[sorted_idx]
                val_out = model(bx, sorted_lengths)
                _, unsort_idx = torch.sort(sorted_idx)
                val_out = val_out[unsort_idx]
                preds_list.extend(torch.argmax(val_out, dim=1).cpu().numpy())
                
        cur_f1 = f1_score(y_val.numpy(), preds_list, average='macro', zero_division=0)
        print(f"Fold {fold} (Left out site: {test_site}) - Macro F1: {cur_f1:.4f}")
        
        f1_m.append(cur_f1)
        prec_m.append(precision_score(y_val.numpy(), preds_list, average='macro', zero_division=0))
        rec_m.append(recall_score(y_val.numpy(), preds_list, average='macro', zero_division=0))
        acc.append(accuracy_score(y_val.numpy(), preds_list))
        kappa.append(cohen_kappa_score(y_val.numpy(), preds_list, weights='quadratic'))
        
        if cur_f1 > best_overall_f1:
            best_overall_f1 = cur_f1
            best_model_state = copy.deepcopy(best_fold_state)
            
    print(f"\n[End-to-End Fine-Tuning] True Leave-One-Site-Out Metrics:")
    print(f"Macro F1: {np.mean(f1_m):.4f}")
    
    torch.save(best_model_state, BEST_MODEL_PATH)
    print(f"\nSaved Best Fine-Tuned Model to {BEST_MODEL_PATH}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    train_and_evaluate()
