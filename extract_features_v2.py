import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import gc
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = ROOT_DIR / "MoCha_baseline_bundle"
CAREPD_DIR = ROOT_DIR / "CARE-PD_github"

# 1. Setup paths to avoid namespace collisions
sys.path.insert(0, str(BASELINE_DIR))
sys.modules.pop('model', None)
import model as baseline_model_module
from submission.preprocess import MotionPreprocessor
from utils.get_opt import get_opt as baseline_get_opt
from model.t2m_eval_wrapper import EvaluatorModelWrapper

sys.path.insert(0, str(CAREPD_DIR))
sys.modules.pop('model', None)
sys.modules.pop('utils', None)
sys.modules.pop('utils.get_opt', None)
from model.momask.model import RVQVAE
from model.momask.get_opt import get_opt as momask_get_opt

def load_pretrained_weights(model, checkpoint):
    import collections
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

def setup_baseline_extractor(device):
    print("Loading Baseline BiGRU Extractor...")
    opt_path = BASELINE_DIR / "weights" / "backbone" / "Comp_v6_KLD005" / "opt.txt"
    chk_path = BASELINE_DIR / "weights" / "backbone" / "motion_encoder_finetuned.pth"
    opt = baseline_get_opt(opt_path, device)
    opt.checkpoints_dir = str(BASELINE_DIR / "weights" / "backbone")
    
    wrapper = EvaluatorModelWrapper(opt)
    state = torch.load(chk_path, map_location=device, weights_only=True)
    wrapper.motion_encoder.load_state_dict(state)
    wrapper.motion_encoder.eval()
    wrapper.movement_encoder.eval()
    return wrapper

def setup_momask_extractor(device):
    print("Loading SOTA MoMask Extractor...")
    opt_path = str(CAREPD_DIR / "assets" / "Pretrained_checkpoints" / "momask" / "opt.txt")
    chk_path = str(CAREPD_DIR / "assets" / "Pretrained_checkpoints" / "momask" / "net_best_fid.tar")
    
    vq_opt = momask_get_opt(opt_path, device=device)
    model = RVQVAE(args=vq_opt, input_width=263, nb_code=vq_opt.nb_code, code_dim=vq_opt.code_dim, 
                   output_emb_width=vq_opt.code_dim, down_t=vq_opt.down_t, stride_t=vq_opt.stride_t, 
                   width=vq_opt.width, depth=vq_opt.depth, dilation_growth_rate=vq_opt.dilation_growth_rate, 
                   activation=vq_opt.vq_act, norm=vq_opt.vq_norm)
    
    checkpoint = torch.load(chk_path, map_location=device)['net']
    load_pretrained_weights(model, checkpoint)
    model.eval()
    model.to(device)
    return model

def extract_time_series_stats(tensor):
    # tensor shape: [B, T, C]
    # We want mean, std, max, min over T
    mean = tensor.mean(dim=1).squeeze(0)
    std = tensor.std(dim=1).squeeze(0)
    max_v, _ = tensor.max(dim=1)
    max_v = max_v.squeeze(0)
    min_v, _ = tensor.min(dim=1)
    min_v = min_v.squeeze(0)
    
    # Fill NaNs from std with 0 if sequence was too short
    std = torch.nan_to_num(std, 0.0)
    
    return torch.cat([mean, std, max_v, min_v])

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    baseline_wrapper = setup_baseline_extractor(device)
    momask_model = setup_momask_extractor(device)
    
    # Needs absolute path for local chumpy shim resolution during pkl load
    sys.path.append(str(BASELINE_DIR))
    
    preprocess = MotionPreprocessor(
        smpl_model_path=BASELINE_DIR / "weights" / "smpl" / "SMPL_NEUTRAL.pkl",
        normalization_dir=BASELINE_DIR / "weights" / "stats" / "pdgam",
        device=device,
        sequence_len=200,
        target_fps=25.0,
        apply_slope_correction=False,
    )
    
    data_dir = ROOT_DIR / "CARE-PD" / "Canonicalized_SMPL_pickles"
    pkl_files = list(data_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files)} pickle files. Starting advanced fusion extraction...")
    
    records = []
    
    for pkl_file in pkl_files:
        print(f"Processing {pkl_file.name}...")
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
            
        for subject_id, walks in tqdm(data.items(), leave=False):
            for walk_id, sample in walks.items():
                label = sample.get("UPDRS_GAIT", None)
                if label is None:
                    continue
                
                motion, length = preprocess(sample)
                motion_tensor = torch.as_tensor(motion[None], dtype=torch.float32, device=device)
                
                with torch.no_grad():
                    # 1. Raw Kinematic Time-Series Stats (263 * 4 = 1052 features)
                    raw_stats = extract_time_series_stats(motion_tensor)
                    
                    # 2. Baseline Embeddings (512 features)
                    length_tensor = torch.as_tensor([length], dtype=torch.long, device=device)
                    baseline_emb = baseline_wrapper.get_motion_embeddings_ordered(motion_tensor, length_tensor)
                    baseline_emb = baseline_emb.squeeze(0)
                    
                    # 3. MoMask Embeddings with Temporal Variance (512 * 2 = 1024 features)
                    momask_out = momask_model(motion_tensor)
                    if isinstance(momask_out, tuple):
                        momask_out = momask_out[0]
                    # momask_out shape: [B, 512, T'] (or wait, momask output is [B, T', 512] normally unless we permuted it)
                    # Let's check shape: RVQVAE returns [B, T', C] because it postprocesses.
                    if momask_out.shape[-1] == 512:
                        momask_stats = extract_time_series_stats(momask_out)
                    else:
                        # shape is [B, 512, T']
                        momask_stats = extract_time_series_stats(momask_out.permute(0, 2, 1))
                        
                # Combine all features
                combined = torch.cat([raw_stats, baseline_emb, momask_stats]).cpu().numpy()
                
                row = {
                    'subject_id': subject_id,
                    'walk_id': walk_id,
                    'label': label
                }
                for i, v in enumerate(combined):
                    row[f'f_{i}'] = v
                
                records.append(row)
        
        # Clean up memory
        gc.collect()

    print("Saving fused features to CSV...")
    df = pd.DataFrame(records)
    df.to_csv(ROOT_DIR / "train_features_fusion.csv", index=False)
    print(f"Extraction Complete. Fused Shape: {df.shape}")

if __name__ == "__main__":
    main()
