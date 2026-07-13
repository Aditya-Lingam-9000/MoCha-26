import os
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# CodaBench automatically injects these paths
DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR.parent))

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class OrdinalDANNModel(nn.Module):
    def __init__(self, input_dim=256, num_domains=4):
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
        self.domain_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )
        
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        severity = self.severity_predictor(features)
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain = self.domain_predictor(reversed_features)
        return severity, domain

class Model:
    def __init__(self):
        self.device = torch.device("cpu")
        print("Initializing MoCha Organizers' baseline encoders...")
        
        # We must load the organizer's evaluators first
        from model.t2m_eval_wrapper import EvaluatorModelWrapper
        from utils.get_opt import get_opt as baseline_get_opt
        
        opt_path = DIR.parent / "weights" / "backbone" / "Comp_v6_KLD005" / "opt.txt"
        chk_path = DIR.parent / "weights" / "backbone" / "motion_encoder_finetuned.pth"
        opt = baseline_get_opt(opt_path, self.device)
        opt.checkpoints_dir = str(DIR.parent / "weights" / "backbone")
        
        self.baseline_wrapper = EvaluatorModelWrapper(opt)
        state = torch.load(chk_path, map_location=self.device, weights_only=True)
        self.baseline_wrapper.motion_encoder.load_state_dict(state)
        self.baseline_wrapper.motion_encoder.eval()
        self.baseline_wrapper.movement_encoder.eval()
        
        print("Loading MoMask RVQVAE...")
        from model.momask.model import RVQVAE
        import model.momask.get_opt as momask_get_opt
        
        momask_opt_path = str(DIR.parent / "weights" / "momask" / "opt.txt")
        momask_chk_path = str(DIR.parent / "weights" / "momask" / "net_best_fid.tar")
        vq_opt = momask_get_opt.get_opt(momask_opt_path, device=self.device)
        self.momask_model = RVQVAE(args=vq_opt, input_width=263, nb_code=vq_opt.nb_code, code_dim=vq_opt.code_dim, 
                       output_emb_width=vq_opt.code_dim, down_t=vq_opt.down_t, stride_t=vq_opt.stride_t, 
                       width=vq_opt.width, depth=vq_opt.depth, dilation_growth_rate=vq_opt.dilation_growth_rate, 
                       activation=vq_opt.vq_act, norm=vq_opt.vq_norm)
        
        momask_checkpoint = torch.load(momask_chk_path, map_location=self.device)['net']
        # Load weights robustly (DDP strip)
        import collections
        new_state_dict = collections.OrderedDict()
        for k, v in momask_checkpoint.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        self.momask_model.load_state_dict(new_state_dict, strict=False)
        self.momask_model.eval()

        print("Loading Ordinal DANN State...")
        self.dann = OrdinalDANNModel(input_dim=256).to(self.device)
        dann_state = torch.load(DIR / "classifier_ordinal_dann.pth", map_location=self.device, weights_only=True)
        
        # Load optimized thresholds calculated during validation!
        self.thresholds = dann_state.pop('optimized_thresholds').numpy()
        print(f"Loaded Optimized Thresholds for Ordinal Classification: {self.thresholds}")
        
        self.dann.load_state_dict(dann_state, strict=False)
        self.dann.eval()
        
        # We need PCA weights
        self.pca_components = torch.load(DIR / "pca_components.pt", map_location=self.device)
        self.pca_mean = torch.load(DIR / "pca_mean.pt", map_location=self.device)
        self.scaler_mean = torch.load(DIR / "scaler_mean.pt", map_location=self.device)
        self.scaler_std = torch.load(DIR / "scaler_std.pt", map_location=self.device)

    def extract_time_series_stats(self, tensor):
        mean = tensor.mean(dim=1).squeeze(0)
        std = tensor.std(dim=1).squeeze(0)
        max_v, _ = tensor.max(dim=1)
        max_v = max_v.squeeze(0)
        min_v, _ = tensor.min(dim=1)
        min_v = min_v.squeeze(0)
        std = torch.nan_to_num(std, 0.0)
        return torch.cat([mean, std, max_v, min_v])

    def predict(self, dataset):
        print("Extracting features from CodaBench Evaluation Set...")
        predictions = {}
        for subject_id, walks in dataset.items():
            predictions[subject_id] = {}
            for walk_id, walk_data in walks.items():
                motion = walk_data["pose"]
                length = len(motion)
                
                motion_tensor = torch.tensor(motion, dtype=torch.float32, device=self.device).unsqueeze(0)
                length_tensor = torch.tensor([length], dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    raw_stats = self.extract_time_series_stats(motion_tensor)
                    
                    baseline_emb = self.baseline_wrapper.get_motion_embeddings_ordered(motion_tensor, length_tensor).squeeze(0)
                    
                    momask_out = self.momask_model(motion_tensor)
                    if isinstance(momask_out, tuple): momask_out = momask_out[0]
                    if momask_out.shape[-1] != 512: momask_out = momask_out.permute(0, 2, 1)
                    momask_stats = self.extract_time_series_stats(momask_out)
                    
                    # 1. Fuse features (2588)
                    x = torch.cat([raw_stats, baseline_emb, momask_stats])
                    
                    # 2. Standard Scale
                    x = (x - self.scaler_mean) / self.scaler_std
                    
                    # 3. PCA Projection (256)
                    x = x - self.pca_mean
                    x = torch.matmul(x, self.pca_components.T).unsqueeze(0)
                    
                    # 4. Ordinal DANN Prediction
                    continuous_score, _ = self.dann(x, alpha=0.0)
                    score = continuous_score.item()
                    
                    # 5. Apply Optimized Thresholds
                    if score < self.thresholds[0]:
                        final_class = 0
                    elif score >= self.thresholds[0] and score < self.thresholds[1]:
                        final_class = 1
                    elif score >= self.thresholds[1] and score < self.thresholds[2]:
                        final_class = 2
                    else:
                        final_class = 3
                        
                predictions[subject_id][walk_id] = int(final_class)
                
        return predictions
