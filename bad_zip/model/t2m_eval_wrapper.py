from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model.t2m_eval_modules import MotionEncoderBiGRUCo, MovementConvEncoder


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_models(opt):
    movement_enc = MovementConvEncoder(
        opt.dim_pose - 4,
        opt.dim_movement_enc_hidden,
        opt.dim_movement_latent,
    )
    motion_enc = MotionEncoderBiGRUCo(
        input_size=opt.dim_movement_latent,
        hidden_size=opt.dim_motion_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device,
    )

    checkpoint_path = Path(opt.checkpoints_dir) / "text_mot_match" / "model" / "finest.tar"
    checkpoint = _torch_load(checkpoint_path, opt.device)
    movement_enc.load_state_dict(checkpoint["movement_encoder"])
    motion_enc.load_state_dict(checkpoint["motion_encoder"])
    print(f"Loading baseline motion encoder completed from {checkpoint_path}")
    return motion_enc, movement_enc


class EvaluatorModelWrapper:
    def __init__(self, opt):
        opt.dim_pose = 263
        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_motion_hidden = 1024
        opt.dim_coemb_hidden = 512

        self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    def get_motion_embeddings_ordered(self, motions: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            original_idx = torch.arange(len(motions), device=motions.device)
            motions = motions.detach().to(self.device).float()
            m_lens = m_lens.detach().to(self.device)

            align_idx = np.argsort(m_lens.detach().cpu().tolist())[::-1].copy()
            align_idx_t = torch.as_tensor(align_idx, dtype=torch.long, device=self.device)
            motions = motions[align_idx_t]
            m_lens = m_lens[align_idx_t]
            original_idx = original_idx[align_idx_t]

            movements = self.movement_encoder(motions[..., :-4]).detach()
            token_lens = torch.clamp(m_lens // self.opt.unit_length, min=1)
            motion_embedding = self.motion_encoder(movements, token_lens)

            _, inverse_idx = torch.sort(original_idx)
            return motion_embedding[inverse_idx]
