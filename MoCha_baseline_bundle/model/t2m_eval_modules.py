from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def init_weight(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: torch.device):
        super().__init__()
        self.device = device
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, inputs: torch.Tensor, m_lens: torch.Tensor) -> torch.Tensor:
        num_samples = inputs.shape[0]
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = m_lens.detach().cpu().tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        _, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)
