import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class GranularityCalibrator(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        hidden_size: int,
        coarse_idx: Sequence[int],
        fine_idx: Sequence[int],
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.coarse_idx = tuple(int(i) for i in coarse_idx)
        self.fine_idx = tuple(int(i) for i in fine_idx)
        if len(self.coarse_idx) == 0 or len(self.fine_idx) == 0:
            raise ValueError("coarse_idx and fine_idx must be non-empty")

        self.common_dim = int(max(len(self.coarse_idx), len(self.fine_idx)))
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.common_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.bias_to_w = nn.Linear(1, 1)
        self.fuse_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out_ln = nn.LayerNorm(self.hidden_size)

    def _pad_to_common(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.common_dim:
            return x
        if x.shape[-1] > self.common_dim:
            return x[..., : self.common_dim]
        pad = self.common_dim - x.shape[-1]
        return F.pad(x, (0, pad), mode="constant", value=0.0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        coarse = states.index_select(-1, torch.tensor(self.coarse_idx, device=states.device))
        fine = states.index_select(-1, torch.tensor(self.fine_idx, device=states.device))
        coarse = self._pad_to_common(coarse)
        fine = self._pad_to_common(fine)

        e_coarse = self.shared_mlp(coarse)
        e_fine = self.shared_mlp(fine)
        cos = F.cosine_similarity(e_coarse, e_fine, dim=-1, eps=1e-8)
        l2 = torch.norm(e_coarse - e_fine, p=2, dim=-1) / (float(self.hidden_size) ** 0.5)
        bias = (1.0 - cos) + l2
        w = torch.sigmoid(self.bias_to_w(bias.unsqueeze(-1))).squeeze(-1)
        e_coarse_cal = e_coarse + w.unsqueeze(-1) * (e_fine - e_coarse)
        g = torch.sigmoid(self.fuse_gate(torch.cat([e_coarse_cal, e_fine], dim=-1)))
        out = g * e_fine + (1.0 - g) * e_coarse_cal
        return self.out_ln(out)

