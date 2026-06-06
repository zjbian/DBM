import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dt import Block, DecisionTransformer


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


class DualBranchEncoderV2(nn.Module):
    """
    修复版双分支编码器，解决 GranularityCalibrator 的三个设计问题：
      1. 独立 MLP：coarse/fine 各自独立权重，无 zero-padding hack
      2. 预算感知 gate：gate 输入显式包含 budget_left，使 gate 值随预算状态变化
      3. SiLU + LayerNorm：每个 branch 内部加 LN，训练更稳定

    budget_idx: state 中表示 normalized budget_left 的维度索引（默认 dim=1）
    """

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_size: int,
        coarse_idx: Sequence[int],
        fine_idx: Sequence[int],
        budget_idx: int = 1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.coarse_idx = tuple(int(i) for i in coarse_idx)
        self.fine_idx   = tuple(int(i) for i in fine_idx)
        self.budget_idx = int(budget_idx)

        # 独立 branch encoder — 各自 input dim，无 padding
        self.coarse_mlp = nn.Sequential(
            nn.Linear(len(self.coarse_idx), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.fine_mlp = nn.Sequential(
            nn.Linear(len(self.fine_idx), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # 预算感知 gate：输入 = cat(e_coarse, e_fine, budget_left)
        # g > 0.5 → market(fine) 主导；g < 0.5 → budget(coarse) 主导
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.out_ln = nn.LayerNorm(hidden_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: (B, T, state_dim) normalized"""
        dev = states.device
        coarse = states.index_select(-1, torch.tensor(self.coarse_idx, device=dev))
        fine   = states.index_select(-1, torch.tensor(self.fine_idx,   device=dev))
        budget_left = states[:, :, self.budget_idx : self.budget_idx + 1]  # (B, T, 1)

        e_coarse = self.coarse_mlp(coarse)   # (B, T, H)
        e_fine   = self.fine_mlp(fine)        # (B, T, H)

        # budget-conditioned gate
        g = torch.sigmoid(self.gate_net(
            torch.cat([e_coarse, e_fine, budget_left], dim=-1)
        ))  # (B, T, H)

        out = g * e_fine + (1.0 - g) * e_coarse
        return self.out_ln(out)


class FiLMDualBranchEncoder(nn.Module):
    """
    FiLM-style dual-branch encoder.

    Replaces GranularityCalibrator's shared_mlp with:
      1. Separate independent MLP per branch (no padding, no weight sharing)
      2. Budget branch produces per-channel scale γ and shift β to modulate
         market branch — asymmetric, matching DBM-Bid's "budget modulates market" claim

    Based on:
      - FiLM: Visual Reasoning with a General Conditioning Layer
        Perez et al., AAAI 2018.  arXiv:1709.07871
      - Cross-Stitch Networks for Multi-task Learning
        Misra et al., CVPR 2016.  arXiv:1604.03539
        (motivation: hard sharing hurts when branch distributions differ)

    Architecture:
      coarse → coarse_mlp → h_c  (B, T, H)
      fine   → fine_mlp   → h_f  (B, T, H)
      h_c    → film_net   → (γ, β) ∈ R^H each
      h_f'   = (1 + γ) ⊙ h_f + β          ← FiLM modulation
      gate   = sigmoid(fuse_gate(cat[h_c, h_f']))
      out    = gate ⊙ h_f' + (1-gate) ⊙ h_c
    """

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_size: int,
        coarse_idx: Sequence[int],
        fine_idx: Sequence[int],
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.coarse_idx  = tuple(int(i) for i in coarse_idx)
        self.fine_idx    = tuple(int(i) for i in fine_idx)

        # Independent branch MLPs — no padding, each uses its real input dim
        self.coarse_mlp = nn.Sequential(
            nn.Linear(len(self.coarse_idx), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.fine_mlp = nn.Sequential(
            nn.Linear(len(self.fine_idx), hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # FiLM network: budget hidden → (γ, β) per channel
        # Init to zero so initial modulation is identity: h_f' = (1+0)⊙h_f + 0 = h_f
        self.film_net = nn.Linear(hidden_size, hidden_size * 2)
        nn.init.zeros_(self.film_net.weight)
        nn.init.zeros_(self.film_net.bias)

        # Learned gate to merge FiLM-modulated fine and coarse
        self.fuse_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.out_ln    = nn.LayerNorm(hidden_size)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: (B, T, state_dim) normalized"""
        dev = states.device
        coarse = states.index_select(-1, torch.tensor(self.coarse_idx, device=dev))
        fine   = states.index_select(-1, torch.tensor(self.fine_idx,   device=dev))

        h_c = self.coarse_mlp(coarse)   # (B, T, H)
        h_f = self.fine_mlp(fine)        # (B, T, H)

        # FiLM: budget branch conditions market branch
        film_params = self.film_net(h_c)                      # (B, T, 2H)
        gamma, beta = film_params.chunk(2, dim=-1)            # each (B, T, H)
        h_f_modulated = (1.0 + gamma) * h_f + beta            # FiLM modulation

        # Learned merge gate
        gate = torch.sigmoid(self.fuse_gate(
            torch.cat([h_c, h_f_modulated], dim=-1)
        ))
        out = gate * h_f_modulated + (1.0 - gate) * h_c
        return self.out_ln(out)


class CrossStitchDualBranchEncoder(nn.Module):
    """
    Cross-Stitch dual-branch encoder.

    Symmetric variant: each branch can selectively absorb features from the other
    at each layer boundary, via learnable per-channel cross-stitch matrices.

    Based on:
      - Cross-Stitch Networks for Multi-task Learning
        Misra et al., CVPR 2016.  arXiv:1604.03539

    Architecture (2-layer MLP with cross-stitch between layers):
      Layer 1: h_c1 = SiLU(W_c1 · coarse),  h_f1 = SiLU(W_f1 · fine)
      Cross-stitch:
        h_c1' = α_cc ⊙ h_c1 + α_cf ⊙ h_f1
        h_f1' = α_fc ⊙ h_c1 + α_ff ⊙ h_f1
      Layer 2: h_c  = LN(W_c2 · h_c1'),  h_f  = LN(W_f2 · h_f1')
      Gate + merge (same as FiLM variant)
    """

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_size: int,
        coarse_idx: Sequence[int],
        fine_idx: Sequence[int],
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.coarse_idx  = tuple(int(i) for i in coarse_idx)
        self.fine_idx    = tuple(int(i) for i in fine_idx)
        H = hidden_size

        # Layer 1 — separate, no padding
        self.coarse_fc1 = nn.Linear(len(self.coarse_idx), H)
        self.fine_fc1   = nn.Linear(len(self.fine_idx),   H)

        # Cross-stitch parameters: 2×2 matrix per hidden channel
        # α[i, j, h] = contribution of branch j to branch i at channel h
        # Init to identity (α_cc=1, α_ff=1, α_cf=α_fc=0) → pure separate at start
        alpha = torch.zeros(2, 2, H)
        alpha[0, 0] = 1.0   # coarse keeps coarse
        alpha[1, 1] = 1.0   # fine  keeps fine
        self.cross_stitch = nn.Parameter(alpha)

        # Layer 2 — separate
        self.coarse_fc2 = nn.Sequential(nn.Linear(H, H), nn.LayerNorm(H))
        self.fine_fc2   = nn.Sequential(nn.Linear(H, H), nn.LayerNorm(H))

        self.fuse_gate = nn.Linear(H * 2, H)
        self.out_ln    = nn.LayerNorm(H)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        dev = states.device
        coarse = states.index_select(-1, torch.tensor(self.coarse_idx, device=dev))
        fine   = states.index_select(-1, torch.tensor(self.fine_idx,   device=dev))

        # Layer 1
        h_c = F.silu(self.coarse_fc1(coarse))   # (B, T, H)
        h_f = F.silu(self.fine_fc1(fine))         # (B, T, H)

        # Cross-stitch: learned linear combination per channel
        # α shape: (2, 2, H);  stack along dim 0 → (B, T, 2, H)
        stacked = torch.stack([h_c, h_f], dim=-2)   # (B, T, 2, H)
        # cross_stitch: (2, 2, H) → einsum over branch dim
        h_c2 = (self.cross_stitch[0, 0] * h_c + self.cross_stitch[0, 1] * h_f)
        h_f2 = (self.cross_stitch[1, 0] * h_c + self.cross_stitch[1, 1] * h_f)

        # Layer 2
        h_c = self.coarse_fc2(h_c2)   # (B, T, H)
        h_f = self.fine_fc2(h_f2)      # (B, T, H)

        gate = torch.sigmoid(self.fuse_gate(torch.cat([h_c, h_f], dim=-1)))
        out  = gate * h_f + (1.0 - gate) * h_c
        return self.out_ln(out)


class ConstraintStructuredStateEncoder(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        hidden_size: int,
        coarse_idx: Sequence[int],
        fine_idx: Sequence[int],
        constraint_idx: Sequence[int],
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.coarse_idx = tuple(int(i) for i in coarse_idx)
        self.fine_idx = tuple(int(i) for i in fine_idx)
        self.constraint_idx = tuple(int(i) for i in constraint_idx)
        if len(self.coarse_idx) == 0 or len(self.fine_idx) == 0 or len(self.constraint_idx) == 0:
            raise ValueError("coarse_idx, fine_idx and constraint_idx must be non-empty")

        self.coarse_proj = nn.Sequential(
            nn.Linear(len(self.coarse_idx), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.fine_proj = nn.Sequential(
            nn.Linear(len(self.fine_idx), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.constraint_proj = nn.Sequential(
            nn.Linear(len(self.constraint_idx), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        gate_in_dim = 8
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 3),
        )
        self.out_ln = nn.LayerNorm(self.hidden_size)

    def _select(self, states: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        return states.index_select(-1, torch.tensor(indices, device=states.device))

    def _feature_or_zero(self, states: torch.Tensor, index: int) -> torch.Tensor:
        if 0 <= int(index) < states.shape[-1]:
            return states[..., index : index + 1]
        return torch.zeros((*states.shape[:-1], 1), dtype=states.dtype, device=states.device)

    def forward(self, states: torch.Tensor, return_features: bool = False):
        coarse_raw = self._select(states, self.coarse_idx)
        fine_raw = self._select(states, self.fine_idx)
        constraint_raw = self._select(states, self.constraint_idx)

        coarse = self.coarse_proj(coarse_raw)
        fine = self.fine_proj(fine_raw)
        constraint = self.constraint_proj(constraint_raw)

        time_left = self._feature_or_zero(states, 0)
        budget_left = self._feature_or_zero(states, 1)
        bid_shift = self._feature_or_zero(states, 3) - self._feature_or_zero(states, 2)
        market_shift = self._feature_or_zero(states, 8) - self._feature_or_zero(states, 4)
        pvalue_shift = self._feature_or_zero(states, 9) - self._feature_or_zero(states, 5)
        conv_shift = self._feature_or_zero(states, 10) - self._feature_or_zero(states, 6)
        recent_pv = self._feature_or_zero(states, 13)
        recent_hist_gap = self._feature_or_zero(states, 14) - self._feature_or_zero(states, 15)

        gate_input = torch.cat(
            [
                time_left,
                budget_left,
                bid_shift,
                market_shift,
                pvalue_shift,
                conv_shift,
                recent_pv,
                recent_hist_gap,
            ],
            dim=-1,
        )
        stream_gate = torch.softmax(self.gate_net(gate_input), dim=-1)

        fused = (
            stream_gate[..., 0:1] * coarse
            + stream_gate[..., 1:2] * fine
            + stream_gate[..., 2:3] * constraint
        )
        fused = self.out_ln(fused)

        if not return_features:
            return fused
        return fused, {
            "coarse_stream": coarse,
            "fine_stream": fine,
            "constraint_stream": constraint,
            "stream_gate": stream_gate,
        }


class CrossGranularityTemporalFusion(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        n_head: int = 4,
        conv_kernel: int = 3,
        conv_stride: int = 2,
        local_window: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_head = int(n_head)
        self.window = nn.Parameter(torch.tensor(float(local_window)))
        self.window_beta = nn.Parameter(torch.tensor(4.0))
        self.window_temp = nn.Parameter(torch.tensor(1.0))

        self.coarse_conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=int(conv_kernel),
            stride=int(conv_stride),
            padding=int(conv_kernel) // 2,
        )
        self.fine_qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.fine_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.fine_dropout = nn.Dropout(float(dropout))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=self.n_head, dropout=float(dropout), batch_first=True
        )
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        self._mask_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        key = ("causal", int(T), 0)
        m = self._mask_cache.get(key)
        if m is None:
            m = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
            self._mask_cache[key] = m.cpu()
        return self._mask_cache[key].to(device)

    def _local_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.arange(T, device=device)
        dist = idx.view(T, 1) - idx.view(1, T)
        future = dist < 0
        dist = torch.clamp(dist, min=0).to(dtype=dtype)
        w = torch.clamp(self.window.to(dtype=dtype), min=1.0, max=float(T))
        beta = torch.clamp(self.window_beta.to(dtype=dtype), min=0.0)
        temp = torch.clamp(self.window_temp.to(dtype=dtype), min=1e-3)
        bias = -beta * F.softplus((dist - w) / temp)
        bias = bias.masked_fill(future, torch.finfo(dtype).min)
        return bias

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, H = x.shape
        if T <= 1:
            return x
        key_padding_mask = None if attention_mask is None else (attention_mask.to(dtype=torch.bool) == 0)

        xc = self.coarse_conv(x.transpose(1, 2))
        xc = F.interpolate(xc, size=T, mode="linear", align_corners=False)
        coarse = xc.transpose(1, 2)

        qkv = self.fine_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        hd = int(self.hidden_size // self.n_head)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        bias = self._local_bias(T, x.device, x.dtype).view(1, 1, T, T)
        if key_padding_mask is not None:
            kp = key_padding_mask.view(B, 1, 1, T)
            bias = bias + kp.to(dtype=x.dtype) * torch.finfo(x.dtype).min
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = torch.softmax(att + bias, dim=-1)
        fine = att @ v
        fine = fine.transpose(1, 2).contiguous().view(B, T, H)
        fine = self.ln1(x + self.fine_dropout(self.fine_proj(fine)))

        fused, _ = self.cross_attn(
            fine,
            coarse,
            coarse,
            attn_mask=self._get_causal_mask(T, x.device),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.ln2(fine + fused)


class MultiScaleDecisionTransformer(DecisionTransformer):
    def __init__(
        self,
        *,
        state_dim: int,
        act_dim: int,
        state_mean,
        state_std,
        action_tanh: bool = False,
        K: int = 20,
        max_ep_len: int = 48,
        scale: float = 40.0,
        target_return: float = 50.0,
        return_dim: int = 1,
        coarse_idx: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        fine_idx: Sequence[int] = (0, 1, 12, 13, 14, 15),
        constraint_idx: Sequence[int] = (0, 1, 4, 8, 10, 12, 13, 14, 15),
        local_window: int = 3,
        n_head: int = 4,
        backbone_variant: str = "legacy",
    ):
        super().__init__(
            state_dim=state_dim,
            act_dim=act_dim,
            state_mean=state_mean,
            state_std=state_std,
            action_tanh=action_tanh,
            K=K,
            max_ep_len=max_ep_len,
            scale=scale,
            target_return=target_return,
            return_dim=return_dim,
        )
        self.backbone_variant = str(backbone_variant)
        context_len = max(128, int(3 * K + 8)) if self.backbone_variant == "csm" else 1024
        block_config = {
            "n_ctx": context_len,
            "n_embd": int(self.hidden_size),
            "n_layer": 3,
            "n_head": int(n_head),
            "n_inner": 512,
            "activation_function": "relu",
            "n_position": context_len,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
        }
        block_config["n_head"] = int(n_head)
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config["n_layer"])])

        if self.backbone_variant == "csm":
            self.state_encoder = ConstraintStructuredStateEncoder(
                state_dim=int(state_dim),
                hidden_size=int(self.hidden_size),
                coarse_idx=coarse_idx,
                fine_idx=fine_idx,
                constraint_idx=constraint_idx,
            )
        else:
            self.state_encoder = GranularityCalibrator(
                state_dim=int(state_dim),
                hidden_size=int(self.hidden_size),
                coarse_idx=coarse_idx,
                fine_idx=fine_idx,
            )
        self.temporal_fusion = CrossGranularityTemporalFusion(
            hidden_size=int(self.hidden_size),
            n_head=int(n_head),
            local_window=int(local_window),
            dropout=0.0,
        )
        if self.backbone_variant == "csm":
            self.granularity_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Sigmoid(),
            )
        else:
            self.granularity_attn = nn.Parameter(torch.zeros((self.hidden_size,)))

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, return_features=False):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        encoder_extras = None
        if self.backbone_variant == "csm":
            state_embeddings, encoder_extras = self.state_encoder(states, return_features=True)
        else:
            state_embeddings = self.state_encoder(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack([attention_mask for _ in range(self.length_times)], dim=1)
        stacked_attention_mask = stacked_attention_mask.permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length)
        stacked_attention_mask = stacked_attention_mask.to(stacked_inputs.dtype)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        x = x.reshape(batch_size, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)
        state_ctx = x[:, 1]
        fused_state_ctx = self.temporal_fusion(state_ctx, attention_mask=attention_mask)
        if self.backbone_variant == "csm":
            adaptive_gate = self.granularity_gate(torch.cat([state_ctx, fused_state_ctx], dim=-1))
            fused_state_ctx = adaptive_gate * fused_state_ctx + (1.0 - adaptive_gate) * state_ctx
        else:
            adaptive_gate = None
            fused_state_ctx = fused_state_ctx * torch.sigmoid(self.granularity_attn).view(1, 1, -1)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(fused_state_ctx)
        extras = None
        if return_features:
            extras = {
                "state_ctx": state_ctx,
                "fused_state_ctx": fused_state_ctx,
                "action_token_ctx": x[:, 2],
            }
            if encoder_extras is not None:
                extras.update(encoder_extras)
            if adaptive_gate is not None:
                extras["adaptive_granularity_gate"] = adaptive_gate
        return state_preds, action_preds, return_preds, extras


# ---------------------------------------------------------------------------
# Auxiliary Separation GranularityCalibrator
# ---------------------------------------------------------------------------

class GranularityCalibrator_AuxSep(nn.Module):
    """
    GranularityCalibrator with auxiliary supervision losses to enforce
    genuine branch specialization.

    Adds two linear probe heads:
      - budget_probe: Linear(H,1) on e_coarse → predicts budget_consumption_ratio
      - market_probe: Linear(H,1) on e_fine   → predicts pValue (state dim 12)

    During training, auxiliary losses push each branch toward its target signal.
    At inference, the probe heads are ignored; only the fused representation is used.

    Architecture is otherwise identical to GranularityCalibrator — same shared_mlp,
    same calibration step, same gate. The only addition is supervised regularization.

    Loss returned (used by method_model.py):
      aux_loss = λ_budget * MSE(budget_pred, budget_ratio)
               + λ_market * MSE(market_pred, pvalue_t)

    References:
      - Alain & Bengio (2016) "Understanding intermediate layers using linear classifier probes"
      - Learning disentangled representations via auxiliary supervision (various)
    """

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_size: int,
        coarse_idx: Sequence[int],
        fine_idx: Sequence[int],
        budget_dim: int = 1,   # state dim index for budget_left (used as probe target)
        pvalue_dim: int = 12,  # state dim index for pValue (market signal probe target)
        aux_lambda_budget: float = 0.1,
        aux_lambda_market: float = 0.1,
    ):
        super().__init__()
        self.state_dim  = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.coarse_idx = tuple(int(i) for i in coarse_idx)
        self.fine_idx   = tuple(int(i) for i in fine_idx)
        self.budget_dim = int(budget_dim)
        self.pvalue_dim = int(pvalue_dim)
        self.aux_lambda_budget = float(aux_lambda_budget)
        self.aux_lambda_market = float(aux_lambda_market)

        # Identical to GranularityCalibrator
        self.common_dim = int(max(len(self.coarse_idx), len(self.fine_idx)))
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.common_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.bias_to_w = nn.Linear(1, 1)
        self.fuse_gate  = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out_ln     = nn.LayerNorm(self.hidden_size)

        # Auxiliary probe heads (only active during training)
        self.budget_probe = nn.Linear(self.hidden_size, 1)
        self.market_probe = nn.Linear(self.hidden_size, 1)

    def _pad_to_common(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.common_dim: return x
        if x.shape[-1] > self.common_dim:  return x[..., :self.common_dim]
        return F.pad(x, (0, self.common_dim - x.shape[-1]))

    def forward(
        self,
        states: torch.Tensor,
        return_aux_loss: bool = False,
    ):
        """
        states: (B, T, state_dim) normalized
        return_aux_loss: if True, also return scalar aux_loss (training only)
        Returns: fused (B, T, H), [aux_loss if return_aux_loss]
        """
        dev = states.device
        coarse = states.index_select(-1, torch.tensor(self.coarse_idx, device=dev))
        fine   = states.index_select(-1, torch.tensor(self.fine_idx,   device=dev))
        coarse = self._pad_to_common(coarse)
        fine   = self._pad_to_common(fine)

        e_c = self.shared_mlp(coarse)
        e_f = self.shared_mlp(fine)

        cos  = F.cosine_similarity(e_c, e_f, dim=-1, eps=1e-8)
        l2   = torch.norm(e_c - e_f, p=2, dim=-1) / (float(self.hidden_size) ** 0.5)
        bias = (1.0 - cos) + l2
        w    = torch.sigmoid(self.bias_to_w(bias.unsqueeze(-1))).squeeze(-1)
        e_cc = e_c + w.unsqueeze(-1) * (e_f - e_c)
        g    = torch.sigmoid(self.fuse_gate(torch.cat([e_cc, e_f], dim=-1)))
        out  = self.out_ln(g * e_f + (1.0 - g) * e_cc)

        if return_aux_loss:
            # Probe targets from raw normalized states
            budget_target = states[..., self.budget_dim:self.budget_dim+1]   # (B,T,1)
            pvalue_target = states[..., self.pvalue_dim:self.pvalue_dim+1]   # (B,T,1)
            budget_pred = self.budget_probe(e_c)   # coarse → budget
            market_pred = self.market_probe(e_f)   # fine   → pValue
            aux_loss = (
                self.aux_lambda_budget * F.mse_loss(budget_pred, budget_target) +
                self.aux_lambda_market * F.mse_loss(market_pred, pvalue_target)
            )
            return out, aux_loss

        return out
