import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dt import Block, DecisionTransformer
from .dbm_backbone import GranularityCalibrator

class CausalTemporalFusionV2(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        n_head: int = 4,
        local_window: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_head = int(n_head)

        self.window = nn.Parameter(torch.tensor(float(local_window)))
        self.window_beta = nn.Parameter(torch.tensor(4.0))
        self.window_temp = nn.Parameter(torch.tensor(1.0))

        self._causal_pad = 2
        self.coarse_conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        self.fine_qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.fine_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.fine_dropout = nn.Dropout(float(dropout))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.n_head,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)

    def _local_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = torch.arange(T, device=device)
        dist = (idx.view(T, 1) - idx.view(1, T)).clamp(min=0).to(dtype)
        future = (idx.view(T, 1) - idx.view(1, T)) < 0
        w = self.window.to(dtype).clamp(min=1.0, max=float(T))
        beta = self.window_beta.to(dtype).clamp(min=0.0)
        temp = self.window_temp.to(dtype).clamp(min=1e-3)
        bias = -beta * F.softplus((dist - w) / temp)
        return bias.masked_fill(future, torch.finfo(dtype).min)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, H = x.shape
        if T <= 1:
            return x

        key_padding_mask = None if attention_mask is None else (attention_mask.to(torch.bool) == 0)

        xt = F.pad(x.transpose(1, 2), (self._causal_pad, 0))
        coarse = self.coarse_conv(xt).transpose(1, 2)

        qkv = self.fine_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        hd = H // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        bias = self._local_bias(T, x.device, x.dtype).view(1, 1, T, T)
        if key_padding_mask is not None:
            bias = bias + key_padding_mask.view(B, 1, 1, T).to(x.dtype) * torch.finfo(x.dtype).min
        att = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(hd) + bias, dim=-1)
        fine = (att @ v).transpose(1, 2).contiguous().view(B, T, H)
        fine = self.ln1(x + self.fine_dropout(self.fine_proj(fine)))

        fused, _ = self.cross_attn(
            fine, coarse, coarse,
            attn_mask=self._causal_mask(T, x.device),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.ln2(fine + fused)

class ConstraintModulator(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        budget = states[:, :, 1:2]
        time = states[:, :, 0:1]
        urgency = 1.0 - torch.sigmoid(budget)
        feat = torch.cat([budget, time, urgency], dim=-1)
        mod = self.proj(feat)
        gate = torch.sigmoid(self.gate(x))
        cpa_scale = getattr(self, "_cpa_scale", None)
        if cpa_scale is not None:
            mod = mod * cpa_scale
        return x + gate * mod

class MultiScaleDecisionTransformerV2(DecisionTransformer):

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
        backbone_variant: str = "v2",
        hidden_size: int = 64,
        n_layers: int = 3,
        n_inner: int = 0,
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

        self.hidden_size = int(hidden_size)
        n_layers = int(n_layers)
        _n_inner = int(n_inner) if n_inner and n_inner > 0 else self.hidden_size * 4
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return   = nn.Linear(self.return_dim, self.hidden_size)
        self.embed_reward   = nn.Linear(1, self.hidden_size)
        self.embed_state    = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action   = nn.Linear(self.act_dim, self.hidden_size)
        self.embed_ln       = nn.LayerNorm(self.hidden_size)
        self.predict_state  = nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Linear(self.hidden_size, self.act_dim)
        self.predict_return = nn.Linear(self.hidden_size, 1)

        context_len = 1024
        block_config = {
            "n_ctx": context_len,
            "n_embd": self.hidden_size,
            "n_layer": n_layers,
            "n_head": int(n_head),
            "n_inner": _n_inner,
            "activation_function": "relu",
            "n_position": context_len,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
        }
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(n_layers)])

        self.state_encoder = GranularityCalibrator(
            state_dim=int(state_dim), hidden_size=int(self.hidden_size),
            coarse_idx=coarse_idx, fine_idx=fine_idx,
        )

        self.temporal_fusion = CausalTemporalFusionV2(
            hidden_size=int(self.hidden_size),
            n_head=int(n_head),
            local_window=int(local_window),
        )

        self.granularity_gate_net = nn.Linear(int(self.hidden_size), int(self.hidden_size))

        self.constraint_mod = ConstraintModulator(int(self.hidden_size))

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        cpa_constraint: Optional[torch.Tensor] = None,
    ):
        B, T = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long, device=states.device)

        state_embeddings = self.state_encoder(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(B, 3 * T, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_mask = (
            torch.stack([attention_mask] * self.length_times, dim=1)
            .permute(0, 2, 1)
            .reshape(B, self.length_times * T)
            .to(stacked_inputs.dtype)
        )

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_mask)

        x = x.reshape(B, T, self.length_times, self.hidden_size).permute(0, 2, 1, 3)
        state_ctx = x[:, 1]

        fused = self.temporal_fusion(state_ctx, attention_mask=attention_mask)

        fused = self.constraint_mod(fused, states)

        gate = torch.sigmoid(self.granularity_gate_net(state_ctx))
        fused_state_ctx = gate * fused + (1.0 - gate) * state_ctx

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
        return state_preds, action_preds, return_preds, extras

    def take_actions(self, state, target_return=None, pre_reward=None, pre_cost=None, cpa_constraint=None):
        self.eval()
        device = self._eval_device()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(device=device, dtype=torch.float32)
            self.eval_target_return = self._format_target_return(target_return, device=device)
        else:
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(device=device, dtype=torch.float32)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)
            self.eval_rewards[-1] = pre_reward
            pred_return = self.eval_target_return[:, -1, :].clone()
            
            pred_return[:, 0] = pred_return[:, 0] - (pre_reward / self.scale)
            
            if self.return_dim > 1 and pre_cost is not None:
                if cpa_constraint is not None:
                    cpa_c = float(cpa_constraint)
                    delta = (-cpa_c * pre_reward + pre_cost) / self.scale
                    pred_return[:, 1] = pred_return[:, 1] + delta
                else:
                    pred_return[:, 1] = torch.clamp(pred_return[:, 1] - (pre_cost / self.scale), min=0.0)
            
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.unsqueeze(1)], dim=1)
            self.eval_timesteps = torch.cat(
                [
                    self.eval_timesteps,
                    torch.ones((1, 1), dtype=torch.long, device=device) * self.eval_timesteps[:, -1] + 1
                ],
                dim=1,
            )
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim, device=device)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1, device=device)])

        action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - self.state_mean.to(device=device, dtype=torch.float32))
            / self.state_std.to(device=device, dtype=torch.float32),
            self.eval_actions.to(dtype=torch.float32, device=device),
            self.eval_rewards.to(dtype=torch.float32, device=device),
            self.eval_target_return.to(dtype=torch.float32, device=device),
            self.eval_timesteps.to(dtype=torch.long, device=device)
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action
