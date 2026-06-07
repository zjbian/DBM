"""DBM-Bid model wrapper.

`ResearchDBMModel` wraps the DBM-Bid backbone (``MultiScaleDecisionTransformerV2``)
with the constraint-aware training objective used in the paper:

* feature/temporal dual-branch encoding is performed inside the backbone;
* the wrapper adds (i) AWR-style per-sample weighting, (ii) selective-imitation
  quality weighting/loss, (iii) a budget-feasibility (C2) token weight,
  (iv) a prefix-feasibility token weight, and (v) a conservative regularizer that
  discourages over-bidding once the budget is tight.

Only the components used by the paper's model are kept here.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dbm_v2 import MultiScaleDecisionTransformerV2

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (x * mask).sum(dim=1) / denom

class ResearchDBMModel(nn.Module):
    def __init__(self, *, config: Dict):
        super().__init__()
        self.config = dict(config)
        self.state_dim = int(config["state_dim"])
        self.act_dim = int(config["act_dim"])
        self.meta_dim = int(config.get("meta_dim", 2))
        self.return_dim = int(config.get("return_dim", 1))

        zeros = torch.zeros((self.state_dim,), dtype=torch.float32)
        ones = torch.ones((self.state_dim,), dtype=torch.float32)
        backbone_kwargs = {
            "state_dim": self.state_dim,
            "act_dim": self.act_dim,
            "state_mean": zeros,
            "state_std": ones,
            "action_tanh": False,
            "K": int(config["K"]),
            "max_ep_len": int(config["max_ep_len"]),
            "scale": float(config["scale"]),
            "target_return": float(config["target_return"]),
            "return_dim": self.return_dim,
            "coarse_idx": tuple(config["coarse_idx"]),
            "fine_idx": tuple(config["fine_idx"]),
            "constraint_idx": tuple(config.get("constraint_idx", (0, 1, 4, 8, 10, 12, 13, 14, 15))),
            "local_window": int(config["local_window"]),
            "n_head": int(config["n_head"]),
            "backbone_variant": "v2",
        }
        self.backbone = MultiScaleDecisionTransformerV2(**backbone_kwargs)
        self.backbone_variant = "v2"
        self.hidden_size = int(self.backbone.hidden_size)

        self.input_adapter = nn.Sequential(
            nn.Linear(self.state_dim + self.meta_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.state_ln = nn.LayerNorm(self.state_dim)

        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode_states(self, states: torch.Tensor, meta: torch.Tensor, retrieval_context: torch.Tensor = None) -> torch.Tensor:
        meta_expand = meta.unsqueeze(1).expand(-1, states.shape[1], -1)
        encoded = states + self.input_adapter(torch.cat([states, meta_expand], dim=-1))
        return self.state_ln(encoded)

    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        rtg = batch["rtg"]
        timesteps = batch["timesteps"]
        mask = batch["mask"].long()
        meta = batch["meta"]
        retrieval_context = batch["retrieval_context"]
        heuristic_actions = batch["heuristic_actions"]

        encoded_states = self.encode_states(states, meta, retrieval_context)
        _, action_preds, _, extras = self.backbone.forward(
            encoded_states, actions, rewards, rtg[:, :-1], timesteps,
            attention_mask=mask, return_features=True,
        )
        if extras is None:
            extras = {"fused_state_ctx": encoded_states}
        final_action_preds = action_preds

        valid = mask.reshape(-1) > 0
        pred_flat = final_action_preds.reshape(-1, self.act_dim)[valid]
        target_flat = actions.reshape(-1, self.act_dim)[valid]
        per_token_loss = ((pred_flat - target_flat) ** 2).mean(dim=-1)

        token_weights = batch["sample_weight"].unsqueeze(1).expand(-1, mask.shape[1]).reshape(-1)[valid]
        pooled = masked_mean(extras["fused_state_ctx"], mask)

        quality_pred = torch.sigmoid(self.quality_head(pooled)).squeeze(-1) * 2.0
        quality_expand = quality_pred.unsqueeze(1).expand(-1, mask.shape[1]).reshape(-1)[valid]
        token_weights = token_weights * (0.5 + quality_expand / 2.0)

        budget_left = states[:, :, 1].reshape(-1)[valid]
        c2_scale = float(self.config.get("c2_budget_scale", 3.0))
        token_weights = token_weights * (0.5 + 0.5 * torch.sigmoid(budget_left * c2_scale))

        pfeas = batch["prefix_feasibility"].squeeze(-1).reshape(-1)[valid]
        pfeas_scale = float(self.config.get("pfeas_scale", 1.5))
        token_weights = token_weights * (0.5 + pfeas_scale * pfeas)

        token_weights = token_weights / token_weights.mean().clamp_min(1e-6)
        action_loss = (per_token_loss * token_weights).mean()

        total_loss = action_loss
        out = {"loss": total_loss, "action_loss": action_loss.detach()}

        if self.config.get("use_conservative_reg", False):
            budget_left_seq = states[:, :, 1]
            tight_threshold = float(self.config.get("cons_tight_threshold", 0.3))
            tight_mask = (budget_left_seq < tight_threshold).float().unsqueeze(-1)
            overbid = torch.relu(final_action_preds - heuristic_actions)
            cons_loss = (overbid ** 2 * tight_mask * mask.float().unsqueeze(-1)).sum() / (
                tight_mask * mask.float().unsqueeze(-1)).sum().clamp_min(1.0)
            cons_weight = float(self.config.get("cons_weight", 0.3))
            total_loss = total_loss + cons_weight * cons_loss
            out["cons_loss"] = cons_loss.detach()

        if self.config.get("use_selective_imitation", False):
            quality_target = batch["quality_target"].clamp(0.0, 2.0)
            selective_loss = F.smooth_l1_loss(quality_pred, quality_target)
            total_loss = total_loss + float(self.config.get("selective_weight", 0.35)) * selective_loss
            out["selective_loss"] = selective_loss.detach()

        out["loss"] = total_loss
        return out

    def init_eval(self):
        self.backbone.init_eval()

    @torch.no_grad()
    def predict_step(
        self,
        *,
        state_norm: torch.Tensor,
        meta: torch.Tensor,
        retrieval_context: torch.Tensor,
        heuristic_action_norm: torch.Tensor,
        target_return: float,
        target_budget_return: float = None,
        pre_reward: float = None,
        pre_cost: float = None,
        cpa_constraint: float = None,
        pfeas_step: float = None,
    ) -> Dict[str, torch.Tensor]:
        if state_norm.ndim == 1:
            state_norm = state_norm.unsqueeze(0).unsqueeze(0)
        if meta.ndim == 1:
            meta = meta.unsqueeze(0)
        if retrieval_context.ndim == 1:
            retrieval_context = retrieval_context.unsqueeze(0)

        encoded = self.encode_states(state_norm, meta, retrieval_context).squeeze(0).squeeze(0)
        target_payload = float(target_return) if target_return is not None else None
        take_kwargs = {"target_return": target_payload, "pre_reward": pre_reward, "pre_cost": pre_cost}
        if getattr(self.backbone, "supports_pfeas_attention", False) and pfeas_step is not None:
            take_kwargs["pfeas_step"] = float(pfeas_step)
        pred = self.backbone.take_actions(encoded.detach().cpu().numpy(), **take_kwargs)
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().item()
        else:
            pred = float(pred.squeeze().item())

        return {
            "action_norm": torch.tensor(pred, dtype=torch.float32),
            "alpha_scale": torch.tensor(1.0, dtype=torch.float32),
            "target_scale": torch.tensor(1.0, dtype=torch.float32),
        }
