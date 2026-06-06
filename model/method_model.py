from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dt import DecisionTransformer
from .msdt_backbone import MultiScaleDecisionTransformer
from .msdt_v2 import MultiScaleDecisionTransformerV2


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (x * mask).sum(dim=1) / denom


def smooth_l1_per_token(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    beta = max(float(beta), 1e-6)
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    return loss.mean(dim=-1)


class StructuralStateEncoder(nn.Module):
    def __init__(self, state_dim: int, meta_dim: int):
        super().__init__()
        self.group_indices = [
            [0, 1, 13, 14, 15],
            [2, 3, 4, 8],
            [5, 9, 12],
            [6, 7, 10, 11],
        ]
        self.group_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(len(group), 16),
                    nn.ReLU(),
                    nn.Linear(16, 16),
                )
                for group in self.group_indices
            ]
        )
        self.meta_proj = nn.Sequential(nn.Linear(meta_dim, 16), nn.ReLU(), nn.Linear(16, 16))
        self.out = nn.Sequential(nn.Linear(16 * (len(self.group_indices) + 1), 32), nn.ReLU(), nn.Linear(32, state_dim))
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, states: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        meta_expand = meta.unsqueeze(1).expand(-1, states.shape[1], -1)
        parts = []
        for group, mlp in zip(self.group_indices, self.group_mlps):
            parts.append(mlp(states[..., group]))
        parts.append(self.meta_proj(meta_expand))
        fused = torch.cat(parts, dim=-1)
        return self.ln(self.out(fused))


class DynamicsStateAdapter(nn.Module):
    def __init__(self, state_dim: int, meta_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8 + meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim),
        )
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, states: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        meta_expand = meta.unsqueeze(1).expand(-1, states.shape[1], -1)
        deltas = torch.stack(
            [
                states[..., 3] - states[..., 2],
                states[..., 8] - states[..., 4],
                states[..., 9] - states[..., 5],
                states[..., 10] - states[..., 6],
                states[..., 11] - states[..., 7],
                states[..., 13],
                states[..., 14],
                states[..., 15],
            ],
            dim=-1,
        )
        return self.ln(self.net(torch.cat([deltas, meta_expand], dim=-1)))


class ResearchMSDTModel(nn.Module):
    def __init__(self, *, config: Dict):
        super().__init__()
        self.config = dict(config)
        self.state_dim = int(config["state_dim"])
        self.act_dim = int(config["act_dim"])
        self.meta_dim = int(config.get("meta_dim", 2))
        self.retrieval_dim = int(config.get("retrieval_dim", 8))
        self.return_dim = int(config.get("return_dim", 1))
        self.future_context_dim = self.state_dim + self.meta_dim + self.retrieval_dim + self.act_dim

        zeros = torch.zeros((self.state_dim,), dtype=torch.float32)
        ones = torch.ones((self.state_dim,), dtype=torch.float32)
        backbone_variant = str(config.get("backbone_variant", "v2"))
        # DBM-Bid uses the v2 backbone (the only variant retained in this release).
        if backbone_variant == "base_dt":
            backbone_cls = DecisionTransformer
        elif backbone_variant != "v2":
            backbone_cls = MultiScaleDecisionTransformer
        else:
            backbone_cls = MultiScaleDecisionTransformerV2
        # base_dt uses only the minimal kwargs that DecisionTransformer accepts
        _base_kwargs = {
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
        }
        if backbone_variant == "base_dt":
            backbone_kwargs = _base_kwargs
        elif backbone_variant == "v3":
            # H-MSDT v3: 真正分层架构，不使用 coarse_idx/fine_idx
            backbone_kwargs = {
                **_base_kwargs,
                "hidden_size": int(config.get("hidden_size", 64)),
                "n_layers": int(config.get("n_layers", 3)),
                "n_head": int(config.get("n_head", 4)),
                "n_inner": int(config.get("n_inner", 0)),
                "local_window": int(config.get("local_window", 3)),
                "cpa_emb_dim": int(config.get("cpa_emb_dim", 16)),
                "slow_idx": tuple(config.get("slow_idx", (0,1,2,3,4,5,6,7,8,9,10,11))),
                "fast_idx": tuple(config.get("fast_idx", (0,1,12,13,14,15))),
                "disable_hierarchical": bool(config.get("disable_hierarchical", False)),
                "disable_fast_path": bool(config.get("disable_fast_path", False)),
                "disable_cpa_gate": bool(config.get("disable_cpa_gate", False)),
                "use_cpa_film": bool(config.get("use_cpa_film", False)),
                "use_feasibility_head": bool(config.get("use_feasibility_head", False)),
                "feas_max_delta": float(config.get("feas_max_delta", 0.5)),
                "backbone_variant": "v3",
            }
        elif backbone_variant == "wave":
            # MSDT-Wave: 小波多分辨率编码器 + 标准 DT, 不使用 coarse_idx/fine_idx
            backbone_kwargs = {
                **_base_kwargs,
                "n_layers": int(config.get("n_layers", 3)),
                "n_head": int(config.get("n_head", 4)),
                "n_levels": int(config.get("wave_n_levels", 3)),
                "dropout": float(config.get("wave_dropout", 0.1)),
            }
        else:
            backbone_kwargs = {
                **_base_kwargs,
                "coarse_idx": tuple(config["coarse_idx"]),
                "fine_idx": tuple(config["fine_idx"]),
                "constraint_idx": tuple(config.get("constraint_idx", (0, 1, 4, 8, 10, 12, 13, 14, 15))),
                "local_window": int(config["local_window"]),
                "n_head": int(config["n_head"]),
                "backbone_variant": str(config.get("backbone_variant", "legacy")),
            }
        if backbone_variant == "hierarchical_v1":
            backbone_kwargs["macro_idx"] = tuple(config.get("macro_idx", config["coarse_idx"]))
            backbone_kwargs["micro_idx"] = tuple(config.get("micro_idx", config["fine_idx"]))
            backbone_kwargs["control_dim"] = int(config.get("control_dim", 3))
        if backbone_variant == "v2" and "use_dual_path_action_head" in config:
            backbone_kwargs["use_dual_path_action_head"] = bool(config["use_dual_path_action_head"])
        if backbone_variant == "v2" and "use_asymmetric_action_head" in config:
            backbone_kwargs["use_asymmetric_action_head"] = bool(config["use_asymmetric_action_head"])
        if backbone_variant == "v2" and config.get("use_rtg_histogram", False):
            backbone_kwargs["use_rtg_histogram"] = True
            backbone_kwargs["rtg_num_bins"] = int(config.get("rtg_num_bins", 20))
            backbone_kwargs["rtg_bin_min"] = float(config.get("rtg_bin_min", 0.0))
            backbone_kwargs["rtg_bin_max"] = float(config.get("rtg_bin_max", 2.0))
        if backbone_variant == "v2" and config.get("disable_cross_granularity", False):
            backbone_kwargs["disable_cross_granularity"] = True
        if backbone_variant == "v2" and config.get("disable_multiscale", False):
            backbone_kwargs["disable_multiscale"] = True
        if backbone_variant == "v2" and config.get("disable_dyn_gate", False):
            backbone_kwargs["disable_dyn_gate"] = True
        if backbone_variant == "v2" and config.get("use_dual_branch_v2", False):
            backbone_kwargs["use_dual_branch_v2"] = True
        if backbone_variant == "v2" and config.get("use_film_encoder", False):
            backbone_kwargs["use_film_encoder"] = True
        if backbone_variant == "v2" and config.get("use_cross_stitch_encoder", False):
            backbone_kwargs["use_cross_stitch_encoder"] = True
        if backbone_variant == "v2" and config.get("use_aux_sep", False):
            backbone_kwargs["use_aux_sep"] = True
            backbone_kwargs["aux_lambda_budget"] = float(config.get("aux_lambda_budget", 0.1))
            backbone_kwargs["aux_lambda_market"] = float(config.get("aux_lambda_market", 0.1))
        if backbone_variant == "v2" and config.get("use_cpa_moe", False):
            backbone_kwargs["use_cpa_moe"] = True
            backbone_kwargs["cpa_moe_num_experts"] = int(config.get("cpa_moe_num_experts", 3))
            backbone_kwargs["cpa_moe_emb_dim"] = int(config.get("cpa_moe_emb_dim", 16))
            backbone_kwargs["cpa_moe_expert_ratio"] = float(config.get("cpa_moe_expert_ratio", 1.0))
        if backbone_variant == "v2" and config.get("use_cpa_mdn", False):
            # MDN action head: replaces predict_action with PerCPAMDN
            backbone_kwargs["use_cpa_mdn"] = True
            backbone_kwargs["cpa_mdn_num_components"] = int(config.get("cpa_mdn_num_components", 3))
            backbone_kwargs["cpa_mdn_emb_dim"] = int(config.get("cpa_mdn_emb_dim", 16))
            backbone_kwargs["cpa_mdn_sigma_min"] = float(config.get("cpa_mdn_sigma_min", 0.01))
            backbone_kwargs["cpa_mdn_sigma_max"] = float(config.get("cpa_mdn_sigma_max", 2.0))
            backbone_kwargs["cpa_mdn_infer_mode"] = str(config.get("cpa_mdn_infer_mode", "max_pi"))
        if backbone_variant == "v2" and "hidden_size" in config:
            backbone_kwargs["hidden_size"] = int(config["hidden_size"])
        if backbone_variant == "v2" and "n_layers" in config:
            backbone_kwargs["n_layers"] = int(config["n_layers"])
        if backbone_variant == "v2" and "n_inner" in config:
            backbone_kwargs["n_inner"] = int(config["n_inner"])

        # v2_ablate: 复用 v2 的 else-kwargs(coarse/fine/...) + backbone 拆解开关
        if backbone_variant == "v2_ablate":
            for _k in ("disable_cross_granularity", "disable_multiscale",
                       "disable_dyn_gate", "disable_constraint_mod"):
                if config.get(_k, False):
                    backbone_kwargs[_k] = True

        # v2 优化 L3/L4：推理 MC-dropout 参数（不影响训练）
        if backbone_variant in ("v2_l3", "v2_l4"):
            backbone_kwargs["mc_samples"] = int(config.get("mc_samples", 8))
            backbone_kwargs["mc_uncertainty_coef"] = float(config.get("mc_uncertainty_coef", 0.0))

        # v2_gran：粗/细分歧利用（DG1 压价 / DG2 双视角 / DG3 多样性）
        if backbone_variant == "v2_gran":
            backbone_kwargs["dual_view"] = bool(config.get("dual_view", False))
            backbone_kwargs["disagree_shade"] = bool(config.get("disagree_shade", False))
            backbone_kwargs["disagree_k"] = float(config.get("disagree_k", 0.1))
            backbone_kwargs["disagree_ref"] = float(config.get("disagree_ref", 1.0))

        self.backbone = backbone_cls(**backbone_kwargs)
        self.backbone_variant = backbone_variant

        self.hidden_size = int(self.backbone.hidden_size)

        self.input_adapter = nn.Sequential(
            nn.Linear(self.state_dim + self.meta_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.structural_encoder = StructuralStateEncoder(self.state_dim, self.meta_dim)
        self.dynamics_adapter = DynamicsStateAdapter(self.state_dim, self.meta_dim)
        self.retrieval_proj = nn.Sequential(
            nn.Linear(self.retrieval_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.meta_proj = nn.Sequential(
            nn.Linear(self.meta_dim, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.state_ln = nn.LayerNorm(self.state_dim)

        self.preference_head = nn.Sequential(
            nn.Linear(self.hidden_size + self.act_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.future_head = nn.Sequential(
            nn.Linear(self.future_context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.act_dim),
        )
        self.meta_head = nn.Sequential(
            nn.Linear(self.state_dim + self.meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.q_head1 = nn.Sequential(
            nn.Linear(self.future_context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.q_head2 = nn.Sequential(
            nn.Linear(self.future_context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.energy_head1 = nn.Sequential(
            nn.Linear(self.future_context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.energy_head2 = nn.Sequential(
            nn.Linear(self.future_context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.bid2x_head = nn.Sequential(
            nn.Linear(self.future_context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.expert_head = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.act_dim),
        )
        self.router_in_dim = self.state_dim + self.meta_dim
        self.num_router_experts = int(self.config.get("num_router_experts", 3))
        self.router_head = nn.Sequential(
            nn.Linear(self.router_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_router_experts),
        )
        self.router_expert_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.state_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, self.act_dim),
                )
                for _ in range(self.num_router_experts)
            ]
        )
        self.feasibility_head = nn.Sequential(
            nn.Linear(self.future_context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Direction 3: CPA-Pressure Adaptive Gate
        # Input: [cpa_ratio_norm, time_left_norm, budget_left_norm] → scalar gate ∈ [0, 1]
        # gate≈1 (CPA healthy) → keep full action; gate≈0 (CPA tight) → reduce action
        if self.config.get("use_cpa_gate", False):
            self.cpa_gate_net = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        # 方向B: Contrastive trajectory learning
        # Projects hidden states to a lower-dim space for InfoNCE contrastive loss
        if self.config.get("use_contrastive_loss", False):
            proj_dim = int(self.config.get("contrastive_proj_dim", 64))
            self.contrastive_proj = nn.Sequential(
                nn.Linear(self.hidden_size, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )

    def _adaptive_constraint_risk(self, batch: Dict[str, torch.Tensor], states: torch.Tensor) -> torch.Tensor:
        budget_left = states[:, :, 1]
        time_left = states[:, :, 0]
        prefix_feasibility = batch["prefix_feasibility"].squeeze(-1)
        burn_rate = torch.relu(batch["costs"].squeeze(-1))
        if self.config.get("use_cpa_state_features", False):
            cpa_pressure = torch.relu(states[:, :, -1])
        else:
            cpa_pressure = 1.0 - prefix_feasibility
        risk_logit = (
            float(self.config.get("risk_budget_coeff", 2.5)) * (float(self.config.get("cons_tight_threshold", 0.3)) - budget_left)
            + float(self.config.get("risk_cpa_coeff", 1.5)) * cpa_pressure
            + float(self.config.get("risk_burn_coeff", 0.75)) * burn_rate
            - float(self.config.get("risk_time_coeff", 1.25)) * (1.0 - time_left)
            + float(self.config.get("risk_pfeas_coeff", 1.5)) * (1.0 - prefix_feasibility)
        )
        return torch.sigmoid(risk_logit)

    def encode_states(self, states: torch.Tensor, meta: torch.Tensor, retrieval_context: torch.Tensor) -> torch.Tensor:
        meta_expand = meta.unsqueeze(1).expand(-1, states.shape[1], -1)
        encoded = states + self.input_adapter(torch.cat([states, meta_expand], dim=-1))

        if self.config.get("use_structural_encoder", False):
            structural = self.structural_encoder(states, meta)
            encoded = encoded + float(self.config.get("structural_mix", 0.35)) * structural

        if self.config.get("use_dynamics_adapter", False):
            encoded = encoded + 0.35 * self.dynamics_adapter(states, meta)

        if self.config.get("use_retrieval_aug", False):
            retrieval_expand = retrieval_context.unsqueeze(1).expand(-1, states.shape[1], -1)
            encoded = encoded + self.retrieval_proj(retrieval_expand)

        if self.config.get("use_meta_calibration", False):
            encoded = encoded + self.meta_proj(meta_expand)

        return self.state_ln(encoded)

    def _future_context(
        self,
        encoded_states: torch.Tensor,
        meta: torch.Tensor,
        retrieval_context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        meta_expand = meta.unsqueeze(1).expand(-1, encoded_states.shape[1], -1)
        retrieval_expand = retrieval_context.unsqueeze(1).expand(-1, encoded_states.shape[1], -1)
        return torch.cat([encoded_states, meta_expand, retrieval_expand, actions], dim=-1)

    def _route_actions(self, encoded_states: torch.Tensor, meta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        meta_expand = meta.unsqueeze(1).expand(-1, encoded_states.shape[1], -1)
        router_inp = torch.cat([encoded_states, meta_expand], dim=-1)
        router_logits = self.router_head(router_inp)
        router_weights = torch.softmax(router_logits, dim=-1)
        expert_deltas = torch.stack([head(encoded_states) for head in self.router_expert_heads], dim=2)
        routed_delta = torch.sum(router_weights.unsqueeze(-1) * expert_deltas, dim=2)
        return routed_delta, router_logits, router_weights

    def _support_center(self, heuristic_actions: torch.Tensor, expert_actions: torch.Tensor) -> torch.Tensor:
        center = heuristic_actions
        if expert_actions is not None:
            center = 0.5 * center + 0.5 * expert_actions
        return center

    def _support_loss(
        self,
        pred_actions: torch.Tensor,
        heuristic_actions: torch.Tensor,
        expert_actions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        center = self._support_center(heuristic_actions, expert_actions)
        band = float(self.config.get("support_band", 0.45))
        gap = F.relu(torch.abs(pred_actions - center) - band)
        valid = mask.unsqueeze(-1).float()
        return ((gap ** 2) * valid).sum() / valid.sum().clamp_min(1.0)

    def _feasibility_loss(
        self,
        encoded_states: torch.Tensor,
        meta: torch.Tensor,
        retrieval_context: torch.Tensor,
        actions: torch.Tensor,
        prefix_feasibility: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feas_inp = self._future_context(encoded_states, meta, retrieval_context, actions)
        feas_logit = self.feasibility_head(feas_inp).squeeze(-1)
        target = prefix_feasibility.squeeze(-1)
        valid = mask > 0
        loss = F.binary_cross_entropy_with_logits(feas_logit[valid], target[valid])
        return loss, feas_logit

    def _apply_support_clip(
        self,
        action_value: float,
        heuristic_action_norm: torch.Tensor,
        expert_action: torch.Tensor = None,
        router_weights: torch.Tensor = None,
    ) -> float:
        if not self.config.get("use_support_clip", False):
            return float(action_value)
        center = float(heuristic_action_norm.item())
        if expert_action is not None and expert_action.numel() > 0:
            center = 0.5 * center + 0.5 * float(expert_action.item())
        band = float(self.config.get("support_band", 0.45)) * float(self.config.get("support_clip_scale", 1.0))
        if router_weights is not None:
            scales = torch.tensor(
                self.config.get("route_expert_scales", [0.75, 1.0, 1.25]),
                dtype=router_weights.dtype,
                device=router_weights.device,
            )
            if scales.numel() == router_weights.numel():
                band = band * float(torch.sum(router_weights * scales).item())
        return float(min(max(action_value, center - band), center + band))

    def _eval_route_adjustment(self, encoded_state: torch.Tensor, meta: torch.Tensor) -> Tuple[float, torch.Tensor]:
        routed_delta, _, router_weights = self._route_actions(encoded_state.unsqueeze(0).unsqueeze(0), meta)
        return float(routed_delta[0, 0, 0].item()), router_weights[0, 0]

    def _build_preference_pairs(self, pooled: torch.Tensor, quality: torch.Tensor) -> List[Tuple[int, int]]:
        batch_size = int(pooled.shape[0])
        if batch_size < 2:
            return []
        if self.config.get("preference_pairing", "adjacent") != "similarity":
            return [(i, i + 1) for i in range(0, batch_size - 1, 2)]

        pooled_norm = F.normalize(pooled, dim=-1, eps=1e-6)
        sim = pooled_norm @ pooled_norm.transpose(0, 1)
        sim.fill_diagonal_(-1e9)
        pairs: List[Tuple[int, int]] = []
        used = set()
        for i in range(batch_size):
            if i in used:
                continue
            row = sim[i].clone()
            if used:
                row[list(used)] = -1e9
            j = int(torch.argmax(row).item())
            if j == i or row[j].item() < -1e8:
                continue
            used.add(i)
            used.add(j)
            pair = (min(i, j), max(i, j))
            pairs.append(pair)
        return pairs

    def _pairwise_preference_loss(self, pooled: torch.Tensor, action_feat: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        pairs = self._build_preference_pairs(pooled, quality)
        if not pairs:
            return pooled.new_tensor(0.0)
        left_idx = torch.tensor([i for i, _ in pairs], device=pooled.device, dtype=torch.long)
        right_idx = torch.tensor([j for _, j in pairs], device=pooled.device, dtype=torch.long)
        left = torch.cat([pooled[left_idx], action_feat[left_idx]], dim=-1)
        right = torch.cat([pooled[right_idx], action_feat[right_idx]], dim=-1)
        left_score = self.preference_head(left).squeeze(-1)
        right_score = self.preference_head(right).squeeze(-1)
        diff = quality[left_idx] - quality[right_idx]
        label = torch.sign(diff)
        valid = torch.abs(diff) > float(self.config.get("quality_margin", 0.1))
        if valid.sum() == 0:
            return pooled.new_tensor(0.0)
        margin = (left_score - right_score) * label
        return F.softplus(-margin[valid]).mean()

    def _action_loss(self, pred_flat: torch.Tensor, target_flat: torch.Tensor, uncertainty_flat: torch.Tensor = None) -> torch.Tensor:
        if self.config.get("use_expectile_loss", False):
            tau = float(self.config.get("expectile_tau", 0.75))
            diff = pred_flat - target_flat
            per_token = (torch.abs(tau - (diff < 0).float()) * diff ** 2).mean(dim=-1)
        elif self.config.get("use_robust_loss", False):
            per_token = smooth_l1_per_token(pred_flat, target_flat, beta=float(self.config.get("robust_beta", 1.0)))
        else:
            per_token = ((pred_flat - target_flat) ** 2).mean(dim=-1)
        if uncertainty_flat is None:
            return per_token
        log_var = torch.clamp(uncertainty_flat.mean(dim=-1), min=-4.0, max=4.0)
        return torch.exp(-log_var) * per_token + float(self.config.get("uncertainty_weight", 0.05)) * log_var

    def _smoothness_loss(self, pred_actions: torch.Tensor, target_actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pair_mask = (mask[:, 1:] * mask[:, :-1]).unsqueeze(-1)
        if pair_mask.sum() <= 0:
            return pred_actions.new_tensor(0.0)
        pred_diff = pred_actions[:, 1:] - pred_actions[:, :-1]
        target_diff = target_actions[:, 1:] - target_actions[:, :-1]
        loss = ((pred_diff - target_diff) ** 2) * pair_mask
        return loss.sum() / pair_mask.sum().clamp_min(1.0)

    def _apply_meta_scale(self, pred: torch.Tensor) -> torch.Tensor:
        target_bound = float(self.config.get("calibration_scale", 0.15))
        alpha_bound = float(self.config.get("calibration_alpha_scale", 0.1))
        return torch.stack(
            [
                1.0 + target_bound * torch.tanh(pred[:, 0]),
                1.0 + alpha_bound * torch.tanh(pred[:, 1]),
            ],
            dim=-1,
        )

    def _judge_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        next_reward = batch["next_rewards"].squeeze(-1)
        next_cost = batch["next_costs"].squeeze(-1)
        next_budget = batch["next_budget_left"].squeeze(-1)
        quality_bonus = (batch["quality_target"].unsqueeze(1) - 1.0)
        return (
            next_reward
            - float(self.config.get("q_cost_weight", 0.35)) * next_cost
            + float(self.config.get("q_budget_weight", 0.05)) * next_budget
            + float(self.config.get("q_quality_weight", 0.2)) * quality_bonus
        )

    def _build_negative_actions(self, actions: torch.Tensor, heuristic_actions: torch.Tensor) -> torch.Tensor:
        deltas = torch.tensor(
            [float(x) for x in self.config.get("candidate_deltas", [-0.75, -0.35, 0.0, 0.35, 0.75])],
            dtype=actions.dtype,
            device=actions.device,
        )
        noise = deltas[torch.randint(0, deltas.shape[0], (actions.shape[0], actions.shape[1]), device=actions.device)].unsqueeze(-1)
        span = float(self.config.get("candidate_span", 0.45))
        base_shift = span * noise
        heuristic_shift = 0.25 * (heuristic_actions - actions)
        return actions + base_shift + heuristic_shift

    def _q_loss(
        self,
        encoded_states: torch.Tensor,
        meta: torch.Tensor,
        retrieval_context: torch.Tensor,
        action_tensor: torch.Tensor,
        judge_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        judge_inp = self._future_context(encoded_states, meta, retrieval_context, action_tensor)
        q1 = self.q_head1(judge_inp).squeeze(-1)
        q2 = self.q_head2(judge_inp).squeeze(-1)
        valid = mask > 0
        target = judge_target[valid]
        loss = F.smooth_l1_loss(q1[valid], target) + F.smooth_l1_loss(q2[valid], target)
        return loss, torch.minimum(q1, q2)

    def _energy_loss(
        self,
        encoded_states: torch.Tensor,
        meta: torch.Tensor,
        retrieval_context: torch.Tensor,
        pos_actions: torch.Tensor,
        neg_actions: torch.Tensor,
        quality_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_inp = self._future_context(encoded_states, meta, retrieval_context, pos_actions)
        neg_inp = self._future_context(encoded_states, meta, retrieval_context, neg_actions)
        pos_e1 = self.energy_head1(pos_inp).squeeze(-1)
        pos_e2 = self.energy_head2(pos_inp).squeeze(-1)
        neg_e1 = self.energy_head1(neg_inp).squeeze(-1)
        neg_e2 = self.energy_head2(neg_inp).squeeze(-1)
        pos_energy = 0.5 * (pos_e1 + pos_e2)
        neg_energy = 0.5 * (neg_e1 + neg_e2)
        valid = mask > 0
        qual = quality_target.unsqueeze(1).expand_as(pos_energy)
        margin = float(self.config.get("energy_margin", 0.25))
        loss = F.relu(margin + pos_energy - neg_energy) * valid.float() * (0.5 + qual / 2.0)
        denom = valid.float().sum().clamp_min(1.0)
        return loss.sum() / denom, pos_energy, neg_energy

    def _bid2x_loss(
        self,
        encoded_states: torch.Tensor,
        meta: torch.Tensor,
        retrieval_context: torch.Tensor,
        actions: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        judge_inp = self._future_context(encoded_states, meta, retrieval_context, actions)
        pred = self.bid2x_head(judge_inp)
        target_quality = batch["quality_target"].unsqueeze(1).unsqueeze(1).expand(-1, actions.shape[1], 1)
        target = torch.cat([batch["next_rewards"], batch["next_costs"], batch["next_budget_left"], target_quality], dim=-1)
        valid = mask.unsqueeze(-1).float()
        loss = (((pred - target) ** 2) * valid).sum() / valid.sum().clamp_min(1.0)
        return loss, pred

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
        adaptive_risk = None

        encoded_states = self.encode_states(states, meta, retrieval_context)
        if self.config.get("use_adaptive_constraint_weights", False):
            adaptive_risk = self._adaptive_constraint_risk(batch, states)

        # CPA-Arch: inject cpa_constraint into ConstraintModulator via meta
        # This makes the backbone itself constraint-aware, not just the loss
        if self.config.get("use_cpa_conditioned_mod", False) and hasattr(self.backbone, "constraint_mod"):
            cpa_val = batch.get("cpa_constraint_val", None)
            if cpa_val is not None:
                # Normalize cpa to [0,1] range (cpa typically 60-130)
                cpa_norm = (cpa_val.float() - 60.0) / 70.0  # (B,)
                # Tight constraint -> stronger modulation scale inside the backbone.
                self.backbone.constraint_mod._cpa_scale = (1.0 + (1.0 - cpa_norm).unsqueeze(1).unsqueeze(2))
        pfeas_seq = None
        if self.config.get("use_pfeas_attention", False):
            pfeas_seq = batch["prefix_feasibility"].squeeze(-1)  # (B, T)
        fwd_kwargs = {"attention_mask": mask, "return_features": True}
        if pfeas_seq is not None and getattr(self.backbone, "supports_pfeas_attention", False):
            fwd_kwargs["pfeas_seq"] = pfeas_seq
        if self.config.get("use_cpa_moe", False):
            fwd_kwargs["cpa_constraint"] = batch.get("cpa_constraint_val", None)
        if self.config.get("use_cpa_mdn", False):
            # MDN 头也需要 CPA 标量来调制混合权重和成分参数
            fwd_kwargs["cpa_constraint"] = batch.get("cpa_constraint_val", None)
        if self.backbone_variant == "v3":
            # v3 的 CPAAwareGate / CPA-FiLM / 可行域头训练时都需要真实 CPA，
            # 否则与 eval 分布不一致（eval 必传 CPA）
            fwd_kwargs["cpa_constraint"] = batch.get("cpa_constraint_val", None)
        _, action_preds, _, extras = self.backbone.forward(
            encoded_states, actions, rewards, rtg[:, :-1], timesteps, **fwd_kwargs,
        )
        if extras is None:
            extras = {"fused_state_ctx": encoded_states}

        residual_mix = float(self.config.get("residual_mix", 0.25))
        final_action_preds = action_preds + residual_mix * heuristic_actions if self.config.get("use_residual_policy", False) else action_preds
        router_logits = None
        router_weights = None
        if self.config.get("use_constraint_router", False):
            routed_delta, router_logits, router_weights = self._route_actions(encoded_states, meta)
            final_action_preds = final_action_preds + routed_delta

        # Direction 3: CPA-Pressure Adaptive Gate — modulate action by CPA state
        if self.config.get("use_cpa_gate", False) and self.config.get("use_cpa_state_features", False):
            # states[:, :, -1] = cpa_ratio_norm (last feature when use_cpa_state_features=True)
            cpa_ratio_norm = states[:, :, -1]           # (B, T)
            time_left_norm = states[:, :, 0]            # (B, T)
            budget_left_norm = states[:, :, 1]          # (B, T)
            gate_feats = torch.stack([cpa_ratio_norm, time_left_norm, budget_left_norm], dim=-1)  # (B, T, 3)
            gate = torch.sigmoid(self.cpa_gate_net(gate_feats))  # (B, T, 1)
            # gate≈1 (cpa_ratio low = CPA healthy) → scale=1.2 (aggressive)
            # gate≈0 (cpa_ratio high = CPA tight) → scale=0.7 (conservative)
            gate_scale = 0.7 + 0.5 * gate  # range [0.7, 1.2]
            final_action_preds = final_action_preds * gate_scale
        valid = mask.reshape(-1) > 0
        pred_flat = final_action_preds.reshape(-1, self.act_dim)[valid]
        target_flat = actions.reshape(-1, self.act_dim)[valid]

        # CPA-target correction: for infeasible tokens (CPA already violated),
        # correct the imitation target to be more conservative.
        # This changes WHAT the model learns, not just how much weight it gets.
        if self.config.get("use_cpa_target_corr", False):
            corr = float(self.config.get("cpa_corr_factor", 0.2))
            pfeas = batch["prefix_feasibility"].squeeze(-1)  # (B, T), 1=feasible 0=violated
            infeas = (1.0 - pfeas).unsqueeze(-1)  # (B, T, 1)
            # Push target toward 0 for infeasible steps: target *= (1 - corr)
            corrected = actions - corr * infeas * torch.relu(actions)
            target_flat = corrected.reshape(-1, self.act_dim)[valid]

        uncertainty_flat = None
        if self.config.get("use_uncertainty", False):
            uncertainty_flat = self.uncertainty_head(extras["fused_state_ctx"]).reshape(-1, self.act_dim)[valid]

        if self.config.get("use_cpa_mdn", False) and extras is not None and "mdn_pi" in extras:
            # MDN NLL loss：直接在完整序列上计算，然后用 valid mask 过滤
            # 注意：NLL loss 内部已处理 mask，这里用 per_token 形式以便后续加权
            # 从 extras 中取出 MDN 参数（已在 forward 中计算）
            mdn_pi = extras["mdn_pi"]      # (B, T, K)
            mdn_mu = extras["mdn_mu"]      # (B, T, K, act_dim)
            mdn_sigma = extras["mdn_sigma"]  # (B, T, K, act_dim)

            # 扩展 target 到 K 个成分
            target_exp = actions.unsqueeze(2).expand_as(mdn_mu)  # (B, T, K, act_dim)

            # 各成分对数概率密度
            import math as _math
            log_prob_k = (
                -0.5 * _math.log(2 * _math.pi)
                - torch.log(mdn_sigma)
                - 0.5 * ((target_exp - mdn_mu) / mdn_sigma) ** 2
            ).sum(dim=-1)  # (B, T, K)

            # log Σ_k π_k exp(log_prob_k)，logsumexp 保证数值稳定
            log_pi = torch.log(mdn_pi + 1e-8)  # (B, T, K)
            log_mixture = torch.logsumexp(log_pi + log_prob_k, dim=-1)  # (B, T)

            # per-token NLL（正值，越小越好）
            per_token_loss = -log_mixture.reshape(-1)[valid]  # (N,)

            # MDN NLL 的量级通常比 MSE 大，用缩放系数对齐
            mdn_loss_scale = float(self.config.get("mdn_loss_scale", 0.5))
            per_token_loss = per_token_loss * mdn_loss_scale
        else:
            per_token_loss = self._action_loss(pred_flat, target_flat, uncertainty_flat=uncertainty_flat)

        token_weights = batch["sample_weight"].unsqueeze(1).expand(-1, mask.shape[1]).reshape(-1)[valid]
        pooled = masked_mean(extras["fused_state_ctx"], mask)
        pooled_encoded = masked_mean(encoded_states, mask)
        quality_pred = None
        if self.config.get("use_selective_imitation", False):
            quality_pred = torch.sigmoid(self.quality_head(pooled)).squeeze(-1) * 2.0
            quality_expand = quality_pred.unsqueeze(1).expand(-1, mask.shape[1]).reshape(-1)[valid]
            token_weights = token_weights * (0.5 + quality_expand / 2.0)
        if self.config.get("use_constraint_imitation_weight", False):
            if adaptive_risk is not None:
                risk_flat = adaptive_risk.reshape(-1)[valid]
                min_scale = float(self.config.get("adaptive_c2_min_scale", 0.65))
                max_scale = float(self.config.get("c2_budget_scale", 3.0))
                cw = min_scale + (max_scale - min_scale) * risk_flat
            elif self.config.get("use_budget_aware_c2", False):
                # Use raw budget ratio so early high-budget states are not over-regularized.
                budget_left_raw = batch["raw_states"][:, :, 1].reshape(-1)[valid]
                c2_floor = float(self.config.get("c2_weight_floor", 0.5))
                c2_ceiling = float(self.config.get("c2_weight_ceiling", 1.0))
                c2_center = float(self.config.get("c2_budget_center", 0.35))
                c2_sharpness = float(self.config.get("c2_budget_sharpness", 10.0))
                if c2_ceiling < c2_floor:
                    c2_floor, c2_ceiling = c2_ceiling, c2_floor
                cw = c2_floor + (c2_ceiling - c2_floor) * torch.sigmoid((c2_center - budget_left_raw) * c2_sharpness)
            elif self.config.get("use_pfeas_c2", False):
                # Pfeas-aware C2: scale down weight when CPA prefix is infeasible
                # High budget + CPA feasible → high weight; high budget + CPA violated → reduced weight
                budget_left = states[:, :, 1].reshape(-1)[valid]
                c2_scale = float(self.config.get("c2_budget_scale", 3.0))
                pfeas_flat = batch["prefix_feasibility"].squeeze(-1).reshape(-1)[valid]
                cpa_factor = 0.7 + 0.3 * pfeas_flat  # [0.7, 1.0]
                cw = (0.5 + 0.5 * torch.sigmoid(budget_left * c2_scale)) * cpa_factor
            else:
                # C2-style: upweight tokens where budget constraint is still feasible
                # states[:,:,1] = budget_left (normalized); > 0 means budget not exhausted
                budget_left = states[:, :, 1].reshape(-1)[valid]
                c2_scale = float(self.config.get("c2_budget_scale", 3.0))
                cw = 0.5 + 0.5 * torch.sigmoid(budget_left * c2_scale)
            token_weights = token_weights * cw
        if self.config.get("use_pfeas_weight", False):
            # Prefix-feasibility weight: upweight tokens where cumulative CPA is still feasible
            # prefix_feasibility[t]=1 if cumulative_cost_t <= cpa_constraint*(cumulative_reward_t+1)
            pfeas = batch["prefix_feasibility"].squeeze(-1).reshape(-1)[valid]
            if self.config.get("use_cpa_adaptive_pfeas", False):
                # CPA-Adaptive λ_pfeas: tight-constraint advertisers get higher pfeas weight.
                # λ_pfeas(ĉ) = 1.0 + Δλ × (ĉ_median / ĉ)
                # Tight CPA (small ĉ) → larger λ → stronger separation between feasible/infeasible steps.
                cpa_vals = batch["cpa_constraint_val"]  # (B,) raw CPA constraint
                cpa_median = float(self.config.get("cpa_adaptive_median", 80.0))
                delta_lambda = float(self.config.get("cpa_adaptive_delta_lambda", 1.0))
                lambda_max = float(self.config.get("cpa_adaptive_lambda_max", 3.0))
                # Per-trajectory λ, clamped to [1.0, lambda_max]
                per_traj_lambda = (1.0 + delta_lambda * (cpa_median / cpa_vals.clamp_min(1.0))).clamp(1.0, lambda_max)
                # Expand to (B, T) then flatten and index by valid
                per_token_lambda = per_traj_lambda.unsqueeze(1).expand(-1, mask.shape[1]).reshape(-1)[valid]
                pfeas_weight = 0.5 + per_token_lambda * pfeas  # 0.5 infeasible, (0.5+λ) feasible
            else:
                pfeas_scale = float(self.config.get("pfeas_scale", 1.5))
                pfeas_weight = 0.5 + pfeas_scale * pfeas  # 0.5 infeasible, 2.0 feasible (default)
            token_weights = token_weights * pfeas_weight
        if self.config.get("use_quality_aware_weight", False):
            # Quality-aware weight: upweight tokens from high-pValue (high-quality) timesteps
            # raw_states[:,:,12] = current pValue (index 12 per coarse/fine_idx convention)
            pvalue = batch["raw_states"][:, :, 12].reshape(-1)[valid]
            pvalue_norm = (pvalue - pvalue.mean()) / (pvalue.std() + 1e-6)
            qa_scale = float(self.config.get("quality_aware_scale", 2.0))
            qa_weight = 1.0 + torch.sigmoid(pvalue_norm * qa_scale) * 0.5  # range [1.0, 1.5]
            token_weights = token_weights * qa_weight
        if self.config.get("use_advantage_weight", False):
            # AWSM: per-step advantage weight
            # advantage[t] = reward[t] - mean_reward_of_traj
            # upweight steps where the agent did better than average
            adv = batch["step_advantage"].reshape(-1)[valid]  # (N,)
            adv_scale = float(self.config.get("advantage_scale", 2.0))
            adv_weight = 1.0 + torch.sigmoid(adv * adv_scale)  # range (1.0, 2.0)
            token_weights = token_weights * adv_weight
        token_weights = token_weights / token_weights.mean().clamp_min(1e-6)
        action_loss = (per_token_loss * token_weights).mean()

        total_loss = action_loss
        out = {"loss": total_loss, "action_loss": action_loss.detach()}

        # Auxiliary separation loss (GranularityCalibrator_AuxSep)
        if extras is not None and "aux_sep_loss" in extras:
            aux_sep = extras["aux_sep_loss"]
            total_loss = total_loss + aux_sep
            out["aux_sep_loss"] = aux_sep.detach()

        # 用法3 (DG3)：粗/细双视角的多样性集成
        # 关键：必须给每个视角头各自 MSE 锚定到目标，否则"平均对、个头发散"会让损失→-∞。
        # 去相关项用有界的误差相关系数(∈[-1,1])，而非无界乘积。
        if self.config.get("use_granularity_diversity", False) and extras is not None and "alpha_coarse" in extras:
            a_c = extras["alpha_coarse"].reshape(-1, self.act_dim)[valid]
            a_f = extras["alpha_fine"].reshape(-1, self.act_dim)[valid]
            tgt = actions.reshape(-1, self.act_dim)[valid]
            ec = a_c - tgt
            ef = a_f - tgt
            head_mse = ec.pow(2).mean() + ef.pow(2).mean()        # 锚定两头到目标
            corr = (ec * ef).mean() / (
                ec.pow(2).mean().sqrt() * ef.pow(2).mean().sqrt() + 1e-6
            )                                                      # ∈[-1,1]，最小化→误差去相关
            head_w = float(self.config.get("granularity_head_weight", 0.5))
            corr_w = float(self.config.get("granularity_diversity_weight", 0.1))
            total_loss = total_loss + head_w * head_mse + corr_w * corr
            out["gran_head_mse"] = head_mse.detach()
            out["gran_corr"] = corr.detach()

        # H-RTG: classification loss — predict which RTG bin comes next
        if self.config.get("use_rtg_histogram", False) and extras is not None and "rtg_bin_logits" in extras:
            rtg_bin_logits = extras["rtg_bin_logits"]  # (B, T, num_bins)
            rtg_bin_idx = extras["rtg_bin_idx"]        # (B, T)
            # Predict bin at t from action token at t (i.e., predict rtg[t] from context up to t)
            # Shift: logits[:, :-1] predict bin_idx[:, 1:]
            if rtg_bin_logits.shape[1] > 1:
                logits_flat = rtg_bin_logits[:, :-1].reshape(-1, rtg_bin_logits.shape[-1])
                target_flat = rtg_bin_idx[:, 1:].reshape(-1)
                valid_hrtg = mask[:, 1:].reshape(-1) > 0
                if valid_hrtg.sum() > 0:
                    hrtg_loss = F.cross_entropy(logits_flat[valid_hrtg], target_flat[valid_hrtg])
                    total_loss = total_loss + float(self.config.get("hrtg_weight", 0.1)) * hrtg_loss
                    out["hrtg_loss"] = hrtg_loss.detach()

        # ICTS: demo prefix imitation loss
        if self.config.get("use_demo_prefix", False) and "demo_actions" in batch:
            demo_states = batch["demo_states"]   # (B, demo_len, state_dim)
            demo_actions = batch["demo_actions"] # (B, demo_len, act_dim)
            demo_rtg = batch["demo_rtg"]         # (B, demo_len, return_dim)
            demo_len = demo_states.shape[1]
            if demo_len > 0:
                demo_mask = (demo_actions.abs().sum(-1) > 0).long()  # (B, demo_len)
                if demo_mask.sum() > 0:
                    demo_meta = meta
                    demo_retrieval = retrieval_context
                    demo_enc = self.encode_states(demo_states, demo_meta, demo_retrieval)
                    demo_ts = torch.zeros((demo_states.shape[0], demo_len), dtype=torch.long, device=demo_states.device)
                    _, demo_pred, _, _ = self.backbone.forward(
                        demo_enc, demo_actions, torch.zeros_like(demo_actions[..., 0]), demo_rtg, demo_ts,
                        attention_mask=demo_mask, return_features=False,
                    )
                    demo_valid = demo_mask.reshape(-1) > 0
                    if demo_valid.sum() > 0:
                        demo_loss = ((demo_pred.reshape(-1, self.act_dim)[demo_valid] - demo_actions.reshape(-1, self.act_dim)[demo_valid]) ** 2).mean()
                        total_loss = total_loss + float(self.config.get("demo_prefix_weight", 0.1)) * demo_loss
                        out["demo_prefix_loss"] = demo_loss.detach()

        if self.config.get("use_conservative_reg", False):
            budget_left_seq = states[:, :, 1]  # (B, T) normalized budget
            tight_threshold = float(self.config.get("cons_tight_threshold", 0.3))
            if self.config.get("use_cpa_adaptive_cons", False):
                # CPA-Adaptive Conservative Reg:
                #   1. Tight-constraint advertisers get a higher trigger threshold (more conservative earlier).
                #      θ_ĉ = base_threshold × (ĉ_median / ĉ), clamped to [base, 0.7]
                #   2. Loose-constraint advertisers get an encouragement loss (bid more aggressively).
                #      L_encourage = λ⁻ × E[1[cpa_slack > θ_slack] × max(0, â_t - a_t)²]
                cpa_vals = batch["cpa_constraint_val"]  # (B,)
                cpa_median = float(self.config.get("cpa_adaptive_median", 80.0))
                # Per-trajectory adaptive threshold: tight CPA → higher threshold (trigger earlier)
                per_traj_thresh = (tight_threshold * (cpa_median / cpa_vals.clamp_min(1.0))).clamp(tight_threshold, 0.7)
                # Expand to (B, T)
                per_step_thresh = per_traj_thresh.unsqueeze(1).expand_as(budget_left_seq)
                tight_mask = (budget_left_seq < per_step_thresh).float().unsqueeze(-1)  # (B, T, 1)
                # Encouragement loss for loose-constraint advertisers (CPA > loose_thresh)
                loose_cpa_thresh = float(self.config.get("cpa_adaptive_loose_thresh", 100.0))
                is_loose = (cpa_vals > loose_cpa_thresh).float()  # (B,)
                if is_loose.sum() > 0:
                    # CPA slack: how much room is left relative to constraint
                    # Use pfeas as proxy: pfeas=1 means CPA is still feasible (slack > 0)
                    pfeas_seq = batch["prefix_feasibility"].squeeze(-1)  # (B, T)
                    cpa_slack_mask = pfeas_seq * is_loose.unsqueeze(1)  # (B, T): loose AND feasible
                    underbid = torch.relu(heuristic_actions.squeeze(-1) - final_action_preds.squeeze(-1))
                    encourage_loss = (underbid ** 2 * cpa_slack_mask * mask.float()).sum() / (
                        (cpa_slack_mask * mask.float()).sum().clamp_min(1.0)
                    )
                    encourage_weight = float(self.config.get("cpa_encourage_weight", 0.1))
                    total_loss = total_loss + encourage_weight * encourage_loss
                    out["encourage_loss"] = encourage_loss.detach()
                    out["encourage_n_loose"] = is_loose.sum().detach()
            elif adaptive_risk is not None:
                tight_mask = adaptive_risk.unsqueeze(-1)
            elif self.config.get("use_pfeas_cons", False):
                # pfeas-triggered conservative reg: fire when CPA prefix is infeasible
                # (cumulative cost > cpa_constraint * cumulative_reward).
                # This covers tight-CPA violations that happen at budget_left≈1.0,
                # which the original budget_left<0.3 threshold completely misses.
                pfeas_seq = batch["prefix_feasibility"].squeeze(-1)  # (B, T) binary
                cpa_risk = (1.0 - pfeas_seq)                         # 1 when infeasible
                budget_risk = (budget_left_seq < tight_threshold).float()
                tight_mask = torch.clamp(budget_risk + cpa_risk, 0.0, 1.0).unsqueeze(-1)
            elif self.config.get("use_cpa_aware_cons", False):
                # CPA-Aware: 2D risk = budget_risk × cpa_risk × time_factor
                # budget_risk: high when budget is low
                budget_risk = torch.sigmoid((tight_threshold - budget_left_seq) * 10.0)
                # cpa_risk: high when CPA prefix is infeasible (1 - pfeas)
                pfeas_seq = batch["prefix_feasibility"].squeeze(-1)  # (B, T) binary
                cpa_risk = 1.0 - pfeas_seq
                # time_factor: near end of episode, spending is expected — reduce penalty
                time_left_seq = states[:, :, 0]
                time_factor = torch.sigmoid((time_left_seq - 0.15) * 10.0)
                cpa_budget_w = float(self.config.get("cpa_aware_budget_w", 0.4))
                cpa_risk_w = float(self.config.get("cpa_aware_cpa_w", 0.6))
                risk_score = (budget_risk * cpa_budget_w + cpa_risk * cpa_risk_w) * time_factor
                tight_mask = risk_score.unsqueeze(-1)
            elif self.config.get("use_soft_cons", False):
                # Soft continuous weight: sigmoid transition instead of hard threshold
                tight_mask = torch.sigmoid((tight_threshold - budget_left_seq) * 8.0).unsqueeze(-1)
            else:
                tight_mask = (budget_left_seq < tight_threshold).float().unsqueeze(-1)  # (B, T, 1)
            # Penalize predicted > heuristic when budget is tight
            overbid = torch.relu(final_action_preds - heuristic_actions)
            cons_loss = (overbid ** 2 * tight_mask * mask.float().unsqueeze(-1)).sum() / (tight_mask * mask.float().unsqueeze(-1)).sum().clamp_min(1.0)
            cons_weight = float(self.config.get("cons_weight", 0.3))
            if adaptive_risk is not None:
                cons_weight = cons_weight * float(self.config.get("adaptive_cons_scale", 1.0))
                out["adaptive_risk_mean"] = adaptive_risk[mask > 0].mean().detach()
            elif self.config.get("use_cpa_aware_cons", False):
                out["cpa_aware_risk_mean"] = tight_mask[mask.unsqueeze(-1) > 0].mean().detach()
            total_loss = total_loss + cons_weight * cons_loss
            out["cons_loss"] = cons_loss.detach()

        # --- Improvement 1: Per-Step CPA Violation Loss ---
        # Directly penalizes overbidding at steps where cumulative CPA is already violated.
        # More precise than conservative_reg (which uses budget as a proxy):
        # fires exactly when cpa_ratio > threshold, regardless of budget level.
        # This is the key fix for CPA=60/70 groups that violate at 74% rate.
        if self.config.get("use_cpa_violation_loss", False):
            # prefix_feasibility[t]=1 when cumulative_cost <= cpa_constraint*(cumulative_reward+1)
            pfeas = batch["prefix_feasibility"].squeeze(-1)  # (B, T) in [0,1]
            cpa_violated_mask = (1.0 - pfeas)               # 1 when CPA already violated
            # Penalize predicted action > heuristic when CPA is violated
            overbid = torch.relu(final_action_preds.squeeze(-1) - heuristic_actions.squeeze(-1))
            cpa_viol_loss = (overbid ** 2 * cpa_violated_mask * mask.float()).sum() / (
                (cpa_violated_mask * mask.float()).sum().clamp_min(1.0)
            )
            cpa_viol_weight = float(self.config.get("cpa_viol_weight", 0.4))
            total_loss = total_loss + cpa_viol_weight * cpa_viol_loss
            out["cpa_viol_loss"] = cpa_viol_loss.detach()

        # --- Improvement 2: Budget Utilization Loss ---
        # Penalizes underbidding when CPA has plenty of headroom.
        # Fixes CPA=110-130 groups that have cpa_ratio=0.53-0.73 (wasting budget).
        # Uses smooth_l1 (not MSE) to avoid dominating action_loss when underbid is large.
        if self.config.get("use_budget_util_loss", False):
            pfeas = batch["prefix_feasibility"].squeeze(-1)  # (B, T)
            budget_left = states[:, :, 1]
            util_threshold = float(self.config.get("util_budget_threshold", 0.4))
            loose_mask = pfeas * (budget_left > util_threshold).float()  # (B, T)
            underbid = torch.relu(heuristic_actions.squeeze(-1) - final_action_preds.squeeze(-1))
            # smooth_l1 caps gradient for large underbid values (beta=0.5)
            util_loss_per = F.smooth_l1_loss(
                final_action_preds.squeeze(-1) * loose_mask,
                heuristic_actions.squeeze(-1) * loose_mask,
                beta=0.5, reduction="none",
            )
            n_loose = (loose_mask * mask.float()).sum().clamp_min(1.0)
            util_loss = (util_loss_per * loose_mask * mask.float()).sum() / n_loose
            util_weight = float(self.config.get("util_weight", 0.02))
            total_loss = total_loss + util_weight * util_loss
            out["util_loss"] = util_loss.detach()

        # --- Improvement 3: CPA-Group Contrastive RTG Loss ---
        # Ensures the model learns that same absolute RTG means different aggressiveness
        # for different CPA groups. Penalizes action variance within same RTG bucket
        # across different CPA groups (they should bid differently for same RTG).
        # Implemented as: for tight-CPA trajectories, add extra penalty when
        # predicted action exceeds cpa-scaled heuristic.
        if self.config.get("use_cpa_scaled_cons", False):
            # Scale the heuristic by cpa_constraint/scale to get CPA-relative target
            cpa_vals = batch["cpa_constraint_val"].unsqueeze(1).unsqueeze(2)  # (B,1,1)
            cpa_scale_factor = (cpa_vals / float(self.config.get("scale", 40.0))).clamp(0.5, 3.0)
            # Tight CPA (cpa_constraint < tight_thresh) → scale down heuristic target
            tight_thresh = float(self.config.get("cpa_scaled_tight_thresh", 80.0))
            is_tight = (batch["cpa_constraint_val"] < tight_thresh).float().unsqueeze(1).unsqueeze(2)
            # For tight CPA: penalize if pred > heuristic * cpa_scale_factor
            scaled_heuristic = heuristic_actions * cpa_scale_factor
            overbid_scaled = torch.relu(final_action_preds - scaled_heuristic)
            cpa_scaled_loss = (overbid_scaled ** 2 * is_tight * mask.float().unsqueeze(-1)).sum() / (
                (is_tight * mask.float().unsqueeze(-1)).sum().clamp_min(1.0)
            )
            cpa_scaled_weight = float(self.config.get("cpa_scaled_weight", 0.3))
            total_loss = total_loss + cpa_scaled_weight * cpa_scaled_loss
            out["cpa_scaled_loss"] = cpa_scaled_loss.detach()
            smooth_loss = self._smoothness_loss(final_action_preds, actions, mask.float())
            total_loss = total_loss + float(self.config.get("smoothness_weight", 0.1)) * smooth_loss
            out["smoothness_loss"] = smooth_loss.detach()

        if self.config.get("use_constraint_router", False):
            route_targets = batch["route_targets"]
            route_loss = F.cross_entropy(router_logits.reshape(-1, self.num_router_experts)[valid], route_targets.reshape(-1)[valid])
            total_loss = total_loss + float(self.config.get("router_weight", 0.2)) * route_loss
            out["router_loss"] = route_loss.detach()
            usage = router_weights[mask > 0].mean(dim=0)
            balance_target = torch.full_like(usage, 1.0 / float(self.num_router_experts))
            balance_loss = torch.mean((usage - balance_target) ** 2)
            total_loss = total_loss + float(self.config.get("router_balance_weight", 0.05)) * balance_loss
            out["router_balance_loss"] = balance_loss.detach()

        if self.config.get("use_preference_loss", False):
            action_feat = masked_mean(final_action_preds, mask)
            preference_loss = self._pairwise_preference_loss(pooled_encoded, action_feat, batch["quality_target"])
            total_loss = total_loss + float(self.config["preference_weight"]) * preference_loss
            out["preference_loss"] = preference_loss.detach()

        if self.config.get("use_future_model", False):
            future_inp = self._future_context(encoded_states, meta, retrieval_context, final_action_preds)
            future_pred = self.future_head(future_inp)
            future_valid = mask.unsqueeze(-1)
            future_target = torch.cat([batch["next_rewards"], batch["next_costs"], batch["next_budget_left"]], dim=-1)
            future_loss = (((future_pred - future_target) ** 2) * future_valid).sum() / future_valid.sum().clamp_min(1.0)
            total_loss = total_loss + float(self.config["future_weight"]) * future_loss
            out["future_loss"] = future_loss.detach()

        if self.config.get("use_selective_imitation", False):
            quality_target = batch["quality_target"].clamp(0.0, 2.0)
            selective_loss = F.smooth_l1_loss(quality_pred, quality_target)
            total_loss = total_loss + float(self.config["selective_weight"]) * selective_loss
            out["selective_loss"] = selective_loss.detach()

        if self.config.get("use_support_regularization", False):
            support_loss = self._support_loss(final_action_preds, heuristic_actions, batch["expert_actions"], mask.float())
            total_loss = total_loss + float(self.config.get("support_weight", 0.08)) * support_loss
            out["support_loss"] = support_loss.detach()

        if self.config.get("use_feasibility_head", False):
            feasibility_loss, feasibility_logit = self._feasibility_loss(
                encoded_states=encoded_states,
                meta=meta,
                retrieval_context=retrieval_context,
                actions=final_action_preds,
                prefix_feasibility=batch["prefix_feasibility"],
                mask=mask,
            )
            total_loss = total_loss + float(self.config.get("feasibility_weight", 0.15)) * feasibility_loss
            out["feasibility_loss"] = feasibility_loss.detach()
            out["feasibility_mean"] = torch.sigmoid(feasibility_logit[mask > 0]).mean().detach()

        if self.config.get("use_q_judge", False):
            judge_target = self._judge_target(batch)
            q_loss, q_pred = self._q_loss(
                encoded_states=encoded_states,
                meta=meta,
                retrieval_context=retrieval_context,
                action_tensor=final_action_preds.detach() if self.config.get("use_q_regularization", False) else final_action_preds,
                judge_target=judge_target,
                mask=mask,
            )
            total_loss = total_loss + float(self.config.get("q_weight", 0.2)) * q_loss
            out["q_loss"] = q_loss.detach()
            if self.config.get("use_q_regularization", False):
                q_reg = -(q_pred[mask > 0]).mean()
                total_loss = total_loss + float(self.config.get("q_reg_weight", 0.05)) * q_reg
                out["q_reg_loss"] = q_reg.detach()

        if self.config.get("use_energy_judge", False):
            negative_actions = self._build_negative_actions(actions, heuristic_actions)
            energy_loss, pos_energy, neg_energy = self._energy_loss(
                encoded_states=encoded_states,
                meta=meta,
                retrieval_context=retrieval_context,
                pos_actions=actions,
                neg_actions=negative_actions,
                quality_target=batch["quality_target"],
                mask=mask,
            )
            total_loss = total_loss + float(self.config.get("energy_weight", 0.15)) * energy_loss
            out["energy_loss"] = energy_loss.detach()
            out["energy_gap"] = (neg_energy[mask > 0].mean() - pos_energy[mask > 0].mean()).detach()

        if self.config.get("use_bid2x_judge", False):
            bid2x_loss, bid2x_pred = self._bid2x_loss(
                encoded_states=encoded_states,
                meta=meta,
                retrieval_context=retrieval_context,
                actions=final_action_preds,
                batch=batch,
                mask=mask,
            )
            total_loss = total_loss + float(self.config.get("bid2x_weight", 0.2)) * bid2x_loss
            out["bid2x_loss"] = bid2x_loss.detach()
            out["bid2x_reward_pred"] = bid2x_pred[..., 0][mask > 0].mean().detach()

        if self.config.get("use_expert_guidance", False):
            expert_pred = self.expert_head(extras["fused_state_ctx"])
            expert_target = batch["expert_actions"]
            valid_mask = mask.unsqueeze(-1).float()
            expert_loss = (((expert_pred - expert_target) ** 2) * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
            total_loss = total_loss + float(self.config.get("expert_weight", 0.2)) * expert_loss
            out["expert_loss"] = expert_loss.detach()

        if self.config.get("use_meta_calibration", False):
            meta_inp = torch.cat([pooled_encoded, meta], dim=-1)
            meta_pred = self.meta_head(meta_inp)
            meta_scale = self._apply_meta_scale(meta_pred)
            quality_centered = (batch["quality_target"] - 1.0).clamp(-1.0, 1.0)
            sample_centered = (batch["sample_weight"] - 1.0).clamp(-1.0, 1.0)
            target_scale_target = 1.0 + float(self.config.get("calibration_scale", 0.15)) * quality_centered
            alpha_scale_target = 1.0 + float(self.config.get("calibration_alpha_scale", 0.1)) * sample_centered
            meta_target = torch.stack([target_scale_target, alpha_scale_target], dim=-1)
            meta_loss = F.smooth_l1_loss(meta_scale, meta_target)
            total_loss = total_loss + float(self.config["meta_weight"]) * meta_loss
            out["meta_loss"] = meta_loss.detach()

        # ── 方向A: Dual RTG supervision ──────────────────────────────────────────
        # When return_dim=2, rtg[:,:,0]=reward-RTG, rtg[:,:,1]=constraint-RTG (cpa_slack).
        # We add an auxiliary loss that encourages the model to be sensitive to the
        # constraint dimension: penalize action when constraint-RTG is negative (CPA tight)
        # and reward when it is positive (CPA slack available).
        if self.config.get("use_dual_rtg_loss", False) and int(self.config.get("return_dim", 1)) >= 2:
            # rtg shape: (B, T+1, 2); rtg[:, :-1, 1] aligns with action tokens
            constraint_rtg = rtg[:, :-1, 1]  # (B, T) — cpa_slack RTG
            # Tight constraint (constraint_rtg < 0): penalize overbidding above heuristic
            tight_mask = (constraint_rtg < 0.0).float()
            # Loose constraint (constraint_rtg > 0.3): penalize underbidding below heuristic
            loose_mask = (constraint_rtg > 0.3).float()
            overbid = torch.relu(final_action_preds.squeeze(-1) - heuristic_actions.squeeze(-1))
            underbid = torch.relu(heuristic_actions.squeeze(-1) - final_action_preds.squeeze(-1))
            dual_rtg_loss = (
                (overbid ** 2 * tight_mask * mask.float()).sum() / (tight_mask * mask.float()).sum().clamp_min(1.0)
                + 0.5 * (underbid ** 2 * loose_mask * mask.float()).sum() / (loose_mask * mask.float()).sum().clamp_min(1.0)
            )
            dual_rtg_weight = float(self.config.get("dual_rtg_constraint_weight", 0.3))
            total_loss = total_loss + dual_rtg_weight * dual_rtg_loss
            out["dual_rtg_loss"] = dual_rtg_loss.detach()
            out["dual_rtg_tight_frac"] = tight_mask[mask > 0].mean().detach()

        # ── 方向B: Contrastive trajectory learning ───────────────────────────────
        # InfoNCE loss in projected hidden space.
        # Positive pairs: tokens with prefix_feasibility=1 (CPA compliant)
        # Negative pairs: tokens with prefix_feasibility=0 (CPA violated)
        # For each compliant token, pull it toward other compliant tokens in the batch
        # and push it away from violated tokens.
        if self.config.get("use_contrastive_loss", False):
            fused_ctx = extras["fused_state_ctx"]  # (B, T, hidden_size)
            pfeas_seq = batch["prefix_feasibility"].squeeze(-1)  # (B, T) binary
            valid_mask_2d = mask > 0  # (B, T)

            # Flatten to (N, hidden_size) keeping only valid tokens
            flat_ctx = fused_ctx.reshape(-1, self.hidden_size)
            flat_pfeas = pfeas_seq.reshape(-1)
            flat_valid = valid_mask_2d.reshape(-1)

            valid_idx = flat_valid.nonzero(as_tuple=True)[0]
            if valid_idx.numel() >= 4:
                ctx_valid = flat_ctx[valid_idx]          # (N_valid, H)
                pfeas_valid = flat_pfeas[valid_idx]      # (N_valid,)

                proj = self.contrastive_proj(ctx_valid)  # (N_valid, proj_dim)
                proj = F.normalize(proj, dim=-1, eps=1e-6)

                pos_mask = pfeas_valid > 0.5             # compliant tokens
                neg_mask = pfeas_valid < 0.5             # violated tokens

                n_pos = int(pos_mask.sum().item())
                n_neg = int(neg_mask.sum().item())

                if n_pos >= 2 and n_neg >= 1:
                    temp = float(self.config.get("contrastive_temperature", 0.07))
                    pos_proj = proj[pos_mask]   # (n_pos, D)
                    neg_proj = proj[neg_mask]   # (n_neg, D)

                    # For each positive token: similarity to all other positives (in-class)
                    # vs similarity to all negatives (cross-class)
                    # Use mean of negatives as the denominator anchor
                    neg_mean = neg_proj.mean(dim=0, keepdim=True)  # (1, D)

                    # sim(pos_i, pos_j) for i≠j
                    sim_pp = (pos_proj @ pos_proj.T) / temp  # (n_pos, n_pos)
                    sim_pp.fill_diagonal_(-1e9)

                    # sim(pos_i, neg_mean)
                    sim_pn = (pos_proj @ neg_mean.T) / temp  # (n_pos, 1)

                    # InfoNCE: for each pos token, maximize similarity to other pos tokens
                    # relative to neg_mean
                    logits = torch.cat([sim_pp, sim_pn.expand(-1, 1)], dim=1)  # (n_pos, n_pos+1)
                    # Positive targets: any of the other n_pos-1 positions (use mean log-sum)
                    pos_logits = sim_pp.logsumexp(dim=1)  # (n_pos,)
                    all_logits = logits.logsumexp(dim=1)  # (n_pos,)
                    contrastive_loss = (all_logits - pos_logits).mean()

                    contrastive_weight = float(self.config.get("contrastive_weight", 0.15))
                    total_loss = total_loss + contrastive_weight * contrastive_loss
                    out["contrastive_loss"] = contrastive_loss.detach()
                    out["contrastive_n_pos"] = pos_proj.new_tensor(float(n_pos))
                    out["contrastive_n_neg"] = neg_proj.new_tensor(float(n_neg))

        out["loss"] = total_loss
        return out

    @torch.no_grad()
    def calibrate(self, pooled_state: torch.Tensor, meta: torch.Tensor):
        if not self.config.get("use_meta_calibration", False):
            return torch.ones((pooled_state.shape[0], 2), dtype=pooled_state.dtype, device=pooled_state.device)
        pred = self.meta_head(torch.cat([pooled_state, meta], dim=-1))
        return self._apply_meta_scale(pred)

    def _score_candidates(
        self,
        encoded_state: torch.Tensor,
        meta: torch.Tensor,
        retrieval_context: torch.Tensor,
        candidate_actions: torch.Tensor,
        cpa_constraint: float = None,
        expert_action: torch.Tensor = None,
    ) -> torch.Tensor:
        n = candidate_actions.shape[0]
        state_feat = encoded_state.unsqueeze(0).expand(n, -1)
        meta_feat = meta.expand(n, -1)
        retrieval_feat = retrieval_context.expand(n, -1)
        judge_inp = torch.cat([state_feat, meta_feat, retrieval_feat, candidate_actions], dim=-1)
        reward_pred = torch.zeros((n,), dtype=encoded_state.dtype, device=encoded_state.device)
        cost_pred = torch.zeros((n,), dtype=encoded_state.dtype, device=encoded_state.device)
        budget_pred = torch.zeros((n,), dtype=encoded_state.dtype, device=encoded_state.device)
        future_score = torch.zeros((n,), dtype=encoded_state.dtype, device=encoded_state.device)
        if self.config.get("use_future_model", False) or self.config.get("use_consequence_rerank", False):
            future_pred = self.future_head(judge_inp)
            reward_pred = future_pred[:, 0]
            cost_pred = torch.clamp(future_pred[:, 1], min=0.0)
            budget_pred = future_pred[:, 2]
            future_score = (
                reward_pred
                - float(self.config.get("rerank_cost_weight", 0.4)) * cost_pred
                + float(self.config.get("rerank_budget_weight", 0.05)) * budget_pred
            )
        if cpa_constraint is not None:
            scaled_cpa = float(cpa_constraint) / float(self.backbone.scale)
            cpa_gap = F.relu(cost_pred - scaled_cpa * reward_pred)
        else:
            cpa_gap = torch.zeros_like(cost_pred)

        score = future_score - float(self.config.get("rerank_cpa_weight", 1.0)) * cpa_gap

        if self.config.get("use_q_judge", False):
            q1 = self.q_head1(judge_inp).squeeze(-1)
            q2 = self.q_head2(judge_inp).squeeze(-1)
            q_score = torch.minimum(q1, q2)
            if self.config.get("use_gas_search", False):
                score = float(self.config.get("search_blend", 0.5)) * score + (1.0 - float(self.config.get("search_blend", 0.5))) * q_score
            else:
                score = score + q_score
            if self.config.get("use_uncertainty_filter", False):
                disagreement = torch.abs(q1 - q2)
                score = score - float(self.config.get("judge_uncertainty_penalty", 0.3)) * disagreement

        if self.config.get("use_energy_judge", False):
            e1 = self.energy_head1(judge_inp).squeeze(-1)
            e2 = self.energy_head2(judge_inp).squeeze(-1)
            score = score - 0.5 * (e1 + e2)
            if self.config.get("use_uncertainty_filter", False):
                disagreement = torch.abs(e1 - e2)
                score = score - float(self.config.get("judge_uncertainty_penalty", 0.3)) * disagreement

        if self.config.get("use_bid2x_judge", False):
            bid2x_pred = self.bid2x_head(judge_inp)
            reward_term = bid2x_pred[:, 0]
            cost_term = torch.clamp(bid2x_pred[:, 1], min=0.0)
            budget_term = bid2x_pred[:, 2]
            quality_term = bid2x_pred[:, 3]
            score = score + float(self.config.get("judge_reward_weight", 1.0)) * reward_term
            score = score - float(self.config.get("rerank_cost_weight", 0.4)) * cost_term
            score = score + float(self.config.get("rerank_budget_weight", 0.05)) * budget_term
            score = score + float(self.config.get("judge_quality_weight", 0.2)) * quality_term

        if self.config.get("use_expert_guidance", False) and expert_action is not None:
            proximity = -torch.abs(candidate_actions.squeeze(-1) - expert_action.squeeze(-1))
            score = score + float(self.config.get("expert_blend", 0.25)) * proximity

        if self.config.get("use_feasibility_head", False):
            feas_logit = self.feasibility_head(judge_inp).squeeze(-1)
            score = score + float(self.config.get("feasibility_score_weight", 0.5)) * (2.0 * torch.sigmoid(feas_logit) - 1.0)

        return score

    def _build_eval_candidates(
        self,
        base_action: float,
        heuristic_action_norm: torch.Tensor,
        encoded_state: torch.Tensor,
        expert_action: torch.Tensor = None,
    ) -> torch.Tensor:
        deltas = self.config.get("candidate_deltas", [-0.75, -0.35, 0.0, 0.35, 0.75])
        span = float(self.config.get("candidate_span", 0.45))
        if heuristic_action_norm.numel() > 0:
            span = max(span, 0.5 * abs(float(base_action) - float(heuristic_action_norm.item())))
        candidate_vals = [float(base_action) + float(delta) * span for delta in deltas]
        if heuristic_action_norm.numel() > 0:
            heuristic = float(heuristic_action_norm.item())
            candidate_vals.extend(
                [
                    heuristic,
                    0.5 * (heuristic + float(base_action)),
                    0.7 * float(base_action) + 0.3 * heuristic,
                ]
            )
        if expert_action is not None and expert_action.numel() > 0:
            expert = float(expert_action.item())
            candidate_vals.extend(
                [
                    expert,
                    0.5 * (expert + float(base_action)),
                ]
            )
        uniq = []
        seen = set()
        for value in candidate_vals:
            key = round(float(value), 6)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(float(value))
        return torch.tensor(uniq, dtype=encoded_state.dtype, device=encoded_state.device).reshape(-1, self.act_dim)

    @torch.no_grad()
    def _rerank_action(
        self,
        base_action: float,
        encoded_state: torch.Tensor,
        meta: torch.Tensor,
        retrieval_context: torch.Tensor,
        heuristic_action_norm: torch.Tensor,
        cpa_constraint: float = None,
        expert_action: torch.Tensor = None,
    ) -> float:
        if not (self.config.get("use_consequence_rerank", False) or self.config.get("use_multi_action_candidates", False)):
            return float(base_action)
        candidates = self._build_eval_candidates(base_action, heuristic_action_norm, encoded_state, expert_action=expert_action)
        scores = self._score_candidates(
            encoded_state=encoded_state,
            meta=meta,
            retrieval_context=retrieval_context,
            candidate_actions=candidates,
            cpa_constraint=cpa_constraint,
            expert_action=expert_action,
        )
        best_idx = int(torch.argmax(scores).item())
        return float(candidates[best_idx, 0].item())

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
        pooled = encoded.unsqueeze(0)
        calibration = self.calibrate(pooled, meta).squeeze(0)
        target_scale = float(calibration[0].item())
        alpha_scale = float(calibration[1].item())
        expert_action = None
        if self.config.get("use_expert_guidance", False):
            zeros_action = torch.zeros((1, 1, self.act_dim), dtype=state_norm.dtype, device=state_norm.device)
            zeros_reward = torch.zeros((1, 1, 1), dtype=state_norm.dtype, device=state_norm.device)
            if self.config.get("use_dual_return", False):
                rtg_seed = torch.tensor(
                    [[[float(target_return or 0.0), float(target_budget_return or 0.0)]]],
                    dtype=state_norm.dtype,
                    device=state_norm.device,
                )
            else:
                rtg_seed = torch.tensor(
                    [[[float(target_return or 0.0)]]],
                    dtype=state_norm.dtype,
                    device=state_norm.device,
                )
            timesteps = torch.zeros((1, 1), dtype=torch.long, device=state_norm.device)
            _, _, _, eval_extras = self.backbone.forward(
                state_norm, zeros_action, zeros_reward, rtg_seed, timesteps,
                attention_mask=torch.ones((1, 1), dtype=torch.long, device=state_norm.device),
                return_features=True,
            )
            if eval_extras is None:
                eval_extras = {"fused_state_ctx": state_norm}
            expert_action = self.expert_head(eval_extras["fused_state_ctx"][:, -1, :]).squeeze(0)

        if self.config.get("use_dual_return", False):
            target_payload = None
            if target_return is not None:
                # CPA-slack RTG: 推理时设 0.0（恰好在约束边界）
                second_rtg = 0.0 if self.config.get("use_cpa_slack_rtg", False) else (
                    float(target_budget_return) if target_budget_return is not None else 0.0
                )
                target_payload = [float(target_return) * target_scale, second_rtg]
        else:
            target_payload = float(target_return) * target_scale if target_return is not None else None

        take_kwargs = {"target_return": target_payload, "pre_reward": pre_reward, "pre_cost": pre_cost}
        if self.config.get("use_cpa_slack_rtg", False) and cpa_constraint is not None:
            take_kwargs["cpa_constraint"] = cpa_constraint
        if self.backbone_variant == "v3" and cpa_constraint is not None:
            # v3 的 CPA 模块（gate/film/feas）eval 时必须拿到真实 CPA
            take_kwargs["cpa_constraint"] = cpa_constraint
        if getattr(self.backbone, "supports_pfeas_attention", False) and pfeas_step is not None:
            take_kwargs["pfeas_step"] = float(pfeas_step)
        pred = self.backbone.take_actions(encoded.detach().cpu().numpy(), **take_kwargs)
        if isinstance(pred, torch.Tensor):
            pred = pred.squeeze().item()
        else:
            pred = float(pred.squeeze().item())

        if self.config.get("use_residual_policy", False):
            pred = float(pred + float(self.config.get("residual_mix", 0.25)) * heuristic_action_norm.item())

        router_weights = None
        if self.config.get("use_constraint_router", False):
            route_delta, router_weights = self._eval_route_adjustment(encoded, meta)
            pred = float(pred + route_delta)

        pred = self._rerank_action(
            base_action=float(pred),
            encoded_state=encoded,
            meta=meta,
            retrieval_context=retrieval_context,
            heuristic_action_norm=heuristic_action_norm,
            cpa_constraint=cpa_constraint,
            expert_action=expert_action,
        )
        pred = self._apply_support_clip(pred, heuristic_action_norm, expert_action=expert_action, router_weights=router_weights)

        # Direction 3: CPA-Pressure Gate at inference
        if self.config.get("use_cpa_gate", False) and self.config.get("use_cpa_state_features", False):
            # state_norm shape: (1, 1, state_dim) after unsqueeze above
            gate_inp = torch.tensor(
                [
                    float(state_norm[0, 0, -1].item()),  # cpa_ratio_norm (last feature)
                    float(state_norm[0, 0, 0].item()),   # time_left_norm
                    float(state_norm[0, 0, 1].item()),   # budget_left_norm
                ],
                dtype=torch.float32,
                device=encoded.device,
            )
            gate = float(torch.sigmoid(self.cpa_gate_net(gate_inp)).item())
            pred = pred * (0.7 + 0.5 * gate)

        return {
            "action_norm": torch.tensor(pred, dtype=torch.float32),
            "alpha_scale": torch.tensor(alpha_scale, dtype=torch.float32),
            "target_scale": torch.tensor(target_scale, dtype=torch.float32),
        }
