"""
MSDTv2: Multi-Scale Decision Transformer v2
对原始 MSDT 的三项关键改进:
  1. 因果卷积 —— 修复 CrossGranularityTemporalFusion 中的非因果 Conv1d
  2. 输入相关粒度门控 —— 修复 granularity_attn 静态参数
  3. 约束感知调制 —— 将预算余量/时间余量显式注入状态表示
"""
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dt import Block, DecisionTransformer
from .msdt_backbone import (GranularityCalibrator, DualBranchEncoderV2,
                             FiLMDualBranchEncoder, CrossStitchDualBranchEncoder,
                             GranularityCalibrator_AuxSep)


# ---------------------------------------------------------------------------
# 1. 因果卷积版时序融合
# ---------------------------------------------------------------------------

class CausalTemporalFusionV2(nn.Module):
    """
    改进的跨粒度时序融合模块。

    改动点（vs CrossGranularityTemporalFusion）:
    - coarse_conv 改为因果卷积（仅对过去做局部平均，不泄漏未来）
    - 去掉 stride=2 + interpolate（stride=2 不改变因果性问题，这里统一用 stride=1）
    """

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

        # 可学习局部窗口参数（同原版）
        self.window = nn.Parameter(torch.tensor(float(local_window)))
        self.window_beta = nn.Parameter(torch.tensor(4.0))
        self.window_temp = nn.Parameter(torch.tensor(1.0))

        # --- 修复 1: 因果 Conv1d（kernel=3, stride=1, 左侧填充 kernel-1=2） ---
        self._causal_pad = 2  # kernel_size - 1
        self.coarse_conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            padding=0,  # 手动左侧填充
        )

        # 细粒度局部注意力（同原版）
        self.fine_qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.fine_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.fine_dropout = nn.Dropout(float(dropout))

        # 跨粒度注意力（coarse 作为 K/V，fine 作为 Q）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.n_head,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)

    def _local_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """单调衰减的局部注意力偏置（只允许关注过去）。"""
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

        # --- 因果粗粒度特征（左侧填充，不看未来）---
        xt = F.pad(x.transpose(1, 2), (self._causal_pad, 0))  # (B, H, T+2)
        coarse = self.coarse_conv(xt).transpose(1, 2)           # (B, T, H)

        # --- 细粒度局部注意力 ---
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

        # --- 跨粒度注意力（coarse → fine）---
        fused, _ = self.cross_attn(
            fine, coarse, coarse,
            attn_mask=self._causal_mask(T, x.device),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.ln2(fine + fused)


# ---------------------------------------------------------------------------
# 2. 约束感知调制模块
# ---------------------------------------------------------------------------

class ConstraintModulator(nn.Module):
    """
    预算/时间约束感知调制器。

    从状态向量中提取:
      - dim 0: time_left  (归一化剩余时间)
      - dim 1: budget_left (归一化剩余预算)
    计算"紧迫感"信号并通过输入相关的门控叠加到状态表示上。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, T, H)  已融合的状态表示
        states: (B, T, state_dim) 归一化状态（budget_left = dim1, time_left = dim0）
        """
        budget = states[:, :, 1:2]            # 归一化预算余量
        time = states[:, :, 0:1]              # 归一化时间余量
        urgency = 1.0 - torch.sigmoid(budget) # 预算紧迫程度（预算越少越高）
        feat = torch.cat([budget, time, urgency], dim=-1)  # (B, T, 3)
        mod = self.proj(feat)                  # (B, T, H)
        gate = torch.sigmoid(self.gate(x))    # 输入相关门控
        # CPA-Arch: scale modulation by per-sample CPA tightness signal
        # _cpa_scale is set externally by method_model.py when use_cpa_conditioned_mod=True
        cpa_scale = getattr(self, "_cpa_scale", None)
        if cpa_scale is not None:
            mod = mod * cpa_scale
        return x + gate * mod


# ---------------------------------------------------------------------------
# 3. MSDTv2 主模型
# ---------------------------------------------------------------------------

class ConstraintAwareActionHead(nn.Module):
    """
    双路径约束感知动作头。
    将动作预测解耦为：
      1. 策略路径：基于上下文预测原始出价（alpha）
      2. 约束路径：基于状态（如剩余预算、时间）和上下文预测动作可行性得分（0~1）
    最终动作 = 策略动作 * 可行性得分
    """
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.policy_path = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_dim)
        )
        self.constraint_path = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        policy_action = self.policy_path(x)
        # 取 states 的前两维（假设为 time_left 和 budget_left）
        constraint_feat = states[:, :, 0:2]
        constraint_input = torch.cat([x, constraint_feat], dim=-1)
        feasibility_score = self.constraint_path(constraint_input)
        return policy_action * feasibility_score


class DecoupledGranularityHead(nn.Module):
    """
    粗细粒度分离动作头。
    - 粗粒度路径（state_ctx，transformer 直接输出）：预测方向缩放因子 [0.5, 2.0]
    - 细粒度路径（fused，经过 CausalFusion + ConstraintMod）：预测基础出价幅度
    最终动作 = direction_scale × magnitude
    """
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.coarse_path = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, act_dim),
            nn.Sigmoid(),  # → [0, 1]，映射到 [0.5, 2.0]
        )
        self.fine_path = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, act_dim),
        )

    def forward(self, fused: torch.Tensor, state_ctx: torch.Tensor) -> torch.Tensor:
        direction = 0.5 + 1.5 * self.coarse_path(state_ctx)  # [0.5, 2.0]
        magnitude = self.fine_path(fused)
        return direction * magnitude


class AsymmetricActionHead(nn.Module):
    """
    非对称分布动作头。
    预测动作的均值 (mu) 和方差 (sigma) 以进行更保守或激进的探索。
    在训练时预测分布参数，测试时可以选择更高回报的采样。
    """
    def __init__(self, hidden_size: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size // 2, act_dim)
        self.log_sig = nn.Linear(hidden_size // 2, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        mu = self.mu(feat)
        log_sig = self.log_sig(feat).clamp(-20, 2)
        # 简单起见，这里直接返回 mu 作为动作预测，或者返回带有方差的合并向量。
        # 考虑到 base_dt 的 step 是按 MSE 计算的，我们返回 mu。
        # 在推理时可以通过特殊逻辑进行上采样，或直接依赖训练后的非对称分布。
        return mu


# ---------------------------------------------------------------------------
# Per-CPA Soft MoE block
# ---------------------------------------------------------------------------

class PerCPASoftMoE(nn.Module):
    """
    在 action head 之前插入的轻量 Soft MoE 层。

    与 CPA-state features 失败方案的关键区别：
    - 显式分配容量：3 个专家各自专注 tight/medium/loose CPA 策略
    - 不依赖模型从 16+1 维 state 里自己学会区分 CPA 差异
    - Soft routing（加权求和）：梯度流向所有专家，训练稳定
    - 参数量仅 +25K，不改变 backbone 接口

    Gate 由 CPA 标量经 embedding 后 softmax 得到，初始化接近均匀，
    训练后自动分化：tight CPA → Expert A 权重高，loose CPA → Expert C 权重高。
    """

    def __init__(self, hidden_size: int, num_experts: int = 3,
                 cpa_emb_dim: int = 16, expert_hidden_ratio: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        expert_hidden = max(16, int(hidden_size * expert_hidden_ratio))

        # CPA scalar → embedding → gate logits (B, num_experts)
        self.cpa_embed = nn.Sequential(
            nn.Linear(1, cpa_emb_dim),
            nn.ReLU(),
            nn.Linear(cpa_emb_dim, num_experts),
        )

        # Expert FFNs: hidden → expert_hidden → hidden
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, hidden_size),
            )
            for _ in range(num_experts)
        ])

        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor,
                cpa_constraint: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:              (B, T, H)
        cpa_constraint: (B,) raw CPA values, or None → uniform mixing
        Returns:        (B, T, H)
        """
        if cpa_constraint is None:
            gate = x.new_full((x.shape[0], self.num_experts), 1.0 / self.num_experts)
        else:
            # Normalize: typical range 60-130, center at 80, scale by 30
            cpa_norm = (cpa_constraint.float() - 80.0) / 30.0  # (B,)
            gate = torch.softmax(self.cpa_embed(cpa_norm.unsqueeze(-1)), dim=-1)  # (B, E)

        # (B, E) → (B, 1, E) for broadcasting over T
        gate = gate.unsqueeze(1)

        # Stack expert outputs: (B, T, H, E), then weighted sum
        expert_outs = torch.stack([e(x) for e in self.experts], dim=-1)  # (B, T, H, E)
        mixed = (expert_outs * gate.unsqueeze(2)).sum(dim=-1)             # (B, T, H)

        return self.out_norm(x + mixed)


class PerCPAMDN(nn.Module):
    """
    方案B：Per-CPA 混合密度网络（Mixture Density Network）动作头。

    核心思想（来自 Bishop 1994）：
      不预测单一动作均值 E[a|s]，而是预测 K 个高斯成分的混合分布：
        p(a|s, ĉ) = Σ_k π_k(s,ĉ) · N(a | μ_k(s,ĉ), σ_k(s,ĉ)²)

    K=3 的物理含义（对应约束异质性）：
      - 成分 0：保守模式（低 μ，小 σ）→ 紧约束广告主（CPA≤70）
      - 成分 1：均衡模式（中 μ，中 σ）→ 中间广告主（CPA=80-100）
      - 成分 2：激进模式（高 μ，大 σ）→ 松约束广告主（CPA≥110）

    与 PerCPASoftMoE 的区别：
      - MoE 在隐层做软路由（改变表征），MDN 在输出层做分布建模（改变预测目标）
      - MDN 直接建模动作的不确定性，可用于安全降级（高 σ → 保守动作）
      - MDN 的可视化价值更高：可以画出"三个成分随 CPA 变化的 μ/π 曲线"

    训练：NLL loss = -log Σ_k π_k N(a_target | μ_k, σ_k)
    推理：默认取最高权重成分的 μ（确定性），可选按 CPA 选分位数

    参数量：约 hidden_size * K * 3 ≈ 64 * 3 * 3 = 576（极轻量）
    """

    def __init__(
        self,
        hidden_size: int,
        act_dim: int = 1,
        num_components: int = 3,
        cpa_emb_dim: int = 16,
        sigma_min: float = 0.01,
        sigma_max: float = 2.0,
    ):
        super().__init__()
        self.num_components = num_components
        self.act_dim = act_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # CPA 标量 → embedding，用于调制混合权重和各成分参数
        # 输入：归一化后的 CPA 标量 (B, 1)
        # 输出：CPA 上下文向量 (B, cpa_emb_dim)
        self.cpa_embed = nn.Sequential(
            nn.Linear(1, cpa_emb_dim),
            nn.Tanh(),  # Tanh 比 ReLU 更平滑，适合连续标量输入
            nn.Linear(cpa_emb_dim, cpa_emb_dim),
            nn.Tanh(),
        )

        # 条件化输入：将隐层表征与 CPA embedding 拼接
        # (B, T, hidden_size + cpa_emb_dim) → 各成分参数
        cond_dim = hidden_size + cpa_emb_dim

        # 混合权重 logits：(B, T, K)
        # 初始化为接近均匀分布，训练后自动分化
        self.pi_head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2),
            nn.ReLU(),
            nn.Linear(cond_dim // 2, num_components),
        )

        # 各成分均值：(B, T, K * act_dim)
        # 每个成分独立预测均值，允许不同 CPA 组学到不同出价水平
        self.mu_head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2),
            nn.ReLU(),
            nn.Linear(cond_dim // 2, num_components * act_dim),
        )

        # 各成分对数标准差：(B, T, K * act_dim)
        # 用 log_sigma 参数化，避免 sigma 为负；clamp 防止坍缩或爆炸
        self.log_sigma_head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2),
            nn.ReLU(),
            nn.Linear(cond_dim // 2, num_components * act_dim),
        )

        # 初始化：让 mu_head 最后一层接近零，避免初始预测偏离数据范围
        nn.init.zeros_(self.mu_head[-1].weight)
        nn.init.zeros_(self.mu_head[-1].bias)
        # 初始化 log_sigma 为 0（sigma=1），给模型足够的初始不确定性
        nn.init.zeros_(self.log_sigma_head[-1].weight)
        nn.init.zeros_(self.log_sigma_head[-1].bias)

    def _get_cpa_context(
        self, x: torch.Tensor, cpa_constraint: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        将 CPA 标量 embedding 与隐层表征拼接，得到条件化输入。

        x:              (B, T, H)
        cpa_constraint: (B,) raw CPA values，或 None（使用中性值 80）
        Returns:        (B, T, H + cpa_emb_dim)
        """
        B, T, H = x.shape
        if cpa_constraint is None:
            # 无 CPA 信息时，使用中性值（归一化后为 0）
            cpa_norm = x.new_zeros(B, 1)
        else:
            # 归一化：中心 80，范围 60-130，scale=30
            # CPA=60 → -0.67，CPA=80 → 0，CPA=130 → +1.67
            cpa_norm = (cpa_constraint.float() - 80.0) / 30.0  # (B,)
            cpa_norm = cpa_norm.unsqueeze(-1)  # (B, 1)

        cpa_emb = self.cpa_embed(cpa_norm)  # (B, cpa_emb_dim)
        # 扩展到序列维度：每个时间步使用相同的 CPA embedding
        cpa_emb_expanded = cpa_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, cpa_emb_dim)
        return torch.cat([x, cpa_emb_expanded], dim=-1)  # (B, T, H + cpa_emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        cpa_constraint: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播，返回 MDN 参数。

        x:              (B, T, H) 隐层表征
        cpa_constraint: (B,) raw CPA values

        Returns:
          pi:    (B, T, K)          混合权重（已 softmax）
          mu:    (B, T, K, act_dim) 各成分均值
          sigma: (B, T, K, act_dim) 各成分标准差（已 clamp）
        """
        B, T, _ = x.shape
        K = self.num_components

        cond = self._get_cpa_context(x, cpa_constraint)  # (B, T, H + cpa_emb_dim)

        # 混合权重
        pi = torch.softmax(self.pi_head(cond), dim=-1)  # (B, T, K)

        # 各成分均值
        mu = self.mu_head(cond)  # (B, T, K * act_dim)
        mu = mu.view(B, T, K, self.act_dim)  # (B, T, K, act_dim)

        # 各成分标准差：exp(log_sigma) + sigma_min，并 clamp 上界
        log_sigma = self.log_sigma_head(cond)  # (B, T, K * act_dim)
        log_sigma = log_sigma.view(B, T, K, self.act_dim)
        # clamp log_sigma 防止数值不稳定
        log_sigma = torch.clamp(log_sigma, min=-4.0, max=2.0)
        sigma = torch.exp(log_sigma) + self.sigma_min  # (B, T, K, act_dim)
        sigma = torch.clamp(sigma, max=self.sigma_max)

        return pi, mu, sigma

    def predict_action(
        self,
        x: torch.Tensor,
        cpa_constraint: Optional[torch.Tensor] = None,
        mode: str = "max_pi",
    ) -> torch.Tensor:
        """
        推理时从 MDN 中提取确定性动作。

        mode:
          "max_pi"   — 取混合权重最高的成分的 μ（默认，最稳定）
          "mean"     — 加权均值 Σ_k π_k μ_k（期望值，更平滑）
          "cpa_select" — 按 CPA 直接选成分：
                         tight(CPA≤70)→成分0, mid→成分1, loose(CPA≥110)→成分2

        Returns: (B, T, act_dim)
        """
        pi, mu, sigma = self.forward(x, cpa_constraint)  # pi:(B,T,K), mu:(B,T,K,D)

        if mode == "mean":
            # 加权均值：Σ_k π_k μ_k
            action = (pi.unsqueeze(-1) * mu).sum(dim=2)  # (B, T, act_dim)

        elif mode == "cpa_select" and cpa_constraint is not None:
            # 按 CPA 阈值选成分
            # tight: CPA < 75 → 成分 0（保守）
            # loose: CPA > 105 → 成分 2（激进）
            # mid:   其余 → 成分 1（均衡）
            B, T, K, D = mu.shape
            cpa = cpa_constraint.float()  # (B,)
            comp_idx = torch.ones(B, dtype=torch.long, device=x.device)  # 默认成分 1
            comp_idx[cpa < 75.0] = 0   # 紧约束 → 保守成分
            comp_idx[cpa > 105.0] = 2  # 松约束 → 激进成分
            # 按选定成分索引 mu
            comp_idx_exp = comp_idx.view(B, 1, 1, 1).expand(B, T, 1, D)
            action = mu.gather(2, comp_idx_exp).squeeze(2)  # (B, T, act_dim)

        else:
            # max_pi：取权重最高成分的 μ
            best_k = pi.argmax(dim=-1)  # (B, T)
            best_k_exp = best_k.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, self.act_dim
            )  # (B, T, 1, act_dim)
            action = mu.gather(2, best_k_exp).squeeze(2)  # (B, T, act_dim)

        return action

    def nll_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        cpa_constraint: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        负对数似然损失：-log Σ_k π_k N(a_target | μ_k, σ_k)

        x:      (B, T, H)
        target: (B, T, act_dim)
        mask:   (B, T) 有效 token 掩码（1=有效，0=padding）

        Returns: 标量损失
        """
        pi, mu, sigma = self.forward(x, cpa_constraint)
        # pi:    (B, T, K)
        # mu:    (B, T, K, act_dim)
        # sigma: (B, T, K, act_dim)

        # 扩展 target 到 K 个成分
        target_exp = target.unsqueeze(2).expand_as(mu)  # (B, T, K, act_dim)

        # 各成分的对数概率密度：log N(a | μ_k, σ_k)
        # = -0.5 * log(2π) - log(σ_k) - 0.5 * ((a - μ_k) / σ_k)²
        log_prob_k = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((target_exp - mu) / sigma) ** 2
        )  # (B, T, K, act_dim)

        # 对 act_dim 求和（act_dim=1 时无影响）
        log_prob_k = log_prob_k.sum(dim=-1)  # (B, T, K)

        # log Σ_k π_k exp(log_prob_k)，用 logsumexp 保证数值稳定
        log_pi = torch.log(pi + 1e-8)  # (B, T, K)
        log_mixture = torch.logsumexp(log_pi + log_prob_k, dim=-1)  # (B, T)

        # 取负号得到 NLL
        nll = -log_mixture  # (B, T)

        if mask is not None:
            valid = mask.float()
            loss = (nll * valid).sum() / valid.sum().clamp_min(1.0)
        else:
            loss = nll.mean()

        return loss


class MultiScaleDecisionTransformerV2(DecisionTransformer):
    """
    MSDTv2: 在 MSDT 基础上的三项改进。

    相比 legacy MSDT:
    ✓ 因果卷积（不泄漏未来信息）
    ✓ 输入相关粒度门控（取代静态 granularity_attn 参数）
    ✓ 约束感知调制（预算/时间信号显式注入状态表示）

    相比 improved MSDT:
    ✓ 更轻量，单流编码器（不引入 ThreeStreamEncoder 的额外复杂度）
    ✓ 直接在原始 MSDT 骨干上打补丁，更易验证改进效果
    """

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
        backbone_variant: str = "v2",  # 接受但不使用此参数（保持接口兼容）
        use_dual_path_action_head: bool = False,
        use_asymmetric_action_head: bool = False,
        use_decoupled_head: bool = False,
        use_rtg_histogram: bool = False,
        rtg_num_bins: int = 20,
        rtg_bin_min: float = 0.0,
        rtg_bin_max: float = 2.0,
        # 消融开关
        disable_cross_granularity: bool = False,  # 去掉跨粒度注意力（temporal_fusion）
        disable_multiscale: bool = False,          # 去掉多尺度融合（state_encoder→单线性层）
        disable_dyn_gate: bool = False,            # 去掉动态门控（固定 gate=0.5）
        # 改进版双分支编码器
        use_dual_branch_v2: bool = False,       # 独立MLP + 预算感知gate
        use_film_encoder: bool = False,          # FiLM调制 (Perez et al., AAAI 2018)
        use_cross_stitch_encoder: bool = False,  # Cross-Stitch (Misra et al., CVPR 2016)
        use_aux_sep: bool = False,               # Auxiliary separation supervision
        aux_lambda_budget: float = 0.1,
        aux_lambda_market: float = 0.1,
        # Per-CPA Soft MoE
        use_cpa_moe: bool = False,
        cpa_moe_num_experts: int = 3,
        cpa_moe_emb_dim: int = 16,
        cpa_moe_expert_ratio: float = 1.0,
        # Per-CPA MDN action head (Scheme B: Mixture Density Network)
        use_cpa_mdn: bool = False,
        cpa_mdn_num_components: int = 3,
        cpa_mdn_emb_dim: int = 16,
        cpa_mdn_sigma_min: float = 0.01,
        cpa_mdn_sigma_max: float = 2.0,
        cpa_mdn_infer_mode: str = "max_pi",  # "max_pi" | "mean" | "cpa_select"
        # Model capacity
        hidden_size: int = 64,
        n_layers: int = 3,
        n_inner: int = 0,  # 0 = auto (hidden_size * 4); set explicitly for compat with old ckpts
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

        # Override hidden_size (parent hardcodes 64) and rebuild dependent layers
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

        # --- 替换 transformer blocks（父类用默认 n_head=1，这里用参数化 n_head）---
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

        # --- 状态编码器 ---
        self.disable_multiscale       = bool(disable_multiscale)
        self.use_dual_branch_v2       = bool(use_dual_branch_v2)
        self.use_film_encoder         = bool(use_film_encoder)
        self.use_cross_stitch_encoder = bool(use_cross_stitch_encoder)
        self.use_aux_sep              = bool(use_aux_sep)
        _enc_kwargs = dict(state_dim=int(state_dim), hidden_size=int(self.hidden_size),
                           coarse_idx=coarse_idx, fine_idx=fine_idx)
        if self.disable_multiscale:
            self.state_encoder = nn.Linear(int(state_dim), int(self.hidden_size))
        elif self.use_film_encoder:
            self.state_encoder = FiLMDualBranchEncoder(**_enc_kwargs)
        elif self.use_cross_stitch_encoder:
            self.state_encoder = CrossStitchDualBranchEncoder(**_enc_kwargs)
        elif self.use_dual_branch_v2:
            self.state_encoder = DualBranchEncoderV2(**_enc_kwargs)
        elif self.use_aux_sep:
            self.state_encoder = GranularityCalibrator_AuxSep(
                **_enc_kwargs,
                aux_lambda_budget=float(aux_lambda_budget),
                aux_lambda_market=float(aux_lambda_market),
            )
        else:
            self.state_encoder = GranularityCalibrator(**_enc_kwargs)

        # --- 改进 1: 因果时序融合 ---
        self.disable_cross_granularity = bool(disable_cross_granularity)
        if not self.disable_cross_granularity:
            self.temporal_fusion = CausalTemporalFusionV2(
                hidden_size=int(self.hidden_size),
                n_head=int(n_head),
                local_window=int(local_window),
            )

        # --- 改进 2: 输入相关粒度门控（取代 granularity_attn 静态参数）---
        self.disable_dyn_gate = bool(disable_dyn_gate)
        if not self.disable_dyn_gate:
            self.granularity_gate_net = nn.Linear(int(self.hidden_size), int(self.hidden_size))

        # --- 改进 3: 约束感知调制器 ---
        self.constraint_mod = ConstraintModulator(int(self.hidden_size))

        self.use_dual_path_action_head = use_dual_path_action_head
        self.use_asymmetric_action_head = use_asymmetric_action_head
        self.use_decoupled_head = use_decoupled_head

        if self.use_decoupled_head:
            self.predict_action = DecoupledGranularityHead(int(self.hidden_size), int(self.act_dim))
        elif self.use_dual_path_action_head:
            self.predict_action = ConstraintAwareActionHead(int(self.hidden_size), int(self.act_dim))
        elif self.use_asymmetric_action_head:
            self.predict_action = AsymmetricActionHead(int(self.hidden_size), int(self.act_dim))

        # H-RTG: replace continuous RTG embedding with discrete bin lookup
        self.use_rtg_histogram = bool(use_rtg_histogram)
        self.rtg_num_bins = int(rtg_num_bins)
        self.rtg_bin_min = float(rtg_bin_min)
        self.rtg_bin_max = float(rtg_bin_max)
        if self.use_rtg_histogram:
            # Replace embed_return (Linear(return_dim, hidden_size)) with bin embedding
            self.rtg_bin_embed = nn.Embedding(self.rtg_num_bins, int(self.hidden_size))
            # Classification head: predict which bin the next RTG falls into
            self.rtg_bin_head = nn.Linear(int(self.hidden_size), self.rtg_num_bins)

        # Per-CPA Soft MoE (optional, inserted before action head)
        self.use_cpa_moe = bool(use_cpa_moe)
        if self.use_cpa_moe:
            self.cpa_moe = PerCPASoftMoE(
                hidden_size=int(self.hidden_size),
                num_experts=int(cpa_moe_num_experts),
                cpa_emb_dim=int(cpa_moe_emb_dim),
                expert_hidden_ratio=float(cpa_moe_expert_ratio),
            )

        # Per-CPA MDN action head (Scheme B: Mixture Density Network)
        # 当启用时，predict_action 被替换为 MDN 头；
        # 训练时用 NLL loss，推理时用 predict_action(mode=cpa_mdn_infer_mode)
        self.use_cpa_mdn = bool(use_cpa_mdn)
        self.cpa_mdn_infer_mode = str(cpa_mdn_infer_mode)
        if self.use_cpa_mdn:
            # MDN 头替换原有的 predict_action（Linear(hidden_size, act_dim)）
            # 注意：MDN 内部包含 CPA embedding，不依赖外部 CPA state features
            self.cpa_mdn_head = PerCPAMDN(
                hidden_size=int(self.hidden_size),
                act_dim=int(self.act_dim),
                num_components=int(cpa_mdn_num_components),
                cpa_emb_dim=int(cpa_mdn_emb_dim),
                sigma_min=float(cpa_mdn_sigma_min),
                sigma_max=float(cpa_mdn_sigma_max),
            )
            # 保留原 predict_action 以兼容父类接口，但训练时不使用其 MSE loss

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        cpa_constraint: Optional[torch.Tensor] = None,  # (B,) raw CPA values for MoE routing
    ):
        B, T = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long, device=states.device)

        # 嵌入
        _aux_sep_loss = None
        if self.use_aux_sep and self.training:
            state_embeddings, _aux_sep_loss = self.state_encoder(states, return_aux_loss=True)
        else:
            state_embeddings = self.state_encoder(states)
        action_embeddings = self.embed_action(actions)
        bin_idx = None
        if self.use_rtg_histogram:
            # Discretize RTG into bins and use embedding lookup
            rtg_scalar = returns_to_go[:, :, 0]  # (B, T) use first dim
            bin_idx = ((rtg_scalar - self.rtg_bin_min) / (self.rtg_bin_max - self.rtg_bin_min) * self.rtg_num_bins).long()
            bin_idx = bin_idx.clamp(0, self.rtg_num_bins - 1)
            returns_embeddings = self.rtg_bin_embed(bin_idx)  # (B, T, H)
        else:
            returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # 交错堆叠 [R, s, a]
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
        state_ctx = x[:, 1]  # (B, T, H)

        # --- 改进 1: 因果时序融合 ---
        if self.disable_cross_granularity:
            fused = state_ctx  # 消融：跳过跨粒度注意力，直接用 transformer 输出
        else:
            fused = self.temporal_fusion(state_ctx, attention_mask=attention_mask)

        # --- 改进 3: 约束感知调制（将 budget/time 信号注入） ---
        fused = self.constraint_mod(fused, states)

        # --- 改进 2: 输入相关粒度门控 ---
        if self.disable_dyn_gate:
            fused_state_ctx = 0.5 * fused + 0.5 * state_ctx  # 消融：固定 gate=0.5
        else:
            gate = torch.sigmoid(self.granularity_gate_net(state_ctx))
            fused_state_ctx = gate * fused + (1.0 - gate) * state_ctx

        # --- 改进 4: Per-CPA Soft MoE (optional) ---
        if self.use_cpa_moe:
            fused_state_ctx = self.cpa_moe(fused_state_ctx, cpa_constraint)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])

        # --- 动作预测：MDN 头 or 标准头 ---
        if self.use_cpa_mdn:
            # MDN 推理：从混合分布中提取确定性动作
            # 训练时 method_model.py 会直接调用 cpa_mdn_head.nll_loss()，
            # 这里返回的 action_preds 仅用于评估/推理
            action_preds = self.cpa_mdn_head.predict_action(
                fused_state_ctx,
                cpa_constraint=cpa_constraint,
                mode=self.cpa_mdn_infer_mode,
            )
        elif getattr(self, "use_decoupled_head", False):
            action_preds = self.predict_action(fused_state_ctx, state_ctx)
        elif getattr(self, "use_dual_path_action_head", False):
            action_preds = self.predict_action(fused_state_ctx, states)
        else:
            action_preds = self.predict_action(fused_state_ctx)

        extras = {}
        if _aux_sep_loss is not None:
            extras["aux_sep_loss"] = _aux_sep_loss
        if return_features:
            extras.update({
                "state_ctx": state_ctx,
                "fused_state_ctx": fused_state_ctx,
                "action_token_ctx": x[:, 2],
            })
        if not extras:
            extras = None
        if extras is not None and return_features:
            if self.use_rtg_histogram:
                extras["rtg_bin_logits"] = self.rtg_bin_head(x[:, 0])  # (B, T, num_bins)
                extras["rtg_bin_idx"] = bin_idx  # (B, T)
            if self.use_cpa_mdn:
                # 将 MDN 参数存入 extras，供 method_model.py 计算 NLL loss
                # 以及供可视化脚本提取成分参数
                pi, mu, sigma = self.cpa_mdn_head.forward(fused_state_ctx, cpa_constraint)
                extras["mdn_pi"] = pi      # (B, T, K)
                extras["mdn_mu"] = mu      # (B, T, K, act_dim)
                extras["mdn_sigma"] = sigma  # (B, T, K, act_dim)
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
            
            # Dimension 0: Reward RTG
            pred_return[:, 0] = pred_return[:, 0] - (pre_reward / self.scale)
            
            if self.return_dim > 1 and pre_cost is not None:
                # Dimension 1: Either CPA-Slack or Budget RTG
                # If cpa_constraint is provided, it's CPA-Slack: cpa_slack_t = cpa_slack_{t-1} - cpa * pre_reward + pre_cost
                if cpa_constraint is not None:
                    cpa_c = float(cpa_constraint)
                    # Notice we also need to scale the cpa_constraint appropriately if needed, 
                    # but cpa_slack in dataset is: cpa_c * reward_rtg - future_cost. 
                    # The delta is - cpa_c * pre_reward + pre_cost
                    delta = (-cpa_c * pre_reward + pre_cost) / self.scale
                    pred_return[:, 1] = pred_return[:, 1] + delta
                else:
                    # Default: Budget RTG
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
