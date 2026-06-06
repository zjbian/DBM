import sys
from pathlib import Path

import numpy as np
import torch

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[2]
sys.path.insert(0, str(ROOT_DIR))

from model import ResearchMSDTModel


class MSDTSamplingBiddingStrategy(BaseBiddingStrategy):
    def __init__(
        self,
        budget=100,
        name="MSDT-Sampling-PlayerStrategy",
        cpa=2,
        category=1,
        load_dir=None,
        target_return_override=None,
    ):
        super().__init__(budget, name, cpa, category)
        root_dir = Path(__file__).resolve().parents[2]
        self.load_dir = Path(load_dir) if load_dir else root_dir / "saved_model" / "MSDT_sampling"
        # Try method-named checkpoint first, fall back to legacy msdt_sampling.pt
        ckpt_path = self.load_dir / "msdt_sampling.pt"
        if not ckpt_path.exists():
            pts = sorted(self.load_dir.glob("*.pt"))
            if pts:
                ckpt_path = pts[0]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MSDT checkpoint not found in: {self.load_dir}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.config = ckpt["config"]

        # --- backward-compat: infer n_inner from checkpoint weights ---
        sd = ckpt["model_state_dict"]
        mlp_key = "backbone.transformer.0.mlp.0.weight"
        if mlp_key in sd and "n_inner" not in self.config:
            self.config["n_inner"] = sd[mlp_key].shape[0]

        # --- backward-compat: remap old Sequential predict_action keys ---
        # Old code: predict_action = nn.Sequential([Linear]) → keys end in ".0.weight"/".0.bias"
        # New code: predict_action = nn.Linear → keys end in ".weight"/".bias"
        remapped = {}
        for k, v in sd.items():
            if k == "backbone.predict_action.0.weight":
                remapped["backbone.predict_action.weight"] = v
            elif k == "backbone.predict_action.0.bias":
                remapped["backbone.predict_action.bias"] = v
            else:
                remapped[k] = v
        ckpt["model_state_dict"] = remapped

        self.model = ResearchMSDTModel(config=self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.state_mean = np.asarray(ckpt["state_mean"], dtype=np.float32)
        self.state_std = np.asarray(ckpt["state_std"], dtype=np.float32)
        self.action_mean = float(ckpt["action_mean"])
        self.action_std = float(ckpt["action_std"])
        self.meta_mean = np.asarray(ckpt.get("meta_mean", np.zeros((2,), dtype=np.float32)), dtype=np.float32)
        self.meta_std = np.asarray(ckpt.get("meta_std", np.ones((2,), dtype=np.float32)), dtype=np.float32)
        self.retrieval_queries = np.asarray(
            ckpt.get("retrieval_queries", np.zeros((1, self.config["state_dim"] + 2), dtype=np.float32)),
            dtype=np.float32,
        )
        self.retrieval_contexts = np.asarray(
            ckpt.get("retrieval_contexts", np.zeros((1, self.config.get("retrieval_dim", 8)), dtype=np.float32)),
            dtype=np.float32,
        )
        self.cpa_ratio_mean = float(ckpt.get("cpa_ratio_mean", 0.0))
        self.cpa_ratio_std = float(ckpt.get("cpa_ratio_std", 1.0))
        if self.config.get("use_cpa_state_features", False):
            self.state_mean = np.append(self.state_mean, self.cpa_ratio_mean).astype(np.float32)
            self.state_std = np.append(self.state_std, self.cpa_ratio_std).astype(np.float32)

        self.target_return = (
            float(target_return_override)
            if target_return_override is not None
            else float(self.config.get("target_return", 50.0))
        )
        # CA-TR: per-CPA-group target return mapping (from oracle analysis on msdt_v2_allcons_s3)
        # Best TR per group: CPA=60→50, 70→50, 80→36, 90→50, 100→36, 110→44, 120→50, 130→32
        self.use_ca_tr = False  # enabled via enable_ca_tr()
        self._ca_tr_map = {60: 50, 70: 50, 80: 36, 90: 50, 100: 36, 110: 44, 120: 50, 130: 32}

    def enable_ca_tr(self, enabled: bool = True):
        """启用 CA-TR：根据广告主 CPA 约束动态选择 target_return。"""
        self.use_ca_tr = enabled

    def set_device(self, device):
        super().set_device(device)
        self.model.eval()

    def reset(self):
        self.remaining_budget = self.budget
        self.total_cost = 0.0
        self.total_conversion = 0.0
        # CA-TR: 在每个 episode 开始时根据 CPA 约束更新 target_return
        if self.use_ca_tr:
            cpa_key = min(self._ca_tr_map.keys(), key=lambda k: abs(k - self.cpa))
            self.target_return = float(self._ca_tr_map[cpa_key])
        self.model.init_eval()

    def _meta_norm(self) -> np.ndarray:
        meta_raw = np.asarray([self.budget, self.cpa], dtype=np.float32)
        return (meta_raw - self.meta_mean) / self.meta_std

    def _build_state(
        self,
        time_step_index,
        history_pvalue_info,
        history_bid,
        history_auction_result,
        history_impression_result,
        history_least_winning_cost,
        pvalues,
    ) -> np.ndarray:
        time_left = (48 - time_step_index) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0.0
        history_xi = [result[:, 0] for result in history_auction_result]
        history_pvalue = [result[:, 0] for result in history_pvalue_info]
        history_conversion = [result[:, 1] for result in history_impression_result]

        def _mean_hist(history):
            return np.mean([np.mean(x) for x in history]) if history else 0.0

        def _mean_last(history, n=3):
            last_n = history[max(0, len(history) - n) :]
            return np.mean([np.mean(x) for x in last_n]) if last_n else 0.0

        current_pvalue_mean = float(np.mean(pvalues)) if len(pvalues) > 0 else 0.0
        current_pv_num = float(len(pvalues))
        historical_pv_num_total = float(sum(len(bids) for bids in history_bid)) if history_bid else 0.0
        last_three_pv_num_total = (
            float(sum(len(history_bid[i]) for i in range(max(0, time_step_index - 3), time_step_index)))
            if history_bid
            else 0.0
        )

        features = [
            time_left,
            budget_left,
            _mean_hist(history_bid),
            _mean_last(history_bid, 3),
            _mean_hist(history_least_winning_cost),
            _mean_hist(history_pvalue),
            _mean_hist(history_conversion),
            _mean_hist(history_xi),
            _mean_last(history_least_winning_cost, 3),
            _mean_last(history_pvalue, 3),
            _mean_last(history_conversion, 3),
            _mean_last(history_xi, 3),
            current_pvalue_mean,
            current_pv_num,
            last_three_pv_num_total,
            historical_pv_num_total,
        ]
        if self.config.get("use_cpa_state_features", False):
            features.append(float(self.total_cost / (self.cpa * (self.total_conversion + 1.0))))
        return np.asarray(features, dtype=np.float32)

    def _heuristic_alpha(self, raw_state: np.ndarray) -> float:
        current_pvalue = max(float(raw_state[12]), 1e-6)
        trailing_lwc = max(float(raw_state[8]), 0.0)
        hist_lwc = max(float(raw_state[4]), 0.0)
        alpha = trailing_lwc / current_pvalue if trailing_lwc > 0 else hist_lwc / current_pvalue
        return float(np.clip(alpha, 0.0, 300.0))

    def _retrieval_context(self, state_norm: np.ndarray, meta_norm: np.ndarray) -> np.ndarray:
        if self.retrieval_queries.shape[0] == 0:
            return np.zeros((self.config.get("retrieval_dim", 8),), dtype=np.float32)
        base_len = self.retrieval_queries.shape[1] - len(meta_norm)
        query = np.concatenate([state_norm[:base_len], meta_norm], axis=0).astype(np.float32)
        query = query / (np.linalg.norm(query) + 1e-6)
        sims = self.retrieval_queries @ query
        topk = int(min(self.config.get("retrieval_topk", 4), self.retrieval_contexts.shape[0]))
        idx = np.argsort(-sims)[: max(topk, 1)]
        return np.mean(self.retrieval_contexts[idx], axis=0).astype(np.float32)

    def _shield_alpha(self, alpha: float, raw_state: np.ndarray) -> float:
        if not self.config.get("use_shield", False):
            return float(max(alpha, 0.0))
        time_left = max(float(raw_state[0]), 1e-3)
        desired_remaining = self.budget * time_left
        budget_ratio = self.remaining_budget / max(desired_remaining, 1e-6)
        budget_factor = float(
            np.clip(
                np.power(max(budget_ratio, 1e-3), self.config.get("shield_budget_beta", 0.5)),
                self.config.get("shield_min_factor", 0.85),
                self.config.get("shield_max_factor", 1.05),
            )
        )
        if self.total_conversion > 0:
            current_cpa = self.total_cost / max(self.total_conversion, 1e-6)
        elif self.total_cost > 0:
            current_cpa = self.cpa * 2.0
        else:
            current_cpa = self.cpa
        cpa_ratio = self.cpa / max(current_cpa, 1e-6)
        cpa_factor = float(
            np.clip(
                np.power(max(cpa_ratio, 1e-3), self.config.get("shield_cpa_beta", 0.5)),
                self.config.get("shield_min_factor", 0.85),
                self.config.get("shield_max_factor", 1.05),
            )
        )
        return float(max(alpha, 0.0) * min(budget_factor, cpa_factor))

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
        device=None,
    ):
        del pValueSigmas, device
        pre_reward = None
        pre_cost = None
        if timeStepIndex > 0 and historyImpressionResult and historyAuctionResult:
            pre_reward = float(np.sum(historyImpressionResult[-1][:, 1]))
            pre_cost = float(np.sum(historyAuctionResult[-1][:, 2]))
            self.total_conversion += pre_reward
            self.total_cost += pre_cost

        raw_state = self._build_state(
            timeStepIndex,
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
            pValues,
        )
        state_norm = (raw_state - self.state_mean) / self.state_std
        meta_norm = self._meta_norm()
        retrieval_context = self._retrieval_context(state_norm, meta_norm)
        heuristic_alpha = self._heuristic_alpha(raw_state)
        heuristic_norm = (heuristic_alpha - self.action_mean) / self.action_std

        time_elapsed = timeStepIndex / 48.0
        feasible_allowance = self.cpa * (self.total_conversion + 1.0) + self.budget * 0.08 * time_elapsed
        pfeas_step = 1.0 if self.total_cost <= feasible_allowance else 0.0

        out = self.model.predict_step(
            state_norm=torch.as_tensor(state_norm, dtype=torch.float32, device=self.device),
            meta=torch.as_tensor(meta_norm, dtype=torch.float32, device=self.device),
            retrieval_context=torch.as_tensor(retrieval_context, dtype=torch.float32, device=self.device),
            heuristic_action_norm=torch.as_tensor([heuristic_norm], dtype=torch.float32, device=self.device),
            target_return=self.target_return if timeStepIndex == 0 else None,
            target_budget_return=(self.remaining_budget / float(self.config.get("scale", 40.0))) if timeStepIndex == 0 else None,
            pre_reward=pre_reward,
            pre_cost=pre_cost,
            cpa_constraint=self.cpa,
            pfeas_step=pfeas_step,
        )
        alpha_norm = float(out["action_norm"].item())
        alpha = max(alpha_norm * self.action_std + self.action_mean, 0.0)
        alpha *= float(out["alpha_scale"].item())
        alpha = self._shield_alpha(alpha, raw_state)
        return alpha * pValues
