import glob
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

def _safe_parse_state(state_str, state_dim: int) -> np.ndarray:
    if pd.isna(state_str):
        return np.zeros((state_dim,), dtype=np.float32)
    s = str(state_str).replace("np.float64(", "").replace(")", "").strip("()")
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) > state_dim:
        vals = vals[:state_dim]
    elif len(vals) < state_dim:
        vals += [0.0] * (state_dim - len(vals))
    arr = np.asarray(vals, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

class StratifiedTrajectorySampler(Sampler[int]):
    def __init__(
        self,
        bucket_indices: List[List[int]],
        *,
        batch_size: int,
        num_samples: int,
        bucket_weights: Optional[List[float]] = None,
        seed: Optional[int] = None,
        traj_weights: Optional[np.ndarray] = None,
    ):
        self.bucket_indices = [list(bucket) for bucket in bucket_indices]
        self.batch_size = int(batch_size)
        self.num_samples = int(num_samples)
        self.seed = seed
        if bucket_weights is None:
            bucket_weights = [1.0] * max(len(self.bucket_indices), 1)
        weights = np.asarray(bucket_weights, dtype=np.float64)
        if weights.size != len(self.bucket_indices):
            raise ValueError("bucket_weights size must match bucket_indices")
        weights = np.clip(weights, 0.0, None)
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        self.bucket_weights = weights / weights.sum()
        self.bucket_traj_probs: List[Optional[np.ndarray]] = []
        if traj_weights is not None:
            traj_weights = np.asarray(traj_weights, dtype=np.float64)
            for bucket in self.bucket_indices:
                if not bucket:
                    self.bucket_traj_probs.append(None)
                else:
                    bw = traj_weights[bucket]
                    s = bw.sum()
                    self.bucket_traj_probs.append((bw / s) if s > 0 else None)
        else:
            self.bucket_traj_probs = [None] * len(self.bucket_indices)

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        all_indices = [idx for bucket in self.bucket_indices for idx in bucket]
        if not all_indices:
            return iter([])
        num_batches = int(np.ceil(self.num_samples / max(self.batch_size, 1)))
        quota = np.floor(self.bucket_weights * self.batch_size).astype(int)
        remainder = int(self.batch_size - quota.sum())
        if remainder > 0:
            frac_order = np.argsort(-(self.bucket_weights * self.batch_size - quota))
            for i in frac_order[:remainder]:
                quota[i] += 1

        sampled = []
        for _ in range(num_batches):
            batch = []
            for bucket_id, count in enumerate(quota):
                if count <= 0:
                    continue
                pool = self.bucket_indices[bucket_id] if self.bucket_indices[bucket_id] else all_indices
                probs = self.bucket_traj_probs[bucket_id] if bucket_id < len(self.bucket_traj_probs) else None
                chosen = rng.choice(pool, size=count, replace=True, p=probs).tolist()
                batch.extend(int(x) for x in chosen)
            if len(batch) < self.batch_size:
                batch.extend(int(x) for x in rng.choice(all_indices, size=self.batch_size - len(batch), replace=True).tolist())
            rng.shuffle(batch)
            sampled.extend(batch)
        return iter(sampled[: self.num_samples])

    def __len__(self):
        return self.num_samples

class MethodReplayBuffer(Dataset):
    def __init__(
        self,
        *,
        state_dim: int,
        act_dim: int,
        data_dir: str,
        train_periods: Optional[List[int]] = None,
        max_ep_len: int = 48,
        scale: float = 40.0,
        K: int = 20,
        reward_key: str = "reward",
        enable_weighted_sampling: bool = True,
        sampling_score_mode: str = "hybrid",
        sampling_awr_beta: float = 20.0,
        loss_weight_mode: str = "traj_score",
        loss_weight_min: float = 0.25,
        loss_weight_max: float = 4.0,
        retrieval_topk: int = 4,
        return_dim: int = 1,
        use_score_rtg: bool = False,
        use_cpa_slack_rtg: bool = False,
        use_cpa_state_features: bool = False,
        use_pfeas_rtg_scale: bool = False,
        use_cpa_compliance_filter: bool = False,
        cpa_compliance_tight_threshold: float = 80.0,
        use_cpa_normalized_rtg: bool = False,
        use_dense_reward_shaping: bool = False,
        dense_reward_scale: float = 0.3,
        use_advantage_weight: bool = False,
        advantage_scale: float = 2.0,
        use_rtg_noise: bool = False,
        rtg_noise_std: float = 0.05,
        use_demo_prefix: bool = False,
        demo_prefix_len: int = 4,
        use_stratified_prefix_sampling: bool = False,
        stratified_bucket_weights: Optional[List[float]] = None,
        safe_prefix_sample_prob: float = 0.75,
        risky_prefix_sample_prob: float = 0.35,
        tight_cpa_threshold: float = 70.0,
        tight_cpa_oversample: float = 2.0,
        use_hindsight_rtg: bool = False,
        use_cpa_scaled_rtg: bool = False,
        cpa_scaled_rtg_mode: str = "linear",
        cpa_scaled_rtg_median: float = 95.0,
        use_hindsight_truncation: bool = False,
        hindsight_truncation_cpa_thresh: float = 80.0,
        hindsight_truncation_min_len: int = 5,
        use_transition_sampling: bool = False,
        transition_sample_prob: float = 0.30,
        transition_window_before: int = 10,
        transition_window_after: int = 5,
        use_cpa_progress_reward: bool = False,
        cpa_progress_alpha: float = 0.1,
        cpa_progress_zero_mean: bool = True,
        cpa_progress_min_conv: int = 3,
        use_cpa_aware_prefix_prob: bool = False,
        loose_cpa_threshold: float = 90.0,
        medium_cpa_threshold: float = 70.0,
        loose_cpa_safe_prob: float = 0.41,
        medium_cpa_safe_prob: float = 0.62,
        sampler_seed: Optional[int] = None,
        use_quality_aware_ht: bool = False,
        quality_ht_low_thresh: float = 0.25,
        quality_ht_high_thresh: float = 0.80,
        use_quality_aware_sampling: bool = False,
        quality_sample_center: float = 0.50,
        quality_sample_width: float = 0.25,
        quality_sample_boost: float = 1.5,
    ):
        super().__init__()
        self.device = "cpu"
        self.use_score_rtg = bool(use_score_rtg)
        self.use_cpa_slack_rtg = bool(use_cpa_slack_rtg)
        self.use_cpa_state_features = bool(use_cpa_state_features)
        self.use_pfeas_rtg_scale = bool(use_pfeas_rtg_scale)
        self.use_cpa_compliance_filter = bool(use_cpa_compliance_filter)
        self.cpa_compliance_tight_threshold = float(cpa_compliance_tight_threshold)
        self.use_cpa_normalized_rtg = bool(use_cpa_normalized_rtg)
        self.use_dense_reward_shaping = bool(use_dense_reward_shaping)
        self.dense_reward_scale = float(dense_reward_scale)
        self.use_advantage_weight = bool(use_advantage_weight)
        self.advantage_scale = float(advantage_scale)
        self.use_rtg_noise = bool(use_rtg_noise)
        self.rtg_noise_std = float(rtg_noise_std)
        self.use_demo_prefix = bool(use_demo_prefix)
        self.demo_prefix_len = int(demo_prefix_len)
        self.use_cpa_scaled_rtg = bool(use_cpa_scaled_rtg)
        self.cpa_scaled_rtg_mode = str(cpa_scaled_rtg_mode)
        self.cpa_scaled_rtg_median = float(cpa_scaled_rtg_median)
        self.use_hindsight_truncation = bool(use_hindsight_truncation)
        self.hindsight_truncation_cpa_thresh = float(hindsight_truncation_cpa_thresh)
        self.hindsight_truncation_min_len = int(hindsight_truncation_min_len)
        self.use_transition_sampling = bool(use_transition_sampling)
        self.transition_sample_prob = float(transition_sample_prob)
        self.transition_window_before = int(transition_window_before)
        self.transition_window_after = int(transition_window_after)
        self.use_stratified_prefix_sampling = bool(use_stratified_prefix_sampling)
        self.stratified_bucket_weights = (
            [float(x) for x in stratified_bucket_weights]
            if stratified_bucket_weights is not None
            else [0.35, 0.35, 0.15, 0.15]
        )
        self.safe_prefix_sample_prob = float(safe_prefix_sample_prob)
        self.risky_prefix_sample_prob = float(risky_prefix_sample_prob)
        self.tight_cpa_threshold = float(tight_cpa_threshold)
        self.tight_cpa_oversample = float(tight_cpa_oversample)
        self.use_hindsight_rtg = bool(use_hindsight_rtg)
        self.use_cpa_progress_reward = bool(use_cpa_progress_reward)
        self.cpa_progress_alpha = float(cpa_progress_alpha)
        self.cpa_progress_zero_mean = bool(cpa_progress_zero_mean)
        self.cpa_progress_min_conv = int(cpa_progress_min_conv)
        self.use_cpa_aware_prefix_prob = bool(use_cpa_aware_prefix_prob)
        self.loose_cpa_threshold = float(loose_cpa_threshold)
        self.medium_cpa_threshold = float(medium_cpa_threshold)
        self.loose_cpa_safe_prob = float(loose_cpa_safe_prob)
        self.medium_cpa_safe_prob = float(medium_cpa_safe_prob)
        self._sampler_seed = sampler_seed
        self.use_quality_aware_ht = bool(use_quality_aware_ht)
        self.quality_ht_low_thresh = float(quality_ht_low_thresh)
        self.quality_ht_high_thresh = float(quality_ht_high_thresh)
        self.use_quality_aware_sampling = bool(use_quality_aware_sampling)
        self.quality_sample_center = float(quality_sample_center)
        self.quality_sample_width = float(quality_sample_width)
        self.quality_sample_boost = float(quality_sample_boost)
        self.base_state_dim = int(state_dim)
        self.state_dim = self.base_state_dim + (1 if use_cpa_state_features else 0)
        self.act_dim = int(act_dim)
        self.max_ep_len = int(max_ep_len)
        self.scale = float(scale)
        self.K = int(K)
        self.reward_key = str(reward_key)
        self.enable_weighted_sampling = bool(enable_weighted_sampling)
        self.sampling_score_mode = str(sampling_score_mode)
        self.sampling_awr_beta = float(sampling_awr_beta)
        self.loss_weight_mode = str(loss_weight_mode)
        self.loss_weight_min = float(loss_weight_min)
        self.loss_weight_max = float(loss_weight_max)
        self.retrieval_topk = int(retrieval_topk)
        self.return_dim = int(return_dim)

        self.data_dir = Path(data_dir)
        self.train_periods = [int(p) for p in train_periods] if train_periods else None

        self.trajectories = self._load_trajectories()
        if len(self.trajectories) == 0:
            raise RuntimeError(f"No trajectories found in {self.data_dir} for periods={self.train_periods}")

        self._build_normalizers()
        self._finalize_trajectory_features()
        self._build_expert_bank()
        self.loss_weights = self._build_loss_weights()
        self.p_sample = self._build_sampling_probs()
        self._build_retrieval_bank()
        self._build_stratified_sampling()
        self._build_hindsight_truncated_trajectories()
        if self.use_hindsight_truncation:
            self._build_stratified_sampling()

    def _iter_files(self):
        if self.train_periods:
            for period in self.train_periods:
                yield self.data_dir / f"period-{period}-rlData.csv"
        else:
            for fp in sorted(glob.glob(str(self.data_dir / "period-*-rlData.csv"))):
                yield Path(fp)

    def _load_trajectories(self):
        trajectories: List[Dict] = []
        for fp in self._iter_files():
            if not fp.exists():
                continue
            df = pd.read_csv(fp)
            for (period, advertiser_id), group in df.groupby(["deliveryPeriodIndex", "advertiserNumber"]):
                group = group.sort_values("timeStepIndex").reset_index(drop=True)
                states_raw = np.stack([_safe_parse_state(v, self.base_state_dim) for v in group["state"]], axis=0).astype(np.float32)
                actions_raw = group["action"].astype(np.float32).to_numpy().reshape(-1, 1)
                rewards_raw = group[self.reward_key].astype(np.float32).to_numpy().reshape(-1, 1)
                sparse_rewards = group["reward"].astype(np.float32).to_numpy().reshape(-1, 1)
                costs_raw = group["cost"].astype(np.float32).to_numpy().reshape(-1, 1) if "cost" in group.columns else np.zeros_like(rewards_raw)
                dones = group["done"].astype(np.int64).to_numpy()
                budget = float(group["budget"].iloc[0]) if "budget" in group.columns else 0.0
                cpa_constraint = float(group["CPAConstraint"].iloc[0]) if "CPAConstraint" in group.columns else 0.0
                traj_score = float(group["traj_score"].iloc[0]) if "traj_score" in group.columns else float(np.sum(sparse_rewards))
                reward_sum = float(np.sum(sparse_rewards))
                cumulative_reward = np.cumsum(sparse_rewards, axis=0).astype(np.float32)
                cumulative_cost = np.cumsum(costs_raw, axis=0).astype(np.float32)
                next_budget_left = np.zeros_like(costs_raw)
                if len(states_raw) > 1:
                    next_budget_left[:-1, 0] = states_raw[1:, 1]
                    next_budget_left[-1, 0] = states_raw[-1, 1]
                else:
                    next_budget_left[:, 0] = states_raw[:, 1]
                trajectories.append(
                    {
                        "period": int(period),
                        "advertiser_id": int(advertiser_id),
                        "states_raw": states_raw,
                        "actions_raw": actions_raw,
                        "rewards_raw": rewards_raw,
                        "sparse_rewards": sparse_rewards,
                        "costs_raw": costs_raw,
                        "next_budget_left": next_budget_left.astype(np.float32),
                        "dones": dones,
                        "budget": float(budget),
                        "cpa_constraint": float(cpa_constraint),
                        "traj_score": float(traj_score),
                        "reward_sum": float(reward_sum),
                        "cumulative_reward": cumulative_reward,
                        "cumulative_cost": cumulative_cost,
                    }
                )
        return trajectories

    def _build_normalizers(self):
        all_states = np.concatenate([traj["states_raw"] for traj in self.trajectories], axis=0)
        all_actions = np.concatenate([traj["actions_raw"] for traj in self.trajectories], axis=0)
        budgets = np.asarray([traj["budget"] for traj in self.trajectories], dtype=np.float32).reshape(-1, 1)
        cpas = np.asarray([traj["cpa_constraint"] for traj in self.trajectories], dtype=np.float32).reshape(-1, 1)
        self.state_mean = np.mean(all_states, axis=0).astype(np.float32)
        self.state_std = (np.std(all_states, axis=0) + 1e-6).astype(np.float32)
        self.action_mean = float(np.mean(all_actions))
        self.action_std = float(np.std(all_actions) + 1e-6)
        self.meta_mean = np.concatenate([np.mean(budgets, axis=0), np.mean(cpas, axis=0)], axis=0).astype(np.float32)
        self.meta_std = np.concatenate([np.std(budgets, axis=0) + 1e-6, np.std(cpas, axis=0) + 1e-6], axis=0).astype(np.float32)
        if self.use_cpa_state_features:
            all_cpa_ratios = []
            for traj in self.trajectories:
                cr = traj["cumulative_reward"]
                cc = traj["cumulative_cost"]
                cpa_c = float(traj["cpa_constraint"])
                cpa_ratio = cc / (cpa_c * (cr + 1.0))
                all_cpa_ratios.append(cpa_ratio)
            all_cpa_ratios = np.concatenate(all_cpa_ratios, axis=0)
            self.cpa_ratio_mean = float(np.mean(all_cpa_ratios))
            self.cpa_ratio_std = float(np.std(all_cpa_ratios) + 1e-6)
        else:
            self.cpa_ratio_mean = 0.0
            self.cpa_ratio_std = 1.0

    def _build_sampling_component(self, xs: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.float32)
        xs = np.clip(xs, 0.0, None) + 1e-3
        return xs / max(float(np.mean(xs)), 1e-6)

    def _build_loss_weights(self) -> np.ndarray:
        if self.loss_weight_mode == "uniform":
            return np.ones((len(self.trajectories),), dtype=np.float32)
        weights = self._build_sampling_component(np.asarray([traj["traj_score"] for traj in self.trajectories], dtype=np.float32))
        weights = np.clip(weights, self.loss_weight_min, self.loss_weight_max)
        return weights.astype(np.float32)

    def _build_sampling_probs(self) -> np.ndarray:
        if not self.enable_weighted_sampling:
            return np.ones((len(self.trajectories),), dtype=np.float32) / float(len(self.trajectories))
        reward_component = self._build_sampling_component(np.asarray([traj["reward_sum"] for traj in self.trajectories], dtype=np.float32))
        traj_component = self._build_sampling_component(np.asarray([traj["traj_score"] for traj in self.trajectories], dtype=np.float32))
        if self.sampling_score_mode == "reward_sum":
            scores = reward_component
        elif self.sampling_score_mode == "traj_score":
            scores = traj_component
        elif self.sampling_score_mode == "awr":
            raw = np.asarray([t["traj_score"] for t in self.trajectories], dtype=np.float32)
            raw_norm = (raw - raw.max()) / (raw.std() + 1e-6)
            scores = np.exp(raw_norm / self.sampling_awr_beta)
        else:
            scores = np.sqrt(reward_component * traj_component)
        if self.use_quality_aware_sampling:
            quality_arr = np.asarray(
                [traj.get("quality_ratio", 0.5) for traj in self.trajectories], dtype=np.float32
            )
            bell = np.exp(-((quality_arr - self.quality_sample_center) ** 2) / (2 * self.quality_sample_width ** 2))
            quality_weights = 1.0 + self.quality_sample_boost * bell
            scores = scores * quality_weights
        scores = scores.astype(np.float32)
        return scores / max(float(np.sum(scores)), 1e-6)

    def _finalize_trajectory_features(self):
        quality_ratios = []
        for traj in self.trajectories:
            traj["states_norm"] = ((traj["states_raw"] - self.state_mean) / self.state_std).astype(np.float32)
            traj["actions_norm"] = ((traj["actions_raw"] - self.action_mean) / self.action_std).astype(np.float32)
            traj["meta_raw"] = np.asarray([traj["budget"], traj["cpa_constraint"]], dtype=np.float32)
            traj["meta_norm"] = ((traj["meta_raw"] - self.meta_mean) / self.meta_std).astype(np.float32)
            costs = traj["costs_raw"].reshape(-1)
            spent_prefix = np.concatenate([[0.0], np.cumsum(costs[:-1])], axis=0)
            remaining_budget = np.clip(traj["budget"] - spent_prefix, 0.0, None).astype(np.float32).reshape(-1, 1)
            traj["remaining_budget_seq"] = remaining_budget
            traj["quality_ratio"] = float(traj["traj_score"] / max(traj["reward_sum"], 1.0)) if traj["reward_sum"] > 0 else 0.0
            quality_ratios.append(traj["quality_ratio"])
            total_cost = float(traj["cumulative_cost"][-1, 0])
            total_reward = float(traj["cumulative_reward"][-1, 0])
            actual_cpa = total_cost / max(total_reward, 1.0)
            traj["cpa_violated"] = actual_cpa > traj["cpa_constraint"]
            summary = np.concatenate(
                [
                    np.mean(traj["states_norm"], axis=0),
                    traj["meta_norm"],
                ],
                axis=0,
            )
            traj["summary_query"] = summary.astype(np.float32)
            retrieval_ctx = np.asarray(
                [
                    np.mean(traj["states_norm"][:, 2]),
                    np.mean(traj["states_norm"][:, 4]),
                    np.mean(traj["states_norm"][:, 6]),
                    np.mean(traj["states_norm"][:, 8]),
                    np.mean(traj["states_norm"][:, 12]),
                    float(traj["reward_sum"]) / self.scale,
                    float(traj["traj_score"]) / self.scale,
                    traj["meta_norm"][1],
                ],
                dtype=np.float32,
            )
            traj["retrieval_context"] = retrieval_ctx
        quality_arr = np.asarray(quality_ratios, dtype=np.float32)
        self.quality_mean = float(np.mean(quality_arr)) if quality_arr.size > 0 else 1.0
        self.quality_std = float(np.std(quality_arr) + 1e-6) if quality_arr.size > 0 else 1.0
        for traj in self.trajectories:
            traj["quality_target"] = float(1.0 + 0.5 * ((traj["quality_ratio"] - self.quality_mean) / self.quality_std))
            mean_r = float(np.mean(traj["rewards_raw"])) if len(traj["rewards_raw"]) > 0 else 0.0
            traj["step_advantage"] = (traj["rewards_raw"].reshape(-1) - mean_r).astype(np.float32)

    def _build_retrieval_bank(self):
        queries = np.stack([traj["summary_query"] for traj in self.trajectories], axis=0).astype(np.float32)
        contexts = np.stack([traj["retrieval_context"] for traj in self.trajectories], axis=0).astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-6
        queries_norm = queries / norms
        sim = queries_norm @ queries_norm.T
        np.fill_diagonal(sim, -1e9)
        nn_idx = np.argsort(-sim, axis=1)[:, : max(self.retrieval_topk, 1)]
        self.retrieval_queries = queries_norm.astype(np.float32)
        self.retrieval_contexts = contexts.astype(np.float32)
        self.neighbor_indices = nn_idx.astype(np.int64)
        for i, traj in enumerate(self.trajectories):
            traj["retrieval_neighbors"] = self.neighbor_indices[i]
            traj["retrieval_context_avg"] = np.mean(self.retrieval_contexts[traj["retrieval_neighbors"]], axis=0).astype(np.float32)

    def _build_stratified_sampling(self):
        traj_scores = np.asarray([traj["traj_score"] for traj in self.trajectories], dtype=np.float32)
        reward_sums = np.asarray([traj["reward_sum"] for traj in self.trajectories], dtype=np.float32)
        q50 = float(np.quantile(traj_scores, 0.50))
        q75 = float(np.quantile(traj_scores, 0.75))
        reward_q50 = float(np.quantile(reward_sums, 0.50))
        self.stratified_bucket_indices: List[List[int]] = [[] for _ in range(4)]
        for idx, traj in enumerate(self.trajectories):
            total_len = len(traj["states_raw"])
            time_axis = np.arange(total_len, dtype=np.float32) / max(float(total_len - 1), 1.0)
            feasible_allowance = (
                float(traj["cpa_constraint"]) * (traj["cumulative_reward"] + 1.0)
                + float(traj["budget"]) * 0.08 * time_axis.reshape(-1, 1)
            )
            prefix_feasible = (traj["cumulative_cost"] <= feasible_allowance).astype(np.float32).reshape(-1)
            traj["prefix_feasible_full"] = prefix_feasible
            traj["prefix_feasible_ratio"] = float(np.mean(prefix_feasible))
            safe_starts = np.where(prefix_feasible > 0.5)[0].astype(np.int64)
            risky_starts = np.where(prefix_feasible <= 0.5)[0].astype(np.int64)
            late_starts = np.where(np.arange(total_len) >= max(total_len - self.K, 0))[0].astype(np.int64)
            traj["safe_start_indices"] = safe_starts
            traj["risky_start_indices"] = risky_starts
            traj["late_start_indices"] = late_starts

            if self.use_transition_sampling and traj.get("cpa_violated", False):
                t_star = self._find_violation_step(traj)
                t_before = getattr(self, "transition_window_before", 10)
                t_after = getattr(self, "transition_window_after", 5)
                t_lo = max(0, t_star - t_before)
                t_hi = min(total_len - self.K, t_star + t_after)
                if t_lo <= t_hi:
                    transition_starts = np.arange(t_lo, t_hi + 1, dtype=np.int64)
                else:
                    transition_starts = np.asarray([], dtype=np.int64)
            else:
                transition_starts = np.asarray([], dtype=np.int64)
            traj["transition_start_indices"] = transition_starts

            if float(traj["traj_score"]) >= q75 and not traj["cpa_violated"]:
                bucket = 0
            elif float(traj["traj_score"]) >= q50 and float(traj["prefix_feasible_ratio"]) >= 0.6:
                bucket = 1
            elif float(traj["reward_sum"]) >= reward_q50 and traj["cpa_violated"]:
                bucket = 2
            else:
                bucket = 3
            traj["traj_bucket"] = int(bucket)
            self.stratified_bucket_indices[bucket].append(idx)

    def _find_violation_step(self, traj: Dict) -> int:
        cpa_c = float(traj["cpa_constraint"])
        cum_cost = traj["cumulative_cost"].reshape(-1)
        cum_reward = traj["cumulative_reward"].reshape(-1)
        allowance = cpa_c * (cum_reward + 1.0)
        violated = cum_cost > allowance
        idxs = np.where(violated)[0]
        return int(idxs[0]) if len(idxs) > 0 else len(cum_cost)

    def _compute_cpa_progress_reward(self, traj: Dict) -> np.ndarray:
        cpa_target = float(traj["cpa_constraint"])
        cum_cost = traj["cumulative_cost"].reshape(-1)
        cum_reward = traj["cumulative_reward"].reshape(-1)
        T = len(cum_cost)

        cum_cpa = cum_cost / np.maximum(cum_reward, 1.0)

        cpa_slack = (cpa_target - cum_cpa) / max(cpa_target, 1e-6)
        r_aux = np.clip(cpa_slack, -1.0, 1.0).astype(np.float32)

        min_conv = self.cpa_progress_min_conv
        conv_gate = np.minimum(cum_reward / max(float(min_conv), 1.0), 1.0)
        r_aux = r_aux * conv_gate

        if self.cpa_progress_zero_mean:
            r_aux = r_aux - np.mean(r_aux)

        return r_aux.reshape(-1, 1).astype(np.float32)

    def _build_hindsight_truncated_trajectories(self):
        if not self.use_hindsight_truncation:
            return

        new_trajs = []
        for orig_idx, traj in enumerate(self.trajectories):
            if not traj.get("cpa_violated", False):
                continue
            if self.use_quality_aware_ht:
                quality = traj.get("quality_ratio", 0.5)
                if quality >= self.quality_ht_high_thresh:
                    continue
                if quality < self.quality_ht_low_thresh:
                    pass
                else:
                    if float(traj["cpa_constraint"]) > self.hindsight_truncation_cpa_thresh:
                        continue
            else:
                if float(traj["cpa_constraint"]) > self.hindsight_truncation_cpa_thresh:
                    continue

            t_star = self._find_violation_step(traj)
            if t_star < self.hindsight_truncation_min_len:
                continue

            T_trunc = t_star

            trunc_reward_sum = float(np.sum(traj["sparse_rewards"][:T_trunc]))
            trunc_traj_score = trunc_reward_sum

            trunc_traj = {
                "period": traj["period"],
                "advertiser_id": traj["advertiser_id"],
                "states_raw": traj["states_raw"][:T_trunc],
                "actions_raw": traj["actions_raw"][:T_trunc],
                "rewards_raw": traj["rewards_raw"][:T_trunc],
                "sparse_rewards": traj["sparse_rewards"][:T_trunc],
                "costs_raw": traj["costs_raw"][:T_trunc],
                "next_budget_left": traj["next_budget_left"][:T_trunc],
                "dones": traj["dones"][:T_trunc],
                "budget": traj["budget"],
                "cpa_constraint": traj["cpa_constraint"],
                "traj_score": trunc_traj_score,
                "reward_sum": trunc_reward_sum,
                "cumulative_reward": traj["cumulative_reward"][:T_trunc],
                "cumulative_cost": traj["cumulative_cost"][:T_trunc],
                "is_truncated": True,
                "truncated_from": orig_idx,
                "cpa_violated": False,
            }

            trunc_traj["states_norm"] = (
                (trunc_traj["states_raw"] - self.state_mean) / self.state_std
            ).astype(np.float32)
            trunc_traj["actions_norm"] = (
                (trunc_traj["actions_raw"] - self.action_mean) / self.action_std
            ).astype(np.float32)
            trunc_traj["meta_raw"] = np.asarray(
                [trunc_traj["budget"], trunc_traj["cpa_constraint"]], dtype=np.float32
            )
            trunc_traj["meta_norm"] = (
                (trunc_traj["meta_raw"] - self.meta_mean) / self.meta_std
            ).astype(np.float32)

            costs = trunc_traj["costs_raw"].reshape(-1)
            spent_prefix = np.concatenate([[0.0], np.cumsum(costs[:-1])], axis=0)
            remaining_budget = np.clip(
                trunc_traj["budget"] - spent_prefix, 0.0, None
            ).astype(np.float32).reshape(-1, 1)
            trunc_traj["remaining_budget_seq"] = remaining_budget

            trunc_traj["quality_ratio"] = float(
                trunc_traj["traj_score"] / max(trunc_traj["reward_sum"], 1.0)
            ) if trunc_traj["reward_sum"] > 0 else 0.0
            trunc_traj["quality_target"] = float(
                1.0 + 0.5 * (
                    (trunc_traj["quality_ratio"] - self.quality_mean) / self.quality_std
                )
            )

            mean_r = float(np.mean(trunc_traj["rewards_raw"])) if len(trunc_traj["rewards_raw"]) > 0 else 0.0
            trunc_traj["step_advantage"] = (
                trunc_traj["rewards_raw"].reshape(-1) - mean_r
            ).astype(np.float32)

            summary = np.concatenate(
                [np.mean(trunc_traj["states_norm"], axis=0), trunc_traj["meta_norm"]], axis=0
            )
            trunc_traj["summary_query"] = summary.astype(np.float32)
            retrieval_ctx = np.asarray([
                np.mean(trunc_traj["states_norm"][:, 2]),
                np.mean(trunc_traj["states_norm"][:, 4]),
                np.mean(trunc_traj["states_norm"][:, 6]),
                np.mean(trunc_traj["states_norm"][:, 8]),
                np.mean(trunc_traj["states_norm"][:, 12]),
                float(trunc_traj["reward_sum"]) / self.scale,
                float(trunc_traj["traj_score"]) / self.scale,
                trunc_traj["meta_norm"][1],
            ], dtype=np.float32)
            trunc_traj["retrieval_context"] = retrieval_ctx
            trunc_traj["retrieval_context_avg"] = retrieval_ctx.copy()

            tlen = min(self.max_ep_len, T_trunc)
            trunc_traj["expert_actions_norm"] = self.expert_action_proto[:tlen].copy()

            new_trajs.append(trunc_traj)

        if not new_trajs:
            return

        self.trajectories.extend(new_trajs)

        self.loss_weights = self._build_loss_weights()
        self.p_sample = self._build_sampling_probs()

    def _build_expert_bank(self):
        qualities = np.asarray([traj["quality_ratio"] for traj in self.trajectories], dtype=np.float32)
        if qualities.size == 0:
            self.expert_action_proto = np.zeros((self.max_ep_len, self.act_dim), dtype=np.float32)
            self.expert_traj_indices = []
            return
        threshold = float(np.quantile(qualities, 0.75))
        expert_trajs = [traj for traj in self.trajectories if traj["quality_ratio"] >= threshold]
        if len(expert_trajs) == 0:
            expert_trajs = list(self.trajectories)
        self.expert_traj_indices = [i for i, t in enumerate(self.trajectories) if t["quality_ratio"] >= threshold]
        expert_proto = np.zeros((self.max_ep_len, self.act_dim), dtype=np.float32)
        expert_count = np.zeros((self.max_ep_len, 1), dtype=np.float32)
        for traj in expert_trajs:
            acts = traj["actions_norm"]
            tlen = min(self.max_ep_len, acts.shape[0])
            expert_proto[:tlen] += acts[:tlen]
            expert_count[:tlen] += 1.0
        expert_proto = expert_proto / np.clip(expert_count, 1.0, None)
        self.expert_action_proto = expert_proto.astype(np.float32)
        for traj in self.trajectories:
            tlen = min(self.max_ep_len, traj["actions_norm"].shape[0])
            traj["expert_actions_norm"] = self.expert_action_proto[:tlen].copy()

    def export_aux_stats(self):
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "meta_mean": self.meta_mean,
            "meta_std": self.meta_std,
            "retrieval_queries": self.retrieval_queries,
            "retrieval_contexts": self.retrieval_contexts,
            "cpa_ratio_mean": float(self.cpa_ratio_mean),
            "cpa_ratio_std": float(self.cpa_ratio_std),
        }

    def discount_cumsum(self, x, gamma=1.0):
        out = np.zeros_like(x)
        out[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            out[t] = x[t] + gamma * out[t + 1]
        return out

    def _heuristic_alpha(self, raw_states: np.ndarray) -> np.ndarray:
        current_pvalue = np.maximum(raw_states[:, 12], 1e-6)
        trailing_lwc = np.maximum(raw_states[:, 8], 0.0)
        hist_lwc = np.maximum(raw_states[:, 4], 0.0)
        alpha = np.where(trailing_lwc > 0, trailing_lwc / current_pvalue, hist_lwc / current_pvalue)
        alpha = np.clip(alpha, 0.0, 300.0).astype(np.float32)
        return alpha.reshape(-1, 1)

    def __len__(self):
        return len(self.trajectories)

    def build_train_sampler(self, *, batch_size: int, num_samples: int):
        if not self.use_stratified_prefix_sampling:
            return None
        if self.tight_cpa_oversample > 1.0:
            weights = np.ones(len(self.trajectories), dtype=np.float64)
            for idx, traj in enumerate(self.trajectories):
                is_tight = float(traj["cpa_constraint"]) < self.tight_cpa_threshold
                is_hq = traj.get("traj_bucket", 3) in (0, 1)
                if is_tight and is_hq:
                    weights[idx] = self.tight_cpa_oversample
            weights = weights / weights.sum()
            num_total = int(np.ceil(num_samples / batch_size)) * batch_size
            sampled = np.random.choice(len(self.trajectories), size=num_total, replace=True, p=weights).tolist()
            return iter(sampled[:num_samples])
        awr_traj_weights = None
        if self.enable_weighted_sampling and self.sampling_score_mode == "awr":
            raw = np.asarray([t["traj_score"] for t in self.trajectories], dtype=np.float64)
            raw_norm = (raw - raw.max()) / (raw.std() + 1e-6)
            awr_traj_weights = np.exp(raw_norm / self.sampling_awr_beta)

        return StratifiedTrajectorySampler(
            self.stratified_bucket_indices,
            batch_size=int(batch_size),
            num_samples=int(num_samples),
            bucket_weights=self.stratified_bucket_weights,
            seed=self._sampler_seed,
            traj_weights=awr_traj_weights,
        )

    def _sample_start_t(self, traj: Dict) -> int:
        max_start = max(len(traj["rewards_raw"]) - 1, 0)
        if not self.use_stratified_prefix_sampling:
            return random.randint(0, max_start)
        bucket = int(traj.get("traj_bucket", 1))
        safe_pool = traj.get("safe_start_indices", np.asarray([], dtype=np.int64))
        risky_pool = traj.get("risky_start_indices", np.asarray([], dtype=np.int64))
        late_pool = traj.get("late_start_indices", np.asarray([], dtype=np.int64))
        transition_pool = traj.get("transition_start_indices", np.asarray([], dtype=np.int64))

        if self.use_cpa_aware_prefix_prob:
            cpa_c = float(traj["cpa_constraint"])
            if cpa_c > self.loose_cpa_threshold:
                effective_safe_prob = self.loose_cpa_safe_prob
            elif cpa_c > self.medium_cpa_threshold:
                effective_safe_prob = self.medium_cpa_safe_prob
            else:
                effective_safe_prob = self.safe_prefix_sample_prob
        else:
            effective_safe_prob = self.safe_prefix_sample_prob

        choose_safe = random.random() < effective_safe_prob
        if (self.use_transition_sampling
                and bucket == 2
                and transition_pool.size > 0
                and random.random() < self.transition_sample_prob):
            return int(np.random.choice(transition_pool))
        if bucket == 2 and risky_pool.size > 0 and random.random() < self.risky_prefix_sample_prob:
            return int(np.random.choice(risky_pool))
        if bucket in (0, 1) and safe_pool.size > 0 and choose_safe:
            return int(np.random.choice(safe_pool))
        if bucket == 3 and late_pool.size > 0 and random.random() < 0.35:
            return int(np.random.choice(late_pool))
        if safe_pool.size > 0 and choose_safe:
            return int(np.random.choice(safe_pool))
        if risky_pool.size > 0:
            return int(np.random.choice(risky_pool))
        return random.randint(0, max_start)

    def __getitem__(self, index):
        traj = self.trajectories[int(index)]
        start_t = self._sample_start_t(traj)
        end_t = min(start_t + self.K, len(traj["rewards_raw"]))

        states_norm = traj["states_norm"][start_t:end_t]
        states_raw = traj["states_raw"][start_t:end_t]
        actions_norm = traj["actions_norm"][start_t:end_t]
        actions_raw = traj["actions_raw"][start_t:end_t]
        rewards_raw = traj["rewards_raw"][start_t:end_t]
        costs_raw = traj["costs_raw"][start_t:end_t]
        remaining_budget_seq = traj["remaining_budget_seq"][start_t:end_t]
        next_budget_left = traj["next_budget_left"][start_t:end_t]
        cumulative_reward = traj["cumulative_reward"][start_t:end_t]
        cumulative_cost = traj["cumulative_cost"][start_t:end_t]
        dones = traj["dones"][start_t:end_t]
        timesteps = np.arange(start_t, start_t + len(states_norm), dtype=np.int64)
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1

        reward_rtg = self.discount_cumsum(traj["rewards_raw"][start_t:], gamma=1.0)[: len(states_norm) + 1]
        if self.use_cpa_progress_reward:
            r_aux_full = self._compute_cpa_progress_reward(traj)
            augmented_reward = traj["rewards_raw"][start_t:] + self.cpa_progress_alpha * r_aux_full[start_t:]
            reward_rtg = self.discount_cumsum(augmented_reward, gamma=1.0)[: len(states_norm) + 1]
        if len(reward_rtg) <= len(states_norm):
            reward_rtg = np.concatenate([reward_rtg, np.zeros((1, 1), dtype=np.float32)], axis=0)
        if self.return_dim > 1:
            if getattr(self, "use_cpa_slack_rtg", False):
                cpa_c = float(traj["cpa_constraint"])
                total_cost = float(np.sum(traj["costs_raw"]))
                future_cost = total_cost - traj["cumulative_cost"][start_t:start_t + len(states_norm)]
                cpa_slack = cpa_c * reward_rtg[:len(states_norm)] - future_cost
                cpa_slack_rtg = np.concatenate([cpa_slack, cpa_slack[-1:]], axis=0)
                rtg = np.concatenate([reward_rtg, cpa_slack_rtg], axis=-1)
            else:
                budget_rtg = np.concatenate([remaining_budget_seq, remaining_budget_seq[-1:]], axis=0)
                rtg = np.concatenate([reward_rtg, budget_rtg], axis=-1)
        else:
            rtg = reward_rtg

        if self.use_score_rtg and float(traj["reward_sum"]) > 0:
            quality_ratio = float(traj["traj_score"]) / max(float(traj["reward_sum"]), 1.0)
            quality_ratio = float(np.clip(quality_ratio, 0.1, 1.0))
            rtg = rtg * quality_ratio

        if self.use_cpa_scaled_rtg:
            cpa_c = float(traj["cpa_constraint"])
            median = self.cpa_scaled_rtg_median
            mode = self.cpa_scaled_rtg_mode
            if mode == "linear":
                cpa_scale = cpa_c / median
            elif mode == "log":
                cpa_scale = float(np.log(cpa_c + 1.0) / np.log(median + 1.0))
            else:
                if cpa_c <= 70.0:
                    cpa_scale = 0.7
                elif cpa_c <= 100.0:
                    cpa_scale = 1.0
                else:
                    cpa_scale = 1.3
            rtg_scaled = rtg.copy()
            rtg_scaled[:, 0:1] = rtg[:, 0:1] * float(cpa_scale)
            rtg = rtg_scaled

        if (getattr(self, "use_hindsight_rtg", False)
                and traj.get("cpa_violated", False)
                and float(traj["cpa_constraint"]) < self.tight_cpa_threshold):
            cpa_c = float(traj["cpa_constraint"])
            total_reward = float(traj["reward_sum"])
            total_cost = float(np.sum(traj["costs_raw"]))
            if total_reward > 0 and total_cost > 0:
                max_compliant_cost = cpa_c * total_reward
                scale_ratio = float(np.clip(max_compliant_cost / total_cost, 0.5, 1.0))
                cf_reward = total_reward * float(np.sqrt(scale_ratio))
                cf_score = cf_reward
                cf_quality_ratio = float(np.clip(cf_score / max(total_reward, 1.0), 0.1, 1.0))
                if self.use_score_rtg and quality_ratio > 0:
                    rtg = rtg / quality_ratio * cf_quality_ratio
                else:
                    rtg = rtg * cf_quality_ratio

        if self.use_dense_reward_shaping:
            pvalue_seq = traj["states_raw"][start_t:, 12].reshape(-1, 1).astype(np.float32)
            pvalue_rtg = self.discount_cumsum(pvalue_seq, gamma=1.0)[: len(states_norm) + 1]
            if len(pvalue_rtg) <= len(states_norm):
                pvalue_rtg = np.concatenate([pvalue_rtg, np.zeros((1, 1), dtype=np.float32)], axis=0)
            pvalue_rtg_norm = pvalue_rtg / (float(np.max(np.abs(pvalue_rtg))) + 1e-6)
            rtg = rtg + self.dense_reward_scale * pvalue_rtg_norm

        if self.use_rtg_noise:
            noise = np.random.randn(*rtg.shape).astype(np.float32) * self.rtg_noise_std
            rtg = rtg + noise

        heuristics_raw = self._heuristic_alpha(states_raw)
        heuristics_norm = ((heuristics_raw - self.action_mean) / self.action_std).astype(np.float32)
        tlen = len(states_norm)

        if self.use_cpa_state_features:
            cpa_c = float(traj["cpa_constraint"])
            cpa_ratio_raw = cumulative_cost / (cpa_c * (cumulative_reward + 1.0))
            cpa_ratio_norm_arr = ((cpa_ratio_raw - self.cpa_ratio_mean) / self.cpa_ratio_std).astype(np.float32)
            states_norm = np.concatenate([states_norm, cpa_ratio_norm_arr], axis=-1)
            states_raw = np.concatenate([states_raw, cpa_ratio_raw], axis=-1)

        budget_gap = states_raw[:, 1] - states_raw[:, 0]
        route_targets = np.ones((tlen,), dtype=np.int64)
        route_margin = 0.08
        route_targets[budget_gap < -route_margin] = 0
        route_targets[budget_gap > route_margin] = 2

        time_elapsed = (1.0 - states_raw[:, 0:1]).astype(np.float32)
        feasible_allowance = (
            float(traj["cpa_constraint"]) * (cumulative_reward + 1.0)
            + float(traj["budget"]) * 0.08 * time_elapsed
        )
        prefix_feasibility = (cumulative_cost <= feasible_allowance).astype(np.float32)

        if (self.use_cpa_compliance_filter
                and traj.get("cpa_violated", False)
                and float(traj["cpa_constraint"]) <= self.cpa_compliance_tight_threshold):
            prefix_feasibility = np.zeros_like(prefix_feasibility)

        if self.use_pfeas_rtg_scale:
            pfeas_scale = (0.3 + 0.7 * prefix_feasibility)
            rtg_body = rtg[:-1] * pfeas_scale
            rtg = np.concatenate([rtg_body, rtg[-1:]], axis=0)

        pad = self.K - tlen

        def _pad2(arr, width):
            return np.concatenate([np.zeros((pad, width), dtype=np.float32), arr], axis=0)

        states_norm = _pad2(states_norm, self.state_dim)
        states_raw = _pad2(states_raw, self.state_dim)
        actions_norm = _pad2(actions_norm, self.act_dim)
        actions_raw = _pad2(actions_raw, self.act_dim)
        rewards_scaled = _pad2(rewards_raw / self.scale, 1)
        rewards_raw = _pad2(rewards_raw, 1)
        costs_scaled = _pad2(costs_raw / self.scale, 1)
        next_rewards = np.concatenate([rewards_raw[1:], rewards_raw[-1:]], axis=0)
        next_costs = np.concatenate([costs_scaled[1:], costs_scaled[-1:]], axis=0)
        next_budget_left = _pad2(next_budget_left, 1)
        rtg = np.concatenate([np.zeros((pad, self.return_dim), dtype=np.float32), rtg], axis=0)
        rtg_scale = float(traj["cpa_constraint"]) if self.use_cpa_normalized_rtg else self.scale
        rtg[:, 0:1] = rtg[:, 0:1] / rtg_scale
        if self.return_dim > 1:
            rtg[:, 1:2] = rtg[:, 1:2] / self.scale
            if getattr(self, "use_cpa_slack_rtg", False):
                rtg[:, 1:2] = np.clip(rtg[:, 1:2], -10.0, 10.0)
        dones = np.concatenate([np.ones((pad,), dtype=np.int64) * 2, dones], axis=0)
        timesteps = np.concatenate([np.zeros((pad,), dtype=np.int64), timesteps], axis=0)
        mask = np.concatenate([np.zeros((pad,), dtype=np.float32), np.ones((tlen,), dtype=np.float32)], axis=0)
        heuristics_norm = _pad2(heuristics_norm, 1)
        expert_actions = _pad2(traj["expert_actions_norm"][start_t:end_t], self.act_dim)
        prefix_feasibility = _pad2(prefix_feasibility, 1)
        route_targets = np.concatenate([np.zeros((pad,), dtype=np.int64), route_targets], axis=0)

        step_adv = traj["step_advantage"][start_t:end_t]
        step_adv = np.concatenate([np.zeros(pad, dtype=np.float32), step_adv], axis=0)

        demo_states = np.zeros((self.demo_prefix_len, self.state_dim), dtype=np.float32)
        demo_actions = np.zeros((self.demo_prefix_len, self.act_dim), dtype=np.float32)
        demo_rtg = np.zeros((self.demo_prefix_len, self.return_dim), dtype=np.float32)
        if self.use_demo_prefix and len(self.expert_traj_indices) > 0:
            demo_idx = self.expert_traj_indices[np.random.randint(len(self.expert_traj_indices))]
            demo_traj = self.trajectories[demo_idx]
            dT = len(demo_traj["states_norm"])
            d_start = np.random.randint(0, max(dT - self.demo_prefix_len, 1))
            d_end = min(d_start + self.demo_prefix_len, dT)
            d_len = d_end - d_start
            demo_states[:d_len] = demo_traj["states_norm"][d_start:d_end]
            demo_actions[:d_len] = demo_traj["actions_norm"][d_start:d_end]
            demo_rtg_full = self.discount_cumsum(demo_traj["rewards_raw"][d_start:], gamma=1.0)
            demo_rtg_slice = demo_rtg_full[:d_len] / self.scale
            if self.use_score_rtg and float(demo_traj["reward_sum"]) > 0:
                qr = float(np.clip(demo_traj["traj_score"] / max(demo_traj["reward_sum"], 1.0), 0.1, 1.0))
                demo_rtg_slice = demo_rtg_slice * qr
            demo_rtg[:d_len, 0] = demo_rtg_slice.reshape(-1)[:d_len]

        quality_target = float(np.clip(traj["quality_target"], 0.0, 2.0))
        retrieval_context = traj["retrieval_context_avg"].astype(np.float32)

        return {
            "states": torch.from_numpy(states_norm).float(),
            "raw_states": torch.from_numpy(states_raw).float(),
            "actions": torch.from_numpy(actions_norm).float(),
            "raw_actions": torch.from_numpy(actions_raw).float(),
            "rewards": torch.from_numpy(rewards_scaled).float(),
            "rewards_raw": torch.from_numpy(rewards_raw).float(),
            "costs": torch.from_numpy(costs_scaled).float(),
            "next_rewards": torch.from_numpy(next_rewards).float(),
            "next_costs": torch.from_numpy(next_costs).float(),
            "next_budget_left": torch.from_numpy(next_budget_left).float(),
            "dones": torch.from_numpy(dones).long(),
            "rtg": torch.from_numpy(rtg).float(),
            "timesteps": torch.from_numpy(timesteps).long(),
            "mask": torch.from_numpy(mask).float(),
            "sample_weight": torch.tensor(float(self.loss_weights[int(index)]), dtype=torch.float32),
            "traj_score": torch.tensor(float(traj["traj_score"]), dtype=torch.float32),
            "reward_sum": torch.tensor(float(traj["reward_sum"]), dtype=torch.float32),
            "meta": torch.from_numpy(traj["meta_norm"]).float(),
            "retrieval_context": torch.from_numpy(retrieval_context).float(),
            "heuristic_actions": torch.from_numpy(heuristics_norm).float(),
            "expert_actions": torch.from_numpy(expert_actions).float(),
            "prefix_feasibility": torch.from_numpy(prefix_feasibility).float(),
            "route_targets": torch.from_numpy(route_targets).long(),
            "quality_target": torch.tensor(float(quality_target), dtype=torch.float32),
            "period": torch.tensor(int(traj["period"]), dtype=torch.long),
            "advertiser_id": torch.tensor(int(traj["advertiser_id"]), dtype=torch.long),
            "step_advantage": torch.from_numpy(step_adv).float(),
            "demo_states": torch.from_numpy(demo_states).float(),
            "demo_actions": torch.from_numpy(demo_actions).float(),
            "demo_rtg": torch.from_numpy(demo_rtg).float(),
            "cpa_constraint_val": torch.tensor(float(traj["cpa_constraint"]), dtype=torch.float32),
        }
