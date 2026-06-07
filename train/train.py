import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CURRENT_DIR))

from model import ResearchMSDTModel
from common_utils import save_normalize_dict
from dataset import MethodReplayBuffer
from method_configs import build_method_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def load_method_overrides(args) -> dict:
    overrides = {}
    if args.config_override_file:
        override_path = Path(args.config_override_file)
        overrides.update(json.loads(override_path.read_text(encoding="utf-8")))
    if args.config_override_json:
        overrides.update(json.loads(args.config_override_json))
    return overrides

def train_model(args):
    import random, numpy as np
    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training args: %s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    method_cfg = build_method_config(args.method)
    method_overrides = load_method_overrides(args)
    if method_overrides:
        method_cfg.update(method_overrides)
        logger.info("Method overrides: %s", json.dumps(method_overrides, ensure_ascii=False, indent=2))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    base_state_dim = int(method_cfg.get("base_state_dim", args.state_dim))
    use_cpa_state = bool(method_cfg.get("use_cpa_state_features", False))
    effective_state_dim = base_state_dim + (1 if use_cpa_state else 0)

    replay_buffer = MethodReplayBuffer(
        state_dim=base_state_dim,
        act_dim=args.act_dim,
        data_dir=args.data_dir,
        train_periods=[int(x) for x in args.train_periods.split(",") if x.strip()],
        max_ep_len=args.max_ep_len,
        scale=args.scale,
        K=args.K,
        reward_key=args.reward_key,
        enable_weighted_sampling=bool(args.enable_weighted_sampling),
        sampling_score_mode=method_cfg["sampling_score_mode"],
        sampling_awr_beta=float(method_cfg.get("sampling_awr_beta", 20.0)),
        loss_weight_mode=method_cfg["loss_weight_mode"],
        loss_weight_min=method_cfg["loss_weight_min"],
        loss_weight_max=method_cfg["loss_weight_max"],
        retrieval_topk=int(method_cfg.get("retrieval_topk", 4)),
        return_dim=method_cfg["return_dim"],
        use_score_rtg=bool(method_cfg.get("use_score_rtg", False)),
        use_cpa_slack_rtg=bool(method_cfg.get("use_cpa_slack_rtg", False)),
        use_cpa_state_features=use_cpa_state,
        use_pfeas_rtg_scale=bool(method_cfg.get("use_pfeas_rtg_scale", False)),
        use_cpa_compliance_filter=bool(method_cfg.get("use_cpa_compliance_filter", False)),
        cpa_compliance_tight_threshold=float(method_cfg.get("cpa_compliance_tight_threshold", 80.0)),
        use_cpa_normalized_rtg=bool(method_cfg.get("use_cpa_normalized_rtg", False)),
        use_dense_reward_shaping=bool(method_cfg.get("use_dense_reward_shaping", False)),
        dense_reward_scale=float(method_cfg.get("dense_reward_scale", 0.3)),
        use_advantage_weight=bool(method_cfg.get("use_advantage_weight", False)),
        advantage_scale=float(method_cfg.get("advantage_scale", 2.0)),
        use_rtg_noise=bool(method_cfg.get("use_rtg_noise", False)),
        rtg_noise_std=float(method_cfg.get("rtg_noise_std", 0.05)),
        use_demo_prefix=bool(method_cfg.get("use_demo_prefix", False)),
        demo_prefix_len=int(method_cfg.get("demo_prefix_len", 4)),
        use_stratified_prefix_sampling=bool(method_cfg.get("use_stratified_prefix_sampling", False)),
        stratified_bucket_weights=method_cfg.get("stratified_bucket_weights", None),
        safe_prefix_sample_prob=float(method_cfg.get("safe_prefix_sample_prob", 0.75)),
        risky_prefix_sample_prob=float(method_cfg.get("risky_prefix_sample_prob", 0.35)),
        tight_cpa_threshold=float(method_cfg.get("tight_cpa_threshold", 70.0)),
        tight_cpa_oversample=float(method_cfg.get("tight_cpa_oversample", 1.0)),
        use_hindsight_rtg=bool(method_cfg.get("use_hindsight_rtg", False)),
        use_cpa_aware_prefix_prob=bool(method_cfg.get("use_cpa_aware_prefix_prob", False)),
        loose_cpa_threshold=float(method_cfg.get("loose_cpa_threshold", 90.0)),
        medium_cpa_threshold=float(method_cfg.get("medium_cpa_threshold", 70.0)),
        loose_cpa_safe_prob=float(method_cfg.get("loose_cpa_safe_prob", 0.41)),
        medium_cpa_safe_prob=float(method_cfg.get("medium_cpa_safe_prob", 0.62)),
        use_cpa_scaled_rtg=bool(method_cfg.get("use_cpa_scaled_rtg", False)),
        cpa_scaled_rtg_mode=str(method_cfg.get("cpa_scaled_rtg_mode", "linear")),
        cpa_scaled_rtg_median=float(method_cfg.get("cpa_scaled_rtg_median", 95.0)),
        use_hindsight_truncation=bool(method_cfg.get("use_hindsight_truncation", False)),
        hindsight_truncation_cpa_thresh=float(method_cfg.get("hindsight_truncation_cpa_thresh", 80.0)),
        hindsight_truncation_min_len=int(method_cfg.get("hindsight_truncation_min_len", 5)),
        use_transition_sampling=bool(method_cfg.get("use_transition_sampling", False)),
        transition_sample_prob=float(method_cfg.get("transition_sample_prob", 0.30)),
        transition_window_before=int(method_cfg.get("transition_window_before", 10)),
        transition_window_after=int(method_cfg.get("transition_window_after", 5)),
        use_cpa_progress_reward=bool(method_cfg.get("use_cpa_progress_reward", False)),
        cpa_progress_alpha=float(method_cfg.get("cpa_progress_alpha", 0.1)),
        cpa_progress_zero_mean=bool(method_cfg.get("cpa_progress_zero_mean", True)),
        cpa_progress_min_conv=int(method_cfg.get("cpa_progress_min_conv", 3)),
        use_quality_aware_ht=bool(method_cfg.get("use_quality_aware_ht", False)),
        quality_ht_low_thresh=float(method_cfg.get("quality_ht_low_thresh", 0.25)),
        quality_ht_high_thresh=float(method_cfg.get("quality_ht_high_thresh", 0.80)),
        use_quality_aware_sampling=bool(method_cfg.get("use_quality_aware_sampling", False)),
        quality_sample_center=float(method_cfg.get("quality_sample_center", 0.50)),
        quality_sample_width=float(method_cfg.get("quality_sample_width", 0.25)),
        quality_sample_boost=float(method_cfg.get("quality_sample_boost", 1.5)),
        sampler_seed=seed,
    )
    logger.info("Replay buffer size: %d", len(replay_buffer))

    val_buffer = None
    if args.val_periods:
        val_buffer = MethodReplayBuffer(
            state_dim=base_state_dim,
            act_dim=args.act_dim,
            data_dir=args.data_dir,
            train_periods=[int(x) for x in args.val_periods.split(",") if x.strip()],
            max_ep_len=args.max_ep_len,
            scale=args.scale,
            K=args.K,
            reward_key=args.reward_key,
            enable_weighted_sampling=False,
            sampling_score_mode=method_cfg["sampling_score_mode"],
            loss_weight_mode="uniform",
            loss_weight_min=1.0,
            loss_weight_max=1.0,
            retrieval_topk=method_cfg["retrieval_topk"],
            return_dim=method_cfg["return_dim"],
            use_score_rtg=bool(method_cfg.get("use_score_rtg", False)),
            use_cpa_slack_rtg=bool(method_cfg.get("use_cpa_slack_rtg", False)),
            use_cpa_state_features=use_cpa_state,
            use_dense_reward_shaping=bool(method_cfg.get("use_dense_reward_shaping", False)),
            dense_reward_scale=float(method_cfg.get("dense_reward_scale", 0.3)),
            use_advantage_weight=bool(method_cfg.get("use_advantage_weight", False)),
            advantage_scale=float(method_cfg.get("advantage_scale", 2.0)),
            use_rtg_noise=False,
            use_demo_prefix=bool(method_cfg.get("use_demo_prefix", False)),
            demo_prefix_len=int(method_cfg.get("demo_prefix_len", 4)),
            use_stratified_prefix_sampling=False,
        )

    aux_stats = replay_buffer.export_aux_stats()
    save_normalize_dict(
        {
            "state_mean": aux_stats["state_mean"],
            "state_std": aux_stats["state_std"],
            "action_mean": aux_stats["action_mean"],
            "action_std": aux_stats["action_std"],
            "meta_mean": aux_stats["meta_mean"],
            "meta_std": aux_stats["meta_std"],
            "retrieval_queries": aux_stats["retrieval_queries"],
            "retrieval_contexts": aux_stats["retrieval_contexts"],
        },
        str(save_dir),
    )

    config = {
        "method": args.method,
        "state_dim": effective_state_dim,
        "act_dim": args.act_dim,
        "K": args.K,
        "max_ep_len": args.max_ep_len,
        "scale": args.scale,
        "target_return": args.target_return,
        "coarse_idx": [int(x) for x in args.coarse_idx.split(",") if x.strip()],
        "fine_idx": [int(x) for x in args.fine_idx.split(",") if x.strip()],
        "local_window": args.local_window,
        "n_head": args.n_head,
        "reward_key": args.reward_key,
        "learning_rate": args.learning_rate,
        "return_dim": int(method_cfg["return_dim"]),
    }
    config.update(method_cfg)
    (save_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    model = ResearchMSDTModel(config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / 10000.0, 1.0))

    resume_start_step = int(getattr(args, "start_step", 0))
    _init_ckpt = getattr(args, "init_ckpt", "")
    if _init_ckpt:
        _ck = torch.load(_init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(_ck["model_state_dict"])
        logger.info("Resumed model weights from %s", _init_ckpt)
    if resume_start_step > 0:
        scheduler.last_epoch = resume_start_step
        _mult = min((resume_start_step + 1) / 10000.0, 1.0)
        for pg in optimizer.param_groups:
            pg["lr"] = float(args.learning_rate) * _mult
        logger.info("Resume: start_step=%d, LR fast-forwarded to %.2e", resume_start_step, float(args.learning_rate) * _mult)

    use_ema = bool(getattr(args, "use_ema", False))
    ema_decay = float(getattr(args, "ema_decay", 0.999))
    ema_start_step = int(getattr(args, "ema_start_step", 6000))
    ema_shadow: dict = {}
    ema_active = False
    if use_ema:
        for name, param in model.named_parameters():
            ema_shadow[name] = param.data.clone().float()
        logger.info("EMA enabled (decay=%.4f, start_step=%d)", ema_decay, ema_start_step)

    use_swa = bool(getattr(args, "use_swa", False))
    swa_start_step = int(getattr(args, "swa_start_step", 12000))
    swa_interval = int(getattr(args, "swa_interval", 1000))
    swa_snapshots: list = []
    if use_swa:
        logger.info("SWA enabled (start_step=%d, interval=%d)", swa_start_step, swa_interval)

    sampler = replay_buffer.build_train_sampler(
        batch_size=args.batch_size,
        num_samples=args.train_steps * args.batch_size,
    )
    if sampler is None:
        sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=args.train_steps * args.batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=args.batch_size)

    model.train()
    best_loss = float("inf")
    best_val_loss = float("inf")
    patience_counter = 0
    best_path = save_dir / f"{args.method}.pt"

    ckpt_start_step = int(getattr(args, "ckpt_start_step", 6000))
    ckpt_interval = int(getattr(args, "ckpt_interval", 1000))
    _ckpt_steps_raw = getattr(args, "ckpt_steps", "")
    ckpt_steps_set: set | None = (
        set(int(x.strip()) for x in _ckpt_steps_raw.split(",") if x.strip())
        if _ckpt_steps_raw else None
    )
    ckpt_best_loss: dict = {}
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_manifest_path = ckpt_dir / "manifest.json"
    ckpt_manifest: list = []

    def save_periodic_ckpt(step: int, loss: float):
        ckpt_path = ckpt_dir / f"{args.method}_ckpt{step:06d}.pt"
        if use_ema and ema_active and ema_shadow:
            orig = {n: p.data.clone() for n, p in model.named_parameters()}
            for name, param in model.named_parameters():
                param.data.copy_(ema_shadow[name].to(param.dtype))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "state_mean": aux_stats["state_mean"],
                "state_std": aux_stats["state_std"],
                "action_mean": aux_stats["action_mean"],
                "action_std": aux_stats["action_std"],
                "meta_mean": aux_stats["meta_mean"],
                "meta_std": aux_stats["meta_std"],
                "retrieval_queries": aux_stats["retrieval_queries"],
                "retrieval_contexts": aux_stats["retrieval_contexts"],
                "cpa_ratio_mean": aux_stats.get("cpa_ratio_mean", 0.0),
                "cpa_ratio_std": aux_stats.get("cpa_ratio_std", 1.0),
                "ckpt_step": step,
                "ckpt_loss": loss,
            },
            ckpt_path,
        )
        if use_ema and ema_active and ema_shadow:
            for name, param in model.named_parameters():
                param.data.copy_(orig[name])
        ckpt_manifest.append({"step": step, "path": str(ckpt_path), "loss": loss})
        ckpt_manifest_path.write_text(
            json.dumps(ckpt_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Periodic ckpt saved: step=%d loss=%.6f path=%s", step, loss, ckpt_path)

    def save_ckpt():
        if use_ema and ema_active and ema_shadow:
            orig = {n: p.data.clone() for n, p in model.named_parameters()}
            for name, param in model.named_parameters():
                param.data.copy_(ema_shadow[name].to(param.dtype))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "state_mean": aux_stats["state_mean"],
                "state_std": aux_stats["state_std"],
                "action_mean": aux_stats["action_mean"],
                "action_std": aux_stats["action_std"],
                "meta_mean": aux_stats["meta_mean"],
                "meta_std": aux_stats["meta_std"],
                "retrieval_queries": aux_stats["retrieval_queries"],
                "retrieval_contexts": aux_stats["retrieval_contexts"],
                "cpa_ratio_mean": aux_stats.get("cpa_ratio_mean", 0.0),
                "cpa_ratio_std": aux_stats.get("cpa_ratio_std", 1.0),
            },
            best_path,
        )
        if use_ema and ema_active and ema_shadow:
            for name, param in model.named_parameters():
                param.data.copy_(orig[name])

    save_smooth_k = int(getattr(args, "save_smooth_k", 500))
    recent_losses: list = []

    for step, batch in enumerate(dataloader, start=1 + resume_start_step):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        losses = model.compute_losses(batch)
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        if use_ema:
            if not ema_active and step >= ema_start_step:
                for name, param in model.named_parameters():
                    ema_shadow[name] = param.data.clone().float()
                ema_active = True
                logger.info("EMA activated at step %d", step)
            if ema_active:
                for name, param in model.named_parameters():
                    ema_shadow[name] = ema_decay * ema_shadow[name] + (1.0 - ema_decay) * param.data.float()

        if use_swa and step >= swa_start_step and (step - swa_start_step) % swa_interval == 0:
            swa_snapshots.append({n: p.data.clone() for n, p in model.named_parameters()})
            logger.info("SWA snapshot collected at step %d (total=%d)", step, len(swa_snapshots))

        if step % args.log_interval == 0:
            metrics = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
            logger.info("Step %d/%d | %s", step, args.train_steps, json.dumps(metrics, ensure_ascii=False))

        if val_buffer is not None and step % args.val_interval == 0:
            model.eval()
            val_loader = DataLoader(val_buffer, batch_size=args.batch_size, shuffle=True)
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    vb = {k: v.to(device) if torch.is_tensor(v) else v for k, v in vb.items()}
                    vl = model.compute_losses(vb)
                    val_losses.append(float(vl["loss"].item()))
                    if len(val_losses) >= 20:
                        break
            val_loss = float(sum(val_losses) / len(val_losses))
            model.train()
            logger.info("Step %d | val_loss=%.6f best_val=%.6f patience=%d/%d",
                        step, val_loss, best_val_loss, patience_counter, args.patience)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_ckpt()
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping at step %d (patience=%d)", step, args.patience)
                    break
        elif val_buffer is None:
            total_loss = float(losses["loss"].detach().cpu().item())
            recent_losses.append(total_loss)
            if len(recent_losses) > save_smooth_k:
                recent_losses.pop(0)
            smoothed = sum(recent_losses) / len(recent_losses)
            if smoothed < best_loss:
                best_loss = smoothed
                save_ckpt()

            _is_ckpt_step = (
                step in ckpt_steps_set if ckpt_steps_set is not None
                else (step >= ckpt_start_step and (step - ckpt_start_step) % ckpt_interval == 0)
            )
            if _is_ckpt_step:
                ckpt_best_loss[step] = smoothed
                save_periodic_ckpt(step, smoothed)
                logger.info(
                    "Periodic ckpt trigger: step=%d smoothed_loss=%.6f", step, smoothed
                )

        if step >= resume_start_step + args.train_steps:
            break

    logger.info("Training complete. Best val_loss: %.6f  Best train_loss: %.6f", best_val_loss, best_loss)

    if use_swa and len(swa_snapshots) >= 2:
        logger.info("SWA: averaging %d snapshots", len(swa_snapshots))
        swa_avg = {}
        for name in swa_snapshots[0]:
            swa_avg[name] = sum(s[name].float() for s in swa_snapshots) / len(swa_snapshots)
        orig = {n: p.data.clone() for n, p in model.named_parameters()}
        for name, param in model.named_parameters():
            param.data.copy_(swa_avg[name].to(param.dtype))
        swa_path = save_dir / f"{args.method}_swa.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "state_mean": aux_stats["state_mean"],
                "state_std": aux_stats["state_std"],
                "action_mean": aux_stats["action_mean"],
                "action_std": aux_stats["action_std"],
                "meta_mean": aux_stats["meta_mean"],
                "meta_std": aux_stats["meta_std"],
                "retrieval_queries": aux_stats["retrieval_queries"],
                "retrieval_contexts": aux_stats["retrieval_contexts"],
                "cpa_ratio_mean": aux_stats.get("cpa_ratio_mean", 0.0),
                "cpa_ratio_std": aux_stats.get("cpa_ratio_std", 1.0),
            },
            swa_path,
        )
        for name, param in model.named_parameters():
            param.data.copy_(orig[name])
        logger.info("SWA checkpoint saved: %s", swa_path)

    print(str(best_path))

def main():
    parser = argparse.ArgumentParser(description="Train new_msdt_method variants")
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/auctionnet/rl_training_data")
    parser.add_argument("--train_periods", type=str, default="7,8,9,10,11,12,13")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state_dim", type=int, default=16)
    parser.add_argument("--act_dim", type=int, default=1)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--max_ep_len", type=int, default=48)
    parser.add_argument("--scale", type=float, default=40.0)
    parser.add_argument("--target_return", type=float, default=50.0)
    parser.add_argument("--reward_key", type=str, default="reward")
    parser.add_argument("--coarse_idx", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--fine_idx", type=str, default="0,1,12,13,14,15")
    parser.add_argument("--local_window", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--enable_weighted_sampling", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=18000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--val_periods", type=str, default="", help="Periods for validation-based early stopping, e.g. '13'")
    parser.add_argument("--val_interval", type=int, default=2000, help="Steps between validation checks")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (# val checks)")
    parser.add_argument("--config_override_file", type=str, default="", help="Path to a JSON file with method config overrides")
    parser.add_argument("--config_override_json", type=str, default="", help="Inline JSON string with method config overrides")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--init_ckpt", type=str, default="", help="Resume: path to a checkpoint .pt to load model weights from")
    parser.add_argument("--start_step", type=int, default=0, help="Resume: starting step offset (e.g. 10000 to continue from a 10k ckpt)")
    parser.add_argument("--use_ema", type=int, default=0, help="Enable EMA weight averaging (0/1)")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--ema_start_step", type=int, default=6000, help="Step to start EMA (warmup period)")
    parser.add_argument("--use_swa", type=int, default=0, help="Enable SWA (0/1)")
    parser.add_argument("--swa_start_step", type=int, default=12000, help="Step to start collecting SWA snapshots")
    parser.add_argument("--swa_interval", type=int, default=1000, help="Steps between SWA snapshots")
    parser.add_argument("--ckpt_start_step", type=int, default=6000,
                        help="Step from which to start saving periodic checkpoints")
    parser.add_argument("--ckpt_interval", type=int, default=1000,
                        help="Interval (in steps) between periodic checkpoint saves")
    parser.add_argument("--ckpt_steps", type=str, default="",
                        help="Exact steps to save checkpoints, comma-separated (overrides start/interval). "
                             "E.g. '6000,8000,12000,16000'")
    args = parser.parse_args()
    if not args.save_dir:
        args.save_dir = str(Path(__file__).resolve().parents[1] / "runs" / args.method)
    train_model(args)

if __name__ == "__main__":
    main()
