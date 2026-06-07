import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

np.random.seed(42)

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EVAL_DIR)
sys.path.insert(0, os.path.dirname(_EVAL_DIR))

from bidding_train_env.offline_eval.offline_env import OfflineEnv
from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.strategy import PlayerBiddingStrategy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def getScore_neurips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward

def evaluate_single_period(agent, period_file: str, device: str = "cuda", budget_scale: float = 1.0, save_actions: bool = False):
    data_loader = TestDataLoader(file_path=period_file)
    env = OfflineEnv()
    keys, test_dict = data_loader.keys, data_loader.test_dict
    advertiser_results = []

    for key in keys:
        df = test_dict[key]
        budget = float(df["budget"].iloc[0]) * float(budget_scale)
        cpa_constraint = float(df["CPAConstraint"].iloc[0])
        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key)

        rewards = np.zeros(num_timeStepIndex)
        history = {
            "historyBids": [],
            "historyAuctionResult": [],
            "historyImpressionResult": [],
            "historyLeastWinningCost": [],
            "historyPValueInfo": [],
        }

        agent.budget = budget
        agent.cpa = cpa_constraint
        agent.reset()

        for timeStep_index in range(num_timeStepIndex):
            pValue = pValues[timeStep_index]
            pValueSigma = pValueSigmas[timeStep_index]
            leastWinningCost = leastWinningCosts[timeStep_index]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0], dtype=np.float32)
            else:
                bid = agent.bidding(
                    timeStep_index,
                    pValue,
                    pValueSigma,
                    history["historyPValueInfo"],
                    history["historyBids"],
                    history["historyAuctionResult"],
                    history["historyImpressionResult"],
                    history["historyLeastWinningCost"],
                    device=device,
                )
                if isinstance(bid, tuple):
                    bid = bid[0]

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid, leastWinningCost)
            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                if len(pv_index) == 0:
                    break
                dropped_pv_index = np.random.choice(
                    pv_index,
                    int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                    replace=False,
                )
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid, leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[timeStep_index] = np.sum(tick_conversion)
            history["historyPValueInfo"].append(np.array([(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            history["historyAuctionResult"].append(np.array([(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])]))
            history["historyImpressionResult"].append(np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])]))

        all_reward = float(np.sum(rewards))
        all_cost = float(agent.budget - agent.remaining_budget)
        cpa_real = all_cost / (all_reward + 1e-10)
        score = getScore_neurips(all_reward, cpa_real, cpa_constraint)
        rec = {
            "key": tuple(int(x) for x in key),
            "reward": all_reward,
            "cost": all_cost,
            "cpa_real": float(cpa_real),
            "cpa_constraint": float(cpa_constraint),
            "score": float(score),
        }
        if save_actions:
            rec["actions"] = [
                float(np.mean(b)) if len(b) > 0 else 0.0
                for b in history["historyBids"]
            ]
        advertiser_results.append(rec)

    avg_reward = float(np.mean([x["reward"] for x in advertiser_results])) if advertiser_results else 0.0
    avg_cost = float(np.mean([x["cost"] for x in advertiser_results])) if advertiser_results else 0.0
    avg_cpa = avg_cost / (avg_reward + 1e-10)
    avg_score = float(np.mean([x["score"] for x in advertiser_results])) if advertiser_results else 0.0
    return {
        "period_file": period_file,
        "avg_reward": avg_reward,
        "avg_cost": avg_cost,
        "avg_cpa": float(avg_cpa),
        "avg_score": avg_score,
        "advertiser_count": len(advertiser_results),
        "advertiser_results": advertiser_results,
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark-native evaluation for DBM + sampling")
    parser.add_argument("--traffic_dir", type=str, default="./data/auctionnet/test_data")
    parser.add_argument("--periods", type=str, default="14,15,16,17,18,19,20")
    parser.add_argument("--device", type=str, default="cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")
    parser.add_argument("--target_return_override", type=float, default=40.0)
    parser.add_argument("--load_dir", type=str, default="")
    parser.add_argument("--budget_scale", type=float, default=1.0)
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--save_actions", action="store_true", default=False,
                        help="Save per-step mean bid in advertiser_results for S4 analysis")
    parser.add_argument("--use_ca_tr", action="store_true", default=False,
                        help="Enable CA-TR: per-CPA-group adaptive target_return")
    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument("--np_seed", type=int, default=42,
                        help="Re-seed numpy in main (overrides import-time seed). Drives the random pValue-drop on budget overflow.")
    args = parser.parse_args()

    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)
    np.random.seed(args.np_seed)

    periods = [int(x) for x in args.periods.split(",") if x.strip()]
    load_dir = args.load_dir if args.load_dir else None
    agent = PlayerBiddingStrategy(load_dir=load_dir, target_return_override=args.target_return_override)
    if args.use_ca_tr:
        agent.enable_ca_tr(True)
        logger.info("CA-TR enabled: per-CPA-group adaptive target_return")
    agent.set_device(torch.device(args.device) if "torch" in globals() else args.device)
    logger.info(
        "evaluation target_return_override=%s budget_scale=%s save_actions=%s",
        args.target_return_override,
        args.budget_scale,
        args.save_actions,
    )

    all_period_results = []
    for period in periods:
        period_file = str(Path(args.traffic_dir) / f"period-{period}.csv")
        result = evaluate_single_period(
            agent,
            period_file=period_file,
            device=args.device,
            budget_scale=args.budget_scale,
            save_actions=args.save_actions,
        )
        logger.info(
            "period-%s | avg_score=%.6f | avg_reward=%.6f | avg_cost=%.6f | avg_cpa=%.6f",
            period,
            result["avg_score"],
            result["avg_reward"],
            result["avg_cost"],
            result["avg_cpa"],
        )
        all_period_results.append(result)

    overall_score = float(np.mean([x["avg_score"] for x in all_period_results])) if all_period_results else 0.0
    logger.info("overall_avg_score=%.6f", overall_score)
    out = {
        "overall_avg_score": overall_score,
        "budget_scale": float(args.budget_scale),
        "period_results": all_period_results,
    }
    if args.output_name:
        if load_dir:
            out_path = Path(load_dir) / args.output_name
        else:
            out_path = Path(args.output_name)
    elif load_dir:
        out_path = Path(load_dir) / "benchmark_eval_results.json"
    else:
        out_path = Path(__file__).resolve().parents[1] / "saved_model" / "DBM_sampling" / "benchmark_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(__import__("json").dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("saved benchmark results to %s", out_path)
    print(overall_score)

if __name__ == "__main__":
    import torch
    main()
