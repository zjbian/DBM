#!/usr/bin/env bash
# Evaluate a trained DBM-Bid checkpoint on the held-out eval periods (14-20).
set -e
cd "$(dirname "$0")"

LOAD_DIR="${1:-./runs/dbm_bid}"   # dir containing the .pt checkpoint, train_config.json, normalize_dict.pkl
# Path to the AuctionNet-Sparse test traffic (edit to your local location).
TRAFFIC_DIR="./data/auctionnet/test_data"

python eval/run_evaluate.py \
    --load_dir "$LOAD_DIR" \
    --traffic_dir "$TRAFFIC_DIR" \
    --periods "14,15,16,17,18,19,20" \
    --target_return_override 40 \
    --budget_scale 1.0 \
    --device "cuda:0"
