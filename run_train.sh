#!/usr/bin/env bash
# Train DBM-Bid (the paper's model) with the reference configuration.
set -e
cd "$(dirname "$0")"

# Path to the AuctionNet-Sparse RL training data (edit to your local location).
DATA_DIR="./data/auctionnet/rl_training_data"
SAVE_DIR="./runs/dbm_bid"

python train/train.py \
    --method dbm_bid \
    --data_dir "$DATA_DIR" \
    --train_periods "7,8,9,10,11,12,13" \
    --save_dir "$SAVE_DIR" \
    --K 20 \
    --scale 40 \
    --train_steps 18000 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --seed 42 \
    --device "cuda:0"

echo "Done. Checkpoints + train_config.json + normalize_dict.pkl in $SAVE_DIR"
