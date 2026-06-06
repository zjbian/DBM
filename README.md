# DBM-Bid

Reference implementation of **DBM-Bid: Dual-Branch Modulated Bidding for Offline
Constrained Auto-Bidding**. This is a clean extraction of the paper's method only —
none of the exploratory variants from the research repo are included.

The model is internally the `v2` backbone with the AWR-weighted training stack
(~227K parameters). A representative configuration is provided in `configs/dbm_bid.json`.

> **Note on reproducibility.** This code runs on top of the AuctionNet bidding
> environment, in which parts of the impression-level signals (e.g.\ value and noise
> terms) are stochastic, so the simulator itself has non-trivial run-to-run variance.
> The provided config is a reasonable reference, not a guaranteed recipe for any exact
> score; expect some spread across seeds and environment instances.

## Layout

```
DBM/
├── configs/
│   └── dbm_bid.json          # reference model + training configuration
├── model/                    # the architecture
│   ├── base_dt.py            # Decision Transformer backbone
│   ├── msdt_backbone.py      # multi-scale DT (legacy backbone, kept for the dispatch)
│   ├── msdt_v2.py            # DBM-Bid backbone: GranularityCalibrator + CausalTemporalFusionV2
│   ├── method_model.py       # ResearchMSDTModel wrapper (backbone + heads + losses)
│   └── __init__.py
├── train/
│   ├── train.py              # training entry point
│   ├── dataset.py            # offline trajectory dataset
│   ├── method_configs.py     # returns the DBM-Bid config for any method name
│   ├── common_utils.py
│   └── __init__.py
├── eval/
│   ├── run_evaluate.py       # evaluation entry point
│   └── bidding_train_env/    # offline auction simulator + bidding strategy
├── run_train.sh
├── run_eval.sh
└── README.md
```

## Architecture (paper §IV)

Two-stage dual-branch separation of *slow* budget signals from *fast* market signals:

1. **Stage 1 — GranularityCalibrator** (feature level): a shared MLP with
   distance-adaptive (cosine/L2) calibration and gated fusion over coarse vs. fine
   state index groups (`coarse_idx`, `fine_idx`).
2. **Stage 2 — CausalTemporalFusionV2** (temporal level): a causal Conv1d *slow*
   stream + a local-attention-with-decay-bias *fast* stream, fused by
   cross-granularity attention (fine = Q, coarse = K/V).
3. **ConstraintModulator**: a gated additive residual on the fused hidden, followed by
   a dynamic gate and a linear action head.

## Requirements

Python 3.9+ and PyTorch. No other special dependencies.

## Dataset

Experiments use the **AuctionNet** large-scale auto-bidding benchmark. Download the
data from the official release and place it locally, then point the scripts at it:

- AuctionNet: <https://github.com/alimama-tech/AuctionNet>

Two inputs are needed: (i) the offline **RL training data** (delivery periods 7–13),
passed via `--data_dir`, and (ii) the **test traffic** for the held-out periods 14–20,
passed via `--traffic_dir`. The `.sh` scripts ship with placeholder paths
(`./data/auctionnet/...`) — edit them to your download location. Note that the
AuctionNet simulator generates parts of each impression stochastically (e.g. value and
noise terms), so scores vary somewhat from run to run.

## Configuration

`configs/dbm_bid.json` holds a single, flat configuration consumed by both training and
evaluation. The main groups:

- **Architecture** — `state_dim`, `K` (context length), `n_head`, `coarse_idx`/`fine_idx`
  (the budget vs. market feature split), `local_window`, `scale`, `target_return`.
- **Optimization / sampling** — `learning_rate`, `loss_weight_mode`,
  `sampling_score_mode` (`awr`) and `sampling_awr_beta`, `enable_weighted_sampling`.
- **Constraint-aware training switches** (`use_*`) with their weights — selective
  imitation, stratified-prefix sampling, the budget-feasibility (C2) weight,
  prefix-feasibility weighting, score-RTG, conservative regularization, the CPA-progress
  reward, and hindsight truncation.

Only the switches the paper's model uses are present; leave them as given to reproduce
the reference model.

## Train

```bash
bash run_train.sh
```

Trains on periods 7–13 of the AuctionNet-Sparse RL data
(`--data_dir` inside the script) and writes to `./runs/dbm_bid/`:
- `dbm_bid.pt` — best checkpoint (self-contained: weights + config + normalization stats)
- `checkpoints/dbm_bid_ckptNNNNNN.pt` — periodic checkpoints
- `train_config.json`, `normalize_dict.pkl`

Reference training settings: `K=20`, `scale=40`, `train_steps=18000`,
`batch_size=128`, `lr=1e-4`, `seed=42`.

## Evaluate

```bash
bash run_eval.sh ./runs/dbm_bid      # arg = dir containing the .pt checkpoint
```

Evaluates on held-out periods 14–20 against the offline auction simulator
(`--traffic_dir`) and writes `benchmark_eval_results.json` into the load dir.
The checkpoint is self-contained, so eval needs only the `.pt` file.

## Notes

- `train/method_configs.py` returns `configs/dbm_bid.json` for any `--method` value,
  so `--method dbm_bid` (or the internal name `msdt_v2_awr_beta5_fixed`) both work.
- The `.sh` scripts use placeholder data paths (`./data/auctionnet/...`); set
  `DATA_DIR` / `TRAFFIC_DIR` (or the `--data_dir` / `--traffic_dir` flags) to your
  local AuctionNet-Sparse data location before running.
