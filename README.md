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
│   ├── dbm_backbone.py      # multi-scale DT (legacy backbone, kept for the dispatch)
│   ├── dbm_v2.py            # DBM-Bid backbone: GranularityCalibrator + CausalTemporalFusionV2
│   ├── method_model.py       # ResearchDBMModel wrapper (backbone + heads + losses)
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

## Results

Main comparison on **AuctionNet-Sparse** (held-out periods 14–20), CPA-penalized
score across five budget levels (% of the original budget). Higher is better; **bold**
marks the best per column.

| Method | 50% | 75% | 100% | 125% | 150% |
|---|:--:|:--:|:--:|:--:|:--:|
| DiffBid † | 9.9 | 15.4 | 19.5 | 25.3 | 30.8 |
| USCB † | 11.5 | 14.9 | 17.5 | 26.7 | 31.3 |
| CQL | 12.8 | 16.7 | 22.2 | 28.6 | 35.8 |
| CDT † | 11.2 | 18.0 | 31.2 | 31.7 | 39.1 |
| DT | 14.8 | 22.9 | 29.6 | 34.3 | 44.5 |
| BCQ | 17.7 | 24.6 | 31.1 | 34.2 | 37.9 |
| IQL | 16.5 | 22.1 | 30.0 | 37.1 | 43.1 |
| GAS † | 18.4 | 27.5 | 36.1 | 40.0 | 46.5 |
| EBaReT †§ | – | – | 36.5 | – | – |
| GAVE † | 19.6 | 28.3 | 37.2 | 42.7 | 47.4 |
| GRAD † | **20.0** | **28.5** | **37.4** | **43.2** | 47.5 |
| **DBM-Bid (ours)** | **20.0** | 28.1 | 36.82 | 42.6 | **48.5** |

DBM-Bid matches the strongest generative bidder at the tight end and leads at the loose
end, while surpassing every offline-RL/DT baseline across budgets. The offline-RL and DT
baselines (CQL, DT, BCQ, IQL) are re-run by us under this identical protocol; methods
marked **†** (DiffBid, USCB, CDT, GAS, GAVE, GRAD) are taken from their original papers;
**§** EBaReT is concurrent and reports only the standard (100%) budget. With ~227K
parameters DBM-Bid is also smaller and faster than a standard encoder carrying the same
training stack (227K vs. 317K params, 26.0 vs. 31.5 ms/batch).

## Hyperparameters

Key settings used to produce the reference model (full set in `configs/dbm_bid.json`):

| Group | Setting |
|---|---|
| Backbone | hidden size 64, 4 attention heads, 3 Transformer layers, context length `K=20` (~227K params) |
| State | `state_dim=16`; coarse/budget indices `0–11`, fine/market indices `{0,1,12–15}` (indices 0,1 = time-left, budget-left) |
| Optimizer | AdamW, lr `1e-4`, weight decay `1e-4` |
| Schedule | `18k` steps, batch size `128`, seed `42` |
| Sampling | advantage-weighted (AWR) sample weighting, `β=5`; score-based RTG |
| Return | `scale=40`; inference target-return searched over `{36, 40, 44, 48}` |
| Data split | train periods `7–13`, evaluate periods `14–20` |
| Hardware | single NVIDIA RTX 4090, ~90 min/run |

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
  so `--method dbm_bid` (or the internal name `dbm_v2_awr_beta5_fixed`) both work.
- The `.sh` scripts use placeholder data paths (`./data/auctionnet/...`); set
  `DATA_DIR` / `TRAFFIC_DIR` (or the `--data_dir` / `--traffic_dir` flags) to your
  local AuctionNet-Sparse data location before running.
