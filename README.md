# BAMoE: Bias-Aware Mixture of Experts for Long-Term Time Series Forecasting

BAMoE is a patch-based Transformer that routes each token through a mixture of heterogeneous attention experts, each encoding a different temporal inductive bias (global, causal, local, periodic). A lightweight learned router selects the top-*k* experts per token and combines their outputs with load-balanced sparse weights.

## Architecture

```
Input (B, L, C)
    │
    ▼
Patch Embedding  (channel-independent, shared across C)
    │
    ▼  ×n_layers
┌─────────────────────────────────────┐
│  LayerNorm → BiasAwareMoE Attention │
│  LayerNorm → FFN (GELU)             │
└─────────────────────────────────────┘
    │
    ▼
LayerNorm → Linear head → Prediction (B, H, C)
```

**Expert types**

| Type       | Mechanism                                      |
|------------|------------------------------------------------|
| `global`   | Standard full self-attention                   |
| `causal`   | Causal (autoregressive) attention mask         |
| `local`    | Sliding-window attention (configurable window) |
| `periodic` | Period-aware attention bias                    |

**Routing modes:** `learned_sparse` (default) · `top1` · `dense` · `uniform` · `random`

**Auxiliary loss:** Switch-Transformer load-balancing loss weighted by `--load_balance_coef`.

---

## Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn scipy tqdm wandb pyyaml
```

Python ≥ 3.10, PyTorch ≥ 2.0. CUDA is recommended; MPS (Apple Silicon) and CPU are also supported.

---

## Data

Download all eight LTSF benchmark datasets (~260 MB total):

```bash
bash scripts/download_data.sh ./data
```

Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Electricity, Exchange Rate.
Source: [THUML Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library) on HuggingFace.

---

## Configuration

Hyperparameters live in YAML files under `configs/`. CLI flags always override YAML values.

| File                       | Used for                                    |
|----------------------------|---------------------------------------------|
| `configs/main.yaml`        | Experiment 2 — main BAMoE results           |
| `configs/preliminary.yaml` | Experiment 1 — single-bias study            |
| `configs/ablation.yaml`    | Experiment 3 — ablation (smaller model)     |
| `configs/sweep_main.yaml`  | Weights & Biases hyperparameter sweep       |

To override a single value without editing the YAML:

```bash
python run.py --config configs/main.yaml --data ETTh1 --pred_len 96 --dropout 0.2
```

---

## Running Experiments

### Single run

```bash
# Experiment 2: main BAMoE
python run.py --config configs/main.yaml --data Weather --pred_len 336

# Experiment 1: single-bias baseline
python run.py --config configs/preliminary.yaml --model SingleBias --bias_type causal \
    --data ETTh1 --pred_len 96

# Train only / test only
python run.py --config configs/main.yaml --data ETTh1 --pred_len 96 --mode train
python run.py --config configs/main.yaml --data ETTh1 --pred_len 96 --mode test

# Resume from checkpoint (skip training if already done)
python run.py --config configs/main.yaml --data ETTh1 --pred_len 96 --resume
```

### Full experiment sweeps

```bash
bash scripts/preliminary/run_preliminary.sh          # Exp 1: 4 biases × 8 datasets × 4 horizons
bash scripts/main/run_main.sh                        # Exp 2: BAMoE × 8 datasets × 4 horizons
bash scripts/ablation/run_ablation.sh                # Exp 3: diversity / routing / K ablations
bash scripts/interpretability/run_interpretability.sh  # Exp 4: routing dynamics (needs Exp 2 checkpoints)
```

Set `DATA_ROOT` to override the data directory:

```bash
DATA_ROOT=/my/data bash scripts/main/run_main.sh
```

Results accumulate in `results/summary.csv`.

---

## Weights & Biases

Add `--wandb` to any run to enable logging. Each run creates one W&B run with all hyperparameters logged as config and per-epoch `train_loss`, `val_loss`, `lr`, `epoch_time_s`, plus final `test_mse`, `test_mae`, `test_crps`, `test_mase`.

```bash
python run.py --config configs/main.yaml --data ETTh1 --pred_len 96 \
    --wandb --wandb_project BAMoE --wandb_entity <your-entity>
```

### Hyperparameter search

```bash
# 1. Create the sweep (prints SWEEP_ID)
wandb sweep configs/sweep_main.yaml

# 2. Launch one or more agents
wandb agent <entity>/BAMoE/<SWEEP_ID>
```

The sweep uses Bayesian optimisation over `learning_rate`, `dropout`, `d_model`, `d_ff`, `n_layers`, `load_balance_coef`, and `top_k`, targeting `val_loss`. Edit `configs/sweep_main.yaml` to change the search space, dataset, or horizon.

---

## Google Colab

Open `BAMoE_Colab.ipynb` for end-to-end cloud training on a free T4 GPU. The notebook mirrors the local scripts exactly — each experiment is a separate cell so you can run steps individually.

**Setup** (Section 0 of the notebook):

```python
CONFIG_FILE   = "configs/main.yaml"   # which YAML to load
CFG_OVERRIDES = dict()                # per-notebook overrides

USE_WANDB     = True
WANDB_PROJECT = "BAMoE"
WANDB_ENTITY  = "your-username"
```

After cloning the repo, the helpers cell imports `build_args` directly from `run.py`, so the notebook and local scripts share a single argument schema.

---

## Programmatic API

`build_args` lets you construct a fully resolved `args` object from Python — useful for notebooks, tests, or custom training loops:

```python
from run import build_args
from exp.exp_forecast import ExpForecast

args = build_args(
    'configs/main.yaml',
    data='ETTh1',
    pred_len=96,
    dropout=0.2,       # overrides the YAML value
)

exp = ExpForecast(args)
exp.train()
mse, mae, crps, mase = exp.test()
```

---

## Project Structure

```
BAMoE/
├── run.py                          # entry point (_make_parser, build_args, get_args)
├── configs/
│   ├── main.yaml                   # Exp 2 hyperparameters
│   ├── preliminary.yaml            # Exp 1 hyperparameters
│   ├── ablation.yaml               # Exp 3 hyperparameters
│   └── sweep_main.yaml             # W&B sweep definition
├── exp/
│   ├── exp_forecast.py             # train / test loop (wandb logging)
│   └── exp_interpretability.py     # routing visualisation
├── models/
│   ├── BAMoE.py                    # main model
│   ├── SingleBias.py               # single-expert baseline
│   └── layers/
│       ├── attention.py            # global / causal / local / periodic attention
│       ├── embed.py                # patch embedding
│       └── moe.py                  # BiasAwareMoE routing layer
├── data_provider/
│   ├── data_factory.py
│   └── data_loader.py
├── utils/
│   ├── metrics.py                  # MSE, MAE, CRPS, MASE
│   ├── tools.py                    # EarlyStopping, LR schedule
│   └── visualize.py
├── scripts/
│   ├── download_data.sh
│   ├── main/run_main.sh
│   ├── preliminary/run_preliminary.sh
│   ├── ablation/run_ablation.sh
│   └── interpretability/run_interpretability.sh
├── BAMoE_Colab.ipynb
├── results/summary.csv             # aggregated test metrics (auto-generated)
└── checkpoints/                    # saved model weights (auto-generated)
```

---

## Key Hyperparameters

| Argument               | Default                        | Description                          |
|------------------------|--------------------------------|--------------------------------------|
| `--expert_types`       | `causal,local,periodic,global` | Comma-separated expert bias types    |
| `--top_k`              | `2`                            | Active experts per token             |
| `--routing`            | `learned_sparse`               | Router mode                          |
| `--load_balance_coef`  | `0.01`                         | Weight for auxiliary balance loss    |
| `--seq_len`            | `336`                          | Look-back window                     |
| `--pred_len`           | `96`                           | Forecast horizon                     |
| `--patch_len`          | `16`                           | Patch size for embedding             |
| `--stride`             | `8`                            | Patch stride                         |
| `--d_model`            | `384`*                         | Model dimension                      |
| `--n_layers`           | `6`*                           | Transformer depth                    |
| `--d_ff`               | `512`*                         | FFN hidden dimension                 |

\* Values from `configs/main.yaml`. `configs/ablation.yaml` uses a smaller model (d_model=128, n_layers=3, d_ff=256).

---

## Metrics

- **MSE / MAE** — standard point-forecast errors
- **CRPS** — Continuous Ranked Probability Score (energy-score formulation); reduces to MAE for point forecasts
- **MASE** — Mean Absolute Scaled Error, normalised by the training-set naïve forecast scale
