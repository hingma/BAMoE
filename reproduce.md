## Section 4:

### 4.1 Isolation Test

16 train+eval runs (4 experts × 4 DGPs), writes a 4×4 MSE CSV + heatmap. The matched diagonal (causal↔trend, alibi↔shock, periodic↔cyclical, global↔longrange) should be the column minimum after full-length training.

### 4.2 Regime Shift

4 models (3 isolated + BAMoE) are all trained on the same regime-switching series, then tested on three splits: `regime-switching overall, pure-cyclical, and pure-shock`.   
The bar chart shows isolated experts thriving in their domain and collapsing in the wrong one, while BAMoE stays as σ increases.

### Slope Sensitivity

8 combinations (6 γ values × 3 noise levels) train ALiBi models on the shock DGP, producing an MSE-vs-γ curve per noise level. The expected shape is U-curves with optimal γ* shifting toward higher values as σ increases.

run the full experiment: 

```bash scripts/synthetic/run_exp4.sh ```


## Section 5:

### Train the baseline models from scratch:

1. Statistical Family: Auto-ARIMA, ETS, Theta Method.
2. Single-Bias DL Family: PatchTST, iTransformer, TimeMixer.
3. Implicit MoE Family: TimeMoE.

```

# Install prerequisites (once)
pip install statsforecast neuralforecast

# Full sweep — all 7 models × 8 datasets × 4 horizons
bash scripts/main_baseline/run_all_baselines.sh

# Single family
FAMILY=dl bash scripts/main_baseline/run_all_baselines.sh

# Subset (e.g. while iterating on one dataset)
<!-- DATASETS="ETTh1" PREDS="96 192" bash scripts/main_baseline/run_all_baselines.sh -->

# After the sweep
python scripts/compute_significance.py --pred_len 96

```

*Notes:*
- --n_stat_windows 200 (default): stat models fit once per test window — expensive for large N. 200 evenly-spaced windows across the test period is standard for ablation-level comparison; use None for full evaluation.
- TimeMoE's model.generate(inputs=ctx_t, max_new_tokens=pred_len) follows the model card API
— verify against https://huggingface.co/Maple728/TimeMoE-50M if the interface has changed.
- Data uses the exact same scaler and splits as BAMoE (reuses data_loader.py directly),
ensuring fair comparison.

### MSE/MAE Comparison

### Significance

1. Run models and save predictions (after training)
  Load the checkpoints, do testing and collect the data.

```
python scripts/collect_preds.py [--config configs/main.yaml] [--train]
```

- --train: Train any experiment whose checkpoint is missing.

1. Compute significance tests

```
python scripts/compute_significance.py [--loss mse] [--alpha 0.05] [--n_boot
1000]
```

## Section 6: Interpretability

### Routing gate trajectories

Loads the raw continuous test array, runs CUSUM peak-to-peak across 300 strided windows to find the window with the largest level shift, then does a single forward pass to get per-patch routing probabilities (averaged over channels).

### Regime embedding

Iterates over the full test loader, accumulates `(B, n_experts)` mean routing vectors and per-sample input 