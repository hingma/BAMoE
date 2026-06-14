#!/usr/bin/env bash
# Experiment 4: Synthetic Validation (Controlled Isolation, Regime Shift,
#                                      ALiBi Sensitivity)
#
# All three sub-experiments write to results/synthetic/:
#   exp41_isolation_matrix.csv    exp41_isolation_heatmap.png
#   exp42_regime_shift.csv        exp42_regime_comparison.png
#   exp43_slope_sensitivity.csv   exp43_slope_sensitivity.png
#
# Runtime estimate (MPS / single GPU):
#   Exp 4.1 : ~15 min  (16 train runs × 30 epochs)
#   Exp 4.2 : ~5  min  (4  train runs × 30 epochs)
#   Exp 4.3 : ~25 min  (18 train runs × 30 epochs)
#
# To run a single sub-experiment:
#   bash scripts/synthetic/run_exp4.sh 41
#   bash scripts/synthetic/run_exp4.sh 42
#   bash scripts/synthetic/run_exp4.sh 43
set -e

PYTHON=${PYTHON:-"/opt/anaconda3/bin/python"}
EXP=${1:-all}   # pass '41', '42', or '43' as first arg, or default to 'all'

echo ">>> BAMoE Experiment 4 (exp=${EXP})"

$PYTHON run_synthetic.py \
  --exp          "${EXP}" \
  --T_synth      5000     \
  --seq_len      336      \
  --pred_len     96       \
  --d_model      64       \
  --n_heads      4        \
  --n_layers     2        \
  --d_ff         128      \
  --dropout      0.1      \
  --patch_len    16       \
  --stride       8        \
  --periodic_period 24    \
  --train_epochs 30       \
  --patience     5        \
  --batch_size   64       \
  --learning_rate 1e-3    \
  --slope_multipliers 0.1,0.5,1.0,2.0,5.0,10.0 \
  --noise_levels 0.5,1.0,2.0                    \
  --results      ./results/

echo ">>> Done. Results in results/synthetic/"
