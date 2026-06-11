#!/usr/bin/env bash
# Experiment 1: Temporal Inductive Bias Analysis
# Trains 4 single-bias variants on all 8 datasets x 4 horizons.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Traffic Electricity Exchange)
BIAS_TYPES=(global causal reverse_causal local alibi periodic_fixed relative trend seasonal)

for BIAS in "${BIAS_TYPES[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for PRED in "${PREDS[@]}"; do
      NAME="SingleBias_${BIAS}_${DATA}_sl336_pl${PRED}"
      echo ">>> ${NAME}"
      python run.py \
        --config configs/preliminary.yaml \
        --bias_type "${BIAS}" \
        --data "${DATA}" \
        --pred_len "${PRED}" \
        --exp_name "${NAME}" \
        --root_path "${ROOT}" \
        --resume
    done
  done
done

echo "Preliminary study done. Results in results/summary.csv"
