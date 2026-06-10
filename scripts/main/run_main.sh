#!/usr/bin/env bash
# Experiment 2: Main BAMoE forecasting results
# BAMoE (heterogeneous experts) on all 8 datasets x 4 horizons.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Traffic Electricity Exchange)

for DATA in "${DATASETS[@]}"; do
  for PRED in "${PREDS[@]}"; do
    NAME="BAMoE_K4_learned_sparse_${DATA}_sl336_pl${PRED}"
    echo ">>> ${NAME}"
    python run.py \
      --config configs/main.yaml \
      --data "${DATA}" \
      --pred_len "${PRED}" \
      --exp_name "${NAME}" \
      --root_path "${ROOT}" \
      --resume
  done
done

echo "Main results done. Results in results/summary.csv"
