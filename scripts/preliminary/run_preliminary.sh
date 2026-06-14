#!/usr/bin/env bash
# Experiment 1: Temporal Inductive Bias Analysis
# Trains 4 single-bias variants on all datasets x 4 horizons.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 ETTm2 Exchange Electricity ILI)
BIAS_TYPES=(global causal causal periodic)

# Channels: Electricity~321, Exchange~8, ETT*~7, ILI~7
# Effective batch = batch_size x n_channels; tune to stay within ~80 GB VRAM.
declare -A BATCH_SIZE
BATCH_SIZE["ETTh1"]=128
BATCH_SIZE["ETTm2"]=128
BATCH_SIZE["Exchange"]=128
BATCH_SIZE["Electricity"]=8
BATCH_SIZE["ILI"]=128

for BIAS in "${BIAS_TYPES[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    BS=${BATCH_SIZE[$DATA]:-16}
    for PRED in "${PREDS[@]}"; do
      NAME="SingleBias_${BIAS}_${DATA}_sl336_pl${PRED}"
      echo ">>> ${NAME}  (batch_size=${BS})"
      python run.py \
        --config configs/preliminary.yaml \
        --bias_type "${BIAS}" \
        --data "${DATA}" \
        --pred_len "${PRED}" \
        --exp_name "${NAME}" \
        --root_path "${ROOT}" \
        --batch_size "${BS}" \
        --resume
    done
  done
done

echo "Preliminary study done. Results in results/summary.csv"
