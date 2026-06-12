#!/usr/bin/env bash
# Experiment 1: Temporal Inductive Bias Analysis
# Trains 4 single-bias variants on all 8 datasets x 4 horizons.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192)
# DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Traffic Electricity Exchange)
DATASETS=(Electricity)
BIAS_TYPES=(global causal reverse_causal local alibi periodic_fixed relative trend seasonal)

# Channels: Traffic~862, Electricity~321, Weather~21, Exchange~8, ETT*~7
# Effective batch = batch_size × n_channels; tune to stay within ~80 GB VRAM.
declare -A BATCH_SIZE
BATCH_SIZE["ETTh1"]=128
BATCH_SIZE["ETTh2"]=128
BATCH_SIZE["ETTm1"]=128
BATCH_SIZE["ETTm2"]=128
BATCH_SIZE["Exchange"]=128
BATCH_SIZE["Weather"]=32
BATCH_SIZE["Electricity"]=8
BATCH_SIZE["Traffic"]=4

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
