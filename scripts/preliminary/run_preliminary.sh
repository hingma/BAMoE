#!/usr/bin/env bash
# Experiment 1: Temporal Inductive Bias Analysis
# Trains 4 single-bias variants on all 8 datasets x 4 horizons.
# Results accumulate in results/summary.csv.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Traffic Electricity Exchange)
BIAS_TYPES=(global causal local periodic)

COMMON="--model SingleBias --seq_len 336 --d_model 384 --n_heads 8
        --n_layers 6 --d_ff 512 --dropout 0.1 --patch_len 16 --stride 8
        --batch_size 128 --learning_rate 1e-4 --train_epochs 20
        --patience 5 --resume --root_path ${ROOT}"

for BIAS in "${BIAS_TYPES[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for PRED in "${PREDS[@]}"; do
      NAME="SingleBias_${BIAS}_${DATA}_sl336_pl${PRED}"
      echo ">>> ${NAME}"
      python run.py ${COMMON} \
        --data "${DATA}" \
        --pred_len "${PRED}" \
        --bias_type "${BIAS}" \
        --exp_name "${NAME}"
    done
  done
done

echo "Preliminary study done. Results in results/summary.csv"
