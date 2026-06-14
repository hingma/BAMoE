#!/usr/bin/env bash
# Experiment 2: Main BAMoE forecasting results
# BAMoE (heterogeneous experts) on all 8 datasets x 4 horizons.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 ETTm2 Exchange Electricity ILI)

# Channels: Electricity~321, Exchange~8, ETT*~7, ILI~7
# Effective batch = batch_size × n_channels; tune to stay within ~80 GB VRAM.
declare -A BATCH_SIZE
BATCH_SIZE["ETTh1"]=64
BATCH_SIZE["ETTm2"]=64
BATCH_SIZE["Exchange"]=64
BATCH_SIZE["Electricity"]=8
BATCH_SIZE["ILI"]=64

for DATA in "${DATASETS[@]}"; do
  BS=${BATCH_SIZE[$DATA]:-4}
  for PRED in "${PREDS[@]}"; do
    NAME="BAMoE_K9_learned_sparse_${DATA}_sl336_pl${PRED}"
    echo ">>> ${NAME}  (batch_size=${BS})"
    python run.py \
      --config configs/main.yaml \
      --data "${DATA}" \
      --pred_len "${PRED}" \
      --exp_name "${NAME}" \
      --root_path "${ROOT}" \
      --batch_size "${BS}" \
      --resume  
  done
done

echo "Main results done. Results in results/summary.csv"
