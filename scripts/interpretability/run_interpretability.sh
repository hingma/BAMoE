#!/usr/bin/env bash
# Experiment 4: Interpretability Analysis
# Requires trained BAMoE checkpoints from Experiment 2.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 ETTm2 Exchange Electricity ILI)

for DATA in "${DATASETS[@]}"; do
  for PRED in "${PREDS[@]}"; do
    NAME="BAMoE_K9_learned_sparse_${DATA}_sl336_pl${PRED}"
    echo ">>> Interpretability for ${NAME}"
    python run.py \
      --config configs/main.yaml \
      --mode interpretability \
      --data "${DATA}" \
      --pred_len "${PRED}" \
      --root_path "${ROOT}" \
      --exp_name "${NAME}"
  done
done

echo "Interpretability analysis done. Figures in results/*/interpretability/"
