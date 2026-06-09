#!/usr/bin/env bash
# Experiment 4: Interpretability Analysis
# Requires trained BAMoE checkpoints from Experiment 2.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 ETTh2 Weather Traffic)

for DATA in "${DATASETS[@]}"; do
  for PRED in "${PREDS[@]}"; do
    NAME="BAMoE_K4_learned_sparse_${DATA}_sl336_pl${PRED}"
    echo ">>> Interpretability for ${NAME}"
    python run.py \
      --mode interpretability \
      --model BAMoE \
      --expert_types causal,local,periodic,global \
      --top_k 2 --routing learned_sparse \
      --seq_len 336 --pred_len "${PRED}" \
      --d_model 128 --n_heads 8 --n_layers 3 --d_ff 256 \
      --patch_len 16 --stride 8 \
      --data "${DATA}" \
      --root_path "${ROOT}" \
      --exp_name "${NAME}"
  done
done

echo "Interpretability analysis done. Figures in results/*/interpretability/"
