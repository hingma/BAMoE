#!/usr/bin/env bash
# Experiment 2: Main BAMoE forecasting results
# BAMoE (heterogeneous experts) on all 8 datasets x 4 horizons.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Traffic Electricity Exchange)

COMMON="--model BAMoE
        --expert_types causal,local,periodic,global
        --top_k 2 --routing learned_sparse
        --seq_len 336 --d_model 384 --n_heads 8
        --n_layers 6 --d_ff 512 --dropout 0.1
        --patch_len 16 --stride 8
        --load_balance_coef 0.01
        --batch_size 128 --learning_rate 1e-4 --train_epochs 20
        --patience 5 --resume --root_path ${ROOT}"

for DATA in "${DATASETS[@]}"; do
  for PRED in "${PREDS[@]}"; do
    NAME="BAMoE_K4_learned_sparse_${DATA}_sl336_pl${PRED}"
    echo ">>> ${NAME}"
    python run.py ${COMMON} \
      --data "${DATA}" \
      --pred_len "${PRED}" \
      --exp_name "${NAME}"
  done
done

echo "Main results done. Results in results/summary.csv"
