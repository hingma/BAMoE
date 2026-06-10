#!/usr/bin/env bash
# Experiment 3: Ablation Studies
# Run on ETTh1, Weather, Traffic at all 4 horizons.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 Weather Traffic)

# ---------- 3a: Expert diversity ablation ----------
echo "=== 3a: Expert diversity ==="
declare -A HOMO_EXPERTS
HOMO_EXPERTS[homo_causal]="causal,causal,causal,causal"
HOMO_EXPERTS[homo_local]="local,local,local,local"
HOMO_EXPERTS[homo_periodic]="periodic,periodic,periodic,periodic"
HOMO_EXPERTS[homo_global]="global,global,global,global"
HOMO_EXPERTS[K1]="causal"
HOMO_EXPERTS[hetero]="causal,local,periodic,global"

for TAG in "${!HOMO_EXPERTS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for PRED in "${PREDS[@]}"; do
      NAME="ablation_diversity_${TAG}_${DATA}_pl${PRED}"
      echo ">>> ${NAME}"
      python run.py \
        --config configs/ablation.yaml \
        --expert_types "${HOMO_EXPERTS[$TAG]}" \
        --data "${DATA}" --pred_len "${PRED}" \
        --exp_name "${NAME}" \
        --root_path "${ROOT}" \
        --resume
    done
  done
done

# ---------- 3b: Routing mechanism ablation ----------
echo "=== 3b: Routing mechanism ==="
ROUTINGS=(uniform random top1 learned_sparse dense)

for ROUTING in "${ROUTINGS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for PRED in "${PREDS[@]}"; do
      K=2
      [[ "$ROUTING" == "top1" ]] && K=1
      NAME="ablation_routing_${ROUTING}_${DATA}_pl${PRED}"
      echo ">>> ${NAME}"
      python run.py \
        --config configs/ablation.yaml \
        --routing "${ROUTING}" \
        --top_k "${K}" \
        --data "${DATA}" --pred_len "${PRED}" \
        --exp_name "${NAME}" \
        --root_path "${ROOT}" \
        --resume
    done
  done
done

# ---------- 3c: Number of experts sweep ----------
echo "=== 3c: Number of experts K ==="
declare -A K_EXPERTS
K_EXPERTS[K2]="causal,global"
K_EXPERTS[K3]="causal,local,periodic"
K_EXPERTS[K4]="causal,local,periodic,global"
K_EXPERTS[K6]="causal,local,periodic,global,causal,local"
K_EXPERTS[K8]="causal,local,periodic,global,causal,local,periodic,global"

for TAG in "${!K_EXPERTS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for PRED in "${PREDS[@]}"; do
      NAME="ablation_Ksweep_${TAG}_${DATA}_pl${PRED}"
      echo ">>> ${NAME}"
      python run.py \
        --config configs/ablation.yaml \
        --expert_types "${K_EXPERTS[$TAG]}" \
        --data "${DATA}" --pred_len "${PRED}" \
        --exp_name "${NAME}" \
        --root_path "${ROOT}" \
        --resume
    done
  done
done

echo "Ablation study done. Results in results/summary.csv"
