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
HOMO_EXPERTS[homo_causal]="causal,causal,causal,causal,causal,causal,causal,causal,causal"
HOMO_EXPERTS[homo_local]="local,local,local,local,local,local,local,local,local"
HOMO_EXPERTS[homo_periodic]="periodic_fixed,periodic_fixed,periodic_fixed,periodic_fixed,periodic_fixed,periodic_fixed,periodic_fixed,periodic_fixed,periodic_fixed"
HOMO_EXPERTS[homo_global]="global,global,global,global,global,global,global,global,global"
HOMO_EXPERTS[homo_reverse_causal]="reverse_causal,reverse_causal,reverse_causal,reverse_causal,reverse_causal,reverse_causal,reverse_causal,reverse_causal,reverse_causal"
HOMO_EXPERTS[homo_alibi]="alibi,alibi,alibi,alibi,alibi,alibi,alibi,alibi,alibi"
HOMO_EXPERTS[homo_relative]="relative,relative,relative,relative,relative,relative,relative,relative,relative"
HOMO_EXPERTS[homo_trend]="trend,trend,trend,trend,trend,trend,trend,trend,trend"
HOMO_EXPERTS[homo_seasonal]="seasonal,seasonal,seasonal,seasonal,seasonal,seasonal,seasonal,seasonal,seasonal"
HOMO_EXPERTS[K1]="causal"
HOMO_EXPERTS[hetero]="causal,reverse_causal,local,alibi,periodic_fixed,relative,trend,seasonal,global"

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
K_EXPERTS[K3]="causal,local,periodic_fixed"
K_EXPERTS[K4]="causal,local,periodic_fixed,global"
K_EXPERTS[K6]="causal,reverse_causal,local,alibi,periodic_fixed,global"
K_EXPERTS[K8]="causal,reverse_causal,local,alibi,periodic_fixed,relative,trend,seasonal"
K_EXPERTS[K9]="causal,reverse_causal,local,alibi,periodic_fixed,relative,trend,seasonal,global"

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
