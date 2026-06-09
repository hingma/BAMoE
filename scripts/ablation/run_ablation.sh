#!/usr/bin/env bash
# Experiment 3: Ablation Studies
# Run on ETTh1, Weather, Traffic at all 4 horizons.

set -e
ROOT=${DATA_ROOT:-"./data"}
PREDS=(96 192 336 720)
DATASETS=(ETTh1 Weather Traffic)

BASE="--seq_len 336 --d_model 128 --n_heads 8 --n_layers 3 --d_ff 256
      --dropout 0.1 --patch_len 16 --stride 8
      --batch_size 128 --learning_rate 1e-4 --train_epochs 20
      --patience 5 --resume --root_path ${ROOT}"

# ---------- 3a: Expert diversity ablation ----------
echo "=== 3a: Expert diversity ==="
declare -A HOMO_EXPERTS
HOMO_EXPERTS[homo_causal]="causal,causal,causal,causal"
HOMO_EXPERTS[homo_local]="local,local,local,local"
HOMO_EXPERTS[homo_periodic]="periodic,periodic,periodic,periodic"
HOMO_EXPERTS[homo_global]="global,global,global,global"
HOMO_EXPERTS[K1]="causal"     # single expert (no MoE)
HOMO_EXPERTS[hetero]="causal,local,periodic,global"  # full BAMoE

for TAG in "${!HOMO_EXPERTS[@]}"; do
  for DATA in "${DATASETS[@]}"; do
    for PRED in "${PREDS[@]}"; do
      NAME="ablation_diversity_${TAG}_${DATA}_pl${PRED}"
      echo ">>> ${NAME}"
      python run.py ${BASE} --model BAMoE \
        --expert_types "${HOMO_EXPERTS[$TAG]}" \
        --top_k 2 --routing learned_sparse \
        --data "${DATA}" --pred_len "${PRED}" --exp_name "${NAME}"
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
      python run.py ${BASE} --model BAMoE \
        --expert_types causal,local,periodic,global \
        --top_k "${K}" --routing "${ROUTING}" \
        --data "${DATA}" --pred_len "${PRED}" --exp_name "${NAME}"
    done
  done
done

# ---------- 3c: Number of experts sweep ----------
echo "=== 3c: Number of experts K ==="
# K=2: two complementary experts
# K=3,4,6,8: select subsets / repeats from the 4 bias types
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
      python run.py ${BASE} --model BAMoE \
        --expert_types "${K_EXPERTS[$TAG]}" \
        --top_k 2 --routing learned_sparse \
        --data "${DATA}" --pred_len "${PRED}" --exp_name "${NAME}"
    done
  done
done

echo "Ablation study done. Results in results/summary.csv"
