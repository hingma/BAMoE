#!/usr/bin/env bash
# =============================================================================
# scripts/run_all_baselines.sh
#
# Train / evaluate ALL baseline model families across every dataset and
# prediction horizon used in the BAMoE paper.  Mirrors the structure of
# scripts/main/run_main.sh so results land in the same results/ tree.
#
# Families
# --------
#   stat    : AutoARIMA, AutoETS, AutoTheta      (statsforecast)
#   dl      : PatchTST, iTransformer, TimeMixer  (neuralforecast)
#   timemoe : TimeMoE-50M                        (HuggingFace, zero-shot)
#
# Usage
# -----
#   # All families, all datasets:
#   bash scripts/run_all_baselines.sh
#
#   # Single family (override via env var):
#   FAMILY=dl bash scripts/run_all_baselines.sh
#
#   # Subset of datasets / horizons:
#   DATASETS="ETTh1 Weather" PREDS="96 192" bash scripts/run_all_baselines.sh
#
#   # Dry-run (print commands, don't execute):
#   DRY_RUN=1 bash scripts/run_all_baselines.sh
#
# Environment variables (all optional)
# -------------------------------------
#   FAMILY          Model family: stat | dl | timemoe | all   [default: all]
#   DATASETS        Space-separated dataset names             [default: all 8]
#   PREDS           Space-separated prediction horizons       [default: 96 192 336 720]
#   DATA_ROOT       Path to data directory                    [default: ./data]
#   RESULTS_DIR     Path to results directory                 [default: ./results]
#   LOG_DIR         Directory for per-run log files           [default: ./logs/baselines]
#   GPU             GPU index for DL / TimeMoE                [default: 0]
#   MAX_STEPS       Max training steps for DL models          [default: 1000]
#   SKIP_EXISTING   1 = skip runs whose losses.csv exists     [default: 1]
#   DRY_RUN         1 = print commands without running        [default: 0]
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration  (all overridable via env vars)
# ---------------------------------------------------------------------------

FAMILY="${FAMILY:-all}"
PREDS="${PREDS:-96 192 336 720}"
DATASETS="${DATASETS:-ETTh1 ETTm2 Exchange Electricity ILI}"
DATA_ROOT="${DATA_ROOT:-./data}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
LOG_DIR="${LOG_DIR:-./logs/baselines}"
GPU="${GPU:-0}"
MAX_STEPS="${MAX_STEPS:-1000}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Per-dataset lookup helpers  (bash 3 compatible — no associative arrays)
# ---------------------------------------------------------------------------

# DL batch size: large channel counts need smaller batches to fit in VRAM.
dl_batch() {
    case "$1" in
        ETTh1|ETTm2|Exchange|ILI) echo 64 ;;
        Electricity)              echo 8  ;;
        *)                        echo 32 ;;
    esac
}

# n_stat_windows: stat models refit per window — cap for tractability.
# Rule of thumb: ≤200 for narrow datasets (<20 channels), ≤100 for wide ones.
stat_windows() {
    case "$1" in
        ETTh1|ETTm2|Exchange|ILI) echo 200 ;;
        Electricity)              echo 100 ;;
        *)                        echo 200 ;;
    esac
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAILED=""
COMPLETED=0

_log() { echo "[$(date '+%H:%M:%S')] $*"; }

_run_one() {
    local DATA="$1"
    local PRED="$2"
    local BS; BS=$(dl_batch "$DATA")
    local SW; SW=$(stat_windows "$DATA")
    local TAG="${DATA}_pl${PRED}_${FAMILY}"
    local LOG="${LOG_DIR}/${TAG}.log"

    local SKIP_FLAG=""
    [ "$SKIP_EXISTING" = "1" ] && SKIP_FLAG="--skip_existing"

    local CMD="python scripts/run_baselines.py \
  --data       $DATA \
  --pred_len   $PRED \
  --seq_len    336 \
  --root_path  $DATA_ROOT \
  --results    $RESULTS_DIR \
  --family     $FAMILY \
  --batch_size $BS \
  --max_steps  $MAX_STEPS \
  --n_stat_windows $SW \
  --gpu        $GPU \
  $SKIP_FLAG"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] $CMD"
        return 0
    fi

    mkdir -p "$LOG_DIR"
    _log "START  $TAG"
    _log "       log → $LOG"

    # Soft-fail: record failures but continue the sweep
    if eval "$CMD" > "$LOG" 2>&1; then
        _log "OK     $TAG"
        COMPLETED=$(( COMPLETED + 1 ))
    else
        _log "FAIL   $TAG  (see $LOG)"
        FAILED="${FAILED} ${TAG}"
    fi
}

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

# Convert space-separated strings to lists
read -r -a DATASET_LIST <<< "$DATASETS"
read -r -a PRED_LIST    <<< "$PREDS"

TOTAL=$(( ${#DATASET_LIST[@]} * ${#PRED_LIST[@]} ))
COUNT=0

_log "===== Baseline sweep started ====="
_log "Family   : $FAMILY"
_log "Datasets : $DATASETS"
_log "Horizons : $PREDS"
_log "Total    : $TOTAL dataset×horizon combinations"
_log "GPU      : $GPU   |   MAX_STEPS : $MAX_STEPS"
_log "Logs     : $LOG_DIR"
echo ""

for DATA in "${DATASET_LIST[@]}"; do
    for PRED in "${PRED_LIST[@]}"; do
        COUNT=$(( COUNT + 1 ))
        _log "--- [$COUNT/$TOTAL]  $DATA  pred_len=$PRED ---"
        _run_one "$DATA" "$PRED"
        echo ""
    done
done

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

echo ""
_log "===== Sweep complete ====="
_log "Completed : $COMPLETED / $TOTAL"

if [ -n "$FAILED" ]; then
    _log "Failed runs ($(echo "$FAILED" | wc -w | tr -d ' ')):"
    for f in $FAILED; do
        _log "  ✗  $f  →  ${LOG_DIR}/${f}.log"
    done
    echo ""
    _log "To re-run failed jobs only, set SKIP_EXISTING=0 and narrow DATASETS/PREDS."
    exit 1
fi

_log "All runs succeeded."
_log "Aggregate results : $RESULTS_DIR/summary.csv"
_log "Per-sample losses : $RESULTS_DIR/<exp_name>/losses.csv"
_log ""
_log "Next step — compute significance tests:"
_log "  python scripts/compute_significance.py --pred_len 96"
