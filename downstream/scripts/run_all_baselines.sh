#!/usr/bin/env bash
# run_all_baselines.sh — Run all Phase 2-A baseline experiments sequentially.
#
# Baselines:
#   1. fMRI-only GNN     (--modality fmri_only)
#   2. sMRI-only Linear  (--modality smri_only)
#   3. PCAG w/o ComBat   (--no_combat)
#   4. Concat fusion     (train_concat_fusion.py)
#   5. SVM + FC tri      (baseline_svm.py)
#
# Each baseline runs on all 3 tasks: NC_vs_AD, NC_vs_MCI, MCI_vs_AD
#
# Usage:
#   bash run_all_baselines.sh [TASKS]
#   bash run_all_baselines.sh NC_vs_AD          # single task
#   bash run_all_baselines.sh NC_vs_AD NC_vs_MCI # two tasks
#   bash run_all_baselines.sh                    # all three tasks (default)

set -euo pipefail
cd "$(dirname "$0")"

TASKS=("${@:-NC_vs_AD NC_vs_MCI MCI_vs_AD}")
# If no args given, default to all three
if [ $# -eq 0 ]; then
    TASKS=(NC_vs_AD NC_vs_MCI MCI_vs_AD)
else
    TASKS=("$@")
fi

mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_task() {
    local label="$1"; shift
    local cmd=("$@")
    log "START  $label"
    "${cmd[@]}" && log "DONE   $label" || { log "FAILED $label"; exit 1; }
}

# ── 1. fMRI-only ──────────────────────────────────────────────────────────────
for TASK in "${TASKS[@]}"; do
    run_task "fmri_only / $TASK" \
        python train_pcag_combat_fusion.py \
            --task "$TASK" --modality fmri_only \
            2>&1 | tee "logs/fmri_only_${TASK}.log"
done

# ── 2. sMRI-only ──────────────────────────────────────────────────────────────
for TASK in "${TASKS[@]}"; do
    run_task "smri_only / $TASK" \
        python train_pcag_combat_fusion.py \
            --task "$TASK" --modality smri_only \
            2>&1 | tee "logs/smri_only_${TASK}.log"
done

# ── 3. PCAG without ComBat ────────────────────────────────────────────────────
for TASK in "${TASKS[@]}"; do
    run_task "no_combat / $TASK" \
        python train_pcag_combat_fusion.py \
            --task "$TASK" --no_combat \
            2>&1 | tee "logs/no_combat_${TASK}.log"
done

# ── 4. Concat fusion ──────────────────────────────────────────────────────────
for TASK in "${TASKS[@]}"; do
    run_task "concat / $TASK" \
        python train_concat_fusion.py \
            --task "$TASK" \
            2>&1 | tee "logs/concat_${TASK}.log"
done

# ── 5. SVM baseline ───────────────────────────────────────────────────────────
for TASK in "${TASKS[@]}"; do
    run_task "svm / $TASK" \
        python baseline_svm.py \
            --task "$TASK" \
            2>&1 | tee "logs/svm_${TASK}.log"
done

log "All baselines finished."
echo ""
echo "Result files written to results/:"
ls results/*_results_v2.json 2>/dev/null | sort || true
