#!/bin/bash
# Task 1-A v2: Run single-site training sequentially per task (5 seeds in parallel per task)
# More stable than 30-parallel; can be resumed after restart

set -e
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream

FMRI_DIR="fmri_combat_v2_nolabel"
SEEDS=(42 123 456 789 2024)
LOG_DIR="logs/site_training"
mkdir -p "$LOG_DIR"

run_task() {
    local SITE=$1; local TASK=$2
    echo "[$(date '+%H:%M:%S')] === $SITE / $TASK ==="
    pids=()
    for SEED in "${SEEDS[@]}"; do
        LOG="$LOG_DIR/${SITE}_${TASK}_s${SEED}.log"
        # skip if already completed (JSON result exists)
        RESULT_JSON="results/pcag_combat_${TASK}_probs_v2_fmricombat_nolabel_site${SITE}.npz"
        if [ -f "$RESULT_JSON" ] && [ "$SEED" = "42" ]; then
            echo "  [SKIP] $SITE $TASK s42 already done"
            continue
        fi
        conda run -n AD python training/train_pcag_combat_fusion.py \
            --task "$TASK" --seed "$SEED" \
            --site_filter "$SITE" \
            --fmri_harmonized --fmri_combat_dir "$FMRI_DIR" \
            --epochs 200 \
            > "$LOG" 2>&1 &
        pids+=($!)
    done
    # wait for all seeds of this task
    for pid in "${pids[@]}"; do wait $pid || true; done

    # print summary
    for SEED in "${SEEDS[@]}"; do
        LOG="$LOG_DIR/${SITE}_${TASK}_s${SEED}.log"
        if [ -f "$LOG" ] && [ -s "$LOG" ]; then
            AUC=$(grep "Test AUC\|test_auc\|auc" "$LOG" 2>/dev/null | grep -o '[0-9]\.[0-9]*' | tail -1)
            echo "  s${SEED}: test_auc=${AUC:-N/A}"
        fi
    done
    echo ""
}

echo "=== Task 1-A: Single-site OVO training (v2) ==="
echo "Started: $(date)"

# Run each site × task combination sequentially (5 seeds in parallel per batch)
for SITE in TPMIC ADNI_new; do
    for TASK in NC_vs_AD NC_vs_MCI MCI_vs_AD; do
        run_task "$SITE" "$TASK"
    done
done

echo "=== All done: $(date) ==="
