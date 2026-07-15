#!/bin/bash
# Extended fusion_dim sweep: {128, 256, 512}
# Complements the original sweep {4,8,16,20,32,64} to find the trade-off peak

set -e
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream

FMRI_DIR="fmri_combat_v2_nolabel"
SEEDS=(42 123 456 789 2024)
TASKS=("NC_vs_AD" "NC_vs_MCI" "MCI_vs_AD")
DIMS=(128 256 512)

LOG_DIR="logs/dim_sweep"
mkdir -p "$LOG_DIR"

run_batch() {
    local DIM=$1; local TASK=$2
    echo "[$(date '+%H:%M:%S')] dim=${DIM} / ${TASK}"
    pids=()
    for SEED in "${SEEDS[@]}"; do
        LOG="$LOG_DIR/dim${DIM}_${TASK}_s${SEED}.log"
        conda run -n AD python training/train_pcag_combat_fusion.py \
            --task "$TASK" --seed "$SEED" \
            --fusion_dim "$DIM" \
            --fmri_harmonized --fmri_combat_dir "$FMRI_DIR" \
            --epochs 200 \
            > "$LOG" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done

    for SEED in "${SEEDS[@]}"; do
        LOG="$LOG_DIR/dim${DIM}_${TASK}_s${SEED}.log"
        AUC=$(grep -o 'auc: [0-9.]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9.]*')
        echo "  s${SEED}: ${AUC:-N/A}"
    done
}

echo "=== fusion_dim extended sweep {128, 256, 512} ==="
echo "dims: ${DIMS[*]}"
echo "Started: $(date)"
echo ""

for DIM in "${DIMS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        run_batch "$DIM" "$TASK"
    done
    echo ""
done

echo "=== Done: $(date) ==="
