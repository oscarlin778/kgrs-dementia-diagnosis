#!/bin/bash
# Ablation: fusion_dim sweep
# Sweeps fusion_dim in {4, 8, 16, 20, 32, 64} across 3 tasks x 5 seeds
# dim=20 is the default; other dims get _dim{N} in exp_tag
# Wall time estimate: 18 batches x ~2 min/batch = ~36 min

set -e
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream

FMRI_DIR="fmri_combat_v2_nolabel"
SEEDS=(42 123 456 789 2024)
TASKS=("NC_vs_AD" "NC_vs_MCI" "MCI_vs_AD")
DIMS=(4 8 16 20 32 64)

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

    # print per-seed AUC
    for SEED in "${SEEDS[@]}"; do
        LOG="$LOG_DIR/dim${DIM}_${TASK}_s${SEED}.log"
        AUC=$(grep -o 'auc: [0-9.]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9.]*')
        echo "  s${SEED}: ${AUC:-N/A}"
    done
}

echo "=== fusion_dim sweep ==="
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
