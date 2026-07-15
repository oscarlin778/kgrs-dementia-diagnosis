#!/bin/bash
# Task 1-A: Single-site training for cross-site generalization analysis
# Trains OVO PCAG models using TPMIC-only and ADNI-only training data
# Evaluates on the FULL held-out test set (both sites) to measure cross-site generalization

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/train_pcag_combat_fusion.py"

FMRI_DIR="/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/fmri_combat_v2_nolabel"
SEEDS=(42 123 456 789 2024)
TASKS=("NC_vs_AD" "NC_vs_MCI" "MCI_vs_AD")
SITES=("TPMIC" "ADNI_new")

LOG_DIR="$SCRIPT_DIR/../logs/site_training"
mkdir -p "$LOG_DIR"

echo "=== Task 1-A: Single-site training ==="
echo "Sites: ${SITES[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo ""

for SITE in "${SITES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            LOG="$LOG_DIR/${SITE}_${TASK}_s${SEED}.log"
            echo "[$(date +%H:%M:%S)] Starting: site=$SITE task=$TASK seed=$SEED"
            conda run -n AD python "$TRAIN_PY" \
                --task "$TASK" \
                --seed "$SEED" \
                --site_filter "$SITE" \
                --fmri_harmonized \
                --fmri_combat_dir "$FMRI_DIR" \
                --epochs 200 \
                > "$LOG" 2>&1
            # extract final AUC from log
            FINAL=$(grep "Test AUC" "$LOG" | tail -1 || echo "N/A")
            echo "  → $FINAL"
        done
        echo ""
    done
done

echo "=== All single-site training complete ==="
echo "Logs: $LOG_DIR"
