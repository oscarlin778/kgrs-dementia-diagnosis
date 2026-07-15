#!/bin/bash
# NC_vs_AD dim=128 + augmentation (mix0.2 + drop_edge 0.2)
# Replicates the original aug setup but with fusion_dim=128
# Expected output suffix: _v2_fmricombat_nolabel_aug_mix0.2_de0.2_dim128

set -e
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream

FMRI_DIR="fmri_combat_v2_nolabel"
SEEDS=(42 123 456 789 2024)
TASK="NC_vs_AD"
DIM=128

LOG_DIR="logs/dim128_aug"
mkdir -p "$LOG_DIR"

echo "=== NC_vs_AD dim=128 augmented ==="
echo "Started: $(date)"
pids=()
for SEED in "${SEEDS[@]}"; do
    LOG="$LOG_DIR/${TASK}_s${SEED}.log"
    conda run -n AD python training/train_pcag_combat_fusion.py \
        --task "$TASK" --seed "$SEED" \
        --fusion_dim "$DIM" \
        --fmri_harmonized --fmri_combat_dir "$FMRI_DIR" \
        --use_mixup --mixup_alpha 0.2 \
        --drop_edge_rate 0.2 \
        --epochs 200 \
        > "$LOG" 2>&1 &
    pids+=($!)
    echo "  launched seed=$SEED (PID $!)"
done

for pid in "${pids[@]}"; do wait "$pid" || true; done

echo ""
echo "Per-seed AUC:"
for SEED in "${SEEDS[@]}"; do
    LOG="$LOG_DIR/${TASK}_s${SEED}.log"
    AUC=$(grep -o 'auc: [0-9.]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9.]*')
    echo "  s${SEED}: ${AUC:-N/A}"
done
echo "=== Done: $(date) ==="
