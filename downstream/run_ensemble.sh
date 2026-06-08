#!/bin/bash
# 跑 5 seeds × 5 fold ensemble per task
# NC_vs_AD: with augmentation (mixup 0.2 + drop_edge 0.2)
# NC_vs_MCI / MCI_vs_AD: no augmentation (baseline)
set -e
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
mkdir -p logs

SEEDS=(42 123 456 789 2024)

run_one() {
    local task=$1
    local seed=$2
    local extra=$3
    local tag=$4
    local log="logs/ensemble_${tag}_${task}_s${seed}.log"
    echo "=== ${tag} | ${task} | seed=${seed} ==="
    conda run -n AD --no-capture-output python -u train_pcag_combat_fusion.py \
        --task "${task}" \
        --fmri_harmonized --fmri_combat_dir fmri_combat_v2_nolabel \
        --ckpt_dir "checkpoints/pcag_ensemble_${tag}" \
        --seed "${seed}" \
        ${extra} > "${log}" 2>&1
    grep -oP "auc:\s*\K[0-9.]+" "${log}" | head -1 | xargs -I {} echo "  AUC={}"
}

# NC_vs_AD：用 augmentation
for seed in "${SEEDS[@]}"; do
    run_one NC_vs_AD $seed "--use_mixup --mixup_alpha 0.2 --drop_edge_rate 0.2" aug
done

# NC_vs_MCI：無 aug
for seed in "${SEEDS[@]}"; do
    run_one NC_vs_MCI $seed "" noaug
done

# MCI_vs_AD：無 aug
for seed in "${SEEDS[@]}"; do
    run_one MCI_vs_AD $seed "" noaug
done

echo "=== ALL ENSEMBLE TRAINING DONE ==="
