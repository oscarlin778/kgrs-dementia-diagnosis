#!/bin/bash
# 跑 ComBat 變體比較：task-specific + no-label，三個 task 各跑一次
set -e
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
mkdir -p logs

run() {
    local task=$1
    local dir=$2
    local tag=$3
    local log=logs/fmricombat_${tag}_${task}.log
    echo "=== ${tag} | ${task} → ${log} ==="
    conda run -n AD --no-capture-output python -u train_pcag_combat_fusion.py \
        --task "${task}" \
        --fmri_harmonized \
        --fmri_combat_dir "${dir}" \
        --ckpt_dir "checkpoints/pcag_fmricombat_${tag}" \
        > "${log}" 2>&1
    grep -E "auc:|Final OOF" "${log}" | head -3
}

# Variant 1: Task-specific (each task uses its own ComBat fit)
run NC_vs_AD  fmri_combat_NC_vs_AD  taskspecific
run NC_vs_MCI fmri_combat_NC_vs_MCI taskspecific
run MCI_vs_AD fmri_combat_MCI_vs_AD taskspecific

# Variant 3: No-label (pure site removal, single fit)
run NC_vs_AD  fmri_combat_v2_nolabel nolabel
run NC_vs_MCI fmri_combat_v2_nolabel nolabel
run MCI_vs_AD fmri_combat_v2_nolabel nolabel

echo "=== ALL DONE ==="
