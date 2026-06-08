#!/bin/bash
# 跑完整 R-Fix pipeline：
#   1. 4 ablation variants × 3 tasks（fmri_only, smri_only, concat, no_combat）all with no-label ComBat
#   2. Q/K/V swap × 3 tasks with no-label ComBat
# 全部用 no-label fMRI ComBat 諧波化
set -e
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
mkdir -p logs

FMRI_DIR="fmri_combat_v2_nolabel"

run_pcag() {
    local task=$1
    local extra_args=$2
    local tag=$3
    local log="logs/revise_${tag}_${task}.log"
    echo "=== ${tag} | ${task} → ${log} ==="
    conda run -n AD --no-capture-output python -u train_pcag_combat_fusion.py \
        --task "${task}" \
        --fmri_harmonized --fmri_combat_dir "${FMRI_DIR}" \
        ${extra_args} > "${log}" 2>&1
    grep -E "auc:|Final OOF" "${log}" | head -3
}

run_concat() {
    local task=$1
    local log="logs/revise_concat_${task}.log"
    echo "=== concat | ${task} → ${log} ==="
    conda run -n AD --no-capture-output python -u train_concat_fusion.py \
        --task "${task}" \
        --fmri_harmonized --fmri_combat_dir "${FMRI_DIR}" \
        > "${log}" 2>&1
    grep -E "auc:|Final OOF" "${log}" | head -3
}

echo "######### Ablation: fmri_only #########"
for task in NC_vs_AD NC_vs_MCI MCI_vs_AD; do
    run_pcag $task "--modality fmri_only --ckpt_dir checkpoints/pcag_combat" fmri_only
done

echo "######### Ablation: smri_only #########"
for task in NC_vs_AD NC_vs_MCI MCI_vs_AD; do
    run_pcag $task "--modality smri_only --ckpt_dir checkpoints/pcag_combat" smri_only
done

echo "######### Ablation: no_combat (no sMRI ComBat) #########"
for task in NC_vs_AD NC_vs_MCI MCI_vs_AD; do
    run_pcag $task "--no_combat --ckpt_dir checkpoints/pcag_combat" no_combat
done

echo "######### Ablation: concat fusion #########"
for task in NC_vs_AD NC_vs_MCI MCI_vs_AD; do
    run_concat $task
done

echo "######### Q/K/V swap (Q=sMRI) — manual edit to PCAGFusion swap K/V back from Q #########"
# train_pcag_combat_swapped.py 已存在，需要傳同樣的 --fmri_harmonized flag
# 但該腳本可能不支援，先檢查
if grep -q "fmri_harmonized" train_pcag_combat_swapped.py 2>/dev/null; then
    for task in NC_vs_AD NC_vs_MCI MCI_vs_AD; do
        log="logs/revise_swapped_${task}.log"
        echo "=== swapped | ${task} → ${log} ==="
        conda run -n AD --no-capture-output python -u train_pcag_combat_swapped.py \
            --task "${task}" \
            --fmri_harmonized --fmri_combat_dir "${FMRI_DIR}" \
            > "${log}" 2>&1
        grep -E "auc:|Final OOF" "${log}" | head -3
    done
else
    echo "  [SKIP] train_pcag_combat_swapped.py needs --fmri_harmonized flag added"
fi

echo "######### ALL ABLATIONS DONE #########"
