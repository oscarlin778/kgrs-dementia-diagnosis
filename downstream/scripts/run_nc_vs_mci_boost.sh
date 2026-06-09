#!/bin/bash
# 救 NC_vs_MCI 的 AUC：兩招併用
#   招 2：gentle augmentation grid (3 variants × seed=42, 看哪個對 MCI 任務有效)
#   招 3：5 個額外 seed for 10-seed ensemble
# 同時對 MCI_vs_AD 也加 5 seed 讓兩個 MCI 任務都有 10-seed ensemble
set -e
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
mkdir -p logs

FMRI="fmri_combat_v2_nolabel"

run() {
    local task=$1 seed=$2 extra="$3" tag="$4"
    local log="logs/boost_${tag}_${task}_s${seed}.log"
    echo "=== ${tag} | ${task} | seed=${seed} ==="
    conda run -n AD --no-capture-output python -u train_pcag_combat_fusion.py \
        --task "${task}" \
        --fmri_harmonized --fmri_combat_dir "${FMRI}" \
        --seed "${seed}" \
        --ckpt_dir "checkpoints/pcag_boost_${tag}" \
        ${extra} > "${log}" 2>&1
    grep -oP "auc:\s*\K[0-9.]+" "${log}" | head -1 | xargs -I{} echo "  Test AUC={}"
}

# ─── 招 2: NC_vs_MCI gentle augmentation grid (seed=42) ───
echo "############## NC_vs_MCI gentle aug grid ##############"
run NC_vs_MCI 42 "--use_mixup --mixup_alpha 0.05 --drop_edge_rate 0.1" mix05de1
run NC_vs_MCI 42 "--use_mixup --mixup_alpha 0.1 --drop_edge_rate 0.0" mix1
run NC_vs_MCI 42 "--drop_edge_rate 0.1" de1only

# ─── 招 3: NC_vs_MCI 額外 5 seeds (no aug) ───
echo "############## NC_vs_MCI extra seeds (no aug) ##############"
for seed in 100 200 300 400 500; do
    run NC_vs_MCI $seed "" noaug_extra
done

# ─── MCI_vs_AD 也加 5 seeds (no aug) — 順便加大 ensemble ───
echo "############## MCI_vs_AD extra seeds (no aug) ##############"
for seed in 100 200 300 400 500; do
    run MCI_vs_AD $seed "" noaug_extra
done

# ─── NC_vs_AD 也加 5 seeds with augmentation ───
echo "############## NC_vs_AD extra seeds (with aug) ##############"
for seed in 100 200 300 400 500; do
    run NC_vs_AD $seed "--use_mixup --mixup_alpha 0.2 --drop_edge_rate 0.2" aug_extra
done

echo "############## ALL BOOST TRAININGS DONE ##############"
