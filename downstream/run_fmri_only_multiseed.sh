#!/bin/bash
# 訓練 fMRI-only 多 seed（123/456/789/2024）並計算 ensemble OOF AUC
# seed=42 已有，只補 4 個
# 總預計時間：~2-3 小時（3 tasks × 4 seeds，每個 ~10-15 分鐘）
set -e
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
mkdir -p logs

TASKS=("NC_vs_AD" "NC_vs_MCI" "MCI_vs_AD")
SEEDS=(123 456 789 2024)
FMRI_DIR="fmri_combat_v2_nolabel"
CKPT_BASE="checkpoints/pcag_combat_v2_fmri_only_fmricombat_nolabel"

echo "=== fMRI-only 多 seed 訓練 ==="
echo "Tasks: ${TASKS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo ""

for TASK in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CKPT_DIR="${CKPT_BASE}_s${SEED}"
        LOG="logs/fmri_only_${TASK}_s${SEED}.log"
        echo "[$(date +%H:%M)] Task=${TASK} Seed=${SEED} → ${LOG}"
        conda run -n AD --no-capture-output python -u train_pcag_combat_fusion.py \
            --task "${TASK}" \
            --modality fmri_only \
            --fmri_harmonized \
            --fmri_combat_dir "${FMRI_DIR}" \
            --seed ${SEED} \
            --ckpt_dir "${CKPT_DIR}" \
            > "${LOG}" 2>&1
        OOF=$(grep -E "Final OOF AUC|oof_auc" "${LOG}" | tail -1)
        echo "  → ${OOF}"
    done
done

echo ""
echo "=== 計算 fMRI-only Ensemble OOF AUC ==="
conda run -n AD --no-capture-output python - <<'PYEOF'
import numpy as np
import json
from sklearn.metrics import roc_auc_score
import os

BASE = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream"
RESULTS = os.path.join(BASE, "results")
SEEDS = [42, 123, 456, 789, 2024]
TASKS = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]

print("\nfMRI-only Ensemble OOF AUC (5 seeds):")
for task in TASKS:
    all_probs = []
    labels = None
    for seed in SEEDS:
        # seed=42 uses original filename (no suffix), others use _s{seed}
        if seed == 42:
            fname = f"pcag_combat_{task}_probs_v2_fmri_only_fmricombat_nolabel.npz"
        else:
            fname = f"pcag_combat_{task}_probs_v2_fmri_only_fmricombat_nolabel_s{seed}.npz"
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path):
            print(f"  [{task}] missing: {fname}")
            continue
        d = np.load(path)
        all_probs.append(d['oof_probs'])
        if labels is None:
            labels = d['oof_labels']
    if not all_probs:
        print(f"  [{task}] no data")
        continue
    ensemble_probs = np.mean(all_probs, axis=0)
    auc = roc_auc_score(labels, ensemble_probs)
    # Also individual seeds
    indiv = [roc_auc_score(labels, p) for p in all_probs]
    print(f"  {task}: ensemble={auc:.4f}  (seeds: {[round(x,4) for x in indiv]})")

# Save result
result = {}
for task in TASKS:
    all_probs = []
    labels = None
    for seed in SEEDS:
        fname = f"pcag_combat_{task}_probs_v2_fmri_only_fmricombat_nolabel{'_s'+str(seed) if seed!=42 else ''}.npz"
        path = os.path.join(RESULTS, fname)
        if not os.path.exists(path): continue
        d = np.load(path)
        all_probs.append(d['oof_probs'])
        if labels is None: labels = d['oof_labels']
    if not all_probs: continue
    ensemble = np.mean(all_probs, axis=0)
    result[task] = {
        "fmri_only_ensemble_oof_auc": round(roc_auc_score(labels, ensemble), 4),
        "n_seeds": len(all_probs),
        "seeds": [s for s in SEEDS[:len(all_probs)]]
    }

out = os.path.join(BASE, "results", "fmri_only_ensemble_summary.json")
json.dump(result, open(out, "w"), indent=2)
print(f"\nSaved to {out}")
PYEOF

echo "=== Done ==="
