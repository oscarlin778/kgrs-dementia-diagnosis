"""
eval_auc_ci.py — Phase 1-C: Bootstrap 95% CI for AUC on v2 test split.

Loads per-task probability files saved during training (pcag_combat_*_probs_v2.npz),
computes point-estimate AUC + 95% CI via 1000-iteration bootstrap.

Usage:
    conda activate AD
    cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
    python eval_auc_ci.py
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

RES = Path("results")
N_BOOT = 1000
SEED = 42
rng = np.random.default_rng(SEED)

TASKS = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]
LABEL_NAMES = {
    "NC_vs_AD":  ("NC", "AD"),
    "NC_vs_MCI": ("NC", "MCI"),
    "MCI_vs_AD": ("MCI", "AD"),
}

def bootstrap_auc(y_true, y_prob, n=N_BOOT):
    n_samples = len(y_true)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

print("=" * 60)
print("AUC with 95% Bootstrap CI  (n_boot=1000, v2 split)")
print("=" * 60)

rows = []
for task in TASKS:
    path = RES / f"pcag_combat_{task}_probs_v2.npz"
    if not path.exists():
        print(f"  {task}: probs file not found ({path.name}), skipping")
        continue

    data = np.load(str(path), allow_pickle=True)
    y_true = data["test_labels"].astype(int)
    y_prob = data["test_probs"].astype(float)   # prob of positive class

    auc = roc_auc_score(y_true, y_prob)
    lo, hi = bootstrap_auc(y_true, y_prob)
    neg, pos = LABEL_NAMES[task]

    row = f"  {neg} vs {pos}: AUC = {auc:.3f}  (95% CI: {lo:.3f}–{hi:.3f})  n={len(y_true)}"
    print(row)
    rows.append({"task": task, "auc": round(auc, 4),
                 "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": int(len(y_true))})

print()
print("LaTeX-ready:")
for r in rows:
    neg, pos = LABEL_NAMES[r["task"]]
    print(f"  {neg} vs {pos} & {r['auc']:.3f} & [{r['ci_lo']:.3f}--{r['ci_hi']:.3f}] & {r['n']} \\\\")
