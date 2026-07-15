#!/usr/bin/env python3
"""
compare_dim_ensemble.py
=======================
dim=20 vs dim=128: 5-seed mean ensemble AUC 比較
所有 probs 檔案已由 dim sweep 產生，不需重新訓練。
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

RES = Path("results")
SEEDS = [42, 123, 456, 789, 2024]
TASKS = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]
DIMS  = [20, 128]
N_BOOT = 2000
RNG = np.random.default_rng(42)


def bootstrap_ci(probs, labels, n=N_BOOT):
    aucs = []
    sz = len(labels)
    for _ in range(n):
        idx = RNG.integers(0, sz, sz)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def load_probs(task, dim, seed):
    dim_tag  = f"_dim{dim}" if dim != 20 else ""
    seed_tag = f"_s{seed}"  if seed != 42 else ""
    p = RES / f"pcag_combat_{task}_probs_v2_fmricombat_nolabel{dim_tag}{seed_tag}.npz"
    if not p.exists():
        return None, None, None
    d = np.load(p, allow_pickle=True)
    return d["test_probs"], d["test_labels"], d["test_subject_ids"]


print("=" * 70)
print(f"{'Task':<12} {'dim':>5}  {'Ensemble AUC':>13}  {'95% CI':>22}  {'vs dim=20':>10}")
print("=" * 70)

summary = {}
for dim in DIMS:
    summary[dim] = {}
    for task in TASKS:
        all_probs = []
        labels = None
        for seed in SEEDS:
            probs, lbls, _ = load_probs(task, dim, seed)
            if probs is None:
                print(f"  ⚠️  Missing: {task} dim={dim} seed={seed}")
                continue
            all_probs.append(probs)
            if labels is None:
                labels = lbls
            single = roc_auc_score(lbls, probs)

        if not all_probs:
            continue

        # Mean ensemble
        ens_probs = np.array(all_probs).mean(axis=0)
        ens_auc   = roc_auc_score(labels, ens_probs)
        lo, hi    = bootstrap_ci(ens_probs, labels)
        summary[dim][task] = ens_auc

        vs = ""
        if dim != 20 and task in summary.get(20, {}):
            delta = ens_auc - summary[20][task]
            vs = f"{delta:+.4f}"

        print(f"{task:<12} {dim:>5}  {ens_auc:>13.4f}  [{lo:.3f}, {hi:.3f}]  {vs:>10}")
    if dim != DIMS[-1]:
        print("-" * 70)

print("=" * 70)
print("\n=== Per-seed individual AUCs ===")
for dim in DIMS:
    print(f"\ndim={dim}:")
    for task in TASKS:
        aucs = []
        for seed in SEEDS:
            probs, lbls, _ = load_probs(task, dim, seed)
            if probs is not None:
                aucs.append(round(roc_auc_score(lbls, probs), 4))
        print(f"  {task:<12}: {aucs}  median={np.median(aucs):.4f}  std={np.std(aucs):.4f}")
