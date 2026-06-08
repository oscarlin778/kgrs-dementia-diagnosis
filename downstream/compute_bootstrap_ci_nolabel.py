"""
compute_bootstrap_ci_nolabel.py
================================
對 no-label fMRI ComBat 重訓的三個 PCAG 模型 + 12 個 ablation baselines
做 95% Bootstrap CI（n=2000 resample）。
輸出 results/bootstrap_ci_v2_nolabel.json。
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

RES_DIR = Path(__file__).parent / "results"
N_BOOT  = 2000
RNG     = np.random.default_rng(42)


def bootstrap_auc(probs, labels, n_boot=N_BOOT):
    aucs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_one(path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    probs, labels = d["test_probs"], d["test_labels"]
    if len(np.unique(labels)) < 2:
        return None
    mean_auc, ci_low, ci_high = bootstrap_auc(probs, labels)
    point = float(roc_auc_score(labels, probs))
    return {"auc": round(point, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "n_test": int(len(labels))}


def main():
    out = {}
    tasks = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]

    # 1. PCAG-ComBat (main): no-label fMRI ComBat
    out["pcag_nolabel"] = {}
    for t in tasks:
        out["pcag_nolabel"][t] = compute_one(RES_DIR / f"pcag_combat_{t}_probs_v2_fmricombat_nolabel.npz")

    # 2. Ablations under no-label ComBat
    ablation_keys = ["fmri_only", "smri_only", "no_combat"]
    for k in ablation_keys:
        out[k + "_nolabel"] = {}
        for t in tasks:
            out[k + "_nolabel"][t] = compute_one(
                RES_DIR / f"pcag_combat_{t}_probs_v2_{k}_fmricombat_nolabel.npz")

    # 3. Concat fusion (no-label ComBat)
    out["concat_nolabel"] = {}
    for t in tasks:
        out["concat_nolabel"][t] = compute_one(
            RES_DIR / f"concat_{t}_probs_v2_fmricombat_nolabel.npz")

    # 4. Swapped (no-label)
    out["swapped_nolabel"] = {}
    for t in tasks:
        out["swapped_nolabel"][t] = compute_one(
            RES_DIR / f"pcag_combat_swapped_{t}_probs_v2_fmricombat_nolabel.npz")

    # 5. Original baseline (for reference)
    out["pcag_baseline"] = {}
    for t in tasks:
        out["pcag_baseline"][t] = compute_one(RES_DIR / f"pcag_combat_{t}_probs_v2.npz")

    out_path = RES_DIR / "bootstrap_ci_v2_nolabel.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVED] {out_path}")

    # Print table
    print(f"\n{'Model':<22} {'NC_vs_AD':>20} {'NC_vs_MCI':>20} {'MCI_vs_AD':>20}")
    print("-" * 86)
    for k, v in out.items():
        cells = []
        for t in tasks:
            cell = v.get(t)
            if cell is None:
                cells.append(f"{'N/A':>20}")
            else:
                cells.append(f"{cell['auc']:>5.3f} [{cell['ci_low']:.2f},{cell['ci_high']:.2f}]".rjust(20))
        print(f"{k:<22} {cells[0]} {cells[1]} {cells[2]}")


if __name__ == "__main__":
    main()
