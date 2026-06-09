"""
aggregate_ensemble.py
=====================
合併 5 seeds × 5 folds = 25 個 model 的預測：
  - 每個 seed 的 train script 都已在 5 fold 內平均，輸出 test_probs (n_test,)
  - 我們對 5 個 seed 的 test_probs 再做平均，得 final ensemble prob
  - 計算最終 AUC + 95% bootstrap CI

輸出：
  results/ensemble_v2_nolabel_results.json
"""
import json, re
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

RES_DIR = Path(__file__).parent / "results"
SEEDS   = [42, 123, 456, 789, 2024]
N_BOOT  = 2000
RNG     = np.random.default_rng(42)

# Per-task config
TASKS = {
    "NC_vs_AD":  {"variant": "aug",   "suffix_extra": "_aug_mix0.2_de0.2"},
    "NC_vs_MCI": {"variant": "noaug", "suffix_extra": ""},
    "MCI_vs_AD": {"variant": "noaug", "suffix_extra": ""},
}


def detect_site(sid):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(sid)) else 'TPMIC'


def bootstrap_auc(probs, labels, n_boot=N_BOOT):
    aucs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2: continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    out = {}
    for task, cfg in TASKS.items():
        print(f"\n=== {task} (variant={cfg['variant']}) ===")
        suffix = "_v2_fmricombat_nolabel" + cfg["suffix_extra"]
        probs_per_seed = []
        labels = None
        sids = None
        missing = []
        for seed in SEEDS:
            seed_tag = "" if seed == 42 else f"_s{seed}"
            npz_path = RES_DIR / f"pcag_combat_{task}_probs{suffix}{seed_tag}.npz"
            if not npz_path.exists():
                missing.append(str(npz_path))
                continue
            d = np.load(npz_path, allow_pickle=True)
            probs_per_seed.append(d["test_probs"])
            if labels is None:
                labels = d["test_labels"]
                sids = d["test_subject_ids"]
            # AUC of this seed alone
            single_auc = roc_auc_score(d["test_labels"], d["test_probs"])
            print(f"  seed={seed}: AUC={single_auc:.4f}")

        if missing:
            print(f"  ⚠️ Missing: {missing}")
            continue
        if len(probs_per_seed) == 0:
            continue

        # Average across seeds
        probs_per_seed = np.array(probs_per_seed)  # (n_seeds, n_test)
        ensemble_probs = probs_per_seed.mean(axis=0)
        ensemble_auc = roc_auc_score(labels, ensemble_probs)

        # Median ensemble (more robust to outlier seeds)
        median_probs = np.median(probs_per_seed, axis=0)
        median_auc = roc_auc_score(labels, median_probs)
        # Top-3 best seeds by their single-AUC, then average those
        single_aucs = np.array([roc_auc_score(labels, p) for p in probs_per_seed])
        top3_idx = np.argsort(single_aucs)[-3:]
        top3_probs = probs_per_seed[top3_idx].mean(axis=0)
        top3_auc = roc_auc_score(labels, top3_probs)
        print(f"  Median ensemble AUC = {median_auc:.4f}")
        print(f"  Top-3 seeds ensemble AUC = {top3_auc:.4f}  (seeds {[SEEDS[i] for i in top3_idx]})")
        mean_auc, ci_lo, ci_hi = bootstrap_auc(ensemble_probs, labels)
        print(f"\n  Mean Ensemble AUC = {ensemble_auc:.4f}")
        print(f"  Bootstrap mean = {mean_auc:.4f}, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

        # Within-site AUC
        sites = np.array([detect_site(s) for s in sids])
        within = {}
        for site in ['ADNI', 'TPMIC']:
            mask = sites == site
            if mask.sum() > 0 and len(np.unique(labels[mask])) == 2:
                within[site] = float(roc_auc_score(labels[mask], ensemble_probs[mask]))
            else:
                within[site] = None
        print(f"  Within-site: ADNI={within['ADNI']}, TPMIC={within['TPMIC']}")

        # Save ensembled probs
        save_path = RES_DIR / f"pcag_combat_{task}_probs_v2_fmricombat_nolabel_ensemble.npz"
        np.savez(save_path,
                 test_probs=ensemble_probs, test_labels=labels, test_subject_ids=sids,
                 seeds=np.array(SEEDS), n_models=len(SEEDS) * 5)
        print(f"  [SAVED] {save_path}")

        out[task] = {
            "variant": cfg["variant"],
            "ensemble_auc": float(ensemble_auc),
            "bootstrap_mean": float(mean_auc),
            "bootstrap_ci": [ci_lo, ci_hi],
            "within_site": within,
            "n_seeds": len(SEEDS),
            "n_test": int(len(labels)),
        }

    # Save summary
    out_path = RES_DIR / "ensemble_v2_nolabel_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    # Summary table
    print("\n" + "="*80)
    print(f"{'Task':<12} {'Ensemble AUC':>14} {'95% CI':>22} {'ADNI':>8} {'TPMIC':>8}")
    print("="*80)
    for task, r in out.items():
        ci_str = f"[{r['bootstrap_ci'][0]:.3f}, {r['bootstrap_ci'][1]:.3f}]"
        a = r['within_site']['ADNI']; t = r['within_site']['TPMIC']
        a_s = f"{a:.3f}" if a is not None else "N/A"
        t_s = f"{t:.3f}" if t is not None else "N/A"
        print(f"{task:<12} {r['ensemble_auc']:>14.4f} {ci_str:>22} {a_s:>8} {t_s:>8}")


if __name__ == "__main__":
    main()
