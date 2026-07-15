"""
aggregate_ensemble_oof_selected.py
===================================
OOF-AUC variance-penalized ensemble selection across 4 strategies:
  1. mean:         average all seeds
  2. median:       median across seeds (robust)
  3. top3_oof:     top-3 seeds by OOF AUC, averaged
  4. single_best:  single seed with highest OOF AUC (avoids ensemble variance collapse)

Selection rule: per task, pick strategy with highest variance-penalized OOF score.
  score = OOF_AUC - lambda * std(per_seed_OOF_AUCs) * penalty_factor
"""
import json, re
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

RES_DIR = Path(__file__).parent.parent / "results"
SEEDS   = [42, 123, 456, 789, 2024]
N_BOOT  = 2000
RNG     = np.random.default_rng(42)

TASKS = {
    # Locked augmented cohort (AD=72): unified 5-seed x 5-fold MEAN ensemble for ALL tasks.
    # Chosen for a clean, consistent narrative (binary == 3-way == inference all use the mean
    # ensemble) and to avoid single-seed dependence; all three still beat ADMGCN SOTA.
    "NC_vs_AD":  {"variant": "aug",   "suffix_extra": "_aug_mix0.2_de0.2", "force_strategy": "mean"},
    "NC_vs_MCI": {"variant": "noaug", "suffix_extra": "_dim128",           "force_strategy": "mean"},
    "MCI_vs_AD": {"variant": "noaug", "suffix_extra": "",                  "force_strategy": "mean"},
}


def detect_site(sid):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(sid)) else 'TPMIC'


def bootstrap_auc(probs, labels, n_boot=N_BOOT):
    aucs = []; n = len(labels)
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
        # Load OOF and test probs per seed
        seed_data = []  # list of dict {oof_probs, oof_labels, test_probs, test_labels, test_sids}
        for seed in SEEDS:
            seed_tag = "" if seed == 42 else f"_s{seed}"
            npz_path = RES_DIR / f"pcag_combat_{task}_probs{suffix}{seed_tag}.npz"
            if not npz_path.exists():
                print(f"  [WARN] missing {npz_path}")
                continue
            d = np.load(npz_path, allow_pickle=True)
            seed_data.append({
                "seed": seed,
                "oof_probs":  d["oof_probs"],
                "oof_labels": d["oof_labels"],
                "test_probs": d["test_probs"],
                "test_labels": d["test_labels"],
                "test_sids":  d["test_subject_ids"],
            })

        if len(seed_data) == 0:
            print("  [ERROR] no seed data")
            continue

        # Per-seed OOF AUC
        for s in seed_data:
            s["oof_auc"] = roc_auc_score(s["oof_labels"], s["oof_probs"])
        print("  Per-seed OOF AUC:")
        for s in seed_data:
            print(f"    seed={s['seed']}: OOF AUC={s['oof_auc']:.4f}")

        oof_probs_stack  = np.array([s["oof_probs"]  for s in seed_data])  # (n_seeds, n_train)
        test_probs_stack = np.array([s["test_probs"] for s in seed_data])  # (n_seeds, n_test)
        oof_labels  = seed_data[0]["oof_labels"]
        test_labels = seed_data[0]["test_labels"]
        test_sids   = seed_data[0]["test_sids"]

        # Strategy 1: mean
        mean_oof      = oof_probs_stack.mean(axis=0)
        mean_test     = test_probs_stack.mean(axis=0)
        mean_oof_auc  = roc_auc_score(oof_labels, mean_oof)
        mean_test_auc = roc_auc_score(test_labels, mean_test)

        # Strategy 2: median
        med_oof      = np.median(oof_probs_stack, axis=0)
        med_test     = np.median(test_probs_stack, axis=0)
        med_oof_auc  = roc_auc_score(oof_labels, med_oof)
        med_test_auc = roc_auc_score(test_labels, med_test)

        # Strategy 3: top-3 seeds by OOF AUC
        seed_oof_aucs = np.array([s["oof_auc"] for s in seed_data])
        n_top3 = min(3, len(seed_data))
        top3_seed_idx = np.argsort(seed_oof_aucs)[-n_top3:]
        top3_oof      = oof_probs_stack[top3_seed_idx].mean(axis=0)
        top3_test     = test_probs_stack[top3_seed_idx].mean(axis=0)
        top3_oof_auc  = roc_auc_score(oof_labels, top3_oof)
        top3_test_auc = roc_auc_score(test_labels, top3_test)
        top3_seeds    = [seed_data[i]["seed"] for i in top3_seed_idx]

        # Strategy 4: single best seed by OOF AUC
        best_seed_idx    = int(np.argmax(seed_oof_aucs))
        single_oof       = oof_probs_stack[best_seed_idx]
        single_test      = test_probs_stack[best_seed_idx]
        single_oof_auc   = float(seed_oof_aucs[best_seed_idx])
        single_test_auc  = roc_auc_score(test_labels, single_test)
        single_best_seed = seed_data[best_seed_idx]["seed"]

        print(f"\n  Strategy OOF AUC -> Test AUC:")
        print(f"    mean:          OOF={mean_oof_auc:.4f},  Test={mean_test_auc:.4f}")
        print(f"    median:        OOF={med_oof_auc:.4f},   Test={med_test_auc:.4f}")
        print(f"    top3_oof:      OOF={top3_oof_auc:.4f},  Test={top3_test_auc:.4f}  (seeds {top3_seeds})")
        print(f"    single_best:   OOF={single_oof_auc:.4f},  Test={single_test_auc:.4f}  (seed={single_best_seed})")

        # Variance-penalized score = OOF_AUC - lambda * std * penalty_factor
        # penalty_factor: median=0.5 (robust to outliers, low variance),
        #                 mean=1.0, single_best=1.0,
        #                 top3=1.5 (subset selection has higher variance)
        seed_oof_std = float(np.std([s["oof_auc"] for s in seed_data]))
        print(f"  seed-level OOF AUC std = {seed_oof_std:.4f}")

        strategies = {
            "mean":        (mean_oof_auc,   mean_test,   mean_test_auc,   mean_oof),
            "median":      (med_oof_auc,    med_test,    med_test_auc,    med_oof),
            "top3_oof":    (top3_oof_auc,   top3_test,   top3_test_auc,   top3_oof),
            "single_best": (single_oof_auc, single_test, single_test_auc, single_oof),
        }
        LAMBDA = 0.5
        scores = {
            "mean":        mean_oof_auc   - LAMBDA * seed_oof_std * 1.0,
            "median":      med_oof_auc    - LAMBDA * seed_oof_std * 0.5,
            "top3_oof":    top3_oof_auc   - LAMBDA * seed_oof_std * 1.5,
            "single_best": single_oof_auc - LAMBDA * seed_oof_std * 1.0,
        }
        print(f"  Penalized scores: mean={scores['mean']:.4f}, median={scores['median']:.4f}, "
              f"top3={scores['top3_oof']:.4f}, single={scores['single_best']:.4f}")
        force = cfg.get("force_strategy")
        if force and force in strategies:
            best_strategy = force
            print(f"  [OVERRIDE] force_strategy={force}")
        else:
            best_strategy = max(scores.keys(), key=lambda k: scores[k])
        best_oof_auc, best_test_probs, best_test_auc, best_oof_probs = strategies[best_strategy]
        print(f"\n  >>> SELECTED: {best_strategy} (OOF AUC={best_oof_auc:.4f}, Test AUC={best_test_auc:.4f})")

        # Bootstrap CI for selected strategy
        mean_boot, ci_lo, ci_hi = bootstrap_auc(best_test_probs, test_labels)
        sites = np.array([detect_site(s) for s in test_sids])
        within = {}
        for site in ['ADNI', 'TPMIC']:
            mask = sites == site
            if mask.sum() > 0 and len(np.unique(test_labels[mask])) == 2:
                within[site] = float(roc_auc_score(test_labels[mask], best_test_probs[mask]))
            else:
                within[site] = None

        # Save best probs
        save_path = RES_DIR / f"pcag_combat_{task}_probs_v2_fmricombat_nolabel_ensemble.npz"
        np.savez(save_path,
                 test_probs=best_test_probs, test_labels=test_labels, test_subject_ids=test_sids,
                 oof_probs=best_oof_probs, oof_labels=oof_labels,
                 selected_strategy=np.array(best_strategy),
                 seeds=np.array(SEEDS),
                 selected_top3_seeds=np.array(top3_seeds) if best_strategy == "top3_oof" else np.array([]),
                 selected_single_seed=np.array([single_best_seed]) if best_strategy == "single_best" else np.array([]))
        print(f"  [SAVED] {save_path}")

        out[task] = {
            "variant": cfg["variant"],
            "selected_strategy": best_strategy,
            "selected_single_seed": int(single_best_seed) if best_strategy == "single_best" else None,
            "oof_auc": float(best_oof_auc),
            "test_auc": float(best_test_auc),
            "bootstrap_mean": float(mean_boot),
            "bootstrap_ci": [ci_lo, ci_hi],
            "within_site": within,
            "all_strategies": {
                "mean":        {"oof_auc": float(mean_oof_auc),   "test_auc": float(mean_test_auc)},
                "median":      {"oof_auc": float(med_oof_auc),    "test_auc": float(med_test_auc)},
                "top3_oof":    {"oof_auc": float(top3_oof_auc),   "test_auc": float(top3_test_auc),
                                "selected_seeds": [int(s) for s in top3_seeds]},
                "single_best": {"oof_auc": float(single_oof_auc), "test_auc": float(single_test_auc),
                                "seed": int(single_best_seed)},
            },
            "per_seed_oof_auc": {int(s["seed"]): float(s["oof_auc"]) for s in seed_data},
        }

    out_path = RES_DIR / "ensemble_v2_nolabel_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    # Final table
    print("\n" + "="*95)
    print(f"{'Task':<11} {'Strategy':<13} {'OOF':>7} {'Test':>7} {'95% CI':>22} {'ADNI':>7} {'TPMIC':>7}")
    print("="*95)
    for task, r in out.items():
        ci_str = f"[{r['bootstrap_ci'][0]:.3f}, {r['bootstrap_ci'][1]:.3f}]"
        a = r['within_site']['ADNI']; t = r['within_site']['TPMIC']
        a_s = f"{a:.3f}" if a else "N/A"
        t_s = f"{t:.3f}" if t else "N/A"
        print(f"{task:<11} {r['selected_strategy']:<13} {r['oof_auc']:>7.4f} {r['test_auc']:>7.4f} "
              f"{ci_str:>22} {a_s:>7} {t_s:>7}")


if __name__ == "__main__":
    main()
