"""
plot_kd_metrics.py
Comprehensive metrics for KD student vs E5 baseline.

Prerequisites: re-run train_kd_fusion.py for all 3 tasks so that
  kd_{task}_{suffix}_probs.npz files exist in results/.

Outputs:
  figures/KD_ROC_curves.png         — ROC curves (3 tasks)
  figures/KD_metrics_comparison.png — bar chart (AUC / F1 / Sens / Spec)
  kd_comprehensive_metrics.json     — full metrics table
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

RESULTS_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results")
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

E5_JSON = "/home/wei-chi/Alzheimers_Project/external_data/scripts/results/E5_clean/metrics.json"

TASKS = {
    "NC_vs_AD":  {"suffix": "a3_T3",  "label": "NC vs AD"},
    "NC_vs_MCI": {"suffix": "a3_T3",  "label": "NC vs MCI"},
    "MCI_vs_AD": {"suffix": "a10_T3", "label": "MCI vs AD"},
}


def cm_metrics(cm_list):
    tn, fp = cm_list[0][0], cm_list[0][1]
    fn, tp = cm_list[1][0], cm_list[1][1]
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn)
    return dict(sens=sens, spec=spec, prec=prec, f1=f1, npv=npv, acc=acc)


def youden_threshold(fpr, tpr, thresholds):
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


e5 = json.load(open(E5_JSON))
summary = {}

# ── Figure 1: ROC curves ──────────────────────────────────────────────
fig_roc, axes = plt.subplots(1, 3, figsize=(15, 5))
fig_roc.suptitle("ROC Curves — fMRI+sMRI KD Student", fontsize=14, fontweight="bold")

for ax, (task, cfg) in zip(axes, TASKS.items()):
    prob_file = RESULTS_DIR / f"kd_{task}_{cfg['suffix']}_probs.npz"

    if not prob_file.exists():
        ax.text(0.5, 0.5, f"Missing:\n{prob_file.name}\n\nRe-run train_kd_fusion.py",
                ha="center", va="center", transform=ax.transAxes,
                color="red", fontsize=10)
        ax.set_title(cfg["label"])
        continue

    d          = np.load(str(prob_file))
    oof_probs  = d["oof_probs"];  oof_labels = d["oof_labels"]
    te_probs   = d["test_probs"]; te_labels  = d["test_labels"]

    # Optimal threshold from OOF (no test-set leakage)
    o_fpr, o_tpr, o_thr = roc_curve(oof_labels, oof_probs)
    opt_thr, _, _        = youden_threshold(o_fpr, o_tpr, o_thr)

    # Test ROC
    fpr, tpr, _ = roc_curve(te_labels, te_probs)
    roc_auc_val = auc(fpr, tpr)

    # Metrics at OOF-optimal threshold
    pred = (te_probs >= opt_thr).astype(int)
    cm   = confusion_matrix(te_labels, pred).tolist()
    m    = cm_metrics(cm)

    # E5 metrics from confusion matrix
    e5_cm  = e5["test"][task]["cm"]
    e5_m   = cm_metrics(e5_cm)
    e5_auc = e5["test"][task]["auc"]

    summary[task] = {
        "kd": {
            "auc":       round(roc_auc_val, 4),
            "threshold": round(opt_thr, 4),
            **{k: round(v, 4) for k, v in m.items()},
            "cm": cm,
        },
        "e5": {
            "auc": round(e5_auc, 4),
            **{k: round(v, 4) for k, v in e5_m.items()},
            "cm": e5_cm,
        },
    }

    # Operating point on ROC
    te_fpr_pt = 1.0 - m["spec"]
    te_tpr_pt = m["sens"]

    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"KD Student  AUC={roc_auc_val:.3f}")
    ax.scatter([te_fpr_pt], [te_tpr_pt], s=140, color="crimson", zorder=6,
               label=f"Opt. pt  Sens={m['sens']:.2f}  Spec={m['spec']:.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.annotate(f"E5 AUC = {e5_auc:.3f}",
                xy=(0.52, 0.06), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("1 − Specificity  (FPR)")
    ax.set_ylabel("Sensitivity  (TPR)")
    ax.set_title(f"{cfg['label']}  (n={len(te_labels)})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
roc_path = FIG_DIR / "KD_ROC_curves.png"
plt.savefig(str(roc_path), dpi=150, bbox_inches="tight")
print(f"[SAVED] {roc_path}")
plt.close()

# ── Figure 2: Bar chart comparison ───────────────────────────────────
tasks_done    = [t for t in TASKS if t in summary]
task_labels   = [TASKS[t]["label"] for t in tasks_done]
metric_keys   = ["auc", "f1", "sens", "spec"]
metric_titles = ["AUC", "F1-Score", "Sensitivity", "Specificity"]

x     = np.arange(len(tasks_done))
width = 0.35

fig_bar, axes2 = plt.subplots(2, 2, figsize=(12, 8))
fig_bar.suptitle("KD Student vs E5 Baseline — Metric Comparison", fontsize=13, fontweight="bold")

for ax, met, mlabel in zip(axes2.flat, metric_keys, metric_titles):
    e5_vals = [summary[t]["e5"][met] for t in tasks_done]
    kd_vals = [summary[t]["kd"][met] for t in tasks_done]

    b1 = ax.bar(x - width / 2, e5_vals, width, label="E5 Baseline",
                color="salmon",    alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width / 2, kd_vals, width, label="KD Student",
                color="steelblue", alpha=0.85, edgecolor="white")
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylim(0, 1.15)
    ax.set_title(mlabel)
    ax.set_ylabel(mlabel)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
bar_path = FIG_DIR / "KD_metrics_comparison.png"
plt.savefig(str(bar_path), dpi=150, bbox_inches="tight")
print(f"[SAVED] {bar_path}")
plt.close()

# ── Save JSON ─────────────────────────────────────────────────────────
out_json = RESULTS_DIR / "kd_comprehensive_metrics.json"
with open(str(out_json), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[SAVED] {out_json}")

# ── Console table ─────────────────────────────────────────────────────
print(f"\n{'='*82}")
print(f"{'Task':<12} {'Model':<10} {'AUC':>6} {'Sens':>6} {'Spec':>6} "
      f"{'F1':>6} {'Prec':>6} {'NPV':>6} {'ACC':>6}")
print(f"{'-'*82}")
for task in tasks_done:
    lbl = TASKS[task]["label"]
    for key, tag in [("e5", "E5 Base."), ("kd", "KD Stu.")]:
        m = summary[task][key]
        print(f"{lbl if key == 'e5' else '':<12} {tag:<10} "
              f"{m['auc']:>6.3f} {m['sens']:>6.3f} {m['spec']:>6.3f} "
              f"{m['f1']:>6.3f} {m['prec']:>6.3f} {m['npv']:>6.3f} {m['acc']:>6.3f}")
    print()
print("Note: KD metrics use OOF-optimal threshold (Youden's J); "
      "E5 metrics from confusion matrix at default threshold.")
