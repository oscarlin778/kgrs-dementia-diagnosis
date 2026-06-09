"""
plot_pcag_results.py
Visualize PCAG fusion results vs KD Student vs E5 Baseline.
Outputs:
  figures/PCAG_metrics_comparison.png  — bar chart (4 metrics × 3 tasks)
  figures/PCAG_radar_chart.png         — radar chart (task-level summary)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results")
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

TASKS       = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]
TASK_LABELS = ["NC vs AD", "NC vs MCI", "MCI vs AD"]

# ── Load data ─────────────────────────────────────────────────────────
with open(RESULTS_DIR / "kd_comprehensive_metrics.json") as f:
    kd_base = json.load(f)

pcag = {}
for task in TASKS:
    with open(RESULTS_DIR / f"pcag_{task}_results.json") as f:
        pcag[task] = json.load(f)

def get_metrics(task):
    e5 = kd_base[task]["e5"]
    kd = kd_base[task]["kd"]
    pm = pcag[task]["test_metrics"]
    return {
        "E5 Baseline": {"auc": e5["auc"], "sens": e5["sens"], "spec": e5["spec"], "f1": e5["f1"]},
        "KD Student":  {"auc": kd["auc"], "sens": kd["sens"], "spec": kd["spec"], "f1": kd["f1"]},
        "PCAG Fusion": {"auc": pm["auc"], "sens": pm["sens"], "spec": pm["spec"], "f1": pm["f1"]},
    }

COLORS  = {"E5 Baseline": "#E07B7B", "KD Student": "#5B9BD5", "PCAG Fusion": "#70AD47"}
METRICS = ["auc", "sens", "spec", "f1"]
MLABELS = ["AUC", "Sensitivity", "Specificity", "F1-Score"]

# ── Figure 1: Grouped bar chart (metric × task) ───────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("PCAG Fusion vs KD Student vs E5 Baseline",
             fontsize=14, fontweight="bold", y=1.02)

x      = np.arange(len(TASKS))
width  = 0.25
models = ["E5 Baseline", "KD Student", "PCAG Fusion"]

for ax, met, mlbl in zip(axes, METRICS, MLABELS):
    for i, model in enumerate(models):
        vals = [get_metrics(t)[model][met] for t in TASKS]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=model, color=COLORS[model], alpha=0.88, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_title(mlbl, fontsize=11, fontweight="bold")
    ax.set_ylabel(mlbl, fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
out1 = FIG_DIR / "PCAG_metrics_comparison.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"[SAVED] {out1}")
plt.close()

# ── Figure 2: AUC summary with OOF annotation ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("AUC Comparison: E5 → KD Student → PCAG Fusion",
             fontsize=13, fontweight="bold")

x     = np.arange(len(TASKS))
width = 0.26

for i, model in enumerate(models):
    vals = [get_metrics(t)[model]["auc"] for t in TASKS]
    bars = ax.bar(x + (i - 1) * width, vals, width,
                  label=model, color=COLORS[model], alpha=0.88, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")

# Annotate OOF AUC for PCAG
for j, task in enumerate(TASKS):
    oof = pcag[task]["oof_auc"]
    ax.annotate(f"OOF={oof:.3f}",
                xy=(x[j] + width, get_metrics(task)["PCAG Fusion"]["auc"] + 0.01),
                fontsize=7, color="#70AD47", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

ax.set_xticks(x)
ax.set_xticklabels(TASK_LABELS, fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_ylabel("AUC", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Highlight MCI_vs_AD PCAG as failed
ax.axvspan(x[2] - 0.45, x[2] + 0.45, alpha=0.07, color="red")
ax.text(x[2], 0.02, "sMRI uninformative\n(CE-only recommended)",
        ha="center", fontsize=7.5, color="red", style="italic")

plt.tight_layout()
out2 = FIG_DIR / "PCAG_AUC_summary.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"[SAVED] {out2}")
plt.close()

# ── Figure 3: Radar chart ─────────────────────────────────────────────
from matplotlib.patches import FancyArrowPatch

radar_metrics  = ["AUC", "Sensitivity", "Specificity", "F1"]
radar_keys     = ["auc",  "sens",        "spec",        "f1"]
N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))
fig.suptitle("PCAG Fusion — Radar Chart per Task", fontsize=13, fontweight="bold")

for ax, task, tlbl in zip(axes, TASKS, TASK_LABELS):
    for model in models:
        m   = get_metrics(task)[model]
        vals = [m[k] for k in radar_keys] + [m[radar_keys[0]]]
        ax.plot(angles, vals, color=COLORS[model], linewidth=2, label=model)
        ax.fill(angles, vals, color=COLORS[model], alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.set_title(tlbl, size=11, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7.5)

plt.tight_layout()
out3 = FIG_DIR / "PCAG_radar_chart.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"[SAVED] {out3}")
plt.close()

# ── Console summary ───────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"{'Task':<12} {'Model':<14} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'F1':>6}")
print(f"{'-'*72}")
for task, tlbl in zip(TASKS, TASK_LABELS):
    for model in models:
        m = get_metrics(task)[model]
        oof_str = f"  (OOF={pcag[task]['oof_auc']:.3f})" if model == "PCAG Fusion" else ""
        print(f"{tlbl if model=='E5 Baseline' else '':<12} {model:<14} "
              f"{m['auc']:>6.3f} {m['sens']:>6.3f} {m['spec']:>6.3f} {m['f1']:>6.3f}{oof_str}")
    print()
