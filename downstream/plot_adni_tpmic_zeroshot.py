"""
plot_adni_tpmic_zeroshot.py
============================
Generate a publication-ready figure for the ADNI-trained → TPMIC zero-shot evaluation.

Output: results/figures/Fig_ADNI_TPMIC_zeroshot.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Font ─────────────────────────────────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(FONT_PATH).exists():
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = "Noto Sans CJK TC"

plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
})

# ── Data ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results")
FIG_DIR     = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / "eval_adni_to_tpmic_zeroshot.json") as f:
    zs = json.load(f)

TASKS = ["NC vs AD", "NC vs MCI", "MCI vs AD"]
COLORS = {
    "NC vs AD":  "#4C72B0",
    "NC vs MCI": "#DD8452",
    "MCI vs AD": "#55A868",
}

metrics = {}
for t in TASKS:
    m = zs["task_metrics"][t]
    metrics[t] = {
        "auc": m["auc"],
        "acc": m["acc"],
        "f1":  m["f1"],
        "n":   m["n"],
        "cm":  np.array(m["confusion_matrix"]),
    }

# Class label pairs per task
TASK_LABELS = {
    "NC vs AD":  ("NC", "AD"),
    "NC vs MCI": ("NC", "MCI"),
    "MCI vs AD": ("MCI", "AD"),
}

# ── Layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("white")

# Grid: 2 rows × 4 cols  (row0=bar charts/summary, row1=confusion matrices)
gs = fig.add_gridspec(2, 4, hspace=0.48, wspace=0.38,
                      left=0.07, right=0.97, top=0.90, bottom=0.08)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Zero-Shot Transfer: ADNI-Trained Model Evaluated on TPMIC (n=136)",
    fontsize=14, fontweight="bold", y=0.97,
)

# ── Panel A: AUC bar chart ────────────────────────────────────────────────────
ax_auc = fig.add_subplot(gs[0, :2])

aucs = [metrics[t]["auc"] for t in TASKS]
bars = ax_auc.bar(
    TASKS, aucs,
    color=[COLORS[t] for t in TASKS],
    width=0.5, zorder=3, edgecolor="white", linewidth=0.8,
)
ax_auc.axhline(0.5, color="grey", linestyle="--", linewidth=1.2, zorder=2, label="Chance (0.5)")
ax_auc.set_ylim(0, 1.0)
ax_auc.set_ylabel("AUC")
ax_auc.set_title("A  AUC per Task", loc="left", fontweight="bold")
ax_auc.set_xticks(range(len(TASKS)))
ax_auc.set_xticklabels(TASKS)
ax_auc.legend(fontsize=9, loc="upper right")
ax_auc.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
ax_auc.set_axisbelow(True)

for bar, auc in zip(bars, aucs):
    ax_auc.text(bar.get_x() + bar.get_width() / 2, auc + 0.02,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ── Panel B: ACC + F1 grouped bar ─────────────────────────────────────────────
ax_acc = fig.add_subplot(gs[0, 2:])

x = np.arange(len(TASKS))
w = 0.3
accs = [metrics[t]["acc"] for t in TASKS]
f1s  = [metrics[t]["f1"]  for t in TASKS]

b1 = ax_acc.bar(x - w/2, accs, w, label="Accuracy",
                color=[COLORS[t] for t in TASKS], alpha=0.9, zorder=3)
b2 = ax_acc.bar(x + w/2, f1s,  w, label="F1-Score",
                color=[COLORS[t] for t in TASKS], alpha=0.45, zorder=3,
                edgecolor=[COLORS[t] for t in TASKS], linewidth=1.2)

ax_acc.set_ylim(0, 1.0)
ax_acc.set_ylabel("Score")
ax_acc.set_title("B  Accuracy & F1-Score per Task", loc="left", fontweight="bold")
ax_acc.set_xticks(x)
ax_acc.set_xticklabels(TASKS)
ax_acc.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
ax_acc.set_axisbelow(True)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="grey",             alpha=0.9, label="Accuracy"),
    Patch(facecolor="grey", edgecolor="dimgrey", alpha=0.45, label="F1-Score"),
]
ax_acc.legend(handles=legend_handles, fontsize=9, loc="upper right")

for b, v in zip(b1, accs):
    ax_acc.text(b.get_x() + b.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
for b, v in zip(b2, f1s):
    ax_acc.text(b.get_x() + b.get_width()/2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)

# ── Panels C / D / E: Confusion matrices ─────────────────────────────────────
CM_TITLES = ["C", "D", "E"]

for col_idx, task in enumerate(TASKS):
    ax_cm = fig.add_subplot(gs[1, col_idx])
    cm = metrics[task]["cm"]
    cls0, cls1 = TASK_LABELS[task]

    im = ax_cm.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

    # Annotations
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / cm[i].sum() * 100
            color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
            ax_cm.text(j, i, f"{val}\n({pct:.0f}%)",
                       ha="center", va="center", fontsize=11,
                       fontweight="bold", color=color)

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels([f"Pred {cls0}", f"Pred {cls1}"], fontsize=9)
    ax_cm.set_yticklabels([f"True {cls0}", f"True {cls1}"], fontsize=9, rotation=90, va="center")
    title_letter = CM_TITLES[col_idx]
    ax_cm.set_title(
        f"{title_letter}  {task}\n"
        f"AUC={metrics[task]['auc']:.3f}  n={metrics[task]['n']}",
        loc="left", fontweight="bold", fontsize=10,
    )
    ax_cm.spines[:].set_visible(False)

# ── Panel F: Summary text box ─────────────────────────────────────────────────
ax_txt = fig.add_subplot(gs[1, 3])
ax_txt.axis("off")

lines = [
    "Dataset Summary",
    "─" * 22,
    "Source:   TPMIC (Taiwan)",
    "Total:    136 subjects",
    "  NC:  42   MCI: 71   AD: 23",
    "",
    "Task        AUC    ACC    F1",
    "─" * 28,
]
for t in TASKS:
    m = metrics[t]
    lines.append(f"{t:<11} {m['auc']:.3f}  {m['acc']:.3f}  {m['f1']:.3f}")

lines += [
    "",
    "Note: Model trained on ADNI only.",
    "Strong NC-bias observed in all",
    "tasks (domain shift effect).",
]

ax_txt.text(0.02, 0.98, "\n".join(lines),
            transform=ax_txt.transAxes,
            va="top", ha="left", fontsize=8.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7F7F7",
                      edgecolor="#CCCCCC", linewidth=1))

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = FIG_DIR / "Fig_ADNI_TPMIC_zeroshot.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"[SAVED] {out_path}")
plt.close(fig)
