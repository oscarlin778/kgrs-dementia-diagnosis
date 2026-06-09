import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────
RESULTS_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results")
E5_METRICS  = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/results/E5_clean/metrics.json")
TEACHER_JSON = RESULTS_DIR / "vitmci_light_finetune_combined_test.json"

TASKS = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]
TASK_LABELS = ["NC vs AD", "NC vs MCI", "MCI vs AD"]

COLORS = ["#4C72B0", "#DD8452", "#55A868"]
MODELS = ["sMRI LightFT", "fMRI E5 GNN", "fMRI KD Student"]

# ── Load Data ──────────────────────────────────────────────────────
with open(TEACHER_JSON, "r") as f:
    teacher_data = json.load(f)["test"]

with open(E5_METRICS, "r") as f:
    e5_data = json.load(f)["test"]

student_data = {}
for task in TASKS:
    # Handle filename differences
    if task == "MCI_vs_AD":
        fname = f"kd_{task}_a10_T3.json"
    else:
        fname = f"kd_{task}_a3_T3.json"
    
    with open(RESULTS_DIR / fname, "r") as f:
        student_data[task] = json.load(f)["student_test"]

# ── Extraction ─────────────────────────────────────────────────────
# Metrics: metrics[row][col][model]
# row 0: AUC, row 1: ACC
metrics = np.zeros((2, 3, 3))

for i, task in enumerate(TASKS):
    t_label = TASK_LABELS[i]
    # Teacher
    metrics[0, i, 0] = teacher_data[t_label]["auc"]
    metrics[1, i, 0] = teacher_data[t_label]["acc"]
    
    # E5 Baseline
    metrics[0, i, 1] = e5_data[task]["auc"]
    metrics[1, i, 1] = e5_data[task]["acc"]
    
    # KD Student
    metrics[0, i, 2] = student_data[task]["auc"]
    metrics[1, i, 2] = student_data[task]["acc"]

# ── Plotting ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
x = np.arange(len(MODELS))
width = 0.6

for row, metric_name in enumerate(["AUC", "Accuracy"]):
    for col, task_label in enumerate(TASK_LABELS):
        ax = axes[row, col]
        vals = metrics[row, col]
        
        bars = ax.bar(x, vals, color=COLORS, width=width, alpha=0.9)
        
        # Labels and formatting
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=15, fontsize=9)
        ax.set_ylim(0, 1.05)
        
        if row == 0:
            ax.set_title(task_label, fontsize=14, fontweight='bold')
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
        if col == 0:
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            
        # Add values on top
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle("Cross-Modal KD: sMRI Teacher → fMRI GNN Student", fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save
OUT_PATH = RESULTS_DIR / "figures" / "Fig_KD_comparison.png"
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"[PLOT] Saved to {OUT_PATH}")
