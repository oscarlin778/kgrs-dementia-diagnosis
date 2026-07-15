#!/usr/bin/env python3
"""
Analyze fusion_dim sweep results → plot + update results JSON
Run after run_fusion_dim_sweep.sh completes.
"""
import numpy as np
import json
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")
LOG_DIR = Path("logs/dim_sweep")
SEEDS   = [42, 123, 456, 789, 2024]
TASKS   = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]
DIMS    = [4, 8, 16, 20, 32, 64, 128, 256, 512]

# ── Collect AUCs from result JSON files ───────────────────────────────────
def get_auc_for_dim_task(dim, task):
    """Read test AUC from result JSON files for each seed."""
    aucs = []
    for seed in SEEDS:
        dim_tag  = f"_dim{dim}" if dim != 20 else ""
        seed_tag = f"_s{seed}"  if seed != 42 else ""
        suffix   = f"_v2_fmricombat_nolabel{dim_tag}{seed_tag}"
        jpath    = RESULTS / f"pcag_combat_{task}_results{suffix}.json"
        if jpath.exists():
            with open(jpath) as f:
                d = json.load(f)
            aucs.append(d["test_metrics"]["auc"])
        else:
            # fallback: parse from log
            log = LOG_DIR / f"dim{dim}_{task}_s{seed}.log"
            if log.exists():
                txt = log.read_text()
                m = re.search(r'^\s+auc:\s+([0-9.]+)', txt, re.MULTILINE)
                if m:
                    aucs.append(float(m.group(1)))
    return aucs

# ── Build result table ─────────────────────────────────────────────────────
print("fusion_dim sweep — collecting results...\n")
table = {}  # dim → {task → [aucs]}
for dim in DIMS:
    table[dim] = {}
    for task in TASKS:
        aucs = get_auc_for_dim_task(dim, task)
        table[dim][task] = aucs
        med = np.median(aucs) if aucs else float('nan')
        print(f"  dim={dim:3d}  {task:12s}  n={len(aucs)}  median={med:.4f}  "
              f"seeds={[round(a,3) for a in aucs]}")

# ── Save JSON ──────────────────────────────────────────────────────────────
out = {}
for dim in DIMS:
    out[str(dim)] = {}
    for task in TASKS:
        aucs = table[dim][task]
        out[str(dim)][task] = {
            "seeds": aucs,
            "median": float(np.median(aucs)) if aucs else None,
            "mean":   float(np.mean(aucs))   if aucs else None,
            "std":    float(np.std(aucs))     if aucs else None,
        }
with open(RESULTS / "fusion_dim_sweep.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: results/fusion_dim_sweep.json")

# ── Plot ───────────────────────────────────────────────────────────────────
TASK_COLORS  = {"NC_vs_AD": "#4472C4", "NC_vs_MCI": "#ED7D31", "MCI_vs_AD": "#70AD47"}
TASK_MARKERS = {"NC_vs_AD": "o",       "NC_vs_MCI": "s",       "MCI_vs_AD": "^"}
TASK_LABELS  = {"NC_vs_AD": "NC vs AD", "NC_vs_MCI": "NC vs MCI", "MCI_vs_AD": "MCI vs AD"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Ablation: fusion_dim Trade-off Analysis", fontsize=14, fontweight="bold")

# ── Left: median AUC line chart ────────────────────────────────────────────
ax = axes[0]
for task in TASKS:
    medians = [np.median(table[d][task]) if table[d][task] else np.nan for d in DIMS]
    stds    = [np.std(table[d][task])    if table[d][task] else np.nan for d in DIMS]
    ax.plot(DIMS, medians, color=TASK_COLORS[task], marker=TASK_MARKERS[task],
            linewidth=2, markersize=7, label=TASK_LABELS[task])
    ax.fill_between(DIMS,
                    [m-s for m, s in zip(medians, stds)],
                    [m+s for m, s in zip(medians, stds)],
                    color=TASK_COLORS[task], alpha=0.12)

ax.axvline(20, color='gray', ls='--', lw=1.2, alpha=0.7, label='Default (dim=20)')
ax.set_xlabel("fusion_dim", fontsize=11)
ax.set_ylabel("Test AUC (median ± std, 5 seeds)", fontsize=11)
ax.set_title("Median AUC vs fusion_dim", fontsize=12, fontweight="bold")
ax.set_xticks(DIMS)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ── Right: heatmap (dim × task) ────────────────────────────────────────────
ax2 = axes[1]
data_mat = np.array([
    [np.median(table[d][t]) if table[d][t] else np.nan for t in TASKS]
    for d in DIMS
])
im = ax2.imshow(data_mat, cmap='RdYlGn', aspect='auto',
                vmin=max(0.5, np.nanmin(data_mat)-0.02),
                vmax=min(1.0, np.nanmax(data_mat)+0.02))
ax2.set_xticks(range(len(TASKS)))
ax2.set_xticklabels([t.replace('_vs_', '\nvs\n') for t in TASKS], fontsize=9)
ax2.set_yticks(range(len(DIMS)))
ax2.set_yticklabels([str(d) + (" ←default" if d == 20 else "") for d in DIMS], fontsize=9)
ax2.set_xlabel("Task", fontsize=11)
ax2.set_ylabel("fusion_dim", fontsize=11)
ax2.set_title("AUC Heatmap (median)", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax2, shrink=0.8, label="AUC")
# annotate cells
for i, dim in enumerate(DIMS):
    for j, task in enumerate(TASKS):
        val = data_mat[i, j]
        if not np.isnan(val):
            ax2.text(j, i, f"{val:.3f}", ha='center', va='center',
                     fontsize=8, color='black', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS / "figures/fusion_dim_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/fusion_dim_sweep.png")

# ── Print best dim per task ────────────────────────────────────────────────
print("\n=== Best fusion_dim per task ===")
overall = np.nanmean(data_mat, axis=1)
best_overall = DIMS[int(np.nanargmax(overall))]
for j, task in enumerate(TASKS):
    col = data_mat[:, j]
    best_i = int(np.nanargmax(col))
    print(f"  {task}: best dim={DIMS[best_i]} (AUC={col[best_i]:.4f})")
print(f"  Overall best dim (mean across tasks): {best_overall} (mean AUC={np.nanmax(overall):.4f})")
print(f"  Default dim=20 mean AUC: {np.nanmean(data_mat[DIMS.index(20)]):.4f}")
