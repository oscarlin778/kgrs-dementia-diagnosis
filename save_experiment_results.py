"""
Experiment result saving framework.
Provides functions for saving metrics, ROC curves, diagnosis plots,
confusion matrices, and cross-experiment comparison charts.

Usage:
    import save_experiment_results as ser
    ser.save_metrics("E1_balanced_sampling", metrics_dict)

metrics_dict structure (per-task keyed):
    {
        "NC_vs_AD": {
            "auc": float,
            "acc": float,
            "fpr": list,   # from roc_curve()
            "tpr": list,
        },
        ...
    }
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

RESULTS_DIR = "/home/wei-chi/Data/script/results"
TASKS = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _serialize(v):
    if isinstance(v, dict):
        return {kk: _serialize(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize(i) for i in v]
    return _to_serializable(v)


# ── 1. save_metrics ───────────────────────────────────────────────────────────

def save_metrics(experiment_name: str, metrics_dict: dict):
    """Save metrics_dict to results/{experiment_name}/metrics.json."""
    out_dir = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(_serialize(metrics_dict), f, indent=2)
    print(f"Saved metrics → {path}")


# ── 2. plot_roc_comparison ───────────────────────────────────────────────────

def plot_roc_comparison(experiment_name: str, current_metrics: dict,
                        previous_metrics: dict = None, baseline_metrics: dict = None):
    """
    Plot ROC curves for all tasks side by side.

    Each metrics dict: {"NC_vs_AD": {"fpr": list, "tpr": list, "auc": float}, ...}
    Optionally overlay previous and baseline curves for comparison.
    Saved to results/{experiment_name}/roc_curve.png.
    """
    out_dir = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    tasks = [t for t in TASKS if t in current_metrics]
    if not tasks:
        print("plot_roc_comparison: no matching tasks in current_metrics.")
        return

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        cur = current_metrics[task]
        ax.plot(cur["fpr"], cur["tpr"],
                label=f"Current (AUC={cur['auc']:.3f})", lw=2, color="#2196F3")

        if baseline_metrics and task in baseline_metrics:
            bsl = baseline_metrics[task]
            if "fpr" in bsl and "tpr" in bsl:
                ax.plot(bsl["fpr"], bsl["tpr"], linestyle="--",
                        label=f"Baseline (AUC={bsl.get('auc', bsl.get('GNN_AUC_overall','?')):.3f})",
                        lw=1.5, color="#9E9E9E")

        if previous_metrics and task in previous_metrics:
            prv = previous_metrics[task]
            ax.plot(prv["fpr"], prv["tpr"], linestyle=":",
                    label=f"Previous (AUC={prv['auc']:.3f})", lw=1.5, color="#FF9800")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(task.replace("_", " "))
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle(f"ROC Curves — {experiment_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve → {path}")


# ── 3. plot_diagnosis ────────────────────────────────────────────────────────

def _classify_source(path: str) -> str:
    lp = path.lower()
    return "ADNI" if ("adni" in lp or "old_dswau" in lp) else "TPMIC"


def plot_diagnosis(experiment_name: str, gnn_probs: np.ndarray, resnet_probs: np.ndarray,
                   labels: np.ndarray, matrix_paths: list, subset: str = "TPMIC"):
    """
    4-panel diagnostic plot filtered to the given data source subset.

    Panels: GNN histogram | ResNet histogram | GNN-ResNet scatter | GNN ranking bar
    Saved to results/{experiment_name}/diagnosis_{subset}.png.
    """
    out_dir = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    sources = np.array([_classify_source(p) for p in matrix_paths])
    mask = sources == subset
    n = mask.sum()

    if n == 0:
        print(f"plot_diagnosis: no {subset} subjects found, skipping.")
        return

    gnn_s = gnn_probs[mask]
    res_s = resnet_probs[mask]
    lab_s = labels[mask]
    palette = {0: "steelblue", 1: "tomato"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"GNN Failure Diagnosis ({subset} subset, N={n})",
                 fontsize=14, fontweight="bold")

    # Panel A: GNN histogram
    ax = axes[0, 0]
    for lbl in [0, 1]:
        ax.hist(gnn_s[lab_s == lbl, 1], bins=15, alpha=0.6,
                color=palette[lbl], label=f"class {lbl}")
    ax.set(xlabel="GNN P(positive)", ylabel="Count",
           title="GNN P(positive) by label")
    ax.legend()

    # Panel B: ResNet histogram
    ax = axes[0, 1]
    for lbl in [0, 1]:
        ax.hist(res_s[lab_s == lbl, 1], bins=15, alpha=0.6,
                color=palette[lbl], label=f"class {lbl}")
    ax.set(xlabel="ResNet P(positive)", ylabel="Count",
           title="ResNet P(positive) by label")
    ax.legend()

    # Panel C: GNN vs ResNet scatter
    ax = axes[1, 0]
    colors = [palette[l] for l in lab_s]
    ax.scatter(gnn_s[:, 1], res_s[:, 1], c=colors, alpha=0.7,
               edgecolors="w", linewidths=0.3, s=60)
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8)
    ax.axvline(0.5, color="gray", linestyle="--", lw=0.8)
    ax.set(xlabel="GNN P(positive)", ylabel="ResNet P(positive)",
           title="GNN vs ResNet (blue=class0, red=class1)",
           xlim=(0, 1), ylim=(0, 1))

    # Panel D: Ranked bar
    ax = axes[1, 1]
    order = np.argsort(gnn_s[:, 1])
    bar_colors = [palette[lab_s[i]] for i in order]
    ax.bar(range(n), gnn_s[order, 1], color=bar_colors, width=0.8)
    ax.axhline(0.5, color="black", linestyle="--", lw=0.8)
    ax.set(xlabel="Subject (ranked by GNN score)", ylabel="GNN P(positive)",
           title="GNN ranking (red=class1, blue=class0)")

    plt.tight_layout()
    path = os.path.join(out_dir, f"diagnosis_{subset}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnosis → {path}")


# ── 4. plot_confusion_matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(experiment_name: str, y_true: np.ndarray,
                          y_pred: np.ndarray, task: str):
    """
    Save a confusion matrix heatmap for a single task.
    Saved to results/{experiment_name}/confusion_matrix_{task}.png.
    """
    out_dir = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    _, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(task.replace("_", " "))
    plt.tight_layout()

    safe_task = task.replace(" ", "_")
    path = os.path.join(out_dir, f"confusion_matrix_{safe_task}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix → {path}")


# ── 5. update_comparison_chart ────────────────────────────────────────────────

def update_comparison_chart():
    """
    Read all experiments' metrics.json and plot a cross-experiment AUC bar chart.
    Saved to results/comparison_all_experiments.png.
    """
    experiments = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
        and os.path.exists(os.path.join(RESULTS_DIR, d, "metrics.json"))
    ])

    if not experiments:
        print("update_comparison_chart: no experiments with metrics.json found.")
        return

    task_aucs = {t: {} for t in TASKS}
    for exp in experiments:
        with open(os.path.join(RESULTS_DIR, exp, "metrics.json")) as f:
            m = json.load(f)
        for task in TASKS:
            if task in m:
                entry = m[task]
                auc = entry.get("auc_meta") or entry.get("auc") or 0.0
                task_aucs[task][exp] = float(auc)

    active_tasks = [t for t in TASKS if task_aucs[t]]
    if not active_tasks:
        print("update_comparison_chart: no AUC data found.")
        return

    fig, axes = plt.subplots(1, len(active_tasks),
                             figsize=(5 * len(active_tasks), 5), sharey=True)
    if len(active_tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, active_tasks):
        exp_names = list(task_aucs[task].keys())
        aucs = [task_aucs[task][e] for e in exp_names]
        colors = ["#4C72B0" if "baseline" in e.lower() else "#DD8452"
                  for e in exp_names]
        bars = ax.bar(range(len(exp_names)), aucs, color=colors, width=0.6)
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(task.replace("_", " "), fontsize=11)
        ax.set_ylabel("AUC")
        ax.grid(axis="y", alpha=0.3)
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{auc:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("AUC Comparison Across Experiments", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_all_experiments.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart → {path}")
