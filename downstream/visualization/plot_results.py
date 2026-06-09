"""
plot_results.py — Generate publication-ready figures from results/
Run: python3 plot_results.py
Outputs saved to: results/figures/
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.font_manager as fm

# ── Font: prefer Noto Sans CJK TC (has CJK) ──────────────────────────────
_cjk_font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_cjk_font)
    _prop = fm.FontProperties(fname=_cjk_font)
    matplotlib.rcParams["font.family"] = _prop.get_name()
except Exception:
    pass   # fall back to default

matplotlib.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "figure.dpi":         100,
})

RESULTS_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

TASKS  = ["NC vs AD", "NC vs MCI", "MCI vs AD"]
COLORS = {"NC vs AD": "#2196F3", "NC vs MCI": "#FF9800", "MCI vs AD": "#4CAF50"}


def load(fname):
    with open(RESULTS_DIR / fname) as f:
        return json.load(f)


def get_task_aucs(data, key="task_metrics"):
    tm = data.get(key, data)
    return [tm.get(t, {}).get("auc", float("nan")) for t in TASKS]


def grouped_bar(ax, labels, data_dict, title, ylabel="AUC",
                ylim=(0.3, 1.05), hline=0.5, note="", val_rot=90):
    n_groups = len(labels)
    width = 0.22
    x = np.arange(n_groups)

    for i, task in enumerate(TASKS):
        vals = data_dict[task]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=task, color=COLORS[task], alpha=0.87,
                      edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=val_rot)

    if hline is not None:
        ax.axhline(hline, color="gray", linestyle=":", linewidth=1.3,
                   alpha=0.75, label=f"Chance ({hline:.1f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    if note:
        ax.text(0.99, 0.02, note, transform=ax.transAxes,
                fontsize=7, ha="right", va="bottom",
                color="#555555", style="italic")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1  System model evolution — TPMIC test (n=27)
# ═══════════════════════════════════════════════════════════════════════════
def fig1_system_evolution():
    exps = [
        ("eval_combined_tpmic_test.json",             "Combined\nModel"),
        ("eval_finetune_tpmic_test.json",              "Finetune\nv1"),
        ("eval_finetune_full_tpmic_test.json",         "Finetune\nFull"),
        ("eval_soft_fusion_combined_tpmic_test.json",  "Soft Fusion\n(Combined)"),
        ("eval_mciad_soft_fusion_tpmic_test.json",     "Soft Fusion\n(MCI+AD)"),
    ]

    labels = [e[1] for e in exps]
    data_dict = {t: [] for t in TASKS}
    ovo_accs  = []

    for fname, _ in exps:
        d = load(fname)
        for i, t in enumerate(TASKS):
            data_dict[t].append(get_task_aucs(d)[i])
        ovo_accs.append(d["ovo_analysis"]["ovo_accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("System Pipeline Evolution — TPMIC Test Set (n=27)",
                 fontsize=13, fontweight="bold")

    grouped_bar(ax1, labels, data_dict,
                title="OVO Binary Classification AUC",
                note="Higher is better  |  chance=0.5")

    # 3-class OVO accuracy
    palette = ["#607D8B", "#795548", "#9C27B0", "#E91E63", "#00BCD4"]
    bars = ax2.bar(range(len(labels)), ovo_accs,
                   color=palette[:len(labels)], alpha=0.87, edgecolor="white")
    for bar, val in zip(bars, ovo_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)

    ax2.axhline(1/3, color="gray", linestyle=":", linewidth=1.3,
                alpha=0.75, label="Chance (0.333)")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylim(0, 0.75)
    ax2.set_ylabel("3-class OVO Accuracy", fontsize=10)
    ax2.set_title("Three-class OVO Overall Accuracy", fontsize=11, fontweight="bold", pad=8)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "Fig1_system_evolution_tpmic.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2  BrainIAC frozen feature extractor comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig2_brainiac():
    vitmci       = load("eval_vitmci_tpmic_test.json")
    brainiac_te  = load("brainiac_test_predictions.json")
    brainiac_oof = load("brainiac_oof_predictions.json")

    resnet_oof = {"NC vs AD": 0.771, "NC vs MCI": 0.731, "MCI vs AD": 0.775}

    exps = {
        "ResNet50\n(OOF baseline)":       [resnet_oof[t] for t in TASKS],
        "BrainIAC.ckpt\n(OOF, n=164)":   [brainiac_oof[t]["auc"] for t in TASKS],
        "BrainIAC.ckpt\n(Test, n=27)":   [brainiac_te[t]["auc"]  for t in TASKS],
        "vit_mci.ckpt\n(OOF, n=164)":    [vitmci["oof"][t]["auc"] for t in TASKS],
        "vit_mci.ckpt\n(Test, n=27)":    [vitmci["test"][t]["auc"] for t in TASKS],
    }

    labels    = list(exps.keys())
    data_dict = {t: [exps[l][i] for l in labels] for i, t in enumerate(TASKS)}

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle("BrainIAC sMRI Frozen Feature Extractor — OVO AUC",
                 fontsize=13, fontweight="bold")

    grouped_bar(ax, labels, data_dict,
                title="Binary OVO AUC (LogReg on 768-d features)",
                note="OOF = 5-fold CV on combined_train (n=164)  |  Test = TPMIC (n=27)")

    # dividers
    ax.axvline(0.5, color="#999", linestyle="--", alpha=0.45, linewidth=1)
    ax.axvline(2.5, color="#999", linestyle="--", alpha=0.45, linewidth=1)
    for xpos, txt in [(0, "ResNet\nBaseline"), (1.5, "BrainIAC.ckpt"), (3.5, "vit_mci.ckpt")]:
        ax.text(xpos, 1.02, txt, transform=ax.get_xaxis_transform(),
                fontsize=7.5, ha="center", color="#444", style="italic")

    plt.tight_layout()
    out = FIG_DIR / "Fig2_brainiac_smri.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3  System v10 vs Zero-shot
# ═══════════════════════════════════════════════════════════════════════════
def fig3_system_v10():
    v10      = load("system_eval_v10.json")
    zeroshot = load("eval_adni_to_tpmic_zeroshot.json")

    zero_aucs = get_task_aucs(zeroshot)
    v10_aucs  = get_task_aucs(v10)
    zero_acc  = zeroshot["ovo_analysis"]["ovo_accuracy"]
    v10_acc   = v10["ovo_analysis"]["ovo_accuracy"]

    labels    = ["ADNI Zero-shot\n(n=136)", "System v10\n(n=74, TPMIC+ADNI)"]
    data_dict = {t: [zero_aucs[i], v10_aucs[i]] for i, t in enumerate(TASKS)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Full System Evaluation — Zero-shot vs. Finetuned (v10)",
                 fontsize=13, fontweight="bold")

    grouped_bar(ax1, labels, data_dict,
                title="OVO Binary AUC",
                note="Zero-shot = ADNI-trained only (no TPMIC fine-tuning)")

    accs = [zero_acc, v10_acc]
    bars = ax2.bar(range(2), accs,
                   color=["#607D8B", "#E91E63"], alpha=0.87, edgecolor="white")
    for bar, val in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=10)
    ax2.axhline(1/3, color="gray", linestyle=":", linewidth=1.3, alpha=0.75,
                label="Chance (0.333)")
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 0.9)
    ax2.set_ylabel("3-class OVO Accuracy", fontsize=10)
    ax2.set_title("Three-class OVO Accuracy", fontsize=11, fontweight="bold", pad=8)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "Fig3_system_v10_zeroshot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4  Modality breakdown (System v10)
# ═══════════════════════════════════════════════════════════════════════════
def fig4_modality():
    v10 = load("system_eval_v10.json")
    ms  = v10.get("modality_stratified", {})

    modalities = ["Dual-Modal", "fMRI only", "sMRI only"]
    mod_labels = ["Dual-Modal\n(fMRI+sMRI)", "fMRI Only", "sMRI Only"]

    data_dict = {t: [] for t in TASKS}
    ns = []

    for mod in modalities:
        m = ms.get(mod, {})
        # structure: {task: {"n": int, "auc": float, ...}}
        first_task = m.get("NC vs AD", {})
        n = first_task.get("n", 0)
        ns.append(n)
        for t in TASKS:
            val = m.get(t, {}).get("auc", float("nan"))
            data_dict[t].append(val)

    if all(n == 0 for n in ns):
        print("[SKIP] Fig4: no modality_stratified data (all n=0)")
        return

    mod_labels_n = [f"{ml}\n(n={n})" for ml, n in zip(mod_labels, ns)]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("System v10 — Modality-stratified AUC Analysis",
                 fontsize=13, fontweight="bold")

    grouped_bar(ax, mod_labels_n, data_dict,
                title="OVO AUC by Modality Group",
                note="System v10  |  n=74 total")

    plt.tight_layout()
    out = FIG_DIR / "Fig4_modality_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5  Summary heatmap (all experiments)
# ═══════════════════════════════════════════════════════════════════════════
def fig5_summary_heatmap():
    rows = [
        # (label,                                   NC/AD,  NC/MCI, MCI/AD,  category)
        ("ResNet50 (OOF baseline)",                 0.771,  0.731,  0.775,  "sMRI Feature"),
        ("BrainIAC.ckpt (OOF, n=164)",             0.551,  0.593,  0.500,  "sMRI Feature"),
        ("vit_mci.ckpt (OOF, n=164)",              0.679,  0.598,  0.561,  "sMRI Feature"),
        ("BrainIAC.ckpt (Test, n=27)",             0.938,  0.683,  0.567,  "sMRI Feature"),
        ("vit_mci.ckpt (Test, n=27)",              0.688,  0.458,  0.567,  "sMRI Feature"),
        ("Combined Model (Test, n=27)",            0.875,  0.483,  0.633,  "Full System"),
        ("Finetune v1 (Test, n=27)",               0.688,  0.483,  0.633,  "Full System"),
        ("Finetune Full (Test, n=27)",             0.906,  0.483,  0.633,  "Full System"),
        ("Soft Fusion Combined (Test, n=27)",      0.875,  0.517,  0.633,  "Full System"),
        ("Zero-shot ADNI→TPMIC (n=136)",           0.752,  0.566,  0.553,  "Full System"),
        ("System v10 (n=74, TPMIC+ADNI)",          0.956,  0.593,  0.684,  "Full System"),
    ]

    matrix   = np.array([[r[1], r[2], r[3]] for r in rows])
    ylabels  = [r[0] for r in rows]
    cats     = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.suptitle("All Experiments — OVO AUC Summary", fontsize=13, fontweight="bold")

    cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0.40, vmax=1.0)
    im   = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(len(rows)):
        for j in range(3):
            val = matrix[i, j]
            txt_col = "black" if 0.50 < val < 0.88 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=txt_col)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["NC vs AD", "NC vs MCI", "MCI vs AD"], fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylabels, fontsize=8.5)

    # right axis — category labels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels(cats, fontsize=7.5, color="#666")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.18)
    cbar.set_label("AUC", fontsize=9)

    # divider between sMRI Feature / Full System
    ax.axhline(4.5, color="black", linewidth=1.5, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = FIG_DIR / "Fig5_summary_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6  Learning curve (finetune fraction) — placeholder note
# ═══════════════════════════════════════════════════════════════════════════
def fig6_finetune_curve():
    """
    finetune_frac02~08 results are all identical (known bug — fraction param
    was not applied correctly). Skipping this figure; a note is printed instead.
    """
    print("[SKIP] Fig6_finetune_curve: frac02~08 results identical (bug — data invalid)")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_system_evolution()
    fig2_brainiac()
    fig3_system_v10()
    fig4_modality()
    fig5_summary_heatmap()
    fig6_finetune_curve()
    print(f"\nAll figures saved to: {FIG_DIR}")
