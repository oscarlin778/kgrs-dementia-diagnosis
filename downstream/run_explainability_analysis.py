#!/usr/bin/env python3
"""
Task 3-A / 3-B / 3-C: Explainability Analysis
- 3-A: fMRI GAT attention → top ROIs per task
- 3-B: sMRI BrainIAC attention → top ROIs per diagnosis group
- 3-C: Modality complementarity scatter + t-SNE proxy
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from sklearn.manifold import TSNE
import os

RESULTS = "results"
os.makedirs(f"{RESULTS}/figures", exist_ok=True)

# ── AAL116 region names ─────────────────────────────────────────────────────
with open("/tmp/aal116_names.json") as f:
    AAL_NAMES = json.load(f)

# DMN ROI indices in AAL116 (0-indexed)
DMN_ROIS = [
    AAL_NAMES.index(r) for r in [
        "Frontal_Med_Orb_L","Frontal_Med_Orb_R","Angular_L","Angular_R",
        "Precuneus_L","Precuneus_R","Cingulum_Post_L","Cingulum_Post_R",
        "Hippocampus_L","Hippocampus_R","Parahippocampal_L","Parahippocampal_R",
    ] if r in AAL_NAMES
]
HIPPO_ROIS = [
    AAL_NAMES.index(r) for r in [
        "Hippocampus_L","Hippocampus_R","Parahippocampal_L","Parahippocampal_R",
        "Amygdala_L","Amygdala_R"
    ] if r in AAL_NAMES
]

LABEL_MAP  = {0: "NC", 1: "MCI", 2: "AD"}
TASK_PAIRS = {
    "NC_vs_AD":  (0, 2),
    "NC_vs_MCI": (0, 1),
    "MCI_vs_AD": (1, 2),
}
TASK_IDX   = {"NC_vs_AD": 0, "NC_vs_MCI": 1, "MCI_vs_AD": 2}

# ═══════════════════════════════════════════════════════════════════════════
# TASK 3-A: fMRI GAT Attention → top ROIs per task
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 3-A: fMRI GAT Attention Analysis")
print("="*60)

gat = np.load(f"{RESULTS}/gat_attention_v2_nolabel.npz", allow_pickle=True)
roi_imp   = gat["roi_importance"]   # (49, 3, 116)
net_att   = gat["net_attention"]    # (49, 3, 9)
top_rois  = gat["top_rois"]         # (49, 3, 10)
sub_ids   = gat["subject_ids"]
labels    = gat["labels"]
tasks     = list(gat["tasks"])
net_names = list(gat["network_labels"])

task_results = {}
for task_name, (pos_label, neg_label) in TASK_PAIRS.items():
    tidx = TASK_IDX[task_name]

    # subjects in this task (only positive and negative class)
    mask = (labels == pos_label) | (labels == neg_label)
    pos_mask = labels == pos_label
    neg_mask = labels == neg_label

    # mean attention across ALL subjects in this task
    mean_all  = roi_imp[mask, tidx, :].mean(axis=0)
    mean_pos  = roi_imp[pos_mask, tidx, :].mean(axis=0)
    mean_neg  = roi_imp[neg_mask, tidx, :].mean(axis=0)
    # discriminative = difference in attention between positive and negative
    disc = mean_pos - mean_neg

    top10_all  = np.argsort(mean_all)[::-1][:10]
    top10_disc = np.argsort(np.abs(disc))[::-1][:10]

    # network-level attention
    net_pos = net_att[pos_mask, tidx, :].mean(axis=0)
    net_neg = net_att[neg_mask, tidx, :].mean(axis=0)

    # check DMN presence in top10
    dmn_in_top10 = [i for i in top10_all if i in DMN_ROIS]
    hippo_in_top10 = [i for i in top10_all if i in HIPPO_ROIS]

    task_results[task_name] = {
        "top10_all": [(AAL_NAMES[i], float(mean_all[i])) for i in top10_all],
        "top10_discriminative": [(AAL_NAMES[i], float(disc[i])) for i in top10_disc],
        "dmn_rois_in_top10": [AAL_NAMES[i] for i in dmn_in_top10],
        "hippo_rois_in_top10": [AAL_NAMES[i] for i in hippo_in_top10],
        "network_attention_pos": {net_names[i]: float(net_pos[i]) for i in range(len(net_names))},
        "network_attention_neg": {net_names[i]: float(net_neg[i]) for i in range(len(net_names))},
    }

    print(f"\n[{task_name}] Top-10 ROIs (mean attention):")
    for name, score in task_results[task_name]["top10_all"]:
        tag = " ← DMN" if name in [AAL_NAMES[i] for i in DMN_ROIS] else ""
        tag += " ← Hippo" if name in [AAL_NAMES[i] for i in HIPPO_ROIS] else ""
        print(f"  {name}: {score:.4f}{tag}")
    print(f"  DMN in top10: {task_results[task_name]['dmn_rois_in_top10']}")
    print(f"  Hippo in top10: {task_results[task_name]['hippo_rois_in_top10']}")

# ── Plot: Top-10 ROIs per task (bar chart) ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("fMRI GAT Attention: Top-10 ROIs per Classification Task", fontsize=14, fontweight="bold")

colors_task = {"NC_vs_AD": "#4472C4", "NC_vs_MCI": "#ED7D31", "MCI_vs_AD": "#70AD47"}

for ax, (task_name, res) in zip(axes, task_results.items()):
    rois   = [r[0].replace("_L","(L)").replace("_R","(R)") for r in res["top10_all"]]
    scores = [r[1] for r in res["top10_all"]]
    # highlight DMN and Hippo
    bar_colors = []
    for name, _ in res["top10_all"]:
        orig = name
        if orig in [AAL_NAMES[i] for i in DMN_ROIS]:
            bar_colors.append("#C00000")   # red = DMN
        elif orig in [AAL_NAMES[i] for i in HIPPO_ROIS]:
            bar_colors.append("#7030A0")   # purple = Hippo
        else:
            bar_colors.append(colors_task[task_name])

    bars = ax.barh(rois[::-1], scores[::-1], color=bar_colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean GAT Attention Weight", fontsize=10)
    ax.set_title(f"{task_name}", fontsize=12, fontweight="bold")
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlim(0, max(scores)*1.2)

# legend
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="#C00000", label="DMN ROI"),
    Patch(facecolor="#7030A0", label="Hippocampal ROI"),
    Patch(facecolor="gray",    label="Other ROI"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{RESULTS}/figures/task3a_fmri_top_rois.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[3-A] Saved: results/figures/task3a_fmri_top_rois.png")

# ── Network-level radar chart ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))
fig.suptitle("Network-Level GAT Attention: Positive vs Negative Class", fontsize=13, fontweight="bold")

angles = np.linspace(0, 2*np.pi, len(net_names), endpoint=False).tolist()
angles += angles[:1]

for ax, (task_name, res) in zip(axes, task_results.items()):
    pos_lab, neg_lab = TASK_PAIRS[task_name]
    vals_pos = [res["network_attention_pos"][n] for n in net_names]
    vals_neg = [res["network_attention_neg"][n] for n in net_names]
    vals_pos += vals_pos[:1]
    vals_neg += vals_neg[:1]

    ax.plot(angles, vals_pos, 'r-', linewidth=2, label=LABEL_MAP[pos_lab])
    ax.fill(angles, vals_pos, alpha=0.15, color='red')
    ax.plot(angles, vals_neg, 'b-', linewidth=2, label=LABEL_MAP[neg_lab])
    ax.fill(angles, vals_neg, alpha=0.15, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(net_names, fontsize=8)
    ax.set_title(task_name, fontsize=11, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

plt.tight_layout()
plt.savefig(f"{RESULTS}/figures/task3a_fmri_network_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("[3-A] Saved: results/figures/task3a_fmri_network_radar.png")

with open(f"{RESULTS}/task3a_fmri_analysis.json", "w") as f:
    json.dump(task_results, f, indent=2)
print("[3-A] Saved: results/task3a_fmri_analysis.json")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 3-B: sMRI BrainIAC Attention → top ROIs per group
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 3-B: sMRI BrainIAC Attention Analysis")
print("="*60)

bri = np.load(f"{RESULTS}/brainiac_roi_v2.npz", allow_pickle=True)
roi_att  = bri["roi_attention"]   # (49, 116)
s_labels = bri["labels"]          # (49,)  0=NC,1=MCI,2=AD
s_subs   = bri["subject_ids"]
missing  = bri["missing"]         # (49,) bool

# known AD-atrophy ROIs
AD_ATROPHY_ROIS = [
    AAL_NAMES.index(r) for r in [
        "Hippocampus_L","Hippocampus_R","Parahippocampal_L","Parahippocampal_R",
        "Amygdala_L","Amygdala_R","Cingulum_Post_L","Cingulum_Post_R",
        "Angular_L","Angular_R","Precuneus_L","Precuneus_R",
        "Temporal_Inf_L","Temporal_Inf_R","Temporal_Mid_L","Temporal_Mid_R",
    ] if r in AAL_NAMES
]

smri_results = {}
for grp_label, grp_name in LABEL_MAP.items():
    mask = (s_labels == grp_label) & (~missing)
    if mask.sum() == 0:
        continue
    mean_att = roi_att[mask, :].mean(axis=0)
    top10_idx = np.argsort(mean_att)[::-1][:10]
    atrophy_in_top10 = [i for i in top10_idx if i in AD_ATROPHY_ROIS]

    smri_results[grp_name] = {
        "n_subjects": int(mask.sum()),
        "top10_rois": [(AAL_NAMES[i], float(mean_att[i])) for i in top10_idx],
        "atrophy_rois_in_top10": [AAL_NAMES[i] for i in atrophy_in_top10],
    }

    print(f"\n[sMRI - {grp_name}] Top-10 ROIs (n={mask.sum()}):")
    for name, score in smri_results[grp_name]["top10_rois"]:
        tag = " ← AD-atrophy" if name in [AAL_NAMES[i] for i in AD_ATROPHY_ROIS] else ""
        print(f"  {name}: {score:.4f}{tag}")
    print(f"  AD-atrophy ROIs in top10: {smri_results[grp_name]['atrophy_rois_in_top10']}")

# discriminative between AD and NC
nc_mask = (s_labels == 0) & (~missing)
ad_mask = (s_labels == 2) & (~missing)
mci_mask = (s_labels == 1) & (~missing)
mean_nc = roi_att[nc_mask, :].mean(axis=0)
mean_ad = roi_att[ad_mask, :].mean(axis=0)
mean_mci = roi_att[mci_mask, :].mean(axis=0)

disc_ad_vs_nc = mean_ad - mean_nc
top10_disc = np.argsort(disc_ad_vs_nc)[::-1][:10]
smri_results["discriminative_AD_vs_NC"] = {
    "top10_rois": [(AAL_NAMES[i], float(disc_ad_vs_nc[i])) for i in top10_disc],
}
print("\n[sMRI - AD vs NC discriminative ROIs (AD attention > NC):]")
for name, score in smri_results["discriminative_AD_vs_NC"]["top10_rois"]:
    tag = " ← AD-atrophy" if name in [AAL_NAMES[i] for i in AD_ATROPHY_ROIS] else ""
    print(f"  {name}: {score:+.4f}{tag}")

# ── Plot: side-by-side comparison NC/MCI/AD top-10 ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("sMRI BrainIAC Attention: Top-10 ROIs per Diagnostic Group", fontsize=14, fontweight="bold")

grp_colors = {"NC": "#4472C4", "MCI": "#ED7D31", "AD": "#C00000"}

for ax, (grp_name, res) in zip(axes, [(k, smri_results[k]) for k in ["NC","MCI","AD"]]):
    rois   = [r[0].replace("_L","(L)").replace("_R","(R)") for r in res["top10_rois"]]
    scores = [r[1] for r in res["top10_rois"]]
    bar_colors = []
    for name, _ in res["top10_rois"]:
        if name in [AAL_NAMES[i] for i in AD_ATROPHY_ROIS]:
            bar_colors.append("#C00000" if grp_name != "NC" else "#FF6B6B")
        else:
            bar_colors.append(grp_colors[grp_name])

    ax.barh(rois[::-1], scores[::-1], color=bar_colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean BrainIAC Attention", fontsize=10)
    ax.set_title(f"{grp_name} (n={res['n_subjects']})", fontsize=12, fontweight="bold")
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlim(0, max(scores)*1.2 if scores else 1)

legend_els = [
    Patch(facecolor="#C00000", label="AD-atrophy region (AD/MCI)"),
    Patch(facecolor="#FF6B6B", label="AD-atrophy region (NC)"),
    Patch(facecolor="gray",    label="Other region"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{RESULTS}/figures/task3b_smri_top_rois.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[3-B] Saved: results/figures/task3b_smri_top_rois.png")

with open(f"{RESULTS}/task3b_smri_analysis.json", "w") as f:
    json.dump(smri_results, f, indent=2)
print("[3-B] Saved: results/task3b_smri_analysis.json")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 3-C: Complementarity Analysis + t-SNE proxy
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 3-C: Fusion Complementarity Analysis")
print("="*60)

pred = np.load(f"{RESULTS}/full_predictions_v2_nolabel.npz", allow_pickle=True)
fused_probs = pred["fused_probs"]   # (49, 3)
fmri_probs  = pred["fmri_probs"]    # (49, 3)
smri_probs  = pred["smri_probs"]    # (49, 3)
pred_labels = pred["labels"]        # (49,)
pred_sites  = pred["sites"]         # (49,)
pred_subs   = pred["subject_ids"]

# for each task, classify per modality and fusion at threshold 0.5
comp_stats = {}
for task_name, (pos_l, neg_l) in TASK_PAIRS.items():
    tidx = TASK_IDX[task_name]
    mask = (pred_labels == pos_l) | (pred_labels == neg_l)
    fp  = fused_probs[mask, tidx]
    fmp = fmri_probs[mask, tidx]
    smp = smri_probs[mask, tidx]
    lbl = (pred_labels[mask] == pos_l).astype(int)

    thr = 0.5
    fused_pred = (fp  >= thr).astype(int)
    fmri_pred  = (fmp >= thr).astype(int)
    smri_pred  = (smp >= thr).astype(int)

    fused_cor = (fused_pred == lbl)
    fmri_cor  = (fmri_pred  == lbl)
    smri_cor  = (smri_pred  == lbl)

    # complementarity cases
    both_wrong_fused_right = (~fmri_cor) & (~smri_cor) & fused_cor
    fmri_wrong_smri_right_fused_right = (~fmri_cor) & smri_cor & fused_cor
    smri_wrong_fmri_right_fused_right = fmri_cor & (~smri_cor) & fused_cor
    fusion_rescued = fmri_wrong_smri_right_fused_right | smri_wrong_fmri_right_fused_right

    n = mask.sum()
    comp_stats[task_name] = {
        "n": int(n),
        "fmri_only_acc": float(fmri_cor.mean()),
        "smri_only_acc": float(smri_cor.mean()),
        "fusion_acc": float(fused_cor.mean()),
        "fusion_rescued": int(fusion_rescued.sum()),
        "pct_rescued": float(fusion_rescued.mean()*100),
        "fmri_rescued_by_smri": int(fmri_wrong_smri_right_fused_right.sum()),
        "smri_rescued_by_fmri": int(smri_wrong_fmri_right_fused_right.sum()),
    }
    print(f"\n[{task_name}] n={n}")
    print(f"  fMRI-only accuracy: {comp_stats[task_name]['fmri_only_acc']:.3f}")
    print(f"  sMRI-only accuracy: {comp_stats[task_name]['smri_only_acc']:.3f}")
    print(f"  Fusion accuracy:    {comp_stats[task_name]['fusion_acc']:.3f}")
    print(f"  Fusion rescued {fusion_rescued.sum()} cases ({fusion_rescued.mean()*100:.1f}%)")
    print(f"    → fMRI wrong, sMRI correct, fusion correct: {fmri_wrong_smri_right_fused_right.sum()}")
    print(f"    → sMRI wrong, fMRI correct, fusion correct: {smri_wrong_fmri_right_fused_right.sum()}")

with open(f"{RESULTS}/task3c_complementarity.json", "w") as f:
    json.dump(comp_stats, f, indent=2)
print("\n[3-C] Saved: results/task3c_complementarity.json")

# ── Plot 1: Stacked accuracy comparison ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(3)
task_names = list(TASK_PAIRS.keys())
w = 0.25

fmri_accs  = [comp_stats[t]["fmri_only_acc"] for t in task_names]
smri_accs  = [comp_stats[t]["smri_only_acc"] for t in task_names]
fused_accs = [comp_stats[t]["fusion_acc"] for t in task_names]

ax.bar(x - w, fmri_accs,  width=w, label="fMRI only",  color="#4472C4", edgecolor="white")
ax.bar(x,     smri_accs,  width=w, label="sMRI only",  color="#ED7D31", edgecolor="white")
ax.bar(x + w, fused_accs, width=w, label="Fusion",     color="#70AD47", edgecolor="white")

for i, t in enumerate(task_names):
    rescued_pct = comp_stats[t]["pct_rescued"]
    ax.annotate(f"+{rescued_pct:.0f}%\nrescued", xy=(i+w, fused_accs[i]+0.01),
                ha='center', va='bottom', fontsize=8, color='#375623', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(task_names, fontsize=11)
ax.set_ylabel("Accuracy (threshold=0.5)", fontsize=11)
ax.set_ylim(0, 1.15)
ax.axhline(0.5, ls='--', color='gray', alpha=0.5, label='Chance')
ax.legend(fontsize=10)
ax.set_title("Modality Complementarity: fMRI + sMRI → Fusion", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS}/figures/task3c_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[3-C] Saved: results/figures/task3c_accuracy_comparison.png")

# ── Plot 2: fMRI vs sMRI probability scatter (NC_vs_MCI as example) ──────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Modal Complementarity: fMRI vs sMRI Probability Space\n(Fusion outcome shown by marker)", fontsize=13, fontweight="bold")

for ax, task_name in zip(axes, task_names):
    pos_l, neg_l = TASK_PAIRS[task_name]
    tidx = TASK_IDX[task_name]
    mask = (pred_labels == pos_l) | (pred_labels == neg_l)
    fp   = fused_probs[mask, tidx]
    fmp  = fmri_probs[mask, tidx]
    smp  = smri_probs[mask, tidx]
    lbl  = (pred_labels[mask] == pos_l).astype(int)

    fused_pred = (fp >= 0.5).astype(int)
    correct    = (fused_pred == lbl)

    # color by true label + correct/wrong
    colors = []
    markers = []
    for lab, cor in zip(lbl, correct):
        if lab == 1:  # positive class
            colors.append("#C00000" if cor else "#FF9999")
            markers.append("o")
        else:         # negative class
            colors.append("#4472C4" if cor else "#9DC3E6")
            markers.append("s")

    for xi, yi, c, m in zip(fmp, smp, colors, markers):
        ax.scatter(xi, yi, c=c, marker=m, s=60, alpha=0.8, edgecolors='white', linewidths=0.5)

    ax.axhline(0.5, ls='--', color='gray', alpha=0.4, lw=1)
    ax.axvline(0.5, ls='--', color='gray', alpha=0.4, lw=1)
    ax.set_xlabel("fMRI probability", fontsize=10)
    ax.set_ylabel("sMRI probability", fontsize=10)
    ax.set_title(f"{task_name}", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # quadrant annotations
    ax.text(0.1,  0.92, "fMRI↓\nsMRI↑", fontsize=7, color='gray', ha='center')
    ax.text(0.9,  0.92, "fMRI↑\nsMRI↑", fontsize=7, color='gray', ha='center')
    ax.text(0.1,  0.08, "fMRI↓\nsMRI↓", fontsize=7, color='gray', ha='center')
    ax.text(0.9,  0.08, "fMRI↑\nsMRI↓", fontsize=7, color='gray', ha='center')

# legend
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#C00000', ms=8, label=f'Positive (correct)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#FF9999', ms=8, label=f'Positive (wrong)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#4472C4', ms=8, label=f'Negative (correct)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='#9DC3E6', ms=8, label=f'Negative (wrong)'),
]
fig.legend(handles=legend_els, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.06))
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{RESULTS}/figures/task3c_scatter_fmri_vs_smri.png", dpi=150, bbox_inches="tight")
plt.close()
print("[3-C] Saved: results/figures/task3c_scatter_fmri_vs_smri.png")

# ── Plot 3: t-SNE using ROI attention as feature vector ───────────────────
# Use fMRI roi_importance for NC_vs_AD (most discriminative task)
print("\n[3-C] Computing t-SNE on fMRI ROI attention + sMRI attention features...")
tidx_nca = TASK_IDX["NC_vs_AD"]
mask_3c  = (pred_labels == 0) | (pred_labels == 2)

# Concatenate fMRI and sMRI attention as feature vector
# fMRI: roi_importance for NC_vs_AD task → shape (N, 116)
# sMRI: roi_attention → shape (N, 116)
# Note: re-index to match pred subject order
pred_sub_list = list(pred_subs)
gat_sub_list  = list(sub_ids)
bri_sub_list  = list(s_subs)

# Build feature matrix by aligning subjects
feats_fmri  = []
feats_smri  = []
feat_labels_all = []
feat_sites_all  = []
feat_tasks_all  = []

for sub, lbl, site in zip(pred_subs, pred_labels, pred_sites):
    gi = gat_sub_list.index(sub) if sub in gat_sub_list else -1
    bi = bri_sub_list.index(sub) if sub in bri_sub_list else -1
    if gi >= 0 and bi >= 0 and not missing[bi]:
        feats_fmri.append(roi_imp[gi, :, :].mean(axis=0))  # avg across 3 tasks
        feats_smri.append(roi_att[bi, :])
        feat_labels_all.append(lbl)
        feat_sites_all.append(site)

feats_fmri  = np.array(feats_fmri)
feats_smri  = np.array(feats_smri)
feat_labels_all = np.array(feat_labels_all)
feat_sites_all  = np.array(feat_sites_all)
n_pts = len(feat_labels_all)
print(f"  Aligned subjects: {n_pts}")

# t-SNE on fMRI features, sMRI features, and concatenated
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_fmri   = scaler.fit_transform(feats_fmri)
X_smri   = scaler.fit_transform(feats_smri)
X_concat = scaler.fit_transform(np.hstack([feats_fmri, feats_smri]))

tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, n_pts//4))
emb_fmri   = tsne.fit_transform(X_fmri)
emb_smri   = tsne.fit_transform(X_smri)
emb_concat = tsne.fit_transform(X_concat)

label_colors = {0: "#4472C4", 1: "#ED7D31", 2: "#C00000"}
site_markers = {"ADNI_new": "o", "TPMIC": "^"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("t-SNE of fMRI / sMRI / Fused Feature Space (all 3 labels)", fontsize=13, fontweight="bold")

for ax, (emb, title) in zip(axes, [
    (emb_fmri,   "fMRI Attention Features"),
    (emb_smri,   "sMRI Attention Features"),
    (emb_concat, "Fused (fMRI+sMRI)")
]):
    for lbl_i, lbl_name in LABEL_MAP.items():
        for site, mkr in site_markers.items():
            mask_i = (feat_labels_all == lbl_i) & (feat_sites_all == site)
            if mask_i.sum() > 0:
                ax.scatter(emb[mask_i, 0], emb[mask_i, 1],
                           c=label_colors[lbl_i], marker=mkr, s=60, alpha=0.85,
                           edgecolors='white', linewidths=0.4,
                           label=f"{lbl_name}/{site[:4]}" if lbl_i == 0 else "_")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.tick_params(labelsize=8)

legend_els = []
for lbl_i, lbl_name in LABEL_MAP.items():
    for site, mkr in site_markers.items():
        legend_els.append(
            Line2D([0],[0], marker=mkr, color='w', markerfacecolor=label_colors[lbl_i],
                   ms=8, label=f"{lbl_name} ({site[:4]})")
        )
fig.legend(handles=legend_els, loc="lower center", ncol=6, fontsize=8,
           bbox_to_anchor=(0.5, -0.06))
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{RESULTS}/figures/task3c_tsne.png", dpi=150, bbox_inches="tight")
plt.close()
print("[3-C] Saved: results/figures/task3c_tsne.png")

print("\n" + "="*60)
print("ALL EXPLAINABILITY TASKS COMPLETE")
print("="*60)
print("Outputs:")
print("  results/task3a_fmri_analysis.json")
print("  results/task3b_smri_analysis.json")
print("  results/task3c_complementarity.json")
print("  results/figures/task3a_fmri_top_rois.png")
print("  results/figures/task3a_fmri_network_radar.png")
print("  results/figures/task3b_smri_top_rois.png")
print("  results/figures/task3c_accuracy_comparison.png")
print("  results/figures/task3c_scatter_fmri_vs_smri.png")
print("  results/figures/task3c_tsne.png")
