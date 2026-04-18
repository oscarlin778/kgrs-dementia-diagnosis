"""
GAT Attention-based ROI Importance Visualization
使用 attention difference (disease - control) 顯示各腦區的差異顯著性
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── AAL116 ROI 名稱（標準順序）────────────────────────────────
AAL116_NAMES = [
    'Precentral_L',       'Precentral_R',       'Frontal_Sup_L',      'Frontal_Sup_R',
    'Frontal_Sup_Orb_L',  'Frontal_Sup_Orb_R',  'Frontal_Mid_L',      'Frontal_Mid_R',
    'Frontal_Mid_Orb_L',  'Frontal_Mid_Orb_R',  'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R',
    'Frontal_Inf_Tri_L',  'Frontal_Inf_Tri_R',  'Frontal_Inf_Orb_L',  'Frontal_Inf_Orb_R',
    'Rolandic_Oper_L',    'Rolandic_Oper_R',    'Supp_Motor_Area_L',  'Supp_Motor_Area_R',
    'Olfactory_L',        'Olfactory_R',        'Frontal_Sup_Med_L',  'Frontal_Sup_Med_R',
    'Frontal_Med_Orb_L',  'Frontal_Med_Orb_R',  'Rectus_L',           'Rectus_R',
    'Insula_L',           'Insula_R',           'Cingulum_Ant_L',     'Cingulum_Ant_R',
    'Cingulum_Mid_L',     'Cingulum_Mid_R',     'Cingulum_Post_L',    'Cingulum_Post_R',
    'Hippocampus_L',      'Hippocampus_R',      'ParaHippocampal_L',  'ParaHippocampal_R',
    'Amygdala_L',         'Amygdala_R',         'Calcarine_L',        'Calcarine_R',
    'Cuneus_L',           'Cuneus_R',           'Lingual_L',          'Lingual_R',
    'Occipital_Sup_L',    'Occipital_Sup_R',    'Occipital_Mid_L',    'Occipital_Mid_R',
    'Occipital_Inf_L',    'Occipital_Inf_R',    'Fusiform_L',         'Fusiform_R',
    'Postcentral_L',      'Postcentral_R',      'Parietal_Sup_L',     'Parietal_Sup_R',
    'Parietal_Inf_L',     'Parietal_Inf_R',     'SupraMarginal_L',    'SupraMarginal_R',
    'Angular_L',          'Angular_R',          'Precuneus_L',        'Precuneus_R',
    'Paracentral_Lob_L',  'Paracentral_Lob_R',  'Caudate_L',          'Caudate_R',
    'Putamen_L',          'Putamen_R',          'Pallidum_L',         'Pallidum_R',
    'Thalamus_L',         'Thalamus_R',         'Heschl_L',           'Heschl_R',
    'Temporal_Sup_L',     'Temporal_Sup_R',     'Temporal_Pole_Sup_L','Temporal_Pole_Sup_R',
    'Temporal_Mid_L',     'Temporal_Mid_R',     'Temporal_Pole_Mid_L','Temporal_Pole_Mid_R',
    'Temporal_Inf_L',     'Temporal_Inf_R',     'Cerebelum_Crus1_L',  'Cerebelum_Crus1_R',
    'Cerebelum_Crus2_L',  'Cerebelum_Crus2_R',  'Cerebelum_3_L',      'Cerebelum_3_R',
    'Cerebelum_4_5_L',    'Cerebelum_4_5_R',    'Cerebelum_6_L',      'Cerebelum_6_R',
    'Cerebelum_7b_L',     'Cerebelum_7b_R',     'Cerebelum_8_L',      'Cerebelum_8_R',
    'Cerebelum_9_L',      'Cerebelum_9_R',      'Cerebelum_10_L',     'Cerebelum_10_R',
    'Vermis_1_2',         'Vermis_3',           'Vermis_4_5',         'Vermis_6',
    'Vermis_7',           'Vermis_8',           'Vermis_9',           'Vermis_10',
]

NETWORK_MAP = {
    'DMN':   [34, 35, 66, 67, 64, 65, 22, 23, 24, 25],
    'SMN':   [0, 1, 56, 57, 68, 69],
    'VN':    [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    'SN':    [28, 29, 30, 31, 32, 33],
    'FPN':   [6, 7, 58, 59, 60, 61],
    'LN':    [36, 37, 38, 39, 40, 41],
    'VAN':   [10, 11, 14, 15],
    'BGN':   [70, 71, 72, 73, 74, 75, 76, 77],
    'CereN': list(range(90, 116)),
}

ATTN_DIR = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints"
OUT_DIR  = "/home/wei-chi/Data/script/results/feature_importance"

# ── 讀取資料 ────────────────────────────────────────────────────
def load_attn(task_pair):
    safe = f"{task_pair[0]}_vs_{task_pair[1]}"
    path = os.path.join(ATTN_DIR, f"gnn_attention_{safe}.npy")
    if not os.path.exists(path):
        print(f"  ⚠️  找不到 {path}，請先重新 train GNN")
        return None, None, None
    data = np.load(path, allow_pickle=True).item()
    imps, labels = [], []
    for v in data.values():
        imps.append(v['importance'])
        labels.append(v['label'])
    return np.array(imps), np.array(labels), data


def class_raw_mean(imps, labels, class_idx):
    """指定 class 的平均節點 attention（未歸一化 raw 值）"""
    mask = labels == class_idx
    if mask.sum() == 0:
        return np.zeros(imps.shape[1])
    return imps[mask].mean(axis=0)


def attention_diff(imps, labels):
    """
    回傳 class_b_raw - class_a_raw，以 max(|diff|) 縮放至 [-1, 1]
    正值 = disease class attention 更高；負值 = control class attention 更高
    """
    raw_a = class_raw_mean(imps, labels, 0)
    raw_b = class_raw_mean(imps, labels, 1)
    diff  = raw_b - raw_a
    scale = np.abs(diff).max()
    return diff / (scale + 1e-8)


# ── 圖 1：Top-K ROI attention difference 柱狀圖 ─────────────────
def plot_top_rois(task_pair, top_k=20, ax=None):
    class_a, class_b = task_pair
    imps, labels, _ = load_attn(task_pair)
    if imps is None:
        return

    diff      = attention_diff(imps, labels)
    top_idx   = np.argsort(np.abs(diff))[-top_k:][::-1]
    top_diff  = diff[top_idx]
    top_names = [AAL116_NAMES[i] if i < len(AAL116_NAMES) else f"ROI_{i}"
                 for i in top_idx]

    colors = ['#D65F5F' if d > 0 else '#4878CF' for d in top_diff]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(range(top_k), top_diff[::-1], color=colors[::-1],
            alpha=0.85, edgecolor='white')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(
        f'Attention Difference  (+ = more important for {class_b}  |  - = more important for {class_a})',
        fontsize=8)
    ax.set_title(f'{class_a} vs {class_b} — Top {top_k} Differential ROIs',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(-1.1, 1.1)

    # 標示腦網路歸屬
    roi_to_net = {roi: net for net, rois in NETWORK_MAP.items() for roi in rois}
    for bar_i, roi_i in enumerate(top_idx[::-1]):
        net = roi_to_net.get(roi_i, '?')
        val = top_diff[::-1][bar_i]
        ax.text(0.03 if val < 0 else -0.03,
                bar_i, net, va='center',
                ha='left' if val < 0 else 'right',
                fontsize=7, color='gray')

    if standalone:
        plt.tight_layout()
        out = os.path.join(OUT_DIR,
              f"gnn_attn_top_rois_{task_pair[0]}_vs_{task_pair[1]}.png")
        plt.savefig(out, dpi=180, bbox_inches='tight')
        print(f"  ✅ {out}")


# ── 圖 2：Network-level attention difference heatmap（3 tasks × 9 nets）──
def plot_network_diff_heatmap(tasks):
    """
    一張圖：rows = 3 tasks，cols = 9 networks
    顏色 = disease_attention - control_attention（diverging，red=disease 主導）
    """
    net_names  = list(NETWORK_MAP.keys())
    task_labels = [f"{a} vs {b}" for a, b in tasks]
    mat = np.zeros((len(tasks), len(net_names)))

    for t_i, task_pair in enumerate(tasks):
        imps, labels, _ = load_attn(task_pair)
        if imps is None:
            continue
        diff = attention_diff(imps, labels)          # [-1, 1] per ROI
        for n_i, net in enumerate(net_names):
            mat[t_i, n_i] = diff[NETWORK_MAP[net]].mean()

    # 以整體矩陣的 max|val| 做對稱縮放
    vmax = np.abs(mat).max()

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle(
        'GAT Attention Difference: Network-level Salience  (disease − control)\n'
        'Red = disease class has higher attention  |  Blue = control class has higher attention',
        fontsize=12, fontweight='bold')

    im = ax.imshow(mat, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(net_names)))
    ax.set_xticklabels(net_names, fontsize=11)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(task_labels, fontsize=11)

    for r in range(len(tasks)):
        for c in range(len(net_names)):
            v = mat[r, c]
            txt_color = 'white' if abs(v) > 0.55 * vmax else 'black'
            ax.text(c, r, f'{v:+.2f}', ha='center', va='center',
                    fontsize=9, color=txt_color, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, label='Attention Difference (normalised)')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "gnn_attn_network_diff_heatmap.png")
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f"  ✅ {out}")
    plt.close()


# ── 圖 3：三個 task 的 Top-K ROI 合併在一張圖 ──────────────────
def plot_all_tasks_combined(tasks, top_k=15):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        'GAT Attention Difference: Differential ROI Salience Across Disease Progression\n'
        'Red = more important for disease class  |  Blue = more important for control class\n'
        '(Values: normalised attention_disease − attention_control)',
        fontsize=11, fontweight='bold')
    for ax, task_pair in zip(axes, tasks):
        plot_top_rois(task_pair, top_k=top_k, ax=ax)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "gnn_attn_all_tasks_combined.png")
    plt.savefig(out, dpi=180, bbox_inches='tight')
    print(f"  ✅ {out}")
    plt.close()


def main():
    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    os.makedirs(OUT_DIR, exist_ok=True)

    print("📊 產生 GAT Attention Difference 視覺化...")
    print("\n[1/2] Network-level attention difference heatmap")
    plot_network_diff_heatmap(tasks)

    print("\n[2/2] Top ROI attention difference (combined)")
    plot_all_tasks_combined(tasks, top_k=15)

    print("\n✅ 所有圖表完成，儲存於:", OUT_DIR)


if __name__ == "__main__":
    main()
