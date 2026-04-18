"""
三模態分類效能比較圖
sMRI (ResNet Teacher OOF) vs fMRI (GNN multi-seed) vs Ensemble
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, roc_auc_score

# ── 路徑設定（與主訓練腳本相同）──────────────────────────────────
CSV_PATHS = [
    "/home/wei-chi/Model/_dataset_mapping.csv",
    "/home/wei-chi/Data/dataset_index_116_clean_old.csv",
    "/home/wei-chi/Data/adni_dataset_index_116.csv"
]
MATRIX_DIR       = "/home/wei-chi/Model/processed_116_matrices"
TEACHER_PROBS_DIR = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints"

# ── v10 multi-seed 跑完的 GNN / Ensemble 結果（手動填入）────────
GNN_RESULTS = {
    'NC vs AD':  {'acc': 0.715, 'auc': 0.776},
    'NC vs MCI': {'acc': 0.689, 'auc': 0.708},
    'MCI vs AD': {'acc': 0.767, 'auc': 0.761},
}
ENS_RESULTS = {
    'NC vs AD':  {'acc': 0.826, 'auc': 0.841},
    'NC vs MCI': {'acc': 0.729, 'auc': 0.774},
    'MCI vs AD': {'acc': 0.842, 'auc': 0.853},
}

# ── Subject ID 萃取（與主訓練腳本相同）──────────────────────────
def get_subject_id(path_str):
    basename = os.path.basename(str(path_str))
    clean = re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy|_T1_MNI\.nii\.gz|_T1\.nii\.gz|\.nii\.gz)$', '', basename)
    clean = re.sub(r'^(sub-|sub_|old_dswau)', '', clean)
    return clean.strip()

# ── 載入 fMRI 受試者清單（取 subject_id + diagnosis）────────────
def load_subject_diagnosis():
    valid_data, seen_paths = [], set()
    for path in CSV_PATHS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m_path = (row.get('matrix_path') or
                      (os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy")
                       if pd.notna(row.get('new_id_base')) else None) or
                      (os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy")
                       if pd.notna(row.get('Subject')) else None))
            if not (m_path and os.path.exists(m_path)) or m_path in seen_paths:
                continue
            try:
                if np.load(m_path).shape != (116, 116):
                    continue
                diag = str(row.get('diagnosis', '')).upper()
                sid  = get_subject_id(m_path)
                if diag and sid:
                    valid_data.append({'subject_id': sid, 'diagnosis': diag})
                    seen_paths.add(m_path)
            except:
                continue
    return pd.DataFrame(valid_data)

# ── 從 teacher_logits.npy 計算 sMRI OOF 指標 ──────────────────
def compute_teacher_metrics(task_pair, df_subjects):
    class_a, class_b = task_pair
    task_name = f"{class_a}_vs_{class_b}"
    npy_path  = os.path.join(TEACHER_PROBS_DIR, f"teacher_logits_{task_name}.npy")

    if not os.path.exists(npy_path):
        print(f"  ⚠️  找不到 {npy_path}")
        return None, None

    teacher_dict = np.load(npy_path, allow_pickle=True).item()

    # 只取本 task 有的受試者，且有 teacher 預測的
    df_task = df_subjects[df_subjects['diagnosis'].isin([class_a, class_b])].copy()
    df_task['label'] = df_task['diagnosis'].map({class_a: 0, class_b: 1})

    y_true, y_prob = [], []
    for _, row in df_task.iterrows():
        sid = row['subject_id']
        if sid in teacher_dict:
            y_true.append(row['label'])
            y_prob.append(teacher_dict[sid])

    if len(y_true) < 5:
        print(f"  ⚠️  {task_name} 配對樣本不足 ({len(y_true)})")
        return None, None

    y_prob = np.array(y_prob)
    acc = accuracy_score(y_true, y_prob.argmax(axis=1))
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except:
        auc = float('nan')

    print(f"  sMRI teacher ({task_name}): n={len(y_true)}, acc={acc*100:.1f}%, AUC={auc:.3f}")
    return acc, auc

# ── 畫圖 ─────────────────────────────────────────────────────────
def plot_comparison(smri_results):
    tasks      = ['NC vs AD', 'NC vs MCI', 'MCI vs AD']
    modalities = ['sMRI\n(ResNet Teacher)', 'fMRI\n(GNN)', 'Ensemble\n(GNN + sMRI)']
    colors     = ['#4878CF', '#6ACC65', '#D65F5F']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Brain Disease Classification: Cross-Modal Knowledge Distillation\n'
                 'sMRI ResNet Teacher  |  fMRI GNN Student  |  Inference-time Ensemble',
                 fontsize=13, fontweight='bold', y=1.02)

    bar_width = 0.22
    x = np.arange(len(tasks))

    for ax_idx, metric in enumerate(['acc', 'auc']):
        ax = axes[ax_idx]
        ylabel = 'Accuracy (%)' if metric == 'acc' else 'AUC'
        title  = 'Classification Accuracy' if metric == 'acc' else 'ROC-AUC Score'

        smri_vals = [smri_results[t][metric] for t in tasks]
        gnn_vals  = [GNN_RESULTS[t][metric]  for t in tasks]
        ens_vals  = [ENS_RESULTS[t][metric]  for t in tasks]

        if metric == 'acc':
            smri_vals = [v * 100 for v in smri_vals]
            gnn_vals  = [v * 100 for v in gnn_vals]
            ens_vals  = [v * 100 for v in ens_vals]

        offsets = [-bar_width, 0, bar_width]
        for vals, color, offset, label in zip(
            [smri_vals, gnn_vals, ens_vals], colors, offsets, modalities
        ):
            bars = ax.bar(x + offset, vals, bar_width * 0.9,
                          color=color, alpha=0.85, label=label.replace('\n', ' '),
                          edgecolor='white', linewidth=0.8)
            # 在每個 bar 頂端印數值
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.5 if metric == 'acc' else 0.005),
                        f'{v:.1f}{"%" if metric == "acc" else ""}',
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        # 80% 基準線
        threshold = 80.0 if metric == 'acc' else 0.80
        ax.axhline(threshold, color='gray', linestyle='--', linewidth=1.2,
                   alpha=0.7, label='80% target')

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0 if metric == 'acc' else 0,
                    105 if metric == 'acc' else 1.05)
        ax.legend(fontsize=9, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = '/home/wei-chi/Data/script/results_comparison.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ 圖表已儲存至: {out_path}")
    return out_path


def main():
    print("載入受試者資料...")
    df_subjects = load_subject_diagnosis()
    print(f"  共找到 {len(df_subjects)} 位受試者")

    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    smri_results = {}
    print("\n計算 sMRI Teacher OOF 指標...")
    for task_pair in tasks:
        task_name = f"{task_pair[0]} vs {task_pair[1]}"
        acc, auc = compute_teacher_metrics(task_pair, df_subjects)
        smri_results[task_name] = {
            'acc': acc if acc is not None else 0.0,
            'auc': auc if auc is not None else 0.0,
        }

    print("\n繪製比較圖...")
    plot_comparison(smri_results)

    # 同時印出彙整表
    print("\n" + "="*65)
    print(f"{'Task':<12} {'Modality':<22} {'Accuracy':>10} {'AUC':>8}")
    print("-"*65)
    for task_name in ['NC vs AD', 'NC vs MCI', 'MCI vs AD']:
        for label, results in [('sMRI', smri_results), ('fMRI GNN', GNN_RESULTS), ('Ensemble', ENS_RESULTS)]:
            acc = results[task_name]['acc']
            auc = results[task_name]['auc']
            flag = '✅' if acc >= 0.80 else '  '
            print(f"{flag} {task_name:<10} {label:<22} {acc*100:>8.1f}%  {auc:>7.3f}")
        print()


if __name__ == "__main__":
    main()
