"""
eval_confusion_matrix.py
========================
對 40 個 hold-out 測試樣本跑完整推論，畫出三分類 confusion matrix。

執行：
    conda activate AD
    cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
    python eval_confusion_matrix.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from tqdm import tqdm

from inference_pipeline_v2 import run_multimodal_inference, ModalityInput

# ── 載入測試集 ────────────────────────────────────────────────────
pcag = pd.read_csv('pcag_test_aligned_v2.csv')    # subject_id, matrix_path, label (0=NC,1=MCI,2=AD)
kd   = pd.read_csv('kd_test_aligned_v2.csv')      # subject_id, smri_path, diagnosis

# 合併：pcag 提供 fMRI 路徑與 label，kd 提供 T1 路徑
df = pcag.merge(kd[['subject_id', 'smri_path', 'diagnosis']], on='subject_id', how='left')
df['t1_path'] = df['smri_path'].apply(
    lambda p: p + '.nii.gz' if pd.notna(p) and os.path.exists(p + '.nii.gz') else None
)
LABEL_MAP = {0: 'NC', 1: 'MCI', 2: 'AD'}
df['true_class'] = df['label'].map(LABEL_MAP)

print(f"測試集：{len(df)} 筆  |  NC={sum(df.label==0)}  MCI={sum(df.label==1)}  AD={sum(df.label==2)}")
print(f"有雙模態：{df['t1_path'].notna().sum()}  僅 fMRI：{df['t1_path'].isna().sum()}\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ── 階層式分類（替代 OVO）────────────────────────────────────────
def hierarchical_predict(raw_results: dict) -> str:
    """
    Step 1: NC vs Disease
        P(AD)  >= 0.5  (NC_vs_AD)  OR
        P(MCI) >= 0.5  (NC_vs_MCI) → Disease
        否則 → NC
    Step 2: Disease 內部
        P(AD)  >= 0.5  (MCI_vs_AD) → AD，否則 → MCI
    """
    p_ad  = float(raw_results.get('NC_vs_AD',  {}).get('prob_positive', 0.5))
    p_mci = float(raw_results.get('NC_vs_MCI', {}).get('prob_positive', 0.5))
    p_adg = float(raw_results.get('MCI_vs_AD', {}).get('prob_positive', 0.5))

    is_disease = (p_ad >= 0.5) or (p_mci >= 0.5)
    if not is_disease:
        return 'NC'
    return 'AD' if p_adg >= 0.5 else 'MCI'

# ── 逐筆推論 ──────────────────────────────────────────────────────
y_true, y_pred = [], []
errors = []

for _, row in tqdm(df.iterrows(), total=len(df), desc='Inference'):
    try:
        inp = ModalityInput(
            matrix_path=row['matrix_path'],
            t1_path=row['t1_path'],
            subject_id=row['subject_id'],
        )
        results = run_multimodal_inference(inp, device)
        pred = hierarchical_predict(results)
    except Exception as e:
        errors.append((row['subject_id'], str(e)))
        pred = 'UNKNOWN'

    y_true.append(row['true_class'])
    y_pred.append(pred)

if errors:
    print(f"\n⚠️  {len(errors)} 筆推論失敗：")
    for sid, err in errors:
        print(f"   {sid}: {err}")

# ── 計算指標 ──────────────────────────────────────────────────────
valid = [(t, p) for t, p in zip(y_true, y_pred) if p != 'UNKNOWN']
yt = [v[0] for v in valid]
yp = [v[1] for v in valid]
classes = ['NC', 'MCI', 'AD']

cm = confusion_matrix(yt, yp, labels=classes)
bal_acc = balanced_accuracy_score(yt, yp)

print(f"\n=== 結果（{len(valid)}/{len(df)} 筆有效）===")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print("\nClassification Report:")
print(classification_report(yt, yp, labels=classes, target_names=classes, zero_division=0))
print("Confusion Matrix (row=True, col=Pred):")
print(pd.DataFrame(cm, index=classes, columns=classes))

# ── 畫圖 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, fontsize=13)
ax.set_yticklabels(classes, fontsize=13)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
ax.set_title(f'Confusion Matrix — KGRS System (Hierarchical)\nBalanced Acc = {bal_acc:.3f}  (n={len(valid)})', fontsize=13)

thresh = cm.max() / 2
for i in range(len(classes)):
    for j in range(len(classes)):
        pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
        color = 'white' if cm[i, j] > thresh else 'black'
        ax.text(j, i, f'{cm[i, j]}\n({pct:.0f}%)', ha='center', va='center',
                fontsize=12, color=color, fontweight='bold')

plt.tight_layout()
out_path = 'results/confusion_matrix_kgrs.png'
os.makedirs('results', exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n圖片已儲存：{out_path}")
