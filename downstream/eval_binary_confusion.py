"""
eval_binary_confusion.py
========================
對三個 binary 任務分別畫 confusion matrix。
每個任務只取對應的樣本，直接用 binary 預測結果評估。

執行：
    conda activate AD
    cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
    python eval_binary_confusion.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, balanced_accuracy_score)
from tqdm import tqdm

from inference_pipeline_v2 import run_multimodal_inference, ModalityInput

# ── 載入測試集 ─────────────────────────────────────────────────────
pcag = pd.read_csv('pcag_test_aligned_v2.csv')
kd   = pd.read_csv('kd_test_aligned_v2.csv')
df   = pcag.merge(kd[['subject_id','smri_path','diagnosis']], on='subject_id', how='left')
df['t1_path']    = df['smri_path'].apply(
    lambda p: p+'.nii.gz' if pd.notna(p) and os.path.exists(p+'.nii.gz') else None)
LABEL_MAP        = {0:'NC', 1:'MCI', 2:'AD'}
df['true_class'] = df['label'].map(LABEL_MAP)

TASKS = [
    ('NC_vs_AD',  'NC', 'AD'),
    ('NC_vs_MCI', 'NC', 'MCI'),
    ('MCI_vs_AD', 'MCI', 'AD'),
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"測試集：{len(df)} 筆  NC={sum(df.label==0)}  MCI={sum(df.label==1)}  AD={sum(df.label==2)}\n")

# ── 逐筆推論，收集每個任務的結果 ──────────────────────────────────
task_data = {tk: {'y_true':[], 'y_pred':[], 'y_prob':[]} for tk,_,_ in TASKS}

for _, row in tqdm(df.iterrows(), total=len(df), desc='Inference'):
    try:
        inp = ModalityInput(matrix_path=row['matrix_path'],
                            t1_path=row['t1_path'],
                            subject_id=row['subject_id'])
        res = run_multimodal_inference(inp, device)
    except Exception as e:
        print(f"  ⚠️  {row['subject_id']}: {e}")
        continue

    tc = row['true_class']
    for tk, ca, cb in TASKS:
        if tk not in res:
            continue
        if tc not in (ca, cb):   # 此樣本不屬於這個 binary 任務
            continue
        true_bin = 1 if tc == cb else 0
        task_data[tk]['y_true'].append(true_bin)
        task_data[tk]['y_pred'].append(int(res[tk]['prediction']))
        task_data[tk]['y_prob'].append(float(res[tk]['prob_positive']))

# ── 畫圖：1 行 3 欄 ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Binary Task Confusion Matrices — KGRS System', fontsize=14, fontweight='bold', y=1.02)

TASK_LABELS = {
    'NC_vs_AD':  (['NC','AD'],  'NC vs AD\n(PCAG-ComBat)'),
    'NC_vs_MCI': (['NC','MCI'], 'NC vs MCI\n(PCAG-ComBat)'),
    'MCI_vs_AD': (['MCI','AD'], 'MCI vs AD\n(KD GNN)'),
}

print("=" * 60)
for ax, (tk, ca, cb) in zip(axes, TASKS):
    yt = task_data[tk]['y_true']
    yp = task_data[tk]['y_pred']
    yb = task_data[tk]['y_prob']
    classes, title = TASK_LABELS[tk]
    n = len(yt)

    cm = confusion_matrix(yt, yp)
    bal_acc = balanced_accuracy_score(yt, yp)
    try:
        auc = roc_auc_score(yt, yb)
    except Exception:
        auc = float('nan')

    print(f"\n{title.replace(chr(10),' ')}")
    print(f"  樣本數: {n}  ({ca}={yt.count(0)}  {cb}={yt.count(1)})")
    print(f"  Balanced Acc: {bal_acc:.3f}   AUC: {auc:.3f}")
    print(f"  Confusion Matrix (row=True, col=Pred):")
    print(pd.DataFrame(cm, index=classes, columns=classes))
    print(classification_report(yt, yp, target_names=classes, zero_division=0))

    # 畫 heatmap
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(2)); ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(range(2)); ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'{title}\nBal.Acc={bal_acc:.3f}  AUC={auc:.3f}  (n={n})', fontsize=11)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            pct = cm[i,j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            color = 'white' if cm[i,j] > thresh else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({pct:.0f}%)',
                    ha='center', va='center', fontsize=13,
                    color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

print("=" * 60)
plt.tight_layout()
out_path = 'results/confusion_matrix_binary.png'
os.makedirs('results', exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n圖片已儲存：{out_path}")
