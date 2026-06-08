"""
eval_system_binary.py
=====================
對系統內所有有 ground-truth label 的病患跑推論，畫三張 binary confusion matrix。
Label 來源：CSV splits（優先）→ Neo4j clinical_dx（補充）

執行：
    conda activate AD
    cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
    python eval_system_binary.py
"""
import os, sys, re, glob
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
from tqdm import tqdm
from dotenv import load_dotenv

from inference_pipeline_v2 import run_multimodal_inference, ModalityInput, find_t1_path

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# ── Step 1：建立系統所有病患清單 ──────────────────────────────────
def clean_sid(path):
    n = os.path.basename(path)
    return re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy)$', '', n)

MATRIX_DIRS = [
    '/home/wei-chi/Alzheimers_Project/external_models/processed_116_matrices',
    '/home/wei-chi/Alzheimers_Project/external_data/features/ADNI_processed_116_matrices',
]
all_patients = {}   # sid -> matrix_path
for d in MATRIX_DIRS:
    for p in glob.glob(os.path.join(d, '*.npy')):
        sid = clean_sid(p)
        if sid not in all_patients:
            all_patients[sid] = p

print(f"系統病患總數: {len(all_patients)}")

# ── Step 2：從 CSV 取 label ────────────────────────────────────────
dx_map = {'NC': 0, 'MCI': 1, 'AD': 2}
csv_labels = {}   # sid -> label (int)

for fname in ['pcag_train_aligned.csv', 'pcag_test_aligned.csv',
              'kd_train_aligned.csv',   'kd_test_aligned.csv']:
    df = pd.read_csv(fname)
    if 'diagnosis' in df.columns:
        df['label'] = df['diagnosis'].map(dx_map)
    for _, row in df.iterrows():
        sid = str(row['subject_id'])
        if sid not in csv_labels and pd.notna(row.get('label')):
            csv_labels[sid] = int(row['label'])

print(f"CSV labels: {len(csv_labels)} 筆")

# ── Step 3：Neo4j 補充缺少的 label ────────────────────────────────
neo_labels = {}
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD'))
    )
    with driver.session() as s:
        rows = s.run(
            'MATCH (p:Patient) WHERE p.clinical_dx IS NOT NULL '
            'RETURN p.subject_id AS sid, p.clinical_dx AS dx'
        ).data()
    for r in rows:
        if r['dx'] in dx_map:
            neo_labels[r['sid']] = dx_map[r['dx']]
    print(f"Neo4j labels: {len(neo_labels)} 筆")
except Exception as e:
    print(f"Neo4j 連線失敗，僅使用 CSV labels: {e}")

def get_label(sid):
    """CSV 優先，否則從 Neo4j 找（ADNI: sub-XXX_S_XXXX → XXX_S_XXXX）"""
    if sid in csv_labels:
        return csv_labels[sid]
    neo_key = re.sub(r'^sub-', '', sid)   # sub-002_S_0413 → 002_S_0413
    return neo_labels.get(neo_key) or neo_labels.get(sid)

# ── Step 4：組合最終資料集 ─────────────────────────────────────────
records = []
for sid, mpath in all_patients.items():
    label = get_label(sid)
    if label is None:
        continue
    t1 = find_t1_path(sid)
    records.append({'subject_id': sid, 'matrix_path': mpath,
                    'label': label, 't1_path': t1})

df = pd.DataFrame(records)
LABEL_STR = {0: 'NC', 1: 'MCI', 2: 'AD'}
df['true_class'] = df['label'].map(LABEL_STR)
cnt = df['label'].value_counts().sort_index()
print(f"\n最終納入: {len(df)} 筆")
print(f"  NC={cnt.get(0,0)}  MCI={cnt.get(1,0)}  AD={cnt.get(2,0)}")
print(f"  雙模態: {df['t1_path'].notna().sum()}  僅 fMRI: {df['t1_path'].isna().sum()}\n")

# ── Step 5：推論 ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

TASKS = [
    ('NC_vs_AD',  'NC', 'AD'),
    ('NC_vs_MCI', 'NC', 'MCI'),
    ('MCI_vs_AD', 'MCI', 'AD'),
]
task_data = {tk: {'y_true': [], 'y_pred': [], 'y_prob': []} for tk, _, _ in TASKS}

for _, row in tqdm(df.iterrows(), total=len(df), desc='Inference'):
    try:
        inp = ModalityInput(matrix_path=row['matrix_path'],
                            t1_path=row['t1_path'],
                            subject_id=row['subject_id'])
        res = run_multimodal_inference(inp, device)
    except Exception as e:
        tqdm.write(f"  ⚠️  {row['subject_id']}: {e}")
        continue

    tc = row['true_class']
    for tk, ca, cb in TASKS:
        if tk not in res or tc not in (ca, cb):
            continue
        task_data[tk]['y_true'].append(1 if tc == cb else 0)
        task_data[tk]['y_pred'].append(int(res[tk]['prediction']))
        task_data[tk]['y_prob'].append(float(res[tk]['prob_positive']))

# ── Step 6：畫圖 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Binary Task Confusion Matrices — All System Patients',
             fontsize=14, fontweight='bold', y=1.02)

TASK_META = {
    'NC_vs_AD':  (['NC', 'AD'],   'NC vs AD\n(PCAG-ComBat)'),
    'NC_vs_MCI': (['NC', 'MCI'],  'NC vs MCI\n(PCAG-ComBat)'),
    'MCI_vs_AD': (['MCI', 'AD'],  'MCI vs AD\n(KD GNN)'),
}

print("\n" + "=" * 65)
for ax, (tk, ca, cb) in zip(axes, TASKS):
    yt = task_data[tk]['y_true']
    yp = task_data[tk]['y_pred']
    yb = task_data[tk]['y_prob']
    classes, title = TASK_META[tk]
    n = len(yt)

    cm = confusion_matrix(yt, yp)
    bal_acc = balanced_accuracy_score(yt, yp)
    try:
        auc = roc_auc_score(yt, yb)
    except Exception:
        auc = float('nan')
    sens = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0   # recall of positive class
    spec = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0   # recall of negative class

    # Console output
    print(f"\n{title.replace(chr(10),' ')}")
    print(f"  n={n}  ({ca}={yt.count(0)}  {cb}={yt.count(1)})")
    print(f"  Balanced Acc={bal_acc:.3f}  AUC={auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

    # Heatmap
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(2)); ax.set_xticklabels(classes, fontsize=13)
    ax.set_yticks(range(2)); ax.set_yticklabels(classes, fontsize=13)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(
        f'{title}\nBal.Acc={bal_acc:.3f}  AUC={auc:.3f}  (n={n})',
        fontsize=11
    )
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            pct = cm[i,j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            color = 'white' if cm[i,j] > thresh else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({pct:.0f}%)',
                    ha='center', va='center', fontsize=13,
                    color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

print("\n" + "=" * 65)
plt.tight_layout()
out = 'results/confusion_matrix_system_binary.png'
os.makedirs('results', exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n圖片已儲存：{out}")
