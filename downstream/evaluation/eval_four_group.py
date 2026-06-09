"""
eval_four_group.py
==================
對 NC_vs_AD、NC_vs_MCI、MCI_vs_AD（PCAG-ComBat 雙模態任務）分析四組病患：
    A: fMRI 正確 + sMRI 正確  （兩個模態一致支持正確答案）
    B: fMRI 正確 + sMRI 錯誤  （fMRI 單獨勝出）
    C: fMRI 錯誤 + sMRI 正確  （sMRI 單獨勝出）
    D: fMRI 錯誤 + sMRI 錯誤  （兩個模態均失敗）

「fMRI-only 預測」：將 sMRI features 歸零，讓 Q (fMRI) residual 獨自決定結果。
「sMRI-only 預測」：將 fMRI embedding 歸零，讓 K/V (sMRI) 路徑獨自決定結果。
兩者均使用已訓練的 PCAG-ComBat 模型，不另行訓練新模型。

MCI_vs_AD 現已切換為 PCAG dual-modal，故三個任務均可做四組分析。

執行：
    conda activate AD
    cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
    python eval_four_group.py
"""
import os, sys, re, glob
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dotenv import load_dotenv

from inference_pipeline_v2 import (
    _get_pipeline, find_t1_path, apply_combat,
    DEVICE,
)

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# ── Step 1：建立系統病患清單（同 eval_system_binary.py）────────────────
def clean_sid(path):
    n = os.path.basename(path)
    return re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy)$', '', n)

MATRIX_DIRS = [
    '/home/wei-chi/Alzheimers_Project/external_models/processed_116_matrices',
    '/home/wei-chi/Alzheimers_Project/external_data/features/ADNI_processed_116_matrices',
]
all_patients = {}
for d in MATRIX_DIRS:
    for p in glob.glob(os.path.join(d, '*.npy')):
        sid = clean_sid(p)
        if sid not in all_patients: all_patients[sid] = p

print(f"系統病患總數: {len(all_patients)}")

# ── Step 2：載入 label ────────────────────────────────────────────────
dx_map = {'NC': 0, 'MCI': 1, 'AD': 2}
csv_labels = {}
for fname in ['pcag_train_aligned_v2.csv', 'pcag_test_aligned_v2.csv',
              'kd_train_aligned_v2.csv',   'kd_test_aligned_v2.csv']:
    df = pd.read_csv(fname)
    if 'diagnosis' in df.columns:
        df['label'] = df['diagnosis'].map(dx_map)
    for _, row in df.iterrows():
        sid = str(row['subject_id'])
        if sid not in csv_labels and pd.notna(row.get('label')):
            csv_labels[sid] = int(row['label'])

neo_labels = {}
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI','bolt://localhost:7687'),
        auth=(os.getenv('NEO4J_USER','neo4j'), os.getenv('NEO4J_PASSWORD'))
    )
    with driver.session() as s:
        rows = s.run(
            'MATCH (p:Patient) WHERE p.clinical_dx IS NOT NULL '
            'RETURN p.subject_id AS sid, p.clinical_dx AS dx'
        ).data()
    for r in rows:
        if r['dx'] in dx_map: neo_labels[r['sid']] = dx_map[r['dx']]
except Exception as e:
    print(f"Neo4j 跳過: {e}")

def get_label(sid):
    if sid in csv_labels: return csv_labels[sid]
    neo_key = re.sub(r'^sub-', '', sid)
    return neo_labels.get(neo_key) or neo_labels.get(sid)

# ── Step 3：篩選雙模態且有 label 的病患 ─────────────────────────────
records = []
for sid, mpath in all_patients.items():
    label = get_label(sid)
    if label is None: continue
    t1 = find_t1_path(sid)
    if t1 is None: continue  # 四組分析需要雙模態
    records.append({'subject_id': sid, 'matrix_path': mpath, 'label': label, 't1_path': t1})

df = pd.DataFrame(records)
LABEL_STR = {0:'NC', 1:'MCI', 2:'AD'}
df['true_class'] = df['label'].map(LABEL_STR)
cnt = df['label'].value_counts().sort_index()
print(f"\n雙模態且有 label: {len(df)} 筆")
print(f"  NC={cnt.get(0,0)}  MCI={cnt.get(1,0)}  AD={cnt.get(2,0)}\n")

# ── Step 4：載入 pipeline ────────────────────────────────────────────
pipe = _get_pipeline()

PCAG_TASKS = [
    ('NC_vs_AD',  'NC', 'AD',  pipe.pcag_nc_ad,   pipe.combat_nc_ad),
    ('NC_vs_MCI', 'NC', 'MCI', pipe.pcag_nc_mci,  pipe.combat_nc_mci),
    ('MCI_vs_AD', 'MCI', 'AD', pipe.pcag_mci_ad,  pipe.combat_mci_ad),
]

FMRI_DIM = 1280
SMRI_DIM = 768

@torch.no_grad()
def predict_fmri_only(models, x, adj):
    """PCAG with sMRI zeroed → residual Q (fMRI) dominates."""
    smri_zero = torch.zeros(1, SMRI_DIM, device=DEVICE)
    probs = []
    for m in models:
        fmri_emb = m.fmri_encoder(x, adj)
        out = F.softmax(m.pcag(fmri_emb, smri_zero), dim=1)
        probs.append(out[0, 1].item())
    return float(np.mean(probs))

@torch.no_grad()
def predict_smri_only(models, smri_harmonized):
    """PCAG with fMRI embedding zeroed → K/V (sMRI) path dominates."""
    smri_t    = torch.tensor(smri_harmonized, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    fmri_zero = torch.zeros(1, FMRI_DIM, device=DEVICE)
    probs = []
    for m in models:
        out = F.softmax(m.pcag(fmri_zero, smri_t), dim=1)
        probs.append(out[0, 1].item())
    return float(np.mean(probs))

@torch.no_grad()
def predict_full(models, x, adj, smri_harmonized):
    """Standard PCAG dual-modal prediction."""
    smri_t = torch.tensor(smri_harmonized, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    probs  = []
    for m in models:
        out = F.softmax(m(x, adj, smri_t), dim=1)
        probs.append(out[0, 1].item())
    return float(np.mean(probs))


# ── Step 5：推論 ─────────────────────────────────────────────────────
task_records = {tk: [] for tk, _, _, _, _ in PCAG_TASKS}

for _, row in tqdm(df.iterrows(), total=len(df), desc='Inference'):
    sid = row['subject_id']
    tc  = row['true_class']
    try:
        x, adj = pipe._encode_fmri(row['matrix_path'])
        smri_raw = pipe._encode_smri(row['t1_path'])
    except Exception as e:
        tqdm.write(f"  ⚠️  encode {sid}: {e}")
        continue

    site = 'ADNI' if re.search(r'\d{3}_S_\d{4}', sid) else 'TPMIC'

    for tk, ca, cb, models, combat_params in PCAG_TASKS:
        if tc not in (ca, cb):
            continue
        try:
            smri_h = apply_combat(smri_raw, site, combat_params)
            p_full  = predict_full(models, x, adj, smri_h)
            p_fmri  = predict_fmri_only(models, x, adj)
            p_smri  = predict_smri_only(models, smri_h)
        except Exception as e:
            tqdm.write(f"  ⚠️  predict {sid}/{tk}: {e}")
            continue

        true_bin = 1 if tc == cb else 0
        pred_full  = int(p_full  >= 0.5)
        pred_fmri  = int(p_fmri  >= 0.5)
        pred_smri  = int(p_smri  >= 0.5)

        task_records[tk].append({
            'subject_id': sid,
            'true_class': tc,
            'true_bin':   true_bin,
            'p_full':     p_full,
            'p_fmri':     p_fmri,
            'p_smri':     p_smri,
            'pred_full':  pred_full,
            'pred_fmri':  pred_fmri,
            'pred_smri':  pred_smri,
            'fmri_correct': int(pred_fmri == true_bin),
            'smri_correct': int(pred_smri == true_bin),
        })


# ── Step 6：分組統計 + 畫圖 ──────────────────────────────────────────
GROUP_LABELS = {
    'AA': 'A: Both\ncorrect',
    'AB': 'B: fMRI✓\nsMRI✗',
    'BA': 'C: fMRI✗\nsMRI✓',
    'BB': 'D: Both\nwrong',
}
GROUP_COLORS = {'AA':'#4CAF50', 'AB':'#2196F3', 'BA':'#FF9800', 'BB':'#F44336'}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Four-Group Modality Agreement Analysis — PCAG-ComBat Tasks',
             fontsize=14, fontweight='bold')

print("\n" + "="*70)
for col_idx, (tk, ca, cb, _, _) in enumerate(PCAG_TASKS):
    recs = task_records[tk]
    if not recs:
        print(f"\n{tk}: 無資料")
        continue
    df_tk = pd.DataFrame(recs)
    n = len(df_tk)

    # Group assignment
    def grp(row):
        return ('A' if row['fmri_correct'] else 'B') + ('A' if row['smri_correct'] else 'B')
    df_tk['group'] = df_tk.apply(grp, axis=1)

    counts = df_tk['group'].value_counts()
    for g in ['AA','AB','BA','BB']:
        if g not in counts: counts[g] = 0

    # AUC per mode
    try:    auc_full = roc_auc_score(df_tk['true_bin'], df_tk['p_full'])
    except: auc_full = float('nan')
    try:    auc_fmri = roc_auc_score(df_tk['true_bin'], df_tk['p_fmri'])
    except: auc_fmri = float('nan')
    try:    auc_smri = roc_auc_score(df_tk['true_bin'], df_tk['p_smri'])
    except: auc_smri = float('nan')

    # Accuracy per mode
    acc_full = (df_tk['pred_full'] == df_tk['true_bin']).mean()
    acc_fmri = (df_tk['pred_fmri'] == df_tk['true_bin']).mean()
    acc_smri = (df_tk['pred_smri'] == df_tk['true_bin']).mean()

    print(f"\n{'─'*50}")
    print(f"Task: {ca} vs {cb}  (n={n})")
    print(f"  AUC  — Full:{auc_full:.3f}  fMRI-only:{auc_fmri:.3f}  sMRI-only:{auc_smri:.3f}")
    print(f"  Acc  — Full:{acc_full:.3f}  fMRI-only:{acc_fmri:.3f}  sMRI-only:{acc_smri:.3f}")
    print(f"  Groups:")
    for g, label in GROUP_LABELS.items():
        pct = counts.get(g, 0) / n * 100
        lbl = label.replace('\n', ' ')
        print(f"    {lbl:22s}: {counts.get(g,0):3d} ({pct:.1f}%)")

    # Row 0: Pie chart of group distribution
    ax_pie = axes[0, col_idx]
    sizes  = [counts.get(g, 0) for g in ['AA','AB','BA','BB']]
    colors = [GROUP_COLORS[g] for g in ['AA','AB','BA','BB']]
    labels = [f"{GROUP_LABELS[g].replace(chr(10),' ')}\n({s})" for g, s in zip(['AA','AB','BA','BB'], sizes)]
    wedges, texts, autotexts = ax_pie.pie(
        sizes, colors=colors, labels=labels, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 9}
    )
    ax_pie.set_title(f'{ca} vs {cb}\nGroup Distribution (n={n})\n'
                     f'AUC: Full={auc_full:.3f} | fMRI={auc_fmri:.3f} | sMRI={auc_smri:.3f}',
                     fontsize=10)

    # Row 1: Bar chart — probability distribution per group
    ax_bar = axes[1, col_idx]
    group_order = ['AA','AB','BA','BB']
    group_fmri_prob = [df_tk[df_tk['group']==g]['p_fmri'].mean() for g in group_order]
    group_smri_prob = [df_tk[df_tk['group']==g]['p_smri'].mean() for g in group_order]
    x_pos = np.arange(len(group_order))
    width = 0.35
    bars1 = ax_bar.bar(x_pos - width/2, group_fmri_prob, width,
                       color='#2196F3', alpha=0.8, label='fMRI-only prob')
    bars2 = ax_bar.bar(x_pos + width/2, group_smri_prob, width,
                       color='#FF9800', alpha=0.8, label='sMRI-only prob')
    ax_bar.axhline(0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([GROUP_LABELS[g].replace('\n',' ') for g in group_order], fontsize=8)
    ax_bar.set_ylabel('Mean P(positive class)')
    ax_bar.set_ylim(0, 1)
    ax_bar.legend(fontsize=8)
    ax_bar.set_title(f'Mean Predicted Probability by Group', fontsize=10)

    # Annotate bar counts
    for bar, g in zip(bars1, group_order):
        n_g = counts.get(g, 0)
        ax_bar.text(bar.get_x() + bar.get_width()/2, 0.02, f'n={n_g}',
                    ha='center', va='bottom', fontsize=7, color='black')

print("\n" + "="*70)

# Save per-task DataFrames
os.makedirs('results', exist_ok=True)
for tk, ca, cb, _, _ in PCAG_TASKS:
    recs = task_records[tk]
    if recs:
        pd.DataFrame(recs).to_csv(f'results/four_group_{tk}.csv', index=False)
        print(f"[SAVED] results/four_group_{tk}.csv")

plt.tight_layout()
out = 'results/four_group_analysis.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n圖片已儲存：{out}")
