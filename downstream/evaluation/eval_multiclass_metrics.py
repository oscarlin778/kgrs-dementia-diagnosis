"""
eval_multiclass_metrics.py
==========================
W3 revision: 3-way (NC/MCI/AD) staging metrics complementing the OVO binary AUCs.

Runs every held-out test subject (n=49) through all three label-free PCAG
ensembles (same checkpoints/aggregation that reproduce paper Table 2:
0.791 / 0.719 / 0.672), combines them by OVO majority vote, and reports:
  - 3-way balanced accuracy + overall accuracy
  - macro-averaged OVO AUC (Hand & Till; mean of the three pairwise AUCs)
  - per-class precision/recall/F1 and the 3x3 confusion matrix

Aggregation per task (seed-level, matches paper):
  NC_vs_AD  : top-3-by-OOF seeds [42,2024,123], mean
  NC_vs_MCI : 5 seeds (dim128), mean
  MCI_vs_AD : 5 seeds (noaug), median

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_multiclass_metrics.py
"""
import sys, os, re, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, accuracy_score,
                             confusion_matrix, classification_report)

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
import inference_pipeline_v2 as P

LABELS = {0: "NC", 1: "MCI", 2: "AD"}


def detect_site(s):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(s)) else 'TPMIC'


def harmonized_graph(matrix_path, site, fmri_combat):
    mat = np.load(matrix_path)
    mat = mat[0] if mat.ndim == 3 else mat
    mat = ((mat + mat.T) / 2).astype(np.float32)
    mat = P.apply_combat_fmri(mat, site, fmri_combat)
    x = torch.tensor(P.extract_node_features(mat)).unsqueeze(0).to(P.DEVICE)
    adj = torch.tensor(P.build_adj(mat)).unsqueeze(0).to(P.DEVICE)
    return x, adj


def predict_task(seed_groups, x, adj, smri_feat, strategy):
    st = torch.tensor(smri_feat, dtype=torch.float32).unsqueeze(0).to(P.DEVICE)
    seed_probs = []
    for folds in seed_groups:
        fp = []
        for m in folds:
            with torch.no_grad():
                lg = m(x, adj, st)
                lg = lg[0] if isinstance(lg, tuple) else lg
                fp.append(F.softmax(lg, dim=1)[0, 1].item())
        seed_probs.append(np.mean(fp))
    return float(np.median(seed_probs) if strategy == 'median' else np.mean(seed_probs))


def main():
    fmri_combat = P.load_fmri_combat_estimates()
    groups = {
        'NC_vs_AD':  (P.load_pcag_seed_groups('NC_vs_AD',  P.PCAG_NCAD_DIRS),         'mean'),
        'NC_vs_MCI': (P.load_pcag_seed_groups('NC_vs_MCI', P.PCAG_NCMCI_DIRS),        'mean'),
        'MCI_vs_AD': (P.load_pcag_seed_groups('MCI_vs_AD', P.PCAG_MCIAD_ENSEMBLE_DIRS), 'median'),
    }
    combat = {t: P.load_combat_params(t) for t in groups}

    df = pd.read_csv(BASE / "pcag_test_aligned_v2.csv")
    smri_all = pd.read_csv(BASE / "brainiac_features_combined_test_v2.csv")
    cols = [f"Feature_{i}" for i in range(768)]

    y_true, y_pred = [], []
    # store OVO probs for macro-AUC (per pairwise task, restricted to its 2 classes)
    pair_scores = {t: {'p': [], 'y': []} for t in groups}

    print(f"Running 3-way inference on {len(df)} test subjects...")
    for _, r in df.iterrows():
        sid = r['subject_id']; site = detect_site(sid); true = int(r['label'])
        x, adj = harmonized_graph(r['matrix_path'], site, fmri_combat)
        sf_raw = smri_all.iloc[int(r['smri_feat_row'])][cols].values.astype(np.float32)
        p = {}
        for t, (grp, strat) in groups.items():
            sf = P.apply_combat(sf_raw, site, combat[t])
            p[t] = predict_task(grp, x, adj, sf, strat)

        # OVO majority vote (P(positive) where positive = higher-index class)
        votes = {'NC': 0, 'MCI': 0, 'AD': 0}
        votes['AD' if p['NC_vs_AD']  >= 0.5 else 'NC']  += 1
        votes['MCI' if p['NC_vs_MCI'] >= 0.5 else 'NC'] += 1
        votes['AD' if p['MCI_vs_AD']  >= 0.5 else 'MCI'] += 1
        pred = max(votes, key=votes.get)
        y_true.append(LABELS[true]); y_pred.append(pred)

        # collect pairwise scores for macro-AUC
        for t, (cls_neg, cls_pos) in [('NC_vs_AD', (0, 2)), ('NC_vs_MCI', (0, 1)), ('MCI_vs_AD', (1, 2))]:
            if true in (cls_neg, cls_pos):
                pair_scores[t]['p'].append(p[t])
                pair_scores[t]['y'].append(int(true == cls_pos))

    classes = ['NC', 'MCI', 'AD']
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    pair_auc = {t: roc_auc_score(pair_scores[t]['y'], pair_scores[t]['p']) for t in groups}
    macro_ovo_auc = float(np.mean(list(pair_auc.values())))

    print("\n" + "=" * 60)
    print("3-WAY (NC/MCI/AD) STAGING METRICS")
    print("=" * 60)
    print(f"Balanced accuracy   : {bal_acc:.4f}")
    print(f"Overall accuracy    : {acc:.4f}")
    print(f"Macro OVO AUC       : {macro_ovo_auc:.4f}  "
          f"(NC/AD={pair_auc['NC_vs_AD']:.3f}, NC/MCI={pair_auc['NC_vs_MCI']:.3f}, MCI/AD={pair_auc['MCI_vs_AD']:.3f})")
    print("\nConfusion matrix (row=True, col=Pred):")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    print("\n" + classification_report(y_true, y_pred, labels=classes, target_names=classes, zero_division=0))

    out = {
        "n_test": len(df),
        "balanced_accuracy": float(bal_acc),
        "accuracy": float(acc),
        "macro_ovo_auc": macro_ovo_auc,
        "pairwise_auc": {k: float(v) for k, v in pair_auc.items()},
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }
    out_path = BASE / "results" / "multiclass_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
