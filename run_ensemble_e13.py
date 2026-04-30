"""
E13 GNN + sMRI ResNet v3 Ensemble (Meta-Classifier)

使用 E13 的 GNN OOF 預測（fMRI）搭配 sMRI ResNet v3 的 OOF 預測，
以 Logistic Regression stacking 訓練 meta-classifier，
並輸出每個任務的 5-fold OOF 評估結果。

執行：python run_ensemble_e13.py
輸出：results/E13_Ensemble/ 下的 metrics.json 與模型檔
"""

import os, re, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix

warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/wei-chi/Data/script')
import save_experiment_results as ser

# ── 路徑 ────────────────────────────────────────────────────────────────────
CSV_PATHS = [
    "/home/wei-chi/Model/_dataset_mapping.csv",
    "/home/wei-chi/Data/dataset_index_116_clean_old.csv",
    "/home/wei-chi/Data/adni_dataset_index_116.csv",
]
MATRIX_DIR       = "/home/wei-chi/Model/processed_116_matrices"
UNIFIED_SPLIT    = "/home/wei-chi/Data/script/unified_subject_split.json"
E13_OOF_PATH     = "/home/wei-chi/Data/script/results/E13_GSL/oof_predictions.npy"
RESNET_DIR       = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints"
OUT_DIR          = "/home/wei-chi/Data/script/results/E13_Ensemble"
META_CKPT_DIR    = "/home/wei-chi/Data/script/checkpoints/meta_checkpoints_e13"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(META_CKPT_DIR, exist_ok=True)

TASKS = [
    ("nc_ad",  "NC_vs_AD",  ["NC", "AD"],  {"NC": 0, "AD": 1}),
    ("nc_mci", "NC_vs_MCI", ["NC", "MCI"], {"NC": 0, "MCI": 1}),
    ("mci_ad", "MCI_vs_AD", ["MCI", "AD"], {"MCI": 0, "AD": 1}),
]


# ── 1. 重建 df_full 與 subject ID 映射 ──────────────────────────────────────
def get_subject_id(p):
    return re.sub(r'^(sub-|sub_|old_dswau)', '',
                  re.sub(r'(_matrix.*|.*nii.gz)$', '', os.path.basename(p))).strip()

def build_df_full():
    valid_data, seen = [], set()
    for path in CSV_PATHS:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m_path = (
                row.get('matrix_path') or
                (os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy")
                 if pd.notna(row.get('new_id_base')) else None) or
                (os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy")
                 if pd.notna(row.get('Subject')) else None)
            )
            if not (m_path and os.path.exists(m_path)) or m_path in seen:
                continue
            if (np.load(m_path).shape == (116, 116) and
                    str(row.get('diagnosis', '')).upper() in ['NC', 'MCI', 'AD']):
                valid_data.append({
                    'matrix_path': m_path,
                    'diagnosis':   str(row['diagnosis']).upper(),
                    'source':      'ADNI' if ('adni' in m_path.lower() or
                                               'old_dswau' in m_path.lower()) else 'TPMIC',
                })
                seen.add(m_path)

    df = pd.DataFrame(valid_data)
    df['sid'] = df['matrix_path'].apply(get_subject_id)
    return df


# ── 2. E13 GNN OOF → {sid: (prob_pos, label)} ──────────────────────────────
def build_gnn_lookup(df_full, e13_oof, task_key, diag_classes, label_map):
    """
    E13 OOF 的順序 = df_full 中「在 unified_split 內且 diagnosis 在 diag_classes」
    的受試者，依 df_full 原始順序排列。
    """
    with open(UNIFIED_SPLIT) as f:
        splits = json.load(f)
    in_split = set()
    for ids in splits.values():
        in_split.update(ids)

    oof    = e13_oof[task_key]           # {"true": array, "prob": array}
    probs  = oof["prob"]                 # shape (N,)  — prob of positive class
    labels = oof["true"]                 # shape (N,)

    # 依 df_full 順序篩出應出現在 OOF 中的受試者
    eligible_rows = df_full[
        df_full['sid'].isin(in_split) &
        df_full['diagnosis'].isin(diag_classes)
    ].reset_index(drop=True)

    assert len(eligible_rows) == len(probs), (
        f"[{task_key}] 預期 {len(eligible_rows)} 筆 OOF，實際 {len(probs)} 筆")

    lookup = {}
    for i, row in eligible_rows.iterrows():
        expected_label = label_map[row['diagnosis']]
        assert expected_label == int(labels[i]), (
            f"[{task_key}] sid={row['sid']} label mismatch")
        lookup[row['sid']] = (float(probs[i]), int(labels[i]))

    return lookup


# ── 3. sMRI ResNet OOF → {sid: (prob_array, label)} ────────────────────────
def build_smri_lookup(task_file_key):
    path = os.path.join(RESNET_DIR, f"smri_resnet_v3_{task_file_key}_oof_probs.npy")
    d    = np.load(path, allow_pickle=True).item()
    probs = d['probs']          # (N, 2)
    labels = d['labels']        # (N,)
    paths  = d['image_paths']   # list of N paths

    lookup = {}
    for i, img_p in enumerate(paths):
        m = re.search(r'sub[_](\d{4})(?=[_.])', img_p)
        if not m:
            continue
        sid = m.group(1)
        lookup[sid] = (probs[i], int(labels[i]))
    return lookup


# ── 4. 對齊 ──────────────────────────────────────────────────────────────────
def align(gnn_lookup, smri_lookup, task_key):
    common = sorted(set(gnn_lookup) & set(smri_lookup))
    print(f"  [{task_key}] GNN={len(gnn_lookup)} | sMRI={len(smri_lookup)} | 交集={len(common)}")

    p_gnn, p_smri, labels, sids, skipped = [], [], [], [], []
    for sid in common:
        g_prob, g_lbl = gnn_lookup[sid]
        s_probs, s_lbl = smri_lookup[sid]
        if g_lbl != s_lbl:
            skipped.append(sid)
            continue
        p_gnn.append([1 - g_prob, g_prob])
        p_smri.append(s_probs)
        labels.append(g_lbl)
        sids.append(sid)

    if skipped:
        print(f"  ⚠  label 不一致，排除 {len(skipped)} 筆: {skipped}")

    return (np.array(p_gnn, dtype=np.float32),
            np.array(p_smri, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            sids)


# ── 5. Meta-feature 建構 ────────────────────────────────────────────────────
def _entropy(p):
    return -np.sum(p * np.log2(p + 1e-10), axis=1)

def build_features(p_gnn, p_smri):
    f1 = p_gnn[:, 1]
    f2 = p_gnn[:, 0]
    f3 = p_smri[:, 1]
    f4 = p_smri[:, 0]
    f5 = np.abs(f1 - f3)
    f6 = _entropy(p_gnn)
    f7 = _entropy(p_smri)
    f8 = f1 * f3
    return np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8])


# ── 6. OOF 訓練 ─────────────────────────────────────────────────────────────
def train_meta_oof(X, y, task_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(y), 2), dtype=np.float64)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])
        print(f"    Fold {fold}: val AUC = {roc_auc_score(y[va], oof[va, 1]):.4f}")
    final_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    final_clf.fit(X, y)
    return oof, final_clf


# ── 7. 評估 ─────────────────────────────────────────────────────────────────
def evaluate(y, oof, p_gnn, p_smri, task_name):
    score = oof[:, 1]
    fpr, tpr, thresh = roc_curve(y, score)
    best_thr = float(thresh[np.argmax(tpr - fpr)])
    y_pred   = (score >= best_thr).astype(int)

    auc_meta   = roc_auc_score(y, score)
    auc_gnn    = roc_auc_score(y, p_gnn[:, 1])
    auc_smri   = roc_auc_score(y, p_smri[:, 1])
    acc        = accuracy_score(y, y_pred)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)

    gain_over_best = auc_meta - max(auc_gnn, auc_smri)
    flag = "↑" if gain_over_best >= 0 else "↓"

    print(f"\n  ╔══ [{task_name}] Ensemble 結果 ══")
    print(f"  ║  AUC  : Meta={auc_meta:.4f}  GNN={auc_gnn:.4f}  sMRI={auc_smri:.4f}  ({flag}{abs(gain_over_best):.4f})")
    print(f"  ║  ACC  : {acc:.4f}  (threshold={best_thr:.3f})")
    print(f"  ║  Sens : {sens:.4f}  |  Spec : {spec:.4f}")
    print(f"  ║  N    : {len(y)}  (class0={int((y==0).sum())}, class1={int((y==1).sum())})")
    print(f"  ╚══════════════════════════════════")

    return {
        "auc":         float(auc_meta),
        "auc_gnn":     float(auc_gnn),
        "auc_smri":    float(auc_smri),
        "acc":         float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "threshold":   float(best_thr),
        "n_subjects":  int(len(y)),
        "fpr":         fpr.tolist(),
        "tpr":         tpr.tolist(),
    }


# ── 8. 主程式 ────────────────────────────────────────────────────────────────
def main():
    print("載入資料...")
    df_full  = build_df_full()
    e13_oof  = np.load(E13_OOF_PATH, allow_pickle=True).item()
    exp_metrics = {}

    for task_key, task_file_key, diag_classes, label_map in TASKS:
        print(f"\n{'='*56}")
        print(f"  任務：{task_file_key}")
        print(f"{'='*56}")

        gnn_lookup  = build_gnn_lookup(df_full, e13_oof, task_key, diag_classes, label_map)
        smri_lookup = build_smri_lookup(task_file_key)

        p_gnn, p_smri, y, sids = align(gnn_lookup, smri_lookup, task_key)
        if len(y) < 10:
            print(f"  ⚠ 對齊後樣本不足 ({len(y)})，跳過此任務")
            continue

        X = build_features(p_gnn, p_smri)
        print(f"  特徵維度：{X.shape}，class0={int((y==0).sum())}, class1={int((y==1).sum())}")
        print("  5-fold OOF 訓練：")
        oof, final_clf = train_meta_oof(X, y, task_file_key)

        metrics = evaluate(y, oof, p_gnn, p_smri, task_file_key)
        exp_metrics[task_file_key] = metrics

        # 儲存 meta-classifier
        clf_path = os.path.join(META_CKPT_DIR, f"meta_clf_{task_file_key}.pkl")
        joblib.dump(final_clf, clf_path)
        print(f"  Saved model → {clf_path}")

        # 儲存 OOF 預測
        oof_path = os.path.join(META_CKPT_DIR, f"meta_oof_{task_file_key}.npy")
        np.save(oof_path, {"probs": oof.astype(np.float32), "labels": y,
                           "subject_ids": sids, "auc": metrics["auc"],
                           "task": task_file_key}, allow_pickle=True)
        print(f"  Saved OOF  → {oof_path}")

    # 總結
    print(f"\n{'='*62}")
    print("  總結")
    print(f"{'='*62}")
    print(f"  {'任務':<14} {'GNN AUC':>9} {'sMRI AUC':>10} {'Meta AUC':>10} {'提升':>8}")
    print(f"  {'─'*55}")
    for t_file_key in [t[1] for t in TASKS]:
        if t_file_key not in exp_metrics:
            continue
        m = exp_metrics[t_file_key]
        best   = max(m['auc_gnn'], m['auc_smri'])
        gain   = m['auc'] - best
        flag   = "↑" if gain >= 0 else "↓"
        print(f"  {t_file_key:<14} {m['auc_gnn']:>9.3f} {m['auc_smri']:>10.3f}"
              f" {m['auc']:>10.3f}  {flag}{abs(gain):.3f}")

    ser.save_metrics("E13_Ensemble", exp_metrics)
    ser.update_comparison_chart()
    print(f"\n結果已儲存至 {OUT_DIR}")


if __name__ == "__main__":
    main()
