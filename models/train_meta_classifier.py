"""
Meta-classifier：以 GNN OOF + ResNet OOF 機率為特徵，
對三個二元分類任務各訓練一個 Logistic Regression stacking 模型。

輸入：
  fnp_gnn_v5_{task}_oof_probs.npy
  smri_resnet_v3_{task}_oof_probs.npy

輸出：
  meta_clf_{task}.pkl
  meta_clf_{task}_oof_probs.npy
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")

# ── 路徑設定 ────────────────────────────────────────────────────────────────
GNN_DIR    = "/home/wei-chi/Data/script/checkpoints/fnp_gnn_v5_checkpoints"
RESNET_DIR = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints"
OUT_DIR    = "/home/wei-chi/Data/script/checkpoints/meta_checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

TASKS = [
    ("NC",  "AD",  "NC_vs_AD"),
    ("NC",  "MCI", "NC_vs_MCI"),
    ("MCI", "AD",  "MCI_vs_AD"),
]


# ── 1. 資料對齊 ─────────────────────────────────────────────────────────────
def _extract_sid(path: str) -> str | None:
    """從路徑提取 sub_XXXX 中的四位數字，作為 subject ID。"""
    m = re.search(r"sub[_](\d{4})(?=[_.])", path)
    return m.group(1) if m else None


def align_subjects(gnn_data: dict, resnet_data: dict):
    """
    將 GNN 與 ResNet OOF 資料以 subject ID 對齊。

    回傳：(p_fmri, p_smri, labels, subject_ids)
      p_fmri, p_smri : ndarray (N, 2)
      labels         : ndarray (N,)  int
      subject_ids    : list[str]
    """
    gnn_sids    = [_extract_sid(p) for p in gnn_data["matrix_paths"]]
    resnet_sids = [_extract_sid(p) for p in resnet_data["image_paths"]]

    # 建立 sid → index 查詢表（只保留能提取 ID 的）
    gnn_map    = {sid: i for i, sid in enumerate(gnn_sids)    if sid}
    resnet_map = {sid: i for i, sid in enumerate(resnet_sids) if sid}

    common = sorted(set(gnn_map) & set(resnet_map))
    if not common:
        raise ValueError("No overlapping subjects found between GNN and ResNet OOF data.")

    p_fmri, p_smri, labels, subject_ids = [], [], [], []
    skipped = []

    for sid in common:
        gi = gnn_map[sid]
        ri = resnet_map[sid]
        gl = int(gnn_data["labels"][gi])
        rl = int(resnet_data["labels"][ri])

        if gl != rl:
            skipped.append((sid, gl, rl))
            continue

        p_fmri.append(gnn_data["probs"][gi])
        p_smri.append(resnet_data["probs"][ri])
        labels.append(gl)
        subject_ids.append(sid)

    if skipped:
        print(f"  ⚠ label 不一致，排除 {len(skipped)} 筆：")
        for sid, gl, rl in skipped:
            print(f"    subject {sid}: GNN={gl}, ResNet={rl}")

    return (
        np.array(p_fmri,  dtype=np.float32),
        np.array(p_smri,  dtype=np.float32),
        np.array(labels,  dtype=np.int64),
        subject_ids,
    )


# ── 2. 特徵建構 ─────────────────────────────────────────────────────────────
def _entropy(p: np.ndarray) -> np.ndarray:
    """二元 entropy，shape (N,)。"""
    return -np.sum(p * np.log2(p + 1e-10), axis=1)


def build_features(p_fmri: np.ndarray, p_smri: np.ndarray) -> np.ndarray:
    """
    8 維 meta-feature matrix，shape (N, 8)。
    f1-f4: 各模型各類別機率
    f5:    兩模型對 positive class 的預測分歧
    f6-f7: 各模型預測不確定性（entropy）
    f8:    兩模型 positive 機率交互項
    """
    f1 = p_fmri[:, 1]                        # P(positive | GNN)
    f2 = p_fmri[:, 0]                        # P(negative | GNN)
    f3 = p_smri[:, 1]                        # P(positive | ResNet)
    f4 = p_smri[:, 0]                        # P(negative | ResNet)
    f5 = np.abs(f1 - f3)                     # |disagreement|
    f6 = _entropy(p_fmri)                    # GNN uncertainty
    f7 = _entropy(p_smri)                    # ResNet uncertainty
    f8 = f1 * f3                             # interaction
    return np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8])


# ── 3. OOF 訓練 + 最終模型 ──────────────────────────────────────────────────
def train_meta(X: np.ndarray, y: np.ndarray, task_name: str):
    """
    5-fold stratified OOF 訓練，並用全部資料訓練最終模型。

    回傳：(oof_probs, final_clf)
      oof_probs  : ndarray (N, 2)
      final_clf  : 已 fit 的 LogisticRegression
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(y), 2), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X[tr_idx], y[tr_idx])
        oof_probs[va_idx] = clf.predict_proba(X[va_idx])
        va_auc = roc_auc_score(y[va_idx], oof_probs[va_idx, 1])
        print(f"    Fold {fold}: val AUC = {va_auc:.4f}")

    final_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    final_clf.fit(X, y)
    return oof_probs, final_clf


# ── 4. 評估 ─────────────────────────────────────────────────────────────────
def _best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Youden-J 最佳切點。"""
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresh[np.argmax(j)])


def evaluate(y_true: np.ndarray, oof_probs: np.ndarray,
             p_fmri: np.ndarray, p_smri: np.ndarray,
             task_name: str) -> dict:
    """
    計算 OOF 指標，並與單模型比較。
    positive class = class 1（AD 或 MCI）。
    """
    y_score   = oof_probs[:, 1]
    threshold = _best_threshold(y_true, y_score)
    y_pred    = (y_score >= threshold).astype(int)

    auc  = roc_auc_score(y_true, y_score)
    acc  = accuracy_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)

    # sensitivity = TP / (TP + FN), specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc_gnn    = roc_auc_score(y_true, p_fmri[:, 1])
    auc_resnet = roc_auc_score(y_true, p_smri[:, 1])

    print(f"\n  [{task_name}] OOF 評估結果")
    print(f"  {'':─<46}")
    print(f"  AUC  : Meta={auc:.4f}  |  GNN={auc_gnn:.4f}  |  ResNet={auc_resnet:.4f}")
    print(f"  Acc  : {acc:.4f}  (threshold={threshold:.3f})")
    print(f"  Sens : {sens:.4f}  |  Spec : {spec:.4f}")
    print(f"  Confusion Matrix (TN FP / FN TP):\n"
          f"    {tn:4d} {fp:4d}\n    {fn:4d} {tp:4d}")

    return {
        "task"      : task_name,
        "auc_meta"  : auc,
        "auc_gnn"   : auc_gnn,
        "auc_resnet": auc_resnet,
        "acc"       : acc,
        "sensitivity": sens,
        "specificity": spec,
        "threshold" : threshold,
        "cm"        : cm,
    }


# ── 5. 存檔 ─────────────────────────────────────────────────────────────────
def save_outputs(task_name: str, clf, oof_probs: np.ndarray,
                 y_true: np.ndarray, subject_ids: list, metrics: dict):
    safe = task_name.replace(" ", "_")

    # 最終模型
    clf_path = os.path.join(OUT_DIR, f"meta_clf_{safe}.pkl")
    joblib.dump(clf, clf_path)
    print(f"  Saved model  → {clf_path}")

    # OOF probs（格式與 gnn/resnet oof_probs.npy 一致）
    oof_path = os.path.join(OUT_DIR, f"meta_clf_{safe}_oof_probs.npy")
    np.save(oof_path, {
        "probs"      : oof_probs.astype(np.float32),
        "labels"     : y_true,
        "subject_ids": subject_ids,
        "acc"        : metrics["acc"],
        "auc"        : metrics["auc_meta"],
        "task"       : task_name,
    }, allow_pickle=True)
    print(f"  Saved OOF    → {oof_path}")


# ── 6. 主程式 ────────────────────────────────────────────────────────────────
def run_task(cls_a: str, cls_b: str, task_name: str) -> dict:
    safe = task_name
    print(f"\n{'='*56}")
    print(f"  Task: {task_name}")
    print(f"{'='*56}")

    # 載入
    gnn_data    = np.load(
        os.path.join(GNN_DIR,    f"fnp_gnn_v5_{safe}_oof_probs.npy"),
        allow_pickle=True).item()
    resnet_data = np.load(
        os.path.join(RESNET_DIR, f"smri_resnet_v3_{safe}_oof_probs.npy"),
        allow_pickle=True).item()

    # 對齊
    p_fmri, p_smri, y_true, subject_ids = align_subjects(gnn_data, resnet_data)
    print(f"  對齊後樣本數：{len(y_true)}  "
          f"(class0={int((y_true==0).sum())}, class1={int((y_true==1).sum())})")

    # 特徵
    X = build_features(p_fmri, p_smri)
    print(f"  特徵維度：{X.shape}")

    # 訓練
    print("  OOF 訓練：")
    oof_probs, final_clf = train_meta(X, y_true, task_name)

    # 評估
    metrics = evaluate(y_true, oof_probs, p_fmri, p_smri, task_name)

    # 存檔
    save_outputs(task_name, final_clf, oof_probs, y_true, subject_ids, metrics)

    return metrics


if __name__ == "__main__":
    all_metrics = []
    for cls_a, cls_b, task_name in TASKS:
        m = run_task(cls_a, cls_b, task_name)
        all_metrics.append(m)

    # 總結表格
    print(f"\n{'='*62}")
    print("  總結")
    print(f"{'='*62}")
    header = f"  {'任務':<12} {'GNN AUC':>9} {'ResNet AUC':>11} {'Meta AUC':>10} {'提升':>7}"
    print(header)
    print(f"  {'─'*58}")
    for m in all_metrics:
        best_single = max(m["auc_gnn"], m["auc_resnet"])
        gain = m["auc_meta"] - best_single
        flag = "↑" if gain > 0 else "↓"
        print(f"  {m['task']:<12} {m['auc_gnn']:>9.3f} {m['auc_resnet']:>11.3f}"
              f" {m['auc_meta']:>10.3f} {flag}{abs(gain):>5.3f}")
    print()
