"""
run_vitmci_eval.py
==================
使用 vit_mci.ckpt 重新提取 768-d 特徵，並進行 OVO 三分類評估。

執行方式:
    conda run -n AD python3 run_vitmci_eval.py

流程:
    Step 1  載入凍結的 vit_mci 特徵提取器
    Step 2  提取訓練集特徵 (164 人) → vitmci_features_train.csv
    Step 3  提取測試集特徵 (27 人)  → vitmci_features_test.csv
    Step 4  OVO 5-fold CV + TPMIC 測試集評估
    Step 5  與 ResNet baseline 對比並輸出報告
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
TRAIN_CSV    = SCRIPT_DIR / "brainiac_combined_train.csv"
TEST_CSV     = SCRIPT_DIR / "brainiac_tpmic_test.csv"
FEAT_TRAIN   = SCRIPT_DIR / "vitmci_features_train.csv"
FEAT_TEST    = SCRIPT_DIR / "vitmci_features_test.csv"
RESULTS_DIR  = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 與 BrainIAC.ckpt 的 OOF AUC 對比基準
RESNET_BASELINE = {"NC vs AD": 0.771, "NC vs MCI": 0.731, "MCI vs AD": 0.775}
BRAINIAC_OOF    = {"NC vs AD": 0.551, "NC vs MCI": 0.593, "MCI vs AD": 0.500}
BRAINIAC_TEST   = {"NC vs AD": 0.938, "NC vs MCI": 0.683, "MCI vs AD": 0.567}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 & 2 & 3  特徵提取
# ──────────────────────────────────────────────────────────────────────────────
def extract_all_features(model, transforms, csv_path: Path, out_path: Path,
                         device: torch.device, desc: str) -> pd.DataFrame:
    """
    讀取 CSV，對每個 NIfTI 提取特徵，存成 CSV。
    CSV 格式: pat_id, label, Feature_0 ... Feature_767
    """
    if out_path.exists():
        print(f"[SKIP] {out_path.name} 已存在，直接讀取")
        return pd.read_csv(out_path)

    df = pd.read_csv(csv_path)
    all_rows = []
    skipped  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        nii_path = str(row["pat_id"]) + ".nii.gz"
        if not os.path.exists(nii_path):
            print(f"  [WARN] 找不到檔案，跳過: {nii_path}")
            skipped += 1
            continue

        try:
            img_tensor = transforms(nii_path)          # (1, 96, 96, 96)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)   # 補 channel dim
            img_tensor = img_tensor.unsqueeze(0)       # 補 batch dim → (1,1,96,96,96)
        except Exception as e:
            print(f"  [ERROR] 前處理失敗 {nii_path}: {e}")
            skipped += 1
            continue

        feats = extract_features(model, img_tensor, device)   # (1, 768)
        feat_dict = {f"Feature_{i}": feats[0, i].item() for i in range(768)}
        feat_dict["pat_id"] = row["pat_id"]
        feat_dict["label"]  = int(row["label"])
        all_rows.append(feat_dict)

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(out_path, index=False)
    print(f"[SAVE] {out_path.name}：{len(result_df)} 人（跳過 {skipped} 人）")
    return result_df


# ──────────────────────────────────────────────────────────────────────────────
# Step 4  OVO 評估
# ──────────────────────────────────────────────────────────────────────────────
def run_ovo_evaluation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    import json

    feat_cols = [c for c in train_df.columns if c.startswith("Feature_")]
    X_train_full = train_df[feat_cols].values
    y_train_full = train_df["label"].values
    X_test_full  = test_df[feat_cols].values
    y_test_full  = test_df["label"].values

    # label: 0=NC, 1=MCI, 2=AD
    TASKS = {
        "NC vs AD":  {"classes": [0, 2], "map": {0: 0, 2: 1}},
        "NC vs MCI": {"classes": [0, 1], "map": {0: 0, 1: 1}},
        "MCI vs AD": {"classes": [1, 2], "map": {1: 0, 2: 1}},
    }

    oof_results  = {}
    test_results = {}

    for task_name, info in TASKS.items():
        classes   = info["classes"]
        label_map = info["map"]

        # ── 訓練集過濾 ──────────────────────────────────────────────────────
        mask_tr = np.isin(y_train_full, classes)
        X_tr = X_train_full[mask_tr]
        y_tr = np.array([label_map[l] for l in y_train_full[mask_tr]])

        # ── 測試集過濾 ──────────────────────────────────────────────────────
        mask_te = np.isin(y_test_full, classes)
        X_te = X_test_full[mask_te]
        y_te = np.array([label_map[l] for l in y_test_full[mask_te]])

        print(f"\n  [{task_name}]  訓練: {len(y_tr)} 人  測試: {len(y_te)} 人")

        # ── 5-fold OOF CV ────────────────────────────────────────────────────
        skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_pred = np.zeros(len(y_tr))

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            scaler = StandardScaler()
            X_f_tr = scaler.fit_transform(X_tr[tr_idx])
            X_f_val = scaler.transform(X_tr[val_idx])

            clf = LogisticRegression(
                max_iter=1000, random_state=42, C=1.0, solver="lbfgs"
            )
            clf.fit(X_f_tr, y_tr[tr_idx])
            oof_pred[val_idx] = clf.predict_proba(X_f_val)[:, 1]

        oof_auc = roc_auc_score(y_tr, oof_pred) if len(np.unique(y_tr)) > 1 else 0.5
        oof_acc = accuracy_score(y_tr, (oof_pred >= 0.5).astype(int))
        oof_results[task_name] = {"auc": round(oof_auc, 4), "acc": round(oof_acc, 4)}
        print(f"    OOF AUC: {oof_auc:.4f}  OOF ACC: {oof_acc:.4f}")

        # ── 全量訓練 → 測試集 ────────────────────────────────────────────────
        scaler_full = StandardScaler()
        X_tr_scaled = scaler_full.fit_transform(X_tr)
        X_te_scaled  = scaler_full.transform(X_te)

        clf_full = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, solver="lbfgs"
        )
        clf_full.fit(X_tr_scaled, y_tr)
        test_prob = clf_full.predict_proba(X_te_scaled)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        test_auc = roc_auc_score(y_te, test_prob) if len(np.unique(y_te)) > 1 else 0.5
        test_acc = accuracy_score(y_te, test_pred)
        test_results[task_name] = {
            "auc": round(test_auc, 4),
            "acc": round(test_acc, 4),
            "n_samples": int(len(y_te)),
        }
        print(f"    Test AUC: {test_auc:.4f}  Test ACC: {test_acc:.4f}")

    return oof_results, test_results


# ──────────────────────────────────────────────────────────────────────────────
# Step 5  彙整報告
# ──────────────────────────────────────────────────────────────────────────────
def print_report(oof_results: dict, test_results: dict):
    import json

    print()
    print("=" * 72)
    print("  vit_mci.ckpt 特徵提取器 — OVO 評估報告")
    print("=" * 72)

    # OOF 對比
    print()
    print("【OOF AUC 對比 (5-fold CV on combined_train)】")
    print(f"{'Task':<12} {'ResNet':>10} {'BrainIAC':>10} {'vit_mci':>10} {'vs ResNet':>10}")
    print("-" * 55)
    for task in ["NC vs AD", "NC vs MCI", "MCI vs AD"]:
        r = RESNET_BASELINE[task]
        b = BRAINIAC_OOF[task]
        v = oof_results[task]["auc"]
        diff = v - r
        mark = "↑" if diff > 0 else "↓"
        print(f"{task:<12} {r:>10.3f} {b:>10.3f} {v:>10.3f} {mark}{abs(diff):>8.3f}")

    # Test AUC 對比
    print()
    print("【TPMIC 測試集 AUC 對比 (n=27)】")
    print(f"{'Task':<12} {'BrainIAC_test':>14} {'vit_mci_test':>13}")
    print("-" * 42)
    for task in ["NC vs AD", "NC vs MCI", "MCI vs AD"]:
        b = BRAINIAC_TEST[task]
        v = test_results[task]["auc"]
        print(f"{task:<12} {b:>14.3f} {v:>13.3f}")

    # 儲存 JSON
    out = {
        "oof":  oof_results,
        "test": test_results,
        "baseline_resnet_oof":    RESNET_BASELINE,
        "baseline_brainiac_oof":  BRAINIAC_OOF,
        "baseline_brainiac_test": BRAINIAC_TEST,
    }
    out_path = RESULTS_DIR / "eval_vitmci_tpmic_test.json"
    import json
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"[SAVE] 結果已儲存至 {out_path}")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPT_DIR))
    from brainiac_extractor import (
        load_vitmci_extractor,
        extract_features,
        get_preprocessing_transforms,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")

    # Step 1: 載入模型
    print("\n[Step 1] 載入 vit_mci.ckpt 特徵提取器...")
    model = load_vitmci_extractor(
        "/home/wei-chi/Alzheimers_Project/external_data/scripts/models/checkpoints/"
        "finetune_tpmic_full/vit_mci.ckpt",
        device, verbose=True,
    )
    transforms = get_preprocessing_transforms(already_preprocessed=True)

    # Step 2: 提取訓練集特徵
    print("\n[Step 2] 提取訓練集特徵...")
    train_feat_df = extract_all_features(
        model, transforms, TRAIN_CSV, FEAT_TRAIN, device,
        desc="Train features",
    )

    # Step 3: 提取測試集特徵
    print("\n[Step 3] 提取測試集特徵...")
    test_feat_df = extract_all_features(
        model, transforms, TEST_CSV, FEAT_TEST, device,
        desc="Test features",
    )

    print(f"\n訓練集: {len(train_feat_df)} 人  測試集: {len(test_feat_df)} 人")

    # Step 4: OVO 評估
    print("\n[Step 4] OVO 評估...")
    oof_results, test_results = run_ovo_evaluation(train_feat_df, test_feat_df)

    # Step 5: 輸出報告
    print_report(oof_results, test_results)
