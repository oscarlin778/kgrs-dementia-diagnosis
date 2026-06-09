"""
calibrate_ad_threshold.py
==========================
用 OOF 預測找最佳 AD 決策門檻，提升 AD 召回率。

目前 OVO 投票：score["AD"] = p_nc_ad + p_mci_ad，取 argmax。
改法：score["AD"] >= threshold → 預測 AD（比 argmax 更積極）。

用 OOF 預測 grid search 門檻，目標：最大化 F2-score（偏重 recall）。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix

BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
RES_DIR  = BASE_DIR / "results"

# ── 載入 OOF 機率 ────────────────────────────────────────────────────
# 使用 ensemble OOF（5 seeds 各有 OOF，取平均）
def load_oof_probs():
    """
    讀取 NC_vs_AD 和 MCI_vs_AD 的 OOF probs（ensemble 平均）。
    回傳 DataFrame with columns: subject_id, label, p_nc_ad, p_mci_ad
    """
    from aggregate_ensemble_oof_selected import SEEDS
    import json

    TASKS = {
        "NC_vs_AD":  {"classes": [0, 2], "pos": 2},
        "MCI_vs_AD": {"classes": [1, 2], "pos": 2},
        "NC_vs_MCI": {"classes": [0, 1], "pos": 1},
    }

    df_meta = pd.read_csv(BASE_DIR / "kd_train_aligned_v2.csv")[["subject_id", "label"]]
    df_meta = df_meta.set_index("subject_id")

    seed_probs = {"NC_vs_AD": [], "MCI_vs_AD": [], "NC_vs_MCI": []}
    seed_sids  = {"NC_vs_AD": None, "MCI_vs_AD": None, "NC_vs_MCI": None}

    for seed in SEEDS:
        s_tag = "" if seed == 42 else f"_s{seed}"
        for task in ["NC_vs_AD", "MCI_vs_AD", "NC_vs_MCI"]:
            variant = "aug_mix0.2_de0.2" if task == "NC_vs_AD" else ""
            if variant:
                npz = RES_DIR / f"pcag_combat_{task}_probs_v2_fmricombat_nolabel_aug_{variant}{s_tag}.npz"
            else:
                npz = RES_DIR / f"pcag_combat_{task}_probs_v2_fmricombat_nolabel{s_tag}.npz"
            if not npz.exists():
                # fallback
                npz = RES_DIR / f"pcag_combat_{task}_probs_v2_fmricombat_nolabel{s_tag}.npz"
            if not npz.exists():
                continue
            d = np.load(npz, allow_pickle=True)
            seed_probs[task].append(d["oof_probs"])
            if seed_sids[task] is None:
                seed_sids[task] = d["oof_labels"], np.arange(len(d["oof_labels"]))

    # Average across seeds
    oof = {}
    for task in ["NC_vs_AD", "MCI_vs_AD", "NC_vs_MCI"]:
        if seed_probs[task]:
            oof[task] = np.mean(seed_probs[task], axis=0)
        else:
            oof[task] = None

    return oof, seed_sids


def ovo_predict_3class(p_nc_ad, p_mci_ad, p_nc_mci, ad_threshold=None):
    """
    OVO aggregation。
    ad_threshold: 若 score_AD >= threshold，強制預測 AD。
    若 None，用 argmax（原始邏輯）。
    """
    n = len(p_nc_ad)
    preds = []
    for i in range(n):
        score = {
            "NC":  (1 - p_nc_ad[i]) + (1 - p_nc_mci[i]),
            "MCI": p_nc_mci[i]      + (1 - p_mci_ad[i]),
            "AD":  p_nc_ad[i]       + p_mci_ad[i],
        }
        if ad_threshold is not None and score["AD"] >= ad_threshold:
            preds.append(2)  # AD
        else:
            cls = max(score, key=score.get)
            preds.append({"NC": 0, "MCI": 1, "AD": 2}[cls])
    return np.array(preds)


def f2_score(y_true, y_pred, ad_label=2):
    """F2-score for AD class (偏重 recall)."""
    tp = np.sum((y_pred == ad_label) & (y_true == ad_label))
    fp = np.sum((y_pred == ad_label) & (y_true != ad_label))
    fn = np.sum((y_pred != ad_label) & (y_true == ad_label))
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return (1 + 4) * prec * rec / (4 * prec + rec + 1e-8)  # beta=2


def main():
    print("Loading OOF predictions...")

    # 直接讀已有的 OOF npz（seed=42 最佳 seed for NC_vs_AD, seed=456 for NC_vs_MCI）
    # 用最具代表性的 OOF（single best seeds per task）
    results = {}
    for task, fname_pattern in [
        ("NC_vs_AD",  "pcag_combat_NC_vs_AD_probs_v2_fmricombat_nolabel_aug_mix0.2_de0.2.npz"),
        ("MCI_vs_AD", "pcag_combat_MCI_vs_AD_probs_v2_fmricombat_nolabel.npz"),
        ("NC_vs_MCI", "pcag_combat_NC_vs_MCI_probs_v2_fmricombat_nolabel_s456.npz"),
    ]:
        path = RES_DIR / fname_pattern
        if not path.exists():
            # fallback to seed=42
            path = RES_DIR / f"pcag_combat_{task}_probs_v2_fmricombat_nolabel.npz"
        d = np.load(path, allow_pickle=True)
        results[task] = {
            "oof_probs":  d["oof_probs"],
            "oof_labels": d["oof_labels"],
        }
        print(f"  {task}: n_oof={len(d['oof_probs'])}")

    # Align subjects — find common index
    # NC_vs_AD OOF: NC(0) and AD(2) patients only
    # MCI_vs_AD OOF: MCI(1) and AD(2) patients only
    # NC_vs_MCI OOF: NC(0) and MCI(1) patients only

    # Build full training label array from pcag_train_aligned_v2.csv
    df_train = pd.read_csv(BASE_DIR / "pcag_train_aligned_v2.csv")[["subject_id", "label"]]
    labels_full = df_train["label"].values  # 155 patients
    sids_full   = df_train["subject_id"].values

    # For each binary task, map OOF probs back to all-patient array
    # (patients not in the task have prob = NaN)
    N = len(df_train)
    p_nc_ad  = np.full(N, np.nan)
    p_mci_ad = np.full(N, np.nan)
    p_nc_mci = np.full(N, np.nan)

    for task, p_arr, label_arr, task_labels in [
        ("NC_vs_AD",  p_nc_ad,  results["NC_vs_AD"]["oof_labels"],  [0, 2]),
        ("MCI_vs_AD", p_mci_ad, results["MCI_vs_AD"]["oof_labels"], [1, 2]),
        ("NC_vs_MCI", p_nc_mci, results["NC_vs_MCI"]["oof_labels"], [0, 1]),
    ]:
        task_mask = np.isin(labels_full, task_labels)
        task_idx  = np.where(task_mask)[0]
        probs = results[task]["oof_probs"]
        if len(probs) == len(task_idx):
            p_arr[task_idx] = probs
        else:
            print(f"  [WARN] {task}: oof len mismatch ({len(probs)} vs {len(task_idx)})")

    # Only keep AD patients (label=2) and patients relevant to both NC_vs_AD and MCI_vs_AD
    # For 3-class OVO, we need all three tasks' probs
    valid_mask = ~(np.isnan(p_nc_ad) & np.isnan(p_mci_ad))
    # Fill NaN with 0.5 (neutral) for non-relevant tasks
    p_nc_ad_f  = np.where(np.isnan(p_nc_ad),  0.5, p_nc_ad)
    p_mci_ad_f = np.where(np.isnan(p_mci_ad), 0.5, p_mci_ad)
    p_nc_mci_f = np.where(np.isnan(p_nc_mci), 0.5, p_nc_mci)

    y_true = labels_full  # 0=NC, 1=MCI, 2=AD

    print(f"\nTraining set: NC={np.sum(y_true==0)}, MCI={np.sum(y_true==1)}, AD={np.sum(y_true==2)}")

    # ── Baseline (argmax, no threshold) ──────────────────────────────
    y_base = ovo_predict_3class(p_nc_ad_f, p_mci_ad_f, p_nc_mci_f, ad_threshold=None)
    ad_mask = y_true == 2
    print("\n=== Baseline (argmax, no AD threshold) ===")
    cm = confusion_matrix(y_true, y_base, labels=[0,1,2])
    print(f"Confusion matrix (rows=true, cols=pred):\n{cm}")
    print(f"  AD recall    = {recall_score(y_true, y_base, labels=[2], average='macro'):.3f}")
    print(f"  AD precision = {precision_score(y_true, y_base, labels=[2], average='macro', zero_division=0):.3f}")
    print(f"  F2 (AD)      = {f2_score(y_true, y_base):.3f}")
    print(f"  Balanced acc = {balanced_accuracy_score(y_true, y_base):.3f}")

    # ── Grid search over AD threshold ────────────────────────────────
    print("\n=== Grid Search: AD threshold ===")
    print(f"{'Threshold':>10} {'AD Recall':>10} {'AD Prec':>9} {'F2(AD)':>8} {'Bal-Acc':>9}")
    print("-" * 55)

    best_f2, best_thresh, best_result = 0, None, None
    for thresh in np.arange(0.6, 1.61, 0.05):
        y_pred = ovo_predict_3class(p_nc_ad_f, p_mci_ad_f, p_nc_mci_f, ad_threshold=thresh)
        rec  = recall_score(y_true, y_pred, labels=[2], average='macro')
        prec = precision_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)
        f2   = f2_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        marker = " ← best F2" if f2 > best_f2 else ""
        print(f"{thresh:>10.2f} {rec:>10.3f} {prec:>9.3f} {f2:>8.3f} {bacc:>9.3f}{marker}")
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = thresh
            best_result = (y_pred, rec, prec, f2, bacc)

    print(f"\n=== Best threshold: {best_thresh:.2f} ===")
    y_best, rec, prec, f2, bacc = best_result
    cm_best = confusion_matrix(y_true, y_best, labels=[0,1,2])
    print(f"Confusion matrix:\n{cm_best}")
    print(f"  AD recall    = {rec:.3f}  (was {recall_score(y_true, y_base, labels=[2], average='macro'):.3f})")
    print(f"  AD precision = {prec:.3f}")
    print(f"  F2 (AD)      = {f2:.3f}")
    print(f"  Balanced acc = {bacc:.3f}")
    print(f"\n→ Use ad_threshold={best_thresh:.2f} in the final OVO prediction")


if __name__ == "__main__":
    main()
