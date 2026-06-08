import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from pathlib import Path

# --- Paths ---
BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

FMRI_TRAIN_CSV = BASE_DIR / "splits/combined_train.csv"
FMRI_TEST_CSV  = BASE_DIR / "splits/combined_test.csv"
PCAG_TRAIN_CSV = BASE_DIR / "pcag_train_aligned.csv"
GLOBAL_METRICS_JSON = RESULTS_DIR / "global_oof_hybrid_metrics.json"

TASKS = {
    "NC_vs_AD": {
        "classes": [0, 2], 
        "pos": 2, 
        "use_pcag": True,
        "kd_npz": "kd_NC_vs_AD_a3_T3_probs.npz",
        "pcag_npz": "pcag_combat_NC_vs_AD_probs.npz"
    },
    "NC_vs_MCI": {
        "classes": [0, 1], 
        "pos": 1, 
        "use_pcag": True,
        "kd_npz": "kd_NC_vs_MCI_a3_T3_probs.npz",
        "pcag_npz": "pcag_combat_NC_vs_MCI_probs.npz"
    },
    "MCI_vs_AD": {
        "classes": [1, 2], 
        "pos": 2, 
        "use_pcag": False,
        "kd_npz": "kd_MCI_vs_AD_a10_T3_probs.npz",
        "pcag_npz": None
    },
}

def fit_platt_scaler(oof_probs, oof_labels):
    """Fit Platt scaling: logistic regression on logit-transformed OOF probs."""
    eps = 1e-7
    logits = np.log(oof_probs + eps) - np.log(1 - oof_probs + eps)  # logit transform
    lr = LogisticRegression(C=1e4, solver='lbfgs', max_iter=1000)
    lr.fit(logits.reshape(-1, 1), oof_labels)
    return lr

def apply_platt_scaler(lr, probs):
    eps = 1e-7
    logits = np.log(probs + eps) - np.log(1 - probs + eps)
    return lr.predict_proba(logits.reshape(-1, 1))[:, 1]

def calculate_metrics(y_true, y_prob):
    # Global optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    best_thr = thresholds[idx]
    
    y_pred = (y_prob >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "sens": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "spec": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1": float(f1_score(y_true, y_pred)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "threshold": float(best_thr),
        "cm": cm.tolist()
    }
    return metrics

def main():
    df_tr_full = pd.read_csv(FMRI_TRAIN_CSV)
    df_te_full = pd.read_csv(FMRI_TEST_CSV)
    df_pcag_tr_full = pd.read_csv(PCAG_TRAIN_CSV)
    
    raw_metrics = {}
    if GLOBAL_METRICS_JSON.exists():
        with open(GLOBAL_METRICS_JSON, 'r') as f:
            raw_metrics = json.load(f)

    calibrated_results = {}

    for task_name, cfg in TASKS.items():
        print(f"\nProcessing Task: {task_name}")
        classes = cfg["classes"]
        pos = cfg["pos"]
        
        # 1. Reconstruction of Subject IDs for OOF
        df_tr_task = df_tr_full[df_tr_full["label"].isin(classes)].reset_index(drop=True)
        kd_tr_ids = df_tr_task["subject_id"].tolist()
        kd_tr_labels = (df_tr_task["label"] == pos).astype(int).values
        
        df_pcag_tr_task = df_pcag_tr_full[df_pcag_tr_full["label"].isin(classes)].reset_index(drop=True)
        pcag_tr_ids = df_pcag_tr_task["subject_id"].tolist()
        pcag_tr_labels = (df_pcag_tr_task["label"] == pos).astype(int).values
        
        # 2. Loading Probabilities
        kd_npz = np.load(RESULTS_DIR / cfg["kd_npz"], allow_pickle=True)
        if not np.array_equal(kd_npz["oof_labels"], kd_tr_labels):
            print(f"  [WARNING] KD OOF label mismatch for {task_name}")
        
        # 3. Fit Platt Scalers
        kd_scaler = fit_platt_scaler(kd_npz["oof_probs"], kd_tr_labels)
        kd_tr_calib = apply_platt_scaler(kd_scaler, kd_npz["oof_probs"])
        kd_te_calib = apply_platt_scaler(kd_scaler, kd_npz["test_probs"])
        
        kd_tr_map = {sid: prob for sid, prob in zip(kd_tr_ids, kd_tr_calib)}
        
        kd_te_task = df_te_full[df_te_full["label"].isin(classes)].reset_index(drop=True)
        kd_te_ids = kd_te_task["subject_id"].tolist()
        kd_te_map = {sid: prob for sid, prob in zip(kd_te_ids, kd_te_calib)}
        
        pcag_tr_map = {}
        pcag_te_map = {}
        pcag_tr_calib = None
        pcag_oof_raw = None
        
        if cfg["use_pcag"]:
            pcag_npz = np.load(RESULTS_DIR / cfg["pcag_npz"], allow_pickle=True)
            if not np.array_equal(pcag_npz["oof_labels"], pcag_tr_labels):
                 print(f"  [WARNING] PCAG OOF label mismatch for {task_name}")
            
            pcag_scaler = fit_platt_scaler(pcag_npz["oof_probs"], pcag_tr_labels)
            pcag_tr_calib = apply_platt_scaler(pcag_scaler, pcag_npz["oof_probs"])
            pcag_te_calib = apply_platt_scaler(pcag_scaler, pcag_npz["test_probs"])
            
            pcag_tr_map = {sid: prob for sid, prob in zip(pcag_tr_ids, pcag_tr_calib)}
            pcag_te_map = {sid: prob for sid, prob in zip(pcag_npz["test_subject_ids"], pcag_te_calib)}
            pcag_oof_raw = pcag_npz["oof_probs"]

        # 4. Hybrid Routing (Global)
        all_probs, all_labels = [], []
        n_pcag, n_kd = 0, 0
        
        # Train Routing
        for sid in kd_tr_ids:
            label = 1 if df_tr_full.loc[df_tr_full["subject_id"] == sid, "label"].values[0] == pos else 0
            if cfg["use_pcag"] and sid in pcag_tr_map:
                prob = pcag_tr_map[sid]
                n_pcag += 1
            else:
                prob = kd_tr_map[sid]
                n_kd += 1
            all_probs.append(prob)
            all_labels.append(label)
            
        # Test Routing
        for sid in kd_te_ids:
            label = 1 if df_te_full.loc[df_te_full["subject_id"] == sid, "label"].values[0] == pos else 0
            if cfg["use_pcag"] and sid in pcag_te_map:
                prob = pcag_te_map[sid]
                n_pcag += 1
            else:
                prob = kd_te_map[sid]
                n_kd += 1
            all_probs.append(prob)
            all_labels.append(label)
            
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 5. Metrics
        task_metrics = calculate_metrics(all_labels, all_probs)
        task_metrics.update({
            "n_total": len(all_labels),
            "n_pcag": n_pcag,
            "n_kd": n_kd
        })
        calibrated_results[task_name] = task_metrics

        # 6. Reliability Diagrams
        if cfg["use_pcag"]:
            plt.figure(figsize=(12, 10))
            # (1,1) KD Raw
            plt.subplot(2, 2, 1)
            prob_true, prob_pred = calibration_curve(kd_tr_labels, kd_npz["oof_probs"], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label='KD Raw')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.title('KD OOF Raw Calibration')
            
            # (1,2) KD Calibrated
            plt.subplot(2, 2, 2)
            prob_true, prob_pred = calibration_curve(kd_tr_labels, kd_tr_calib, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label='KD Calibrated', color='green')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.title('KD OOF Calibrated')

            # (2,1) PCAG Raw
            plt.subplot(2, 2, 3)
            prob_true, prob_pred = calibration_curve(pcag_tr_labels, pcag_oof_raw, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label='PCAG Raw')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.title('PCAG OOF Raw Calibration')

            # (2,2) PCAG Calibrated
            plt.subplot(2, 2, 4)
            prob_true, prob_pred = calibration_curve(pcag_tr_labels, pcag_tr_calib, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label='PCAG Calibrated', color='green')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.title('PCAG OOF Calibrated')
            
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"Calibration_Curves_{task_name}.png", dpi=300)
            plt.close()
        else:
            # MCI_vs_AD
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            prob_true, prob_pred = calibration_curve(kd_tr_labels, kd_npz["oof_probs"], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.title('KD OOF Raw Calibration')
            
            plt.subplot(1, 2, 2)
            prob_true, prob_pred = calibration_curve(kd_tr_labels, kd_tr_calib, n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', color='green')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.title('KD OOF Calibrated')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"Calibration_Curves_{task_name}.png", dpi=300)
            plt.close()

    # Save JSON
    with open(RESULTS_DIR / "calibrated_hybrid_metrics.json", "w") as f:
        json.dump(calibrated_results, f, indent=2)
    print(f"\nResults saved to results/calibrated_hybrid_metrics.json")

    # Final Summary Table
    print("\n" + "="*45)
    print(f"{'Task':<12} | {'Raw AUC':<9} | {'Calib AUC':<11} | {'Δ':<7}")
    print("-" * 45)
    for task_name in TASKS.keys():
        raw_auc = raw_metrics.get(task_name, {}).get("auc", 0.0)
        calib_auc = calibrated_results[task_name]["auc"]
        diff = calib_auc - raw_auc
        print(f"{task_name:<12} | {raw_auc:<9.3f} | {calib_auc:<11.3f} | {diff:+.3f}")
    print("="*45)

if __name__ == "__main__":
    main()
