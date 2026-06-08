import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score
from pathlib import Path

# --- Paths ---
BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

FMRI_TRAIN_CSV = BASE_DIR / "splits/combined_train.csv"
FMRI_TEST_CSV  = BASE_DIR / "splits/combined_test.csv"
PCAG_TRAIN_CSV = BASE_DIR / "pcag_train_aligned.csv"
BASELINE_JSON  = RESULTS_DIR / "kd_comprehensive_metrics.json"

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
    
    with open(BASELINE_JSON, 'r') as f:
        baseline_metrics = json.load(f)

    global_metrics = {}
    plot_data_roc = {}
    plot_data_bars = []

    for task_name, cfg in TASKS.items():
        print(f"\nProcessing Task: {task_name}")
        classes = cfg["classes"]
        pos = cfg["pos"]
        
        # 1. Reconstruction of Subject IDs for OOF
        # KD OOF IDs
        df_tr_task = df_tr_full[df_tr_full["label"].isin(classes)].reset_index(drop=True)
        kd_tr_ids = df_tr_task["subject_id"].tolist()
        kd_tr_labels = (df_tr_task["label"] == pos).astype(int).values
        
        # PCAG OOF IDs
        df_pcag_tr_task = df_pcag_tr_full[df_pcag_tr_full["label"].isin(classes)].reset_index(drop=True)
        pcag_tr_ids = df_pcag_tr_task["subject_id"].tolist()
        pcag_tr_labels = (df_pcag_tr_task["label"] == pos).astype(int).values
        
        # 2. Loading Probabilities
        kd_npz = np.load(RESULTS_DIR / cfg["kd_npz"], allow_pickle=True)
        # Verify labels distribution
        if not np.array_equal(kd_npz["oof_labels"], kd_tr_labels):
            print(f"  [ERROR] KD OOF label mismatch for {task_name}")
            # Fallback or exit
        kd_tr_map = {sid: prob for sid, prob in zip(kd_tr_ids, kd_npz["oof_probs"])}
        kd_te_task = df_te_full[df_te_full["label"].isin(classes)].reset_index(drop=True)
        kd_te_ids = kd_te_task["subject_id"].tolist()
        kd_te_map = {sid: prob for sid, prob in zip(kd_te_ids, kd_npz["test_probs"])}
        
        pcag_tr_map = {}
        pcag_te_map = {}
        if cfg["use_pcag"]:
            pcag_npz = np.load(RESULTS_DIR / cfg["pcag_npz"], allow_pickle=True)
            if not np.array_equal(pcag_npz["oof_labels"], pcag_tr_labels):
                 print(f"  [ERROR] PCAG OOF label mismatch for {task_name}")
            pcag_tr_map = {sid: prob for sid, prob in zip(pcag_tr_ids, pcag_npz["oof_probs"])}
            pcag_te_map = {sid: prob for sid, prob in zip(pcag_npz["test_subject_ids"], pcag_npz["test_probs"])}

        # 3. Hybrid Routing (Global)
        all_probs, all_labels = [], []
        n_pcag, n_kd = 0, 0
        
        # Routing Train
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
            
        # Routing Test
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
        
        # 4. Metrics
        task_metrics = calculate_metrics(all_labels, all_probs)
        task_metrics.update({
            "n_total": len(all_labels),
            "n_pcag": n_pcag,
            "n_kd": n_kd
        })
        global_metrics[task_name] = task_metrics
        plot_data_roc[task_name] = (all_labels, all_probs)
        
        # 5. Bar Data
        # KD (80/20) from kd_comprehensive_metrics.json
        kd_8020_auc = baseline_metrics[task_name]["kd"]["auc"]
        # PCAG Combat (80/20) - need to load from its json
        pcag_8020_auc = 0.0
        if cfg["pcag_npz"]:
             p_res_path = RESULTS_DIR / f"pcag_combat_{task_name}_results.json"
             if p_res_path.exists():
                 with open(p_res_path, 'r') as f:
                     pcag_8020_auc = json.load(f)["test_metrics"]["auc"]
        
        # Hybrid Combat (80/20) - load from results/hybrid_system_combat_metrics.json
        hybrid_8020_auc = 0.0
        h_res_path = RESULTS_DIR / "hybrid_system_combat_metrics.json"
        if h_res_path.exists():
            with open(h_res_path, 'r') as f:
                hybrid_8020_auc = json.load(f)[task_name]["hybrid"]["auc"]

        plot_data_bars.append({"Task": task_name, "Model": "KD (80/20)", "AUC": kd_8020_auc})
        plot_data_bars.append({"Task": task_name, "Model": "PCAG Combat (80/20)", "AUC": pcag_8020_auc})
        plot_data_bars.append({"Task": task_name, "Model": "Hybrid Combat (80/20)", "AUC": hybrid_8020_auc})
        plot_data_bars.append({"Task": task_name, "Model": "Global OOF Hybrid", "AUC": task_metrics["auc"]})

    # Output JSON
    with open(RESULTS_DIR / "global_oof_hybrid_metrics.json", "w") as f:
        json.dump(global_metrics, f, indent=2)
    print(f"\nResults saved to results/global_oof_hybrid_metrics.json")

    # Visualization: ROC Curves
    plt.figure(figsize=(18, 5))
    for i, task_name in enumerate(TASKS.keys()):
        plt.subplot(1, 3, i+1)
        y_true, y_prob = plot_data_roc[task_name]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = global_metrics[task_name]["auc"]
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Mark threshold
        thr = global_metrics[task_name]["threshold"]
        # Find point on curve near threshold
        # This is approximated
        idx = np.argmin(np.abs(np.array(_) - thr)) if '_' in locals() else 0
        # Actually I already have best_thr index from calculate_metrics, let's redo it here simply
        fpr_opt, tpr_opt, thresholds_opt = roc_curve(y_true, y_prob)
        idx_opt = np.argmax(tpr_opt - fpr_opt)
        plt.scatter(fpr_opt[idx_opt], tpr_opt[idx_opt], color='red', s=50, label=f'Youden J (T={thr:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Global OOF Hybrid: {task_name}')
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Global_OOF_ROC_curves.png", dpi=300)
    print(f"ROC plot saved to {FIGURES_DIR / 'Global_OOF_ROC_curves.png'}")

    # Visualization: Bar Chart
    df_bars = pd.DataFrame(plot_data_bars)
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_bars, x="Task", y="AUC", hue="Model")
    plt.title("Comparison: 80/20 Test vs. Global OOF Hybrid", fontsize=15)
    plt.ylim(0, 1.1)
    
    # Annotate values
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Global_OOF_vs_8020_comparison.png", dpi=300)
    print(f"Comparison plot saved to {FIGURES_DIR / 'Global_OOF_vs_8020_comparison.png'}")

if __name__ == "__main__":
    main()
