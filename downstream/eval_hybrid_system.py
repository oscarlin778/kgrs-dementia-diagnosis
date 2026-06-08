import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, precision_score

# --- Paths ---
BASE_DIR = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COMBINED_TEST_CSV = os.path.join(BASE_DIR, "splits/combined_test.csv")
BASELINE_JSON = os.path.join(RESULTS_DIR, "kd_comprehensive_metrics.json")

TASK_CFG = {
    'NC_vs_AD': {
        'classes': [0, 2], 
        'pos': 2, 
        'kd_npz': 'kd_NC_vs_AD_a3_T3_probs.npz',
        'pcag_npz': 'pcag_NC_vs_AD_probs.npz'
    },
    'NC_vs_MCI': {
        'classes': [0, 1], 
        'pos': 1, 
        'kd_npz': 'kd_NC_vs_MCI_a3_T3_probs.npz',
        'pcag_npz': 'pcag_NC_vs_MCI_probs.npz'
    },
    'MCI_vs_AD': {
        'classes': [1, 2], 
        'pos': 2, 
        'kd_npz': 'kd_MCI_vs_AD_a10_T3_probs.npz',
        'pcag_npz': 'pcag_MCI_vs_AD_probs.npz'
    },
}

def calculate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "sens": float(sens),
        "spec": float(spec),
        "prec": float(prec),
        "npv": float(npv),
        "acc": float(acc),
        "f1": float(f1),
        "cm": cm.tolist(),
        "threshold": float(threshold)
    }

def main():
    combined_test = pd.read_csv(COMBINED_TEST_CSV)
    with open(BASELINE_JSON, 'r') as f:
        baseline_metrics = json.load(f)
    
    final_results = {}
    plot_data = []

    for task_name, cfg in TASK_CFG.items():
        print(f"\nEvaluating {task_name}...")
        classes, pos = cfg['classes'], cfg['pos']
        
        # Filter task subjects in order
        task_test = combined_test[combined_test['label'].isin(classes)].reset_index(drop=True)
        task_test['bin_label'] = (task_test['label'] == pos).astype(int)
        
        # Step 1: KD Map
        kd_path = os.path.join(RESULTS_DIR, cfg['kd_npz'])
        kd_data = np.load(kd_path, allow_pickle=True)
        kd_map = {sid: prob for sid, prob in zip(task_test['subject_id'], kd_data['test_probs'])}
        
        # Step 2: PCAG Map
        pcag_path = os.path.join(RESULTS_DIR, cfg['pcag_npz'])
        pcag_data = np.load(pcag_path, allow_pickle=True)
        pcag_map = {sid: prob for sid, prob in zip(pcag_data['test_subject_ids'], pcag_data['test_probs'])}
        
        # Step 3: Hybrid System
        hybrid_probs = []
        hybrid_labels = []
        n_pcag = 0
        n_kd = 0
        
        for idx, row in task_test.iterrows():
            sid = row['subject_id']
            label = row['bin_label']
            if sid in pcag_map:
                hybrid_probs.append(pcag_map[sid])
                n_pcag += 1
            else:
                hybrid_probs.append(kd_map[sid])
                n_kd += 1
            hybrid_labels.append(label)
        
        hybrid_probs = np.array(hybrid_probs)
        hybrid_labels = np.array(hybrid_labels)
        
        # Step 4: System AUCs
        hybrid_auc = roc_auc_score(hybrid_labels, hybrid_probs)
        kd_only_auc = roc_auc_score(kd_data['test_labels'], kd_data['test_probs'])
        
        pcag_only_probs = [pcag_map[sid] for sid in task_test['subject_id'] if sid in pcag_map]
        pcag_only_labels = [row['bin_label'] for idx, row in task_test.iterrows() if row['subject_id'] in pcag_map]
        pcag_only_auc = roc_auc_score(pcag_only_labels, pcag_only_probs) if len(set(pcag_only_labels)) > 1 else 0.5
        
        # Step 5: Hybrid Metrics with Youden's J threshold from KD OOF
        oof_fpr, oof_tpr, oof_thr = roc_curve(kd_data['oof_labels'], kd_data['oof_probs'])
        best_thr = oof_thr[np.argmax(oof_tpr - oof_fpr)]
        
        hybrid_metrics = calculate_metrics(hybrid_labels, hybrid_probs, best_thr)
        hybrid_metrics["n"] = len(hybrid_labels)
        hybrid_metrics["n_pcag"] = n_pcag
        hybrid_metrics["n_kd"] = n_kd
        
        # Baselines
        e5_metrics = baseline_metrics[task_name]['e5']
        kd_baseline_metrics = baseline_metrics[task_name]['kd']
        
        final_results[task_name] = {
            "hybrid": hybrid_metrics,
            "kd_only": {"n": len(kd_data['test_labels']), "auc": float(kd_only_auc)},
            "pcag_only": {"n": len(pcag_only_labels), "auc": float(pcag_only_auc)},
            "e5": e5_metrics,
            "kd_baseline": kd_baseline_metrics
        }
        
        # Prepare data for plotting
        plot_data.append({"Task": task_name, "Model": "E5 Baseline", "AUC": e5_metrics['auc']})
        plot_data.append({"Task": task_name, "Model": "KD Student", "AUC": kd_baseline_metrics['auc']})
        plot_data.append({"Task": task_name, "Model": "PCAG (dual only)", "AUC": pcag_only_auc})
        plot_data.append({"Task": task_name, "Model": "Hybrid System", "AUC": hybrid_auc, "n_pcag": n_pcag, "n_total": len(hybrid_labels)})

    # Save JSON
    with open(os.path.join(RESULTS_DIR, "hybrid_system_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to results/hybrid_system_metrics.json")

    # Print Table
    print("\n" + "="*80)
    print(f"{'Task':<15} {'Model':<20} {'N':<5} {'AUC':<8} {'Sens':<8} {'Spec':<8} {'F1':<8}")
    print("-" * 80)
    for task_name in TASK_CFG.keys():
        res = final_results[task_name]
        # E5
        print(f"{task_name:<15} {'E5 Baseline':<20} {res['hybrid']['n']:<5} {res['e5']['auc']:<8.4f} {res['e5']['sens']:<8.4f} {res['e5']['spec']:<8.4f} {res['e5']['f1']:<8.4f}")
        # KD
        print(f"{'':<15} {'KD Student':<20} {res['kd_only']['n']:<5} {res['kd_baseline']['auc']:<8.4f} {res['kd_baseline']['sens']:<8.4f} {res['kd_baseline']['spec']:<8.4f} {res['kd_baseline']['f1']:<8.4f}")
        # PCAG
        print(f"{'':<15} {'PCAG (dual only)':<20} {res['pcag_only']['n']:<5} {res['pcag_only']['auc']:<8.4f} {'-':<8} {'-':<8} {'-':<8}")
        # Hybrid
        print(f"{'':<15} {'Hybrid System':<20} {res['hybrid']['n']:<5} {res['hybrid']['auc']:<8.4f} {res['hybrid']['sens']:<8.4f} {res['hybrid']['spec']:<8.4f} {res['hybrid']['f1']:<8.4f}")
        print("-" * 80)

    # Visualization
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    palette = {
        "E5 Baseline": "salmon",
        "KD Student": "steelblue",
        "PCAG (dual only)": "lightgreen",
        "Hybrid System": "darkgreen"
    }
    
    ax = sns.barplot(data=df_plot, x="Task", y="AUC", hue="Model", palette=palette)
    
    # Annotate Hybrid bars with N_PCAG/N_total
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            # Find the corresponding data row
            # This is a bit tricky with seaborn barplot patches
            pass # We'll do it by looping through plot_data instead

    # Correct way to annotate
    tasks = df_plot['Task'].unique()
    models = ["E5 Baseline", "KD Student", "PCAG (dual only)", "Hybrid System"]
    
    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            val = df_plot[(df_plot['Task'] == task) & (df_plot['Model'] == model)]
            if not val.empty:
                auc_val = val['AUC'].values[0]
                # Calculate x position: i is center of group, offset by model index
                # bar width is roughly 0.8 / 4 = 0.2
                x_pos = i + (j - 1.5) * 0.2
                ax.text(x_pos, auc_val + 0.01, f"{auc_val:.3f}", ha='center', fontsize=9)
                
                if model == "Hybrid System":
                    n_pcag = val['n_pcag'].values[0]
                    n_total = val['n_total'].values[0]
                    ax.text(x_pos, auc_val / 2, f"{n_pcag}/{n_total}\nPCAG", ha='center', color='white', fontweight='bold', fontsize=8)

    plt.title("Hybrid System Evaluation: AUC Comparison", fontsize=15)
    plt.ylim(0, 1.1)
    plt.ylabel("AUC Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(FIGURES_DIR, "Hybrid_system_AUC.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
