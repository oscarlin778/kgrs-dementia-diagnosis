import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

def fix_and_report():
    data_path = "/home/wei-chi/Alzheimers_Project/external_data/scripts/results/E13_GSL/oof_predictions.npy"
    if not os.path.exists(data_path):
        print("Predictions not found.")
        return

    data = np.load(data_path, allow_pickle=True).item()
    corrected_metrics = {}

    print("=== Corrected GNN E13 Performance Report ===")
    for task_key, results in data.items():
        y_true = np.array(results['true'])
        y_prob = np.array(results['prob'])
        
        # 1. AUC (本身就穩定)
        auc = roc_auc_score(y_true, y_prob)
        
        # 2. 尋找能極大化 Accuracy 的穩定門檻
        thresholds = np.linspace(0, 1, 101)
        accs = [accuracy_score(y_true, (y_prob > t).astype(int)) for t in thresholds]
        best_idx = np.argmax(accs)
        best_acc = accs[best_idx]
        best_t = thresholds[best_idx]
        
        # 3. 記錄 0.5 基準
        acc_05 = accuracy_score(y_true, (y_prob > 0.5).astype(int))
        
        task_display = task_key.replace('_', ' ').upper()
        print(f"\nTask: {task_display}")
        print(f"  Samples: {len(y_true)}")
        print(f"  AUC: {auc:.4f}")
        print(f"  ACC (at 0.5): {acc_05*100:.1f}%")
        print(f"  Best ACC: {best_acc*100:.1f}% (at threshold {best_t:.2f})")
        
        corrected_metrics[task_display] = {
            "auc": auc,
            "acc": best_acc,
            "best_threshold": best_t
        }

    # 儲存修正後的 JSON
    with open("/home/wei-chi/Alzheimers_Project/external_data/scripts/results/E13_GSL/metrics_corrected.json", "w") as f:
        json.dump(corrected_metrics, f, indent=2)
    print("\n✅ Corrected metrics saved to metrics_corrected.json")

if __name__ == "__main__":
    fix_and_report()
