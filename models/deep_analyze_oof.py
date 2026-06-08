import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def deep_analyze():
    data_path = "/home/wei-chi/Alzheimers_Project/external_data/scripts/results/E13_GSL/oof_predictions.npy"
    if not os.path.exists(data_path):
        print("File not found.")
        return

    # 讀取資料
    data = np.load(data_path, allow_pickle=True).item()
    
    # 資料結構推測: {task: {'y_true': [...], 'y_prob': [...]}}
    # 或是舊腳本中提到的 seed_probs 結構
    
    for task, results in data.items():
        print(f"\n===== Deep Analysis: {task} =====")
        y_true = np.array(results['true'])
        y_prob = np.array(results['prob'])
        
        # 1. 檢查 Baseline ACC (0.5)
        acc_05 = accuracy_score(y_true, (y_prob > 0.5).astype(int))
        auc = roc_auc_score(y_true, y_prob)
        
        print(f"  Samples: {len(y_true)} (Pos: {sum(y_true)}, Neg: {len(y_true)-sum(y_true)})")
        print(f"  AUC: {auc:.4f}")
        print(f"  ACC (0.5 threshold): {acc_05:.4f}")
        
        # 2. 尋找真正能極大化 ACC 的門檻
        thresholds = np.linspace(0, 1, 101)
        accs = [accuracy_score(y_true, (y_prob > t).astype(int)) for t in thresholds]
        best_acc_idx = np.argmax(accs)
        best_acc = accs[best_acc_idx]
        best_t_acc = thresholds[best_acc_idx]
        
        print(f"  Best possible ACC: {best_acc:.4f} (at threshold {best_t_acc:.2f})")
        
        # 3. 檢查預測值的分佈 (確認是否全都擠在低分區)
        print(f"  Prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
        print(f"  Mean Prob: {y_prob.mean():.4f}")

if __name__ == "__main__":
    deep_analyze()
