import os
import numpy as np
import json
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, roc_curve

def diagnose():
    results_dir = "/home/wei-chi/Alzheimers_Project/external_data/scripts/results/E13_GSL"
    # 這裡假設你的腳本有把 OOF 的 logits 存下來，或是我們可以從已存的 metrics 推導
    # 但為了最精確，我們應該去讀取訓練過程中可能產出的 .npy 檔案
    
    print("🔍 Checking for raw prediction files...")
    # 找尋是否有 teacher_logits 或 oof_predictions 之類的檔案
    files = [f for f in os.listdir(results_dir) if f.endswith('.npy') or f.endswith('.json')]
    print(f"  Found files: {files}")

    # 如果只有 metrics.json，我們無法重新分析原始分佈
    # 但我們可以檢查 models 目錄下是否有 fold-based 的預測結果
    models_dir = "/home/wei-chi/Alzheimers_Project/external_data/scripts/checkpoints/adni_only_gnn"
    if os.path.exists(models_dir):
        gnn_files = [f for f in os.listdir(models_dir) if f.endswith('.npy')]
        print(f"  Found GNN raw files in checkpoints: {gnn_files}")

    # 讀取原本的 metrics.json 看看是否有異狀
    metrics_path = os.path.join(results_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            m = json.load(f)
        
        for task in m:
            print(f"\n--- Task: {task} ---")
            print(f"  Reported AUC: {m[task]['auc']:.4f}")
            print(f"  Reported ACC: {m[task]['acc']:.4f}")
            print(f"  Best Threshold: {m[task]['best_threshold']:.4f}")
            
            # 檢查 FPR/TPR 是否合理
            # 如果 AUC 很高但 ACC 低，通常是門檻切在極端值
            if 'fpr' in m[task] and 'tpr' in m[task]:
                fpr = np.array(m[task]['fpr'])
                tpr = np.array(m[task]['tpr'])
                # 計算 Youden J
                j = tpr - fpr
                best_idx = np.argmax(j)
                print(f"  Verification: Youden J max at index {best_idx}, TPR={tpr[best_idx]:.2f}, FPR={fpr[best_idx]:.2f}")

if __name__ == "__main__":
    diagnose()
