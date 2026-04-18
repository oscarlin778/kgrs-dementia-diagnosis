import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# ================= 1. 資料讀取 =================
CSV_PATHS = [
    "/home/wei-chi/Model/_dataset_mapping.csv", 
    "/home/wei-chi/Data/dataset_index_116_clean_old.csv"
]
MATRIX_DIR = "/home/wei-chi/Model/processed_116_matrices"

def load_data():
    label_dict = {'NC': 0, 'MCI': 1, 'AD': 2}
    valid_data = []
    
    for csv_path in CSV_PATHS:
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            label_str = row.get('diagnosis')
            if label_str not in label_dict: continue
            
            matrix_path = None
            if 'matrix_path' in row and pd.notna(row['matrix_path']):
                matrix_path = row['matrix_path']
            elif 'new_id_base' in row and pd.notna(row['new_id_base']):
                matrix_path = os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy")
            elif 'Subject' in row and pd.notna(row['Subject']):
                matrix_path = os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy")

            if matrix_path and not any(d['matrix_path'] == matrix_path for d in valid_data):
                if os.path.exists(matrix_path):
                    try:
                        if np.load(matrix_path).shape == (116, 116):
                            valid_data.append({'matrix_path': matrix_path, 'diagnosis': label_str})
                    except: pass
    return pd.DataFrame(valid_data)

# ================= 2. 傳統 ML 訓練與驗證 =================
from sklearn.feature_selection import SelectKBest, f_classif

# ================= 2. 傳統 ML 訓練與驗證 (特徵篩選突破 80% 版) =================
def run_classical_baseline(df_full, task_pair):
    class_a, class_b = task_pair
    print(f"\n🎯 啟動傳統 ML 臨床任務: {class_a} vs {class_b} ...")
    
    df_task = df_full[df_full['diagnosis'].isin([class_a, class_b])].copy()
    df_task['label'] = df_task['diagnosis'].map({class_a: 0, class_b: 1})
    df_task = df_task.reset_index(drop=True)
    
    print(f"📊 資料分佈: {df_task['diagnosis'].value_counts().to_dict()}")
    
    # 萃取 6670 條連線特徵
    features, labels = [], []
    for _, row in df_task.iterrows():
        adj = np.load(row['matrix_path'])
        idx = np.triu_indices(116, k=1) 
        features.append(adj[idx])
        labels.append(row['label'])
    
    X = np.array(features)
    y = np.array(labels)
    
    loo = LeaveOneOut()
    
    models = {
        "SVM (RBF)": SVC(kernel='rbf', class_weight='balanced', C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    results = {}
    for name, clf in models.items():
        all_pred, all_true = [], []
        
        total = len(X)
        print(f"  ▶ 正在訓練 {name} (加入 ANOVA 特徵篩選, 共 {total} Folds)...", end="", flush=True)
        
        for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 1. 標準化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 🔥 2. 神技：特徵篩選 (只挑選最具鑑別度的前 150 條連線)
            # 注意：嚴格遵守只能用 train_idx 來挑選，避免 Data Leakage
            selector = SelectKBest(f_classif, k=150)
            X_train_sel = selector.fit_transform(X_train_scaled, y_train)
            X_test_sel = selector.transform(X_test_scaled)
            
            # 3. 訓練與預測
            clf.fit(X_train_sel, y_train)
            all_pred.append(clf.predict(X_test_sel)[0])
            all_true.append(y_test[0])
            
            if (fold + 1) % 40 == 0: print(".", end="", flush=True)
            
        acc = accuracy_score(all_true, all_pred)
        cm = confusion_matrix(all_true, all_pred)
        results[name] = (acc, cm)
        print(f" 完畢！Acc: {acc*100:.1f}%")
        
    return results

# ================= 3. 主程式 =================
def main():
    print("🚀 啟動傳統機器學習 Baseline (SVM & Random Forest) ...")
    df_full = load_data()
    print(f"📦 成功合併載入 {len(df_full)} 筆有效矩陣資料！")
    if len(df_full) == 0: return
    
    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    
    print("\n" + "="*60)
    print("🏆 傳統機器學習 效能排行榜")
    print("="*60)
    
    for task in tasks:
        results = run_classical_baseline(df_full, task)
        for model_name, (acc, cm) in results.items():
            print(f"📌 {task[0]} vs {task[1]} | {model_name:<15} | 準確率: {acc*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()