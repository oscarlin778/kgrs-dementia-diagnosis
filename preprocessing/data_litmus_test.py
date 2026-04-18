import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore') # 忽略除以零等數學警告

# ================= 1. 路徑與參數設定 =================
BASE_DIR = "/home/wei-chi/Data"
INDEX_CSV = os.path.join(BASE_DIR, "dataset_index_116.csv")

# ================= 2. 資料讀取與前處理 =================
def load_and_preprocess_data():
    print("🔍 讀取資料並進行 Fisher Z 轉換...")
    df_raw = pd.read_csv(INDEX_CSV)
    
    # 鎖定 NC vs AD
    df_pair = df_raw[df_raw['diagnosis'].isin(['NC', 'AD'])].copy()
    df_pair['label'] = df_pair['diagnosis'].map({'NC': 0, 'AD': 1})
    df_pair = df_pair.reset_index(drop=True)
    
    features = []
    labels = []
    
    # 提取上三角矩陣索引 (116 * 115 / 2 = 6670 個特徵)
    triu_indices = np.triu_indices(116, k=1)
    
    for i in range(len(df_pair)):
        try:
            adj = np.load(df_pair.iloc[i]['matrix_path'])
            if adj.shape != (116, 116):
                continue
                
            # 1. 抽取上三角特徵
            vec = adj[triu_indices]
            
            # 2. 醫學影像鐵律：Fisher Z-transformation
            # 避免剛好為 1 或 -1 導致 log(0) 錯誤，限制在 [-0.999, 0.999]
            vec_clipped = np.clip(vec, -0.999, 0.999)
            vec_z = np.arctanh(vec_clipped)
            
            features.append(vec_z)
            labels.append(df_pair.iloc[i]['label'])
        except Exception as e:
            pass
            
    X = np.array(features)
    y = np.array(labels)
    
    return X, y

# ================= 3. 極限快篩測試 (LOOCV) =================
def run_litmus_test():
    X, y = load_and_preprocess_data()
    total_samples = len(y)
    
    print("\n" + "="*60)
    print(f"🚀 啟動資料快篩測試 (Data Litmus Test)")
    print(f"📊 總樣本數: {total_samples} (NC: {sum(y==0)}, AD: {sum(y==1)})")
    print(f"🧬 原始特徵維度: {X.shape[1]} 條連線")
    print("="*60)
    
    loo = LeaveOneOut()
    
    # 定義兩個最基礎且強大的傳統模型
    # 1. 線性 SVM (對高維度小樣本最有效)
    svm_preds = []
    # 2. 隨機森林 (對非線性雜訊抵抗力強)
    rf_preds = []
    
    true_labels = []

    print("⏳ 正在執行 LOOCV，這只需要幾秒鐘...")
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # --- 關鍵：嚴格的特徵篩選 (ANOVA F-test) ---
        # 只保留與疾病最相關的 Top 100 條連線，避免 SVM 崩潰
        selector = SelectKBest(f_classif, k=100)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # --- 訓練與預測 ---
        # SVM
        svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
        svm_model.fit(X_train_selected, y_train)
        svm_preds.append(svm_model.predict(X_test_selected)[0])
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf_model.fit(X_train_selected, y_train)
        rf_preds.append(rf_model.predict(X_test_selected)[0])
        
        true_labels.append(y_test[0])

    # ================= 4. 結果分析 =================
    svm_acc = accuracy_score(true_labels, svm_preds)
    rf_acc = accuracy_score(true_labels, rf_preds)
    
    svm_cm = confusion_matrix(true_labels, svm_preds, labels=[0, 1])
    rf_cm = confusion_matrix(true_labels, rf_preds, labels=[0, 1])
    
    print("\n" + "★"*60)
    print("🎯 快篩測試結果出爐")
    print("★"*60)
    
    print(f"\n🧠 【1. 支援向量機 (SVM - Linear)】")
    print(f"準確率: {svm_acc*100:.1f}%")
    print(f"                 預測 NC (0) | 預測 AD (1)")
    print(f"實際 NC (0) |      {svm_cm[0,0]:<9} |      {svm_cm[0,1]:<7}")
    print(f"實際 AD (1) |      {svm_cm[1,0]:<9} |      {svm_cm[1,1]:<7}")
    
    print(f"\n🌲 【2. 隨機森林 (Random Forest)】")
    print(f"準確率: {rf_acc*100:.1f}%")
    print(f"                 預測 NC (0) | 預測 AD (1)")
    print(f"實際 NC (0) |      {rf_cm[0,0]:<9} |      {rf_cm[0,1]:<7}")
    print(f"實際 AD (1) |      {rf_cm[1,0]:<9} |      {rf_cm[1,1]:<7}")
    
    print("\n💡 【診斷建議】:")
    if svm_acc > 0.65 or rf_acc > 0.65:
        print("✅ 資料本身具有可分性！問題在於 GNN 太容易過擬合，建議專注於調校輕量級模型或直接使用傳統 ML 發表專題。")
    else:
        print("❌ 警告：連帶有特徵篩選的 SVM 與隨機森林都在瞎猜 (小於 60%)。")
        print("這代表這 69 筆矩陣資料在數學上『不具備分類特徵』。")
        print("👉 下一步：請務必回頭檢查 dataset_index_116.csv 的標籤是否對錯人，或是 fMRI 前處理是否產生了大量雜訊矩陣！")

if __name__ == "__main__":
    run_litmus_test()