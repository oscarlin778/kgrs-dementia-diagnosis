import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import os

# ==========================================
# 1. 檔案路徑 (從報告中擷取的真實路徑)
# ==========================================
paths = {
    "NC (sub-TPMIC03002)": "/home/wei-chi/Alzheimers_Project/external_data/features/processed_116_clean_matrices/sub-TPMIC03002_matrix_clean_116.npy",
    "MCI (sub-TPMIC03007)": "/home/wei-chi/Alzheimers_Project/external_data/features/processed_116_clean_matrices/sub-TPMIC03007_matrix_clean_116.npy",
    "AD (sub-TPMIC03010)": "/home/wei-chi/Alzheimers_Project/external_data/features/processed_116_clean_matrices/sub-TPMIC03010_matrix_clean_116.npy"
}

# ==========================================
# 2. 提取特徵：取出 116x116 對稱矩陣的右上半部
# ==========================================
def get_upper_triangular(matrix_path):
    if not os.path.exists(matrix_path):
        print(f"❌ 找不到檔案: {matrix_path}")
        return None
    
    mat = np.load(matrix_path)
    # 取出對角線以上的元素索引 (k=1 代表不包含對角線自己)
    idx = np.triu_indices(116, k=1)
    
    # 拉平變成 1D 向量 (長度為 6670)
    vector = mat[idx] 
    return vector

features = {}
for name, path in paths.items():
    feat = get_upper_triangular(path)
    if feat is not None:
        features[name] = feat

# ==========================================
# 3. 計算相似度
# ==========================================
if len(features) == 3:
    print("="*60)
    print(" 🔍 fMRI 矩陣相似度分析 (Cosine & Pearson)")
    print("="*60)

    names = list(features.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1, name2 = names[i], names[j]
            vec1, vec2 = features[name1], features[name2]

            # 1. 餘弦相似度 (Cosine Similarity)
            cos_sim = 1 - cosine(vec1, vec2)
            
            # 2. 皮爾森相關係數 (Pearson Correlation)
            corr, _ = pearsonr(vec1, vec2)
            
            print(f"【{name1}】 vs 【{name2}】")
            print(f"  ▶ 餘弦相似度 : {cos_sim:.4f}")
            print(f"  ▶ 相關係數   : {corr:.4f}")
            print("-" * 50)
            
    print("\n💡 【判讀指南】")
    print(" 🚨 若數值 > 0.95：")
    print("    代表三個人的大腦網路長得幾乎一模一樣！")
    print("    兇手是「前處理流程」：可能過度平滑化、錯誤的 Z-score，或不小心複製到同一個檔。")
    print("\n ✅ 若數值介於 0.30 ~ 0.85 之間：")
    print("    代表這三個矩陣差異夠大，前處理完全沒問題。")
    print("    兇手是「模型崩潰」：請回頭檢查 KD Loss 權重、資料不平衡，或 Soft Label 太模糊。")
else:
    print("⚠️ 無法進行比對，請確認上方路徑檔案是否都存在。")