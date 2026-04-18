import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 路徑設定 =================
BASE_DIR = "/home/wei-chi/Data"
# 讀取我們剛剛處理好的舊資料索引檔
INDEX_CSV = os.path.join(BASE_DIR, "adni_dataset_index_116.csv")
# 輸出的診斷圖名稱
OUTPUT_PLOT_PATH = os.path.join(BASE_DIR, "script", "brain_matrix_diagnostic_clean_old.png")

def run_diagnostics():
    if not os.path.exists(INDEX_CSV):
        print(f"❌ 找不到索引檔: {INDEX_CSV}")
        return

    print("🔍 讀取【舊資料高純度前處理版】中...")
    df_raw = pd.read_csv(INDEX_CSV)
    df_pair = df_raw[df_raw['diagnosis'].isin(['NC', 'AD'])].copy()
    
    nc_matrices = []
    ad_matrices = []
    all_values = []

    # 讀取矩陣
    for i in range(len(df_pair)):
        try:
            adj = np.load(df_pair.iloc[i]['matrix_path'])
            if adj.shape != (116, 116): continue
            
            # 蒐集數值看分佈 (只取上三角)
            triu_vals = adj[np.triu_indices(116, k=1)]
            all_values.extend(triu_vals)
            
            if df_pair.iloc[i]['diagnosis'] == 'NC':
                nc_matrices.append(adj)
            else:
                ad_matrices.append(adj)
        except:
            pass
            
    if not nc_matrices or not ad_matrices:
        print("❌ 讀取矩陣失敗，請檢查路徑。")
        return

    # 計算平均矩陣
    nc_mean = np.mean(nc_matrices, axis=0)
    ad_mean = np.mean(ad_matrices, axis=0)
    
    # 計算差異矩陣
    diff_matrix = np.abs(ad_mean - nc_mean)

    print("\n🎨 正在繪製大腦連線診斷圖...")
    
    # 建立畫布
    fig = plt.figure(figsize=(20, 15))
    
    # 圖 1：數值分佈直方圖
    ax1 = plt.subplot(2, 2, 1)
    sns.histplot(all_values, bins=100, kde=True, ax=ax1, color='teal')
    ax1.set_title("Distribution of Connectivity Values (CLEAN OLD DATA)\n(Verify if it matches the bell curve of the new data)")
    ax1.set_xlabel("Fisher Z-Transformed Connectivity")
    ax1.set_ylabel("Count")
    
    # 設定熱力圖共用的顏色範圍 (Vmin, Vmax)
    vmin, vmax = -0.5, 0.8  
    
    # 圖 2：NC 平均矩陣
    ax2 = plt.subplot(2, 2, 2)
    sns.heatmap(nc_mean, cmap='coolwarm', vmin=vmin, vmax=vmax, square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title(f"Average NC Matrix (N={len(nc_matrices)})\n(Check for block structures)")
    
    # 圖 3：AD 平均矩陣
    ax3 = plt.subplot(2, 2, 3)
    sns.heatmap(ad_mean, cmap='coolwarm', vmin=vmin, vmax=vmax, square=True, ax=ax3, cbar_kws={'shrink': 0.8})
    ax3.set_title(f"Average AD Matrix (N={len(ad_matrices)})")
    
    # 圖 4：差異矩陣
    ax4 = plt.subplot(2, 2, 4)
    sns.heatmap(diff_matrix, cmap='Reds', vmin=0, vmax=0.2, square=True, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title("Absolute Difference |AD - NC|\n(Are the patterns consistent?)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
    plt.close()
    
    print(f"\n✅ 舊資料乾淨版診斷圖表已生成: {OUTPUT_PLOT_PATH}")
    print("💡 【驗收重點】：")
    print("1. 左上角直方圖是否也呈現漂亮的鐘型曲線？")
    print("2. 右側與左下熱力圖是否能看到與新資料類似的深紅色對角區塊 (如 DMN 等網路)？")
    print("👉 如果圖形看起來與新資料相似，我們就可以安心將它們合併訓練了！")

if __name__ == "__main__":
    run_diagnostics()