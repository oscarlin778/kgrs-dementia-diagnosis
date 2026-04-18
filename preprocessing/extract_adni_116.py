import os
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 路徑設定 =================
# 你剛才搬好家的新資料集
MATCHED_DIR = "/home/wei-chi/Data/Matched_Dual_Modal_Dataset"
# 儲存 116 矩陣的新地方
OUTPUT_MATRICES_DIR = "/home/wei-chi/Data/ADNI_processed_116_matrices"
# 儲存索引 CSV 的路徑
OUTPUT_CSV_PATH = "/home/wei-chi/Data/adni_dataset_index_116.csv"

# Nilearn Atlas 快取路徑
AAL_DIR = "/home/wei-chi/Data/nilearn_data"
TR_TIME = 2.0  # ADNI 3 rsfMRI 通常是 2.0 左右

# ================= 2. 準備工作 =================
os.makedirs(OUTPUT_MATRICES_DIR, exist_ok=True)

# 抓取 AAL 116 標籤
atlas = datasets.fetch_atlas_aal(version='SPM12', data_dir=AAL_DIR)
# 建立 Masker
masker = NiftiLabelsMasker(
    labels_img=atlas.maps, 
    standardize='zscore_sample',
    detrend=True, 
    low_pass=0.1, 
    high_pass=0.01, 
    t_r=TR_TIME,
    memory='nilearn_cache', 
    verbose=0
)

# ================= 3. 執行萃取 =================
def run_adni_extraction():
    valid_records = []
    groups = ["AD", "MCI", "NC"]
    label_mapping = {'NC': 0, 'AD': 1, 'MCI': 2}

    print(f"🚀 開始為 41 位配對受試者萃取 116 節點矩陣...")
    
    for group in groups:
        group_path = os.path.join(MATCHED_DIR, group)
        if not os.path.exists(group_path): continue
        
        subjects = os.listdir(group_path)
        for subj in subjects:
            fmri_dir = os.path.join(group_path, subj, "fMRI")
            # 尋找我們轉好的 .nii.gz 檔案
            nii_files = [f for f in os.listdir(fmri_dir) if f.endswith('.nii.gz')]
            
            if not nii_files:
                continue
            
            nii_path = os.path.join(fmri_dir, nii_files[0])
            out_npy_name = f"sub-{subj}_matrix_clean_116.npy"
            out_npy_path = os.path.join(OUTPUT_MATRICES_DIR, out_npy_name)
            
            try:
                # 1. 萃取時間序列
                time_series = masker.fit_transform(nii_path)
                
                # 2. 計算相關矩陣 (Correlation)
                correlation_measure = ConnectivityMeasure(kind='correlation')
                corr_matrix = correlation_measure.fit_transform([time_series])[0]
                
                # 3. Fisher Z 轉換 (對齊你原本的處理手法)
                np.fill_diagonal(corr_matrix, 0)
                z_matrix = np.arctanh(np.clip(corr_matrix, -0.999, 0.999))
                np.fill_diagonal(z_matrix, 1.0) 
                
                # 4. 儲存
                np.save(out_npy_path, z_matrix)
                
                valid_records.append({
                    'subject_id': f"sub-{subj}",
                    'matrix_path': out_npy_path,
                    'diagnosis': group,
                    'label': label_mapping[group]
                })
                print(f"  ✅ {subj} ({group}) 處理完成")
                
            except Exception as e:
                print(f"  ❌ {subj} 處理失敗: {e}")

    # 存成 CSV
    df = pd.DataFrame(valid_records)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n🎉 萃取結束！")
    print(f"總計成功: {len(df)} 筆")
    print(f"索引檔存於: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run_adni_extraction()