import os
import glob
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
import warnings
import ssl
import requests

warnings.filterwarnings("ignore")

# ================= 解決 SSL 憑證驗證失敗的問題 =================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

old_request = requests.Session.request
def new_request(*args, **kwargs):
    kwargs['verify'] = False
    return old_request(*args, **kwargs)
requests.Session.request = new_request

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 1. 路徑設定 =================
BASE_DIR = "/home/wei-chi/Model" 
MAPPING_CSV = os.path.join(BASE_DIR, "_dataset_mapping.csv")
FMRI_DIR = os.path.join(BASE_DIR, "MRI_data", "fMRI")

OUTPUT_DIR = os.path.join(BASE_DIR, "processed_116_matrices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 指向我們訓練腳本要吃的路徑
OUTPUT_INDEX_CSV = "/home/wei-chi/Data/dataset_index_116.csv"
AAL_DIR = "/home/wei-chi/Data/fMRI/nilearn_data"

# ================= 2. 初始化 AAL 圖譜 =================
print(f"🗺️ 正在載入 AAL 116 圖譜...")
try:
    aal = datasets.fetch_atlas_aal(version='SPM12', data_dir=AAL_DIR)
except Exception as e:
    aal = datasets.fetch_atlas_aal(version='SPM12', data_dir=AAL_DIR)

masker = NiftiLabelsMasker(
    labels_img=aal.maps, 
    standardize=True, 
    memory='nilearn_cache', 
    verbose=0
)

# ================= 3. 開始處理資料 (加入【缺失腦區對齊】修復機制) =================
def process_data():
    df = pd.read_csv(MAPPING_CSV)
    df = df[df['diagnosis'].isin(['NC', 'MCI', 'AD'])].copy()
    df['label'] = df['diagnosis'].map({'NC': 0, 'MCI': 1, 'AD': 2})
    
    all_nii_files = glob.glob(os.path.join(FMRI_DIR, "**", "*.nii*"), recursive=True)
    valid_records = []
    
    for idx, row in df.iterrows():
        new_id = str(row['new_id_base'])
        orig_id = str(row['original_id'])
        
        matched_nii = None
        for f in all_nii_files:
            if new_id in f or orig_id in f:
                matched_nii = f
                break
                
        if not matched_nii: continue
            
        out_filename = f"{new_id}_matrix_116.npy"
        out_filepath = os.path.join(OUTPUT_DIR, out_filename)
        
        print(f"⏳ [{new_id}] 正在萃取並對齊 116 腦區特徵...")
        try:
            # 1. 萃取時間序列
            time_series = masker.fit_transform(matched_nii)
            # 2. 計算相關係數矩陣
            corr = np.corrcoef(time_series.T)
            corr = np.nan_to_num(corr)
            
            # 【關鍵修復核心】
            # 檢查這顆大腦是否有缺角 (例如只掃到 111 個腦區)
            if time_series.shape[1] == 116:
                full_corr = corr
                print(f"   ✅ 腦區完整 (116/116)")
            else:
                # 建立一個完美的 116x116 全零矩陣
                full_corr = np.zeros((116, 116))
                # 取得 Nilearn 實際抓到的標籤編號 (AAL的編號是 1~116)
                if hasattr(masker, 'labels_'):
                    kept_labels = masker.labels_
                    
                    # 將算出來的值，一個一個精準填回它應該在的正確位置
                    for i, label_i in enumerate(kept_labels):
                        idx_i = int(float(label_i)) - 1  # 轉成 0~115 的 index
                        for j, label_j in enumerate(kept_labels):
                            idx_j = int(float(label_j)) - 1
                            if 0 <= idx_i < 116 and 0 <= idx_j < 116:
                                full_corr[idx_i, idx_j] = corr[i, j]
                                
                    print(f"   ⚠️ 發現小腦缺失 ({time_series.shape[1]}/116)，已自動修復對齊為 116x116！")
                else:
                    print(f"   ❌ 無法取得標籤對應，跳過。")
                    continue
            
            # 儲存修復後的完美矩陣 (強迫覆寫之前的錯誤檔案)
            np.save(out_filepath, full_corr)
            
        except Exception as e:
            print(f"❌ 處理 {new_id} 時發生錯誤: {e}")
            continue
            
        valid_records.append({
            'subject_id': new_id,
            'original_id': orig_id,
            'diagnosis': row['diagnosis'],
            'label': row['label'],
            'matrix_path': out_filepath,
            'timeseries_path': matched_nii 
        })
        
    res_df = pd.DataFrame(valid_records)
    res_df.to_csv(OUTPUT_INDEX_CSV, index=False)
    
    print("\n" + "="*60)
    print(f"🎉 萃取與修復完成！所有 {len(res_df)} 筆矩陣皆已被強制對齊為標準的 116x116。")
    print(f"📋 索引檔已更新: {OUTPUT_INDEX_CSV}")
    print("="*60)

if __name__ == "__main__":
    process_data()