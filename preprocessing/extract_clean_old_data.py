import os
import re
import numpy as np
import pandas as pd
import warnings

# ================= 0. 破解 SSL 憑證驗證問題 =================
import ssl
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

old_send = requests.Session.send
def new_send(self, request, **kwargs):
    kwargs['verify'] = False
    return old_send(self, request, **kwargs)
requests.Session.send = new_send

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ============================================================

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# ================= 1. 路徑設定 =================
BASE_DIR = "/home/wei-chi/Data"

# 【關鍵修改：指定 32 筆舊資料的確切位置】
OLD_NIFTI_ROOT = "/home/wei-chi/Data/fMRI"

# 輸出的新資料夾 (舊資料的乾淨 116 矩陣)
OUTPUT_MATRICES_DIR = os.path.join(BASE_DIR, "processed_116_clean_old_matrices")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "dataset_index_116_clean_old.csv")

# 本地圖譜位置
POSSIBLE_NILEARN_DIRS = ["/home/wei-chi/Data/nilearn_data", "/home/wei-chi/nilearn_data"]
NILEARN_DIR = next((d for d in POSSIBLE_NILEARN_DIRS if os.path.exists(d)), None)

TR_TIME = 2.0 

# ================= 2. 掃描檔案與設定標籤 =================
def find_nifti_files():
    if not os.path.exists(OLD_NIFTI_ROOT):
        raise FileNotFoundError(f"❌ 找不到舊資料 NIFTI 根目錄: {OLD_NIFTI_ROOT}")
        
    os.makedirs(OUTPUT_MATRICES_DIR, exist_ok=True)
    file_list = []
    
    # 遍歷 fMRI 資料夾下的 AD 和 CN 目錄
    for root, dirs, files in os.walk(OLD_NIFTI_ROOT):
        # 確保只處理 AD 和 CN 目錄下的檔案
        if not ('/AD' in root or '/CN' in root):
            continue

        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_path = os.path.join(root, file)
                
                # 從路徑判斷是 AD 還是 NC (CN)
                if '/AD' in file_path:
                    diagnosis = 'AD'
                    label = 1
                elif '/CN' in file_path:
                    diagnosis = 'NC' # 統一轉為 NC
                    label = 0
                else:
                    continue
                    
                # 取得檔名作為 ID (去除副檔名)
                subj_id = file.split('.')[0]
                # 移除可能的前綴，確保 ID 乾淨
                subj_id = subj_id.replace('sub-', '')
                
                file_list.append({
                    'subject_id': subj_id,
                    'nii_path': file_path,
                    'diagnosis': diagnosis,
                    'label': label
                })
                
    return file_list

# ================= 3. Nilearn 醫學級特徵萃取 =================
def extract_old_clean_matrices():
    file_list = find_nifti_files()
    if not file_list:
        print(f"❌ 在 {OLD_NIFTI_ROOT} 找不到任何 AD 或 CN 的 NIfTI 檔案！")
        return

    print(f"✅ 成功在舊資料夾中找到 {len(file_list)} 筆 AD/NC 的 NIFTI 檔案！")
    print(f"✅ 準備輸出乾淨的 116 矩陣至: {OUTPUT_MATRICES_DIR}")
    
    if NILEARN_DIR:
        print(f"🌍 正在從本地端載入 AAL 116 圖譜...")
        atlas = datasets.fetch_atlas_aal(version='SPM12', data_dir=NILEARN_DIR)
    else:
        print("🌍 未找到本地圖譜，將嘗試從網路下載...")
        atlas = datasets.fetch_atlas_aal(version='SPM12')
        
    atlas_filename = atlas.maps
    
    # 【一模一樣的醫學級前處理】確保基準完全一致
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        standardize='zscore_sample', 
        detrend=True,                
        low_pass=0.1,                
        high_pass=0.01,              
        t_r=TR_TIME,                 
        memory='nilearn_cache', 
        verbose=0
    )

    valid_records = []
    
    print(f"\n🚀 開始萃取舊資料 (共 {len(file_list)} 筆)...")
    
    for idx, item in enumerate(file_list):
        subj_id = item['subject_id']
        nii_path = item['nii_path']
        
        npy_filename = f"old_{subj_id}_matrix_clean_116.npy"
        out_npy_path = os.path.join(OUTPUT_MATRICES_DIR, npy_filename)
        
        print(f"  ▶ [{idx+1}/{len(file_list)}] 處理: {subj_id} ({item['diagnosis']}) ...", end=" ", flush=True)
        
        try:
            # 1. 萃取時間序列
            time_series = masker.fit_transform(nii_path)
            
            # 檢查小腦截斷狀況
            zero_var_rois = np.where(np.var(time_series, axis=0) == 0)[0]
            if len(zero_var_rois) > 0:
                print(f"⚠️ {len(zero_var_rois)}腦區無訊號", end=" ")
            
            # 2. 計算 Pearson 相關矩陣
            correlation_measure = ConnectivityMeasure(kind='correlation')
            corr_matrix = correlation_measure.fit_transform([time_series])[0]
            
            # 3. Fisher Z-Transformation (完美常態化)
            np.fill_diagonal(corr_matrix, 0)
            corr_clipped = np.clip(corr_matrix, -0.999, 0.999)
            z_matrix = np.arctanh(corr_clipped)
            np.fill_diagonal(z_matrix, 1.0) 
            
            # 4. 儲存乾淨矩陣
            np.save(out_npy_path, z_matrix)
            print("✅ 成功")
            
            valid_records.append({
                'subject_id': f"old_{subj_id}",
                'matrix_path': out_npy_path,
                'diagnosis': item['diagnosis'],
                'label': item['label']
            })
            
        except Exception as e:
            print(f"❌ 發生錯誤: {str(e)[:50]}...")

    # ================= 4. 儲存新的索引檔 =================
    if valid_records:
        df_new = pd.DataFrame(valid_records)
        df_new.to_csv(OUTPUT_CSV_PATH, index=False)
        print("\n" + "="*60)
        print(f"🎉 舊資料同化完成！共成功生成 {len(df_new)} 筆高純度 116 矩陣。")
        print(f"💾 專用訓練索引檔已儲存至: {OUTPUT_CSV_PATH}")
        print("="*60)
        print("💡 下一步：我們可以把這份 CSV 跟你新資料的 CSV 合併起來一起訓練了！")
    else:
        print("\n⚠️ 未生成任何有效資料。")

if __name__ == "__main__":
    extract_old_clean_matrices()