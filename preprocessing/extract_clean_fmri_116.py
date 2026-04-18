import os
import re
import numpy as np
import pandas as pd
import warnings
import ssl
import requests
import urllib3

# --- 破解 SSL 與警告環境設定 ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ----------------------------

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# ================= 1. 路徑設定 =================
BASE_DIR = "/home/wei-chi/Data"
MODEL_DIR = "/home/wei-chi/Model/MRI_data/fMRI"
NIFTI_DIR = os.path.join(MODEL_DIR, "nifti")
MAPPING_CSV = os.path.join(MODEL_DIR, "/home/wei-chi/Model/_dataset_mapping.csv")
AAL_DIR = os.path.join(BASE_DIR, "nilearn_data")

OUTPUT_MATRICES_DIR = os.path.join(BASE_DIR, "processed_116_clean_matrices")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "dataset_index_116_clean.csv")

TR_TIME = 2.0 

# ================= 2. 標籤對應 (支援 MCI) =================
def setup_labels():
    label_dict = {}
    df_map = pd.read_csv(MAPPING_CSV)
    for i in range(len(df_map)):
        row = df_map.iloc[i]
        orig_id = str(row['original_id']).strip()
        diag = str(row['diagnosis']).strip().upper()
        # 標籤定義：NC=0, AD=1, MCI=2
        mapping = {'NC': 0, 'AD': 1, 'MCI': 2}
        if diag in mapping:
            label_dict[orig_id] = {'diagnosis': diag, 'label': mapping[diag]}
    return label_dict

# ================= 3. 執行萃取 =================
def run_extraction():
    label_dict = setup_labels()
    os.makedirs(OUTPUT_MATRICES_DIR, exist_ok=True)
    
    atlas = datasets.fetch_atlas_aal(version='SPM12', data_dir=AAL_DIR)
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps, standardize='zscore_sample',
        detrend=True, low_pass=0.1, high_pass=0.01, t_r=TR_TIME,
        memory='nilearn_cache', verbose=0
    )

    valid_records = []
    nifti_files = [f for f in os.listdir(NIFTI_DIR) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    print(f"🚀 開始萃取新資料 (包含 NC, AD, MCI)...")
    for filename in sorted(nifti_files):
        core_id = filename.split('_')[0].replace('sub-', '')
        # 【關鍵修改】：現在允許 MCI
        if core_id not in label_dict:
            continue
            
        nii_path = os.path.join(NIFTI_DIR, filename)
        out_npy_path = os.path.join(OUTPUT_MATRICES_DIR, f"sub-{core_id}_matrix_clean_116.npy")
        
        try:
            time_series = masker.fit_transform(nii_path)
            correlation_measure = ConnectivityMeasure(kind='correlation')
            corr_matrix = correlation_measure.fit_transform([time_series])[0]
            
            # Fisher Z 轉換
            np.fill_diagonal(corr_matrix, 0)
            z_matrix = np.arctanh(np.clip(corr_matrix, -0.999, 0.999))
            np.fill_diagonal(z_matrix, 1.0) 
            
            np.save(out_npy_path, z_matrix)
            valid_records.append({
                'subject_id': f"sub-{core_id}",
                'matrix_path': out_npy_path,
                'diagnosis': label_dict[core_id]['diagnosis'],
                'label': label_dict[core_id]['label']
            })
            print(f"  ✅ {core_id} ({label_dict[core_id]['diagnosis']})")
        except: pass

    pd.DataFrame(valid_records).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"🎉 新資料萃取完成，存至: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run_extraction()