import os
import glob
import re
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure

# ================= 1. 解析座標檔案 =================
def parse_nodes_file(txt_path):
    coords = []
    node_names = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        match = re.search(r'\((.*?)\)', line)
        if match:
            num_str = match.group(1).replace('+', '')
            xyz = tuple(map(float, num_str.split()))
            coords.append(xyz)
            node_names.append(line.split(':')[0])
    return coords, node_names

# ================= 2. 強化版智慧命名解析器 =================
def generate_output_filename(nifti_path):
    """
    掃描完整路徑判斷類別，並精準抓取 S_XXXX 或 sub-XXXX 編號
    """
    full_path_upper = nifti_path.upper()
    filename = os.path.basename(nifti_path)
    
    # 1. 類別判斷：掃描完整路徑 (解決多層目錄問題)
    if '/AD/' in full_path_upper or '_AD_' in full_path_upper:
        diagnosis = 'AD'
    elif '/NC/' in full_path_upper or '_NC_' in full_path_upper:
        diagnosis = 'NC'
    elif '/MCI/' in full_path_upper or '_MCI_' in full_path_upper:
        diagnosis = 'MCI'
    else:
        diagnosis = 'Unknown'
        
    # 2. 編號萃取：優先抓取 S_XXXX (如 S_6648) 或 sub-XXXX
    # 針對 dswausub-027_S_6648... 這種格式，S_6648 通常是真正的 Subject ID
    s_match = re.search(r'S_(\d+)', filename)
    sub_match = re.search(r'sub-(\d+)', filename)
    
    if s_match:
        subject_id = s_match.group(1)
    elif sub_match:
        subject_id = sub_match.group(1)
    else:
        # 如果都沒有，抓取第一個超過 3 位數的數字 (避免抓到像 sub-027 這種序號)
        long_id = re.search(r'\d{4,}', filename)
        if long_id:
            subject_id = long_id.group()
        else:
            first_id = re.search(r'\d+', filename)
            subject_id = first_id.group() if first_id else "unknown"
    
    return f"Pearson_Correlation_Matrix_{diagnosis}_{subject_id}.csv"

# ================= 3. 核心參數設定 =================
INPUT_DIR = "/home/wei-chi/Data/fMRI"  # 設定為最上層資料夾
OUTPUT_DIR = "/home/wei-chi/Model/processed_13nodes_final" 
NODES_TXT_PATH = "/home/wei-chi/Data/AD_nodes.txt"

# ================= 4. 主程式 =================
def main():
    print("🚀 啟動批次處理程序...")
    coords, node_names = parse_nodes_file(NODES_TXT_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    masker = NiftiSpheresMasker(
        seeds=coords, radius=5.0, standardize='zscore_sample', 
        detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.0
    )
    correlation_measure = ConnectivityMeasure(kind='correlation')

    # 使用 **/*.nii.gz 進行深度遞迴搜尋，無論多幾層路徑都能抓到
    nifti_files = glob.glob(os.path.join(INPUT_DIR, "**/*.nii.gz"), recursive=True)
    print(f"📂 深度掃描完成，共找到 {len(nifti_files)} 個檔案。\n")

    success_count = 0
    for nifti_path in nifti_files:
        output_filename = generate_output_filename(nifti_path)
        output_csv_path = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_csv_path):
            print(f"⏩ 跳過已存在檔案: {output_filename}")
            success_count += 1
            continue

        try:
            time_series = masker.fit_transform(nifti_path)
            corr_matrix = correlation_measure.fit_transform([time_series])[0]
            np.fill_diagonal(corr_matrix, np.nan)
            
            df = pd.DataFrame(corr_matrix)
            df.to_csv(output_csv_path, index=False, header=False)
            
            success_count += 1
            print(f"✅ 成功處理: {output_filename}")
            
        except Exception as e:
            print(f"❌ 處理失敗: {nifti_path} | 錯誤: {e}")

    print(f"\n🎉 任務結束！成功產出 {success_count} 個 CSV 檔案至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()