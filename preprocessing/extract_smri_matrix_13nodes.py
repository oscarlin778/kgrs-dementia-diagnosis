import os
import glob
import re
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import coord_transform
from sklearn.preprocessing import StandardScaler

# ================= 1. 解析座標檔案 =================
def parse_nodes_file(txt_path):
    coords = []
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
    return coords

# ================= 2. 智慧命名解析器 =================
def generate_output_filename(nifti_path):
    filename = os.path.basename(nifti_path)
    full_path_upper = nifti_path.upper()
    
    if '/AD/' in full_path_upper or '_AD_' in full_path_upper: diagnosis = 'AD'
    elif '/NC/' in full_path_upper or '_NC_' in full_path_upper: diagnosis = 'NC'
    elif '/MCI/' in full_path_upper or '_MCI_' in full_path_upper: diagnosis = 'MCI'
    else: diagnosis = 'Unknown'
        
    s_match = re.search(r'S_(\d+)', filename)
    sub_match = re.search(r'sub[-_](\d+)', filename, re.IGNORECASE)
    
    if s_match: subject_id = s_match.group(1)
    elif sub_match: subject_id = sub_match.group(1)
    else:
        long_id = re.search(r'\d{4,}', filename)
        if long_id: subject_id = long_id.group()
        else:
            first_id = re.search(r'\d+', filename)
            subject_id = first_id.group() if first_id else "unknown"
            
    return f"sMRI_Structural_Matrix_{diagnosis}_{subject_id}.csv"

# ================= 3. 核心參數設定 =================
INPUT_DIR = "/home/wei-chi/Model/sMRI_data_MultiModal_Aligned_MNI"  
OUTPUT_DIR = "/home/wei-chi/Model/processed_sMRI_13nodes" 
NODES_TXT_PATH = "/home/wei-chi/Data/AD_nodes.txt"

# ================= 4. 主程式 =================
def main():
    print("🚀 啟動 sMRI 結構網路精準萃取程序 (Direct Voxel Method)...")
    
    if not os.path.exists(NODES_TXT_PATH):
        print(f"❌ 找不到座標檔案: {NODES_TXT_PATH}")
        return
        
    coords = parse_nodes_file(NODES_TXT_PATH)
    print(f"✅ 成功載入 {len(coords)} 個核心節點座標。\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔍 正在掃描 T1 影像...")
    all_nifti_files = glob.glob(os.path.join(INPUT_DIR, "**/*.nii*"), recursive=True)
    
    nifti_files = []
    for f in all_nifti_files:
        path_upper = f.upper()
        if 'T1' in path_upper and 'T2' not in path_upper and 'DWI' not in path_upper and 'FLAIR' not in path_upper:
            if 'MASK' not in path_upper and 'SEG' not in path_upper:
                nifti_files.append(f)

    print(f"📂 總共掃描到 {len(nifti_files)} 個 T1 影像檔案，準備批次處理！\n")
    print("-" * 60)

    success_count = 0
    for idx, nifti_path in enumerate(nifti_files):
        output_filename = generate_output_filename(nifti_path)
        output_csv_path = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_csv_path):
            print(f"⏩ 跳過已存在檔案: {output_filename}")
            success_count += 1
            continue

        try:
            # 🔥 直接讀取 NIfTI 底層資料與空間轉換矩陣 (Affine)
            img = nib.load(nifti_path)
            data = img.get_fdata()
            affine = img.affine
            inv_affine = np.linalg.inv(affine)
            
            node_intensities = []
            
            # 手動針對 13 個座標抓取特徵
            for coord in coords:
                # 1. 將 MNI 物理座標轉換為 Numpy 陣列的 [x, y, z] 索引
                voxel_coords = coord_transform(coord[0], coord[1], coord[2], inv_affine)
                x, y, z = int(round(voxel_coords[0])), int(round(voxel_coords[1])), int(round(voxel_coords[2]))
                
                # 2. 抓取該點周圍 3x3x3 空間的平均灰階強度 (等同於模擬小球體)
                # 加入邊界保護，防止座標跑到影像外面
                x_min, x_max = max(0, x-1), min(data.shape[0], x+2)
                y_min, y_max = max(0, y-1), min(data.shape[1], y+2)
                z_min, z_max = max(0, z-1), min(data.shape[2], z+2)
                
                local_mean = np.mean(data[x_min:x_max, y_min:y_max, z_min:z_max])
                node_intensities.append(local_mean)
                
            node_intensities = np.array(node_intensities)
            
            # 💡 偵錯：印出第一筆資料萃取出來的真實數值，證明我們確實抓到了！
            if idx == 0:
                print(f"🔎 [偵測確認] 第一位受試者的 13 個腦區原始強度: \n{np.round(node_intensities, 2)}")
            
            # 確保沒有 NaN (如果座標完全在背景中會發生)
            node_intensities = np.nan_to_num(node_intensities)

            # 標準化與矩陣計算
            scaler = StandardScaler()
            node_intensities_scaled = scaler.fit_transform(node_intensities.reshape(-1, 1)).flatten()
            
            num_nodes = len(node_intensities_scaled)
            struct_matrix = np.zeros((num_nodes, num_nodes))
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    diff = abs(node_intensities_scaled[i] - node_intensities_scaled[j])
                    struct_matrix[i, j] = 1.0 / (1.0 + diff)
            
            np.fill_diagonal(struct_matrix, np.nan)
            
            df = pd.DataFrame(struct_matrix)
            df.to_csv(output_csv_path, index=False, header=False)
            
            success_count += 1
            print(f"✅ 成功處理: {output_filename}")
            
        except Exception as e:
            print(f"❌ 處理失敗: {nifti_path} | 錯誤: {e}")

    print("-" * 60)
    print(f"🎉 任務結束！成功產出 {success_count} 個有內容的結構網路 CSV 檔案！")

if __name__ == "__main__":
    main()