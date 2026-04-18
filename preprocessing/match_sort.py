import os
import re
import shutil
import pandas as pd

# ==========================================
# 1. 路徑參數設定
# ==========================================
RAW_ROOT_DIR = "/home/wei-chi/Data/ADNI_Raw_Download/ADNI" 
CSV_PATH = "/home/wei-chi/DXSUM_10Apr2026.csv" 
OUTPUT_DIR = "/home/wei-chi/Data/Matched_Dual_Modal_Dataset"

# ==========================================
# 2. 解析 CSV 建立診斷字典
# ==========================================
def build_diagnosis_dict(csv_path):
    print(f"📖 正在讀取臨床數據: {csv_path}")
    # ADNI 的 CSV 常有引號，用 pandas 處理
    df = pd.read_csv(csv_path, quotechar='"', skipinitialspace=True)
    
    # 清理欄位名稱（移除可能存在的空格或隱藏字元）
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 篩選 Baseline 資料
    baseline_df = df[df['VISCODE'].isin(['bl', 'm00', 'v01'])]
    
    diag_dict = {}
    
    # 遍歷每一行，根據 DIAGNOSIS 數字進行分類
    for _, row in baseline_df.iterrows():
        ptid = str(row['PTID']).strip()
        
        try:
            dx_val = int(row['DIAGNOSIS'])
            if dx_val == 1:
                dx = 'NC'
            elif dx_val == 2:
                dx = 'MCI'
            elif dx_val == 3:
                dx = 'AD'
            else:
                continue
            diag_dict[ptid] = dx
        except (ValueError, TypeError):
            continue
            
    print(f"✅ 成功載入 {len(diag_dict)} 筆 Baseline 診斷紀錄。")
    return diag_dict

# ==========================================
# 3. 主程式
# ==========================================
def main():
    print("🚀 啟動雙模態資料搜身配對引擎...")
    
    diag_dict = build_diagnosis_dict(CSV_PATH)
    if not diag_dict:
        print("❌ 診斷字典為空，請檢查 CSV 內容！")
        return
    
    if not os.path.exists(RAW_ROOT_DIR):
        print(f"❌ 找不到原始資料目錄: {RAW_ROOT_DIR}")
        return

    all_subjects = [d for d in os.listdir(RAW_ROOT_DIR) 
                    if os.path.isdir(os.path.join(RAW_ROOT_DIR, d)) and re.match(r'\d{3}_S_\d{4}', d)]
    
    print(f"🔍 在原始資料夾中找到了 {len(all_subjects)} 位受試者。")
    
    success_count = 0
    
    for subj in all_subjects:
        subj_dir = os.path.join(RAW_ROOT_DIR, subj)
        subdirs = [d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))]
        
        t1_source_folders = []
        fmri_source_folders = []
        
        for d in subdirs:
            d_upper = d.upper()
            # 針對 ADNI 3 的常見命名過濾
            if 'MPRAGE' in d_upper or 'T1' in d_upper:
                t1_source_folders.append(os.path.join(subj_dir, d))
            elif 'RSFMRI' in d_upper or 'RESTING' in d_upper or 'FMRI' in d_upper:
                fmri_source_folders.append(os.path.join(subj_dir, d))
                
        if t1_source_folders and fmri_source_folders and subj in diag_dict:
            dx = diag_dict[subj]
            
            subj_t1_out = os.path.join(OUTPUT_DIR, dx, subj, "T1")
            subj_fmri_out = os.path.join(OUTPUT_DIR, dx, subj, "fMRI")
            
            os.makedirs(subj_t1_out, exist_ok=True)
            os.makedirs(subj_fmri_out, exist_ok=True)
            
            try:
                for src in t1_source_folders:
                    shutil.copytree(src, os.path.join(subj_t1_out, os.path.basename(src)), dirs_exist_ok=True)
                for src in fmri_source_folders:
                    shutil.copytree(src, os.path.join(subj_fmri_out, os.path.basename(src)), dirs_exist_ok=True)
                
                success_count += 1
                print(f"📦 完美配對並搬運完成: [{dx}] {subj}")
            except Exception as e:
                print(f"❌ 搬運失敗: {subj} | 錯誤: {e}")

    print("=" * 50)
    print(f"🏁 任務結束！成功建立了 {success_count} 個雙模態完美配對資料夾！")
    print(f"路徑: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()