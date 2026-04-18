import pandas as pd

# ================= 檔案路徑 =================
IDA_CSV = "/home/wei-chi/Data/idaSearch_4_12_2026.csv"
EXCLUDE_TXT = "/home/wei-chi/Data/exclude_ids.txt"
# 這次我們改成輸出 txt 檔，方便你直接複製貼上
OUTPUT_TXT = "/home/wei-chi/Data/new_adni_subjects.txt" 

def filter_adni_data():
    print("🚀 啟動 IDA 雙模態過濾引擎 (Subject ID 專用版)...")
    
    # 1. 讀取黑名單
    with open(EXCLUDE_TXT, 'r') as f:
        exclude_ids = set([line.strip() for line in f.readlines() if line.strip()])
    print(f"🛡️ 載入黑名單：共 {len(exclude_ids)} 位受試者需避開。")
    
    # 2. 讀取 IDA 總表
    try:
        df = pd.read_csv(IDA_CSV)
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {IDA_CSV}")
        return

    df.columns = [str(c).strip().replace('"', '') for c in df.columns]
    
    # 強制指定欄位名稱 (根據你剛才的輸出結果)
    subj_col = 'Subject ID'
    desc_col = 'Description'
    
    if subj_col not in df.columns or desc_col not in df.columns:
        print(f"❌ 找不到 '{subj_col}' 或 '{desc_col}' 欄位，請確認 CSV 內容。")
        return
        
    print(f"✅ 成功鎖定欄位：受試者=[{subj_col}], 描述=[{desc_col}]")
    
    # 3. 過濾掉黑名單
    df_new = df[~df[subj_col].isin(exclude_ids)]
    
    # 4. 尋找雙模態受試者
    df_new['desc_lower'] = df_new[desc_col].astype(str).str.lower()
    valid_subjects = []
    
    for subj, group in df_new.groupby(subj_col):
        # 條件 A: 包含 rs-fMRI
        has_fmri = group['desc_lower'].str.contains('rest|fmri|epb', na=False).any()
        # 條件 B: 包含 T1 結構影像
        has_t1 = group['desc_lower'].str.contains('mprage|t1|fspgr', na=False).any()
        
        if has_fmri and has_t1:
            valid_subjects.append(subj)
            
    print(f"🎯 篩選完畢！找到 {len(valid_subjects)} 位「全新」且「同時擁有 T1+fMRI」的受試者！")
    
    if len(valid_subjects) == 0:
        print("⚠️ 沒有找到符合條件的新受試者。")
        return
    
    # 5. 匯出成逗號分隔的純文字檔，方便貼上 IDA 網站
    with open(OUTPUT_TXT, 'w') as f:
        f.write(",".join(valid_subjects))
        
    print(f"\n✅ 已將名單輸出至 {OUTPUT_TXT}")
    print("👉 接下來請打開這個 txt 檔，複製裡面所有的文字 (逗號分隔的 ID)。")
    print("👉 回到 IDA Advanced Search 網頁，貼在 'Subject' 搜尋框裡，直接 Search 就能把這批人打包了！")

if __name__ == "__main__":
    filter_adni_data()