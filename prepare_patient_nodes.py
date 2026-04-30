import pandas as pd

def process_patient_demographics(index_csv_path, demographics_csv_path, output_csv_path):
    print("⏳ 開始讀取與清理 Demographic 原始資料...")
    # 1. 讀取與清理 ADNI CSV (人口統計資料)
    df_demo = pd.read_csv(demographics_csv_path)
    
    # 過濾空值並處理縱向資料 (每個 PTID 只留一筆)
    df_demo = df_demo.dropna(subset=['PTID']).copy()
    df_demo['Gender'] = df_demo.groupby('PTID')['Gender'].ffill().bfill()
    df_demo['RMT_Education'] = df_demo.groupby('PTID')['RMT_Education'].ffill().bfill()
    df_demo['Age_Baseline'] = df_demo.groupby('PTID')['Age_Baseline'].ffill().bfill()
    df_unique_patients = df_demo.drop_duplicates(subset=['PTID'], keep='first').copy()
    
    print(f"✅ 清理完成，共取得 {len(df_unique_patients)} 位獨立病患的基本資料。")

    # 2. 讀取 Index CSV (包含臨床診斷與矩陣路徑)
    print("⏳ 讀取 Dataset Index CSV...")
    df_index = pd.read_csv(index_csv_path)
    
    # 將 subject_id (如 'sub-011_S_6303') 去除 'sub-'，變成與 PTID 相同的格式 ('011_S_6303')
    df_index['PTID'] = df_index['subject_id'].str.replace('sub-', '')
    print(f"✅ 成功載入 {len(df_index)} 筆推論影像索引資料。")

    # 3. 資料對齊與合併 (Left Merge 保留所有索引檔內的病患)
    df_merged = pd.merge(df_index, df_unique_patients, on='PTID', how='left')
    
    # 4. 整理最終輸出的欄位
    gender_map = {1.0: 'Male', 0.0: 'Female'}
    
    df_final = pd.DataFrame({
        'subject_id': df_merged['PTID'], # 使用乾淨的 ID 存入 Neo4j (如 011_S_6303)
        'age': df_merged['Age_Baseline'].fillna(0).astype(int), 
        'sex': df_merged['Gender'].map(gender_map).fillna('Unknown'),
        'education': df_merged['RMT_Education'].fillna(0).astype(int),
        'clinical_dx': df_merged['diagnosis'] # 成功補上真實的診斷標籤！(AD, MCI, NC)
    })
    
    # 5. 輸出乾淨的 CSV 給 Neo4j
    df_final.to_csv(output_csv_path, index=False)
    print(f"🎯 對齊成功！已產出 {len(df_final)} 筆具備臨床診斷的病患節點資料，儲存至 {output_csv_path}")

if __name__ == "__main__":
    # 直接讀取這兩個 CSV 即可，不需要再指定矩陣資料夾路徑了
    INDEX_CSV = "/home/wei-chi/Data/adni_dataset_index_116.csv"
    DEMO_CSV = "/home/wei-chi/Data/script/patient_demographics.csv"
    OUTPUT_CSV = "final_patient_nodes.csv"
    
    process_patient_demographics(INDEX_CSV, DEMO_CSV, OUTPUT_CSV)