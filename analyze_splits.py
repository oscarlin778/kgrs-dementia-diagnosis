import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold

CSV_PATHS = [
    "/home/wei-chi/Model/_dataset_mapping.csv",
    "/home/wei-chi/Data/dataset_index_116_clean_old.csv",
    "/home/wei-chi/Data/adni_dataset_index_116.csv"
]
MATRIX_DIR = "/home/wei-chi/Model/processed_116_matrices"

def get_subject_id(path_str):
    basename = os.path.basename(str(path_str))
    clean = re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy|_T1_MNI\.nii\.gz|_T1\.nii\.gz|\.nii\.gz)$', '', basename)
    clean = re.sub(r'^(sub-|sub_|old_dswau)', '', clean)
    return clean.strip()

def analyze_subjects():
    valid_data = []
    seen_paths = set()
    for path in CSV_PATHS:
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m_path = (row.get('matrix_path') 
                      or (os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy") if pd.notna(row.get('new_id_base')) else None)
                      or (os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy") if pd.notna(row.get('Subject')) else None))
            if not (m_path and os.path.exists(m_path)) or m_path in seen_paths: continue
            try:
                if np.load(m_path).shape != (116, 116): continue
                diag = str(row.get('diagnosis', '')).upper()
                if diag in ['NC', 'MCI', 'AD']:
                    src = 'ADNI' if ('adni' in m_path.lower() or 'old_dswau' in m_path.lower()) else 'TPMIC'
                    subj_id = get_subject_id(m_path)
                    valid_data.append({'subj_id': subj_id, 'diagnosis': diag, 'source': src, 'matrix_path': m_path})
                    seen_paths.add(m_path)
            except: continue
    
    df = pd.DataFrame(valid_data)
    print(f"Total entries: {len(df)}")
    
    # Check for multiple entries per subject (same subject might have multiple scans or be in multiple CSVs)
    subj_df = df.drop_duplicates(subset=['subj_id']).copy()
    print(f"Unique subjects: {len(subj_df)}")
    
    print("\nDiagnosis distribution (Unique Subjects):")
    print(subj_df['diagnosis'].value_counts())
    
    print("\nSource distribution (Unique Subjects):")
    print(subj_df['source'].value_counts())
    
    # Create a stratified split
    subj_df['strata'] = subj_df['diagnosis'] + "_" + subj_df['source']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    split_info = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(subj_df, subj_df['strata'])):
        val_subjs = subj_df.iloc[val_idx]['subj_id'].tolist()
        split_info[f"fold_{fold}"] = val_subjs
        print(f"Fold {fold}: {len(val_subjs)} validation subjects")

    import json
    with open("Data/script/unified_subject_split.json", "w") as f:
        json.dump(split_info, f, indent=4)
    print("\nUnified split saved to Data/script/unified_subject_split.json")

if __name__ == "__main__":
    analyze_subjects()
