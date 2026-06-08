import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
ADNI_NEW_CSV = "/home/wei-chi/Alzheimers_Project/external_data/metadata/adni_dataset_index_116.csv"
ADNI_OLD_CSV = "/home/wei-chi/Alzheimers_Project/external_data/metadata/dataset_index_116_clean_old.csv"
TPMIC_CSV = "/home/wei-chi/Alzheimers_Project/external_models/_dataset_mapping.csv"
OUTPUT_DIR = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits"
MATRIX_DIR_TPMIC = "/home/wei-chi/Alzheimers_Project/external_models/processed_116_matrices"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_id(subj_id):
    # Remove prefix like sub- or old_dswau
    clean = re.sub(r'^(sub-|sub_|old_dswau)', '', subj_id)
    # Remove suffix like _task-rest_bold
    clean = re.sub(r'(_task-rest_bold)', '', clean)
    return clean.strip()

def get_label(diag):
    diag = str(diag).upper()
    if diag == 'NC': return 0
    if diag == 'MCI': return 1
    if diag == 'AD': return 2
    return -1

# 1. Load ADNI subjects
print("Loading ADNI subjects...")
df_adni_new = pd.read_csv(ADNI_NEW_CSV)
df_adni_new['source'] = 'ADNI_new'
df_adni_old = pd.read_csv(ADNI_OLD_CSV)
df_adni_old['source'] = 'ADNI_old'

df_adni_all = pd.concat([df_adni_new, df_adni_old], ignore_index=True)
adni_data = []
for _, row in df_adni_all.iterrows():
    m_path = row['matrix_path']
    if os.path.exists(m_path):
        if np.load(m_path).shape == (116, 116):
            adni_data.append({
                'subject_id': normalize_id(row['subject_id']),
                'matrix_path': m_path,
                'diagnosis': str(row['diagnosis']).upper(),
                'label': get_label(row['diagnosis']),
                'source': row['source']
            })

df_adni = pd.DataFrame(adni_data)

# Ensure uniqueness
df_adni = df_adni.drop_duplicates(subset=['subject_id']).reset_index(drop=True)

# 2. Stratified 80/20 split of ADNI
train_adni, test_adni = train_test_split(
    df_adni, test_size=0.2, random_state=42, stratify=df_adni['diagnosis']
)

# Enforce minimum 1 AD in test set
if (test_adni['diagnosis'] == 'AD').sum() < 1:
    print("Warning: AD count in test set < 1. Re-splitting with manual adjustment if needed.")

# 3. Load TPMIC subjects
print("Loading TPMIC subjects...")
df_tpmic_raw = pd.read_csv(TPMIC_CSV)
tpmic_data = []
for _, row in df_tpmic_raw.iterrows():
    new_id = row['new_id_base']
    diag = row['diagnosis']
    m_path = os.path.join(MATRIX_DIR_TPMIC, f"{new_id}_matrix_116.npy")
    if os.path.exists(m_path) and np.load(m_path).shape == (116, 116):
        tpmic_data.append({
            'subject_id': new_id,
            'matrix_path': m_path,
            'diagnosis': diag,
            'label': get_label(diag),
            'source': 'TPMIC'
        })
df_tpmic = pd.DataFrame(tpmic_data)

# 4. Stratified 80/20 split of TPMIC
train_tpmic, test_tpmic = train_test_split(
    df_tpmic, test_size=0.2, random_state=42, stratify=df_tpmic['diagnosis']
)

# 5. Save files
print(f"Saving splits to {OUTPUT_DIR}...")
train_adni.to_csv(os.path.join(OUTPUT_DIR, "adni_train.csv"), index=False)
test_adni.to_csv(os.path.join(OUTPUT_DIR, "adni_test.csv"), index=False)
train_tpmic.to_csv(os.path.join(OUTPUT_DIR, "tpmic_train.csv"), index=False)
test_tpmic.to_csv(os.path.join(OUTPUT_DIR, "tpmic_test.csv"), index=False)
df_tpmic.to_csv(os.path.join(OUTPUT_DIR, "tpmic_full.csv"), index=False)

# 6. Create Combined Splits
print("\nCreating combined datasets...")
combined_train = pd.concat([train_adni, train_tpmic], ignore_index=True)
combined_test = pd.concat([test_adni, test_tpmic], ignore_index=True)

combined_train.to_csv(os.path.join(OUTPUT_DIR, "combined_train.csv"), index=False)
combined_test.to_csv(os.path.join(OUTPUT_DIR, "combined_test.csv"), index=False)

# Summary for combined_train
ct_counts = combined_train['diagnosis'].value_counts().to_dict()
ct_sources = combined_train['source'].value_counts().to_dict()
print(f"combined_train: n={len(combined_train)}  "
      f"NC={ct_counts.get('NC',0)}  MCI={ct_counts.get('MCI',0)}  AD={ct_counts.get('AD',0)}  "
      f"(ADNI_new={ct_sources.get('ADNI_new',0)}, ADNI_old={ct_sources.get('ADNI_old',0)}, TPMIC={ct_sources.get('TPMIC',0)})")

# 7. Summary Table
def get_counts(df):
    counts = df['diagnosis'].value_counts()
    return counts.get('NC', 0), counts.get('MCI', 0), counts.get('AD', 0), len(df)

print("\nSplit         NC   MCI   AD   Total")
for name, df in [("adni_train", train_adni), ("adni_test", test_adni), 
                 ("tpmic_train", train_tpmic), ("tpmic_test", test_tpmic),
                 ("combined_tr", combined_train), ("combined_ts", combined_test)]:
    nc, mci, ad, total = get_counts(df)
    print(f"{name:<13} {nc:<4} {mci:<5} {ad:<4} {total}")
