import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
TRAIN_FEAT = BASE_DIR / "brainiac_features_train.csv"
TRAIN_COMB = BASE_DIR / "brainiac_combined_train.csv"
TRAIN_KD   = BASE_DIR / "kd_train_aligned.csv"

TEST_FEAT  = BASE_DIR / "brainiac_features_test.csv"
TEST_VIT   = BASE_DIR / "vitmci_features_test.csv"
TEST_COMB  = BASE_DIR / "brainiac_combined_test.csv"
TEST_KD    = BASE_DIR / "kd_test_aligned.csv"

OUT_TRAIN = BASE_DIR / "pcag_train_aligned.csv"
OUT_TEST  = BASE_DIR / "pcag_test_aligned.csv"

def verify_paths(df):
    missing = 0
    for idx, row in df.iterrows():
        if not os.path.exists(row['matrix_path']):
            print(f"  [ERROR] Matrix path not found: {row['matrix_path']}")
            missing += 1
    return missing

print("Starting PCAG data preparation...")

# --- 1. Train Alignment ---
print("\nProcessing Training Set...")
df_tr_feat = pd.read_csv(TRAIN_FEAT)
df_tr_comb = pd.read_csv(TRAIN_COMB)
df_tr_kd   = pd.read_csv(TRAIN_KD)

# brainiac_features_train.csv row i corresponds to brainiac_combined_train.csv row i
df_tr_feat['smri_feat_row'] = df_tr_feat.index
df_tr_feat['orig_sid'] = df_tr_comb['orig_sid']

# Merge with KD aligned to get matrix_path
df_pcag_tr = df_tr_feat[['orig_sid', 'smri_feat_row']].merge(
    df_tr_kd[['subject_id', 'matrix_path', 'label']],
    left_on='orig_sid',
    right_on='subject_id',
    how='inner'
)

# Clean up
df_pcag_tr = df_pcag_tr[['subject_id', 'matrix_path', 'label', 'smri_feat_row']]
df_pcag_tr.to_csv(OUT_TRAIN, index=False)

print(f"Training set aligned: {len(df_pcag_tr)} subjects")
print("Label distribution:")
print(df_pcag_tr['label'].value_counts().sort_index())

# --- 2. Test Alignment ---
print("\nProcessing Test Set...")
df_te_feat = pd.read_csv(TEST_FEAT)
df_te_vit  = pd.read_csv(TEST_VIT)
df_te_comb = pd.read_csv(TEST_COMB)
df_te_kd   = pd.read_csv(TEST_KD)

# vitmci_features_test.csv row i corresponds to brainiac_features_test.csv row i
df_te_vit['smri_feat_row'] = df_te_vit.index

# Merge with brainiac_combined_test to get orig_sid
df_te_aligned_meta = df_te_vit[['pat_id', 'smri_feat_row']].merge(
    df_te_comb[['pat_id', 'orig_sid']],
    on='pat_id',
    how='inner'
)

# Merge with kd_test_aligned to get matrix_path
df_pcag_te = df_te_aligned_meta.merge(
    df_te_kd[['subject_id', 'matrix_path', 'label']],
    left_on='orig_sid',
    right_on='subject_id',
    how='inner'
)

# Clean up
df_pcag_te = df_pcag_te[['subject_id', 'matrix_path', 'label', 'smri_feat_row']]
df_pcag_te.to_csv(OUT_TEST, index=False)

print(f"Test set aligned: {len(df_pcag_te)} subjects")
print("Label distribution:")
print(df_pcag_te['label'].value_counts().sort_index())

# --- 3. Verification ---
print("\nVerifying Data...")
tr_missing = verify_paths(df_pcag_tr)
te_missing = verify_paths(df_pcag_te)

if tr_missing == 0 and te_missing == 0:
    print("All matrix_paths verified.")
else:
    print(f"Verification failed: {tr_missing + te_missing} missing files.")

print(f"smri_feat_row range (Train): {df_pcag_tr['smri_feat_row'].min()} to {df_pcag_tr['smri_feat_row'].max()} (Expected 0 to {len(df_tr_feat)-1})")
print(f"smri_feat_row range (Test):  {df_pcag_te['smri_feat_row'].min()} to {df_pcag_te['smri_feat_row'].max()} (Expected 0 to {len(df_te_feat)-1})")

print(f"\n[DONE] Saved to {OUT_TRAIN} and {OUT_TEST}")
