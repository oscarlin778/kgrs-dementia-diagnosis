import pandas as pd
import os

def check_file(path):
    return os.path.exists(path)

# Training CSV
train_df = pd.read_csv('/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits/combined_train.csv')
brainiac_train_data = []

adni_root = '/home/wei-chi/Alzheimers_Project/external_data/datasets/ADNI_sMRI_Aligned_MNI/'
tpmic_root = '/home/wei-chi/Alzheimers_Project/external_models/sMRI_data_MultiModal_Aligned_MNI/'

for _, row in train_df.iterrows():
    sid = row['subject_id']
    diag = row['diagnosis']
    source = row['source']
    label = row['label']
    
    if source.startswith('ADNI'):
        path_without_ext = os.path.join(adni_root, diag, f"{sid}_T1_MNI")
    else:
        path_without_ext = os.path.join(tpmic_root, diag, f"{sid}_T1")
    
    full_path = path_without_ext + ".nii.gz"
    
    if check_file(full_path):
        brainiac_train_data.append({
            'pat_id': path_without_ext,
            'label': label,
            'source': source,
            'orig_sid': sid,
            'dataset': 'combined'
        })
    else:
        if source.startswith('ADNI'):
            path_without_ext_sub = os.path.join(adni_root, diag, f"sub-{sid}_T1_MNI")
            if check_file(path_without_ext_sub + ".nii.gz"):
                 brainiac_train_data.append({
                    'pat_id': path_without_ext_sub,
                    'label': label,
                    'source': source,
                    'orig_sid': sid,
                    'dataset': 'combined'
                })
                 continue
        print(f"Warning: File not found for {sid} ({source}): {full_path}")

brainiac_train_df = pd.DataFrame(brainiac_train_data)
brainiac_train_df.to_csv('/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/brainiac_combined_train.csv', index=False)
print(f"Saved training CSV with {len(brainiac_train_df)} subjects.")

# Test CSV
test_df = pd.read_csv('/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits/tpmic_test.csv')
brainiac_test_data = []

for _, row in test_df.iterrows():
    sid = row['subject_id']
    diag = row['diagnosis']
    label = row['label']
    
    path_without_ext = os.path.join(tpmic_root, diag, f"{sid}_T1")
    full_path = path_without_ext + ".nii.gz"
    
    if check_file(full_path):
        brainiac_test_data.append({
            'pat_id': path_without_ext,
            'label': label,
            'orig_sid': sid,
            'dataset': 'tpmic'
        })
    else:
        print(f"Warning: File not found for {sid} (TPMIC): {full_path}")

brainiac_test_df = pd.DataFrame(brainiac_test_data)
brainiac_test_df.to_csv('/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/brainiac_tpmic_test.csv', index=False)
print(f"Saved test CSV with {len(brainiac_test_df)} subjects.")
