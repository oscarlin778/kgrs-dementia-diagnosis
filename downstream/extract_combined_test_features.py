import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# Import necessary functions from brainiac_extractor
from brainiac_extractor import (
    load_vitmci_extractor, 
    extract_features, 
    get_preprocessing_transforms,
    CKPT_PATH
)

def main():
    input_csv = "brainiac_combined_test.csv"
    output_csv = "vitmci_features_combined_test.csv"
    
    # Check if output exists
    if os.path.exists(output_csv):
        print(f"[INFO] {output_csv} already exists. Skipping extraction.")
        return

    # 1. Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    model = load_vitmci_extractor(CKPT_PATH, device, verbose=True)
    transforms = get_preprocessing_transforms(already_preprocessed=True)
    
    # 2. Read input CSV
    df = pd.read_csv(input_csv)
    print(f"[INFO] Total subjects to process: {len(df)}")
    
    feature_data = []
    
    # 3. Extract features
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        pat_id = row['pat_id']
        label = row['label']
        nii_path = f"{pat_id}.nii.gz"
        
        if not os.path.exists(nii_path):
            print(f"[WARN] File not found: {nii_path}. Skipping.")
            continue
            
        try:
            # Load and preprocess
            img_tensor = transforms(nii_path) # Returns (1, 96, 96, 96)
            img_tensor = img_tensor.unsqueeze(0) # (1, 1, 96, 96, 96)
            
            # Extract
            feats = extract_features(model, img_tensor, device) # (1, 768)
            feats_np = feats.numpy().flatten()
            
            # Prepare row
            entry = {
                'pat_id': pat_id,
                'label': label
            }
            for i, val in enumerate(feats_np):
                entry[f'Feature_{i}'] = val
            
            feature_data.append(entry)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {pat_id}: {str(e)}")
            continue
            
    # 4. Save results
    df_features = pd.DataFrame(feature_data)
    df_features.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Saved {len(df_features)} features to {output_csv}")

if __name__ == "__main__":
    main()
