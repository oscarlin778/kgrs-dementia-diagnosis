import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, '/home/wei-chi/Data/script')
from models.train_hierarchical_gnn_e6 import FNPGNNv8_KD, fMRIDataset_KD, load_data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints/e6_checkpoints"

def evaluate_alignment(task_pair=('NC', 'MCI')):
    print(f"\nAnalyzing Alignment for Task: {task_pair[0]} vs {task_pair[1]}")
    df_full = load_data()
    df_task = df_full[df_full['diagnosis'].isin(task_pair)].copy()
    df_task['current_task_label'] = df_task['diagnosis'].map({task_pair[0]: 0, task_pair[1]: 1})
    
    dataset = fMRIDataset_KD(df_task)
    
    seeds = [42, 123, 456]
    for seed in seeds:
        ckpt_path = os.path.join(CKPT_DIR, f"e6_{task_pair[0]}_vs_{task_pair[1]}_seed{seed}.pt")
        if not os.path.exists(ckpt_path): continue
        
        model = FNPGNNv8_KD().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        
        features = {k: [] for k in ['TPMIC_NC', 'ADNI_NC', 'TPMIC_MCI', 'ADNI_MCI', 'TPMIC_AD', 'ADNI_AD']}
        with torch.no_grad():
            for i in range(len(dataset)):
                item = dataset[i]
                diag_idx = item['diag_type'].item()
                diag = ['NC', 'MCI', 'AD'][diag_idx] if diag_idx != -1 else 'UNKNOWN'
                src = 'ADNI' if item['domain_label'] == 1 else 'TPMIC'
                key = f"{src}_{diag}"
                
                _, flat = model(item['x'].unsqueeze(0).to(DEVICE), item['adj'].unsqueeze(0).to(DEVICE))
                if key in features:
                    features[key].append(flat.cpu().numpy()[0])
        
        print(f"  [Seed {seed}]")
        if features['TPMIC_NC'] and features['ADNI_NC']:
            d_nc = np.linalg.norm(np.mean(features['TPMIC_NC'], axis=0) - np.mean(features['ADNI_NC'], axis=0))
            print(f"    NC Dist (TPMIC vs ADNI):  {d_nc:.4f}")
        if features['TPMIC_MCI'] and features['ADNI_MCI']:
            d_mci = np.linalg.norm(np.mean(features['TPMIC_MCI'], axis=0) - np.mean(features['ADNI_MCI'], axis=0))
            print(f"    MCI Dist (TPMIC vs ADNI): {d_mci:.4f}")
        if features['TPMIC_AD'] and features['ADNI_AD']:
            d_ad = np.linalg.norm(np.mean(features['TPMIC_AD'], axis=0) - np.mean(features['ADNI_AD'], axis=0))
            print(f"    AD Dist (TPMIC vs ADNI):  {d_ad:.4f}")
        
        # Inter-class distance
        if features['TPMIC_NC'] and features['TPMIC_MCI']:
            d_nc_mci = np.linalg.norm(np.mean(features['TPMIC_NC'], axis=0) - np.mean(features['TPMIC_MCI'], axis=0))
            print(f"    TPMIC (NC vs MCI) Dist:   {d_nc_mci:.4f}")

if __name__ == "__main__":
    evaluate_alignment(('NC', 'MCI'))
    evaluate_alignment(('NC', 'AD'))
    evaluate_alignment(('MCI', 'AD'))
