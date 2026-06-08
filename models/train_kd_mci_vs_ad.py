"""
train_kd_mci_vs_ad.py
=====================
Trains KD GNN for MCI vs AD only, saving per-seed checkpoints.
Checkpoints saved to: resnet_checkpoints/gnn_checkpoints/gnn_MCI_vs_AD_seed{N}.pt

Run:
    cd /home/wei-chi/Alzheimers_Project/external_data/scripts/models
    python train_kd_mci_vs_ad.py
"""
import os, sys
sys.path.insert(0, '/home/wei-chi/Alzheimers_Project/external_data/scripts/models')
sys.path.insert(0, '/home/wei-chi/Alzheimers_Project/external_data/scripts')

import torch
from train_hierarchical_gnn_kd import (
    load_data,
    load_teacher_probs,
    run_task_kd,
    fMRIDataset_KD,
    TEACHER_PROBS_DIR,
    SEEDS,
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    task_pair = ('MCI', 'AD')
    ckpt_dir  = os.path.join(TEACHER_PROBS_DIR, "gnn_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading data...")
    df_full = load_data()
    df_task = df_full[df_full['diagnosis'].isin(list(task_pair))].copy()
    df_task['current_task_label'] = df_task['diagnosis'].map({task_pair[0]: 0, task_pair[1]: 1})
    df_task = df_task.reset_index(drop=True)
    print(f"MCI vs AD samples: {len(df_task)}")

    teacher_dict = load_teacher_probs(task_pair)

    for seed in SEEDS:
        print(f"\n{'='*50}\n  Seed {seed}\n{'='*50}")
        ckpt_path = os.path.join(ckpt_dir, f"gnn_MCI_vs_AD_seed{seed}.pt")
        if os.path.exists(ckpt_path):
            print(f"  Already exists, skipping: {ckpt_path}")
            continue
        run_task_kd(df_task, task_pair, teacher_dict, device, seed, ckpt_dir=ckpt_dir)
        print(f"  Saved: {ckpt_path}")

    print("\nAll done. Checkpoints:")
    for seed in SEEDS:
        p = os.path.join(ckpt_dir, f"gnn_MCI_vs_AD_seed{seed}.pt")
        exists = "✓" if os.path.exists(p) else "✗"
        print(f"  {exists} {p}")

if __name__ == "__main__":
    main()
