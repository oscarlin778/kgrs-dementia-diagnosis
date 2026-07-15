"""
train_concat_fusion.py — Concat fusion baseline (no cross-attention).

Replaces PCAG cross-attention with simple concat + 2-layer MLP.
Everything else identical to train_pcag_combat_fusion.py (same splits,
same ComBat, same GNN encoder, same training loop).

Usage:
    python train_concat_fusion.py --task NC_vs_AD
    python train_concat_fusion.py --task NC_vs_MCI
    python train_concat_fusion.py --task MCI_vs_AD
"""
import os, sys, argparse, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, roc_curve
from pathlib import Path
from neuroCombat import neuroCombat

# Reuse model components from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_pcag_combat_fusion import (
    FMRIEncoder, PCAGDataset,
    extract_node_features, build_adj,
    FMRI_EMBED_DIM, HIDDEN_DIM, DROPOUT, N_ROIS, K_RATIO,
    N_NETWORKS, NODE_FEAT_DIM, POOLING_MAT, NETWORK_MAP,
    net_list, roi_to_net
)

FMRI_DIM = FMRI_EMBED_DIM   # 1280
SMRI_DIM = 768


class ConcatFusion(nn.Module):
    """Simple concat + MLP baseline (no cross-attention)."""
    def __init__(self, fmri_dim=FMRI_DIM, smri_dim=SMRI_DIM,
                 hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(fmri_dim + smri_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, fmri_emb, smri_feat):
        return self.proj(torch.cat([fmri_emb, smri_feat], dim=-1))


class ConcatModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fmri_encoder = FMRIEncoder()
        self.fusion = ConcatFusion(hidden_dim=hidden_dim)

    def forward(self, x_node, adj, smri_feat):
        return self.fusion(self.fmri_encoder(x_node, adj), smri_feat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="NC_vs_MCI",
                        choices=["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fmri_harmonized", action="store_true", default=False,
                        help="使用 no-label fMRI ComBat 諧波化矩陣")
    parser.add_argument("--fmri_combat_dir", type=str, default="fmri_combat_v2_nolabel")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
    RES_DIR  = BASE_DIR / "results"
    CKPT_DIR = BASE_DIR / "checkpoints" / "pcag_concat_v2"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    TASK_CFG = {
        'NC_vs_AD':  {'classes': [0, 2], 'pos': 2},
        'NC_vs_MCI': {'classes': [0, 1], 'pos': 1},
        'MCI_vs_AD': {'classes': [1, 2], 'pos': 2},
    }
    cfg = TASK_CFG[args.task]

    # Load v2 splits
    df_tr_meta = pd.read_csv(BASE_DIR / "kd_train_aligned_v2.csv")
    df_te_meta = pd.read_csv(BASE_DIR / "kd_test_aligned_v2.csv")
    df_tr_pcag = pd.read_csv(BASE_DIR / "pcag_train_aligned_v2.csv")
    df_te_pcag = pd.read_csv(BASE_DIR / "pcag_test_aligned_v2.csv")

    df_tr = df_tr_pcag.merge(df_tr_meta[['subject_id', 'source']], on='subject_id', how='inner')
    df_te = df_te_pcag.merge(df_te_meta[['subject_id', 'source']], on='subject_id', how='inner')
    df_tr = df_tr[df_tr['label'].isin(cfg['classes'])].reset_index(drop=True)
    df_tr['bin_label'] = (df_tr['label'] == cfg['pos']).astype(int)
    df_te = df_te[df_te['label'].isin(cfg['classes'])].reset_index(drop=True)
    df_te['bin_label'] = (df_te['label'] == cfg['pos']).astype(int)

    # F-1 修正：使用 no-label ComBat 諧波化 fMRI 矩陣
    if args.fmri_harmonized:
        fmri_dir = BASE_DIR / args.fmri_combat_dir
        def _swap(sid):
            p = fmri_dir / f"{sid}_combat.npy"
            if not p.exists():
                raise FileNotFoundError(f"Harmonized fMRI not found: {p}")
            return str(p)
        df_tr['matrix_path'] = df_tr['subject_id'].apply(_swap)
        df_te['matrix_path'] = df_te['subject_id'].apply(_swap)
        print(f"  [fMRI harmonized] using {fmri_dir}")

    # Load sMRI features
    smri_cols = [f"Feature_{i}" for i in range(768)]
    smri_tr_raw = pd.read_csv(BASE_DIR / "brainiac_features_train_v2.csv")\
                    .iloc[df_tr['smri_feat_row']][smri_cols].values
    smri_te_raw = pd.read_csv(BASE_DIR / "brainiac_features_combined_test_v2.csv")\
                    .iloc[df_te['smri_feat_row']][smri_cols].values

    # ComBat harmonization (train-only fit)
    all_sites = sorted(list(set(df_tr['source'].unique()) | set(df_te['source'].unique())))
    site_map  = {s: i for i, s in enumerate(all_sites)}
    df_tr['site_idx'] = df_tr['source'].map(site_map)
    df_te['site_idx'] = df_te['source'].map(site_map)

    combat_out = neuroCombat(
        dat=smri_tr_raw.T,
        covars=df_tr[['site_idx', 'bin_label']],
        batch_col='site_idx', categorical_cols=['bin_label']
    )
    smri_tr = combat_out['data'].T
    est = combat_out['estimates']

    def apply_combat_local(dat, site_indices, est):
        g_mean = est['stand.mean'].mean(axis=1)
        v_pool = est['var.pooled'].flatten()
        gamma  = est['gamma.star'][site_indices]
        delta  = est['delta.star'][site_indices]
        x_std  = (dat - g_mean) / (np.sqrt(v_pool) + 1e-8)
        x_adj  = (x_std - gamma) / (np.sqrt(delta) + 1e-8)
        return x_adj * np.sqrt(v_pool) + g_mean

    smri_te = apply_combat_local(smri_te_raw, df_te['site_idx'].values, est)
    print(f"[{args.task}] Train: {len(df_tr)}  Test: {len(df_te)}")

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros(len(df_tr))
    test_probs_folds = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(df_tr, df_tr['bin_label'])):
        print(f"\n--- Fold {fold+1}/5 ---")
        tr_paths  = df_tr.iloc[tr_idx]['matrix_path'].tolist()
        tr_smri   = smri_tr[tr_idx]
        tr_labels = df_tr.iloc[tr_idx]['bin_label'].values
        val_paths = df_tr.iloc[val_idx]['matrix_path'].tolist()
        val_smri  = smri_tr[val_idx]
        val_labels= df_tr.iloc[val_idx]['bin_label'].values

        counts  = np.bincount(tr_labels)
        sampler = WeightedRandomSampler(
            (1.0 / counts)[tr_labels], len(tr_labels))

        tr_loader  = DataLoader(PCAGDataset(tr_paths, tr_smri, tr_labels),
                                batch_size=args.batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(PCAGDataset(val_paths, val_smri, val_labels),
                                batch_size=args.batch_size, shuffle=False)
        te_loader  = DataLoader(PCAGDataset(df_te['matrix_path'].tolist(), smri_te,
                                            df_te['bin_label'].values),
                                batch_size=args.batch_size, shuffle=False)

        model     = ConcatModel(hidden_dim=args.hidden_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
        criterion = nn.CrossEntropyLoss()

        best_auc, patience_counter, best_state = 0, 0, None
        patience = 40
        for epoch in range(args.epochs):
            model.train()
            for x, adj, smri, y in tr_loader:
                x, adj, smri, y = x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x, adj, smri), y)
                loss.backward(); optimizer.step()

            model.eval()
            val_p = []
            with torch.no_grad():
                for x, adj, smri, y in val_loader:
                    out = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE))
                    val_p.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())

            auc = roc_auc_score(val_labels, val_p) if len(set(val_labels)) > 1 else 0.5
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience: break

        print(f"Best Val AUC: {best_auc:.4f}")
        ckpt_path = CKPT_DIR / f"concat_{args.task}_fold{fold}.pt"
        torch.save({'model_state': best_state, 'val_auc': best_auc}, ckpt_path)
        print(f"[SAVED] {ckpt_path}")

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            fold_p = []
            for x, adj, smri, y in val_loader:
                out = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE))
                fold_p.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            oof_probs[val_idx] = fold_p

            te_p = []
            for x, adj, smri, y in te_loader:
                out = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE))
                te_p.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            test_probs_folds.append(te_p)

    # Evaluation
    oof_auc = roc_auc_score(df_tr['bin_label'], oof_probs)
    fpr, tpr, thresholds = roc_curve(df_tr['bin_label'], oof_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]

    avg_test_p = np.mean(test_probs_folds, axis=0)
    test_auc   = roc_auc_score(df_te['bin_label'], avg_test_p)
    test_pred  = (avg_test_p >= best_thresh).astype(int)
    cm = confusion_matrix(df_te['bin_label'], test_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auc":  float(test_auc),
        "sens": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "spec": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        "f1":   float(f1_score(df_te['bin_label'], test_pred)),
        "acc":  float(accuracy_score(df_te['bin_label'], test_pred)),
        "cm":   cm.tolist(),
    }
    print(f"\nConcat Fusion Test Metrics ({args.task}):")
    for k, v in metrics.items(): print(f"  {k}: {v}")

    fmri_tag = "_fmricombat_nolabel" if args.fmri_harmonized else ""
    out_file = RES_DIR / f"concat_{args.task}_results_v2{fmri_tag}.json"
    with open(out_file, "w") as f:
        json.dump({"oof_auc": float(oof_auc), "test_metrics": metrics}, f, indent=2)
    print(f"\n[SAVED] {out_file}")

    _seed_tag = '' if args.seed == 42 else f'_s{args.seed}'
    np.savez(str(RES_DIR / f"concat_{args.task}_probs_v2{fmri_tag}{_seed_tag}.npz"),
             test_probs=avg_test_p, test_labels=df_te['bin_label'].values,
             oof_probs=oof_probs, oof_labels=df_tr['bin_label'].values)

if __name__ == "__main__":
    main()
