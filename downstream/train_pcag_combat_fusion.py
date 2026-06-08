import os
import sys
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pathlib import Path
from neuroCombat import neuroCombat

# --- Constants & ROI Map ---
NETWORK_MAP = {
    'DMN':   [34, 35, 66, 67, 64, 65, 22, 23, 24, 25],
    'SMN':   [0, 1, 56, 57, 68, 69],
    'VN':    [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    'SN':    [28, 29, 30, 31, 32, 33],
    'FPN':   [6, 7, 58, 59, 60, 61],
    'LN':    [36, 37, 38, 39, 40, 41],
    'VAN':   [10, 11, 14, 15],
    'BGN':   [70, 71, 72, 73, 74, 75, 76, 77],
    'CereN': list(range(90, 116))
}
roi_to_net = {}
net_list = list(NETWORK_MAP.keys())
for i, (net, nodes) in enumerate(NETWORK_MAP.items()):
    for node in nodes: roi_to_net[node] = i

N_ROIS = 116
POOLING_MAT = torch.zeros(N_ROIS, len(net_list))
for i, net in enumerate(net_list):
    for node_idx in NETWORK_MAP[net]:
        POOLING_MAT[node_idx, i] = 1.0

HIDDEN_DIM = 128
DROPOUT    = 0.4
K_RATIO    = 0.20
NODE_FEAT_DIM = 116 + 5 + 2 + 2  # 125
N_NETWORKS = 9
FMRI_EMBED_DIM = N_NETWORKS * HIDDEN_DIM + HIDDEN_DIM  # 1280

# --- Feature Extraction Functions ---
def extract_node_features(adj_z):
    N = adj_z.shape[0]; adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO); adj_bin = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]; adj_bin[i, top_idx] = 1.0
    adj_bin = np.maximum(adj_bin, adj_bin.T); degree = adj_bin.sum(axis=1)
    cc = np.diag(adj_bin @ adj_bin @ adj_bin) / (degree * (degree - 1) + 1e-8)
    adj_abs_thresh = adj_abs * adj_bin; pc = np.zeros(N, dtype=np.float32)
    for i in range(N):
        ki = adj_abs_thresh[i].sum()
        if ki > 1e-8:
            pc_i = 1.0
            for net_nodes in NETWORK_MAP.values():
                kim = adj_abs_thresh[i, list(net_nodes)].sum(); pc_i -= (kim / ki) ** 2
            pc[i] = float(np.clip(pc_i, 0.0, 1.0))
    features = []
    for i in range(N):
        row = adj_z[i].copy(); row[i] = 0; fc_feat = row.astype(np.float32)
        stat_feat = np.array([row.mean(), row.std(), (row>0).mean(), (row<0).mean(), (np.abs(row)>0.1).sum()], dtype=np.float32)
        net_i = roi_to_net.get(i, -1)
        if net_i >= 0:
            w_nodes = [r for r in NETWORK_MAP[net_list[net_i]] if r != i]
            b_nodes = [r for r in range(N) if r != i and roi_to_net.get(r, -1) != net_i]
            w_fc = float(np.mean([row[r] for r in w_nodes])) if w_nodes else 0.0
            b_fc = float(np.mean([row[r] for r in b_nodes])) if b_nodes else 0.0
        else: w_fc, b_fc = 0.0, 0.0
        topo_feat = np.array([cc[i], pc[i]], dtype=np.float32)
        features.append(np.concatenate([fc_feat, stat_feat, np.array([w_fc, b_fc]), topo_feat]))
    return np.stack(features, axis=0).astype(np.float32)

def build_adj(adj_z):
    N = adj_z.shape[0]; adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO); adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]; adj[i, top_idx] = adj_z[i, top_idx]
    return np.maximum(adj, adj.T)

# --- Model Components ---
# 全域 DropEdge rate（由 main() 控制，training 時生效）
_DROP_EDGE_RATE = 0.0

def set_drop_edge_rate(rate: float):
    global _DROP_EDGE_RATE
    _DROP_EDGE_RATE = float(rate)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H = num_heads; self.d = out_dim // num_heads; self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(self.d, 1, bias=False); self.a_dst = nn.Linear(self.d, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_dim); self.dropout = nn.Dropout(dropout)
    def forward(self, h, adj):
        B, N, _ = h.shape; Wh_flat = self.W(h); Wh = Wh_flat.view(B, N, self.H, self.d)
        # DropEdge augmentation (training only)
        if self.training and _DROP_EDGE_RATE > 0:
            mask = (torch.rand_like(adj) > _DROP_EDGE_RATE).float()
            # 對稱遮罩：(i→j) 和 (j→i) 同步保留或丟棄
            mask = ((mask + mask.transpose(-1, -2)) > 0).float()
            adj = adj * mask
        e = F.leaky_relu(self.a_src(Wh).squeeze(-1).unsqueeze(2) + self.a_dst(Wh).squeeze(-1).unsqueeze(1), negative_slope=0.2)
        e = e + adj.unsqueeze(-1) * 0.5; e = e.masked_fill((adj.abs() < 1e-6).unsqueeze(-1), -1e9)
        alpha = self.dropout(F.softmax(e, dim=2))
        alpha_t = alpha.permute(0,3,1,2).reshape(B*self.H, N, N); Wh_t = Wh.permute(0,2,1,3).reshape(B*self.H, N, self.d)
        out = torch.bmm(alpha_t, Wh_t).reshape(B, self.H, N, self.d).permute(0,2,1,3).reshape(B, N, self.out_dim)
        out = self.bn(out.reshape(B*N, -1)).reshape(B, N, -1)
        return F.elu(self.dropout(out)) + Wh_flat

class FMRIEncoder(nn.Module):
    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(0.2))
        self.bn_input = nn.BatchNorm1d(hidden_dim); self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.gat1, self.gat2, self.gat3 = [GATLayer(hidden_dim, hidden_dim) for _ in range(3)]
        self.vn_update = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ELU()); self.register_buffer('pooling_mat', POOLING_MAT)
        self.net_attn = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.net_ln = nn.LayerNorm(N_NETWORKS * hidden_dim + hidden_dim); nn.init.normal_(self.virtual_node_emb, std=0.02)
    def forward(self, x, adj):
        B, N, _ = x.shape; h = self.bn_input(self.node_encoder(x).reshape(B*N,-1)).reshape(B, N, -1)
        vn = self.virtual_node_emb.expand(B,-1,-1) + h.mean(dim=1, keepdim=True)
        h = self.gat1(h, adj); vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1)); h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat2(h, adj) + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1)); h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat3(h, adj) + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        pooled = torch.matmul(h.transpose(1,2), self.pooling_mat).transpose(1,2)
        pooled = pooled * torch.softmax(self.net_attn(pooled), dim=1)
        return self.net_ln(torch.cat([pooled.reshape(B,-1), vn.squeeze(1)], dim=1))

class PCAGFusion(nn.Module):
    def __init__(self, fmri_dim=1280, smri_dim=768, fusion_dim=20, num_classes=2):
        super().__init__()
        self.fmri_proj = nn.Linear(fmri_dim, fusion_dim); self.smri_proj_k = nn.Linear(smri_dim, fusion_dim)
        self.smri_proj_v = nn.Linear(smri_dim, fusion_dim); self.W_e = nn.Linear(fusion_dim, fusion_dim)
        self.W_g1 = nn.Linear(fusion_dim, fusion_dim); self.W_g2 = nn.Linear(fusion_dim, fusion_dim)
        self.ln_e = nn.LayerNorm(fusion_dim); self.ln_g = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim, num_classes))
    def forward(self, fmri_emb, smri_feat):
        Q = self.fmri_proj(fmri_emb); K = self.smri_proj_k(smri_feat); V = self.smri_proj_v(smri_feat)
        P = (torch.tanh(Q) * torch.tanh(K) + 1) / 2; A_gated = (Q * K) * P; S = torch.sigmoid(A_gated); V_hat = S * V
        E = F.relu(self.W_e(V_hat)); G = F.relu(self.W_g1(V_hat) + self.W_g2(Q)); C = self.ln_e(E) * self.ln_g(G)
        return self.classifier(C + F.relu(Q))

class PCAGModel(nn.Module):
    def __init__(self, fusion_dim=20, modality='both'):
        super().__init__()
        self.modality = modality
        self.fmri_encoder = FMRIEncoder()
        self.pcag = PCAGFusion(fusion_dim=fusion_dim)
    def forward(self, x_node, adj, smri_feat):
        fmri_emb = self.fmri_encoder(x_node, adj)
        if self.modality == 'smri_only':
            fmri_emb = torch.zeros_like(fmri_emb)
        elif self.modality == 'fmri_only':
            smri_feat = torch.zeros_like(smri_feat)
        return self.pcag(fmri_emb, smri_feat)

# --- Dataset ---
class PCAGDataset(Dataset):
    def __init__(self, matrix_paths, smri_feats, labels):
        self.paths = matrix_paths; self.smri_feats = smri_feats.astype(np.float32); self.labels = labels; self._cache = {}
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        if path not in self._cache:
            mat = np.load(path); feat = extract_node_features(mat); adj = build_adj(mat); self._cache[path] = (feat, adj)
        feat, adj = self._cache[path]
        return (torch.tensor(feat), torch.tensor(adj), torch.tensor(self.smri_feats[i]), torch.tensor(self.labels[i], dtype=torch.long))

# --- Main Training Script ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="NC_vs_MCI", choices=["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--fusion_dim", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/pcag_combat")
    parser.add_argument("--split_v2", action="store_true", default=True, help="Use v2 splits (more AD in test)")
    parser.add_argument("--modality", default="both", choices=["both", "fmri_only", "smri_only"],
                        help="Modality ablation: both (default) / fmri_only / smri_only")
    parser.add_argument("--fmri_harmonized", action="store_true", default=False,
                        help="使用 fmri_combat_v2/ 內 ComBat 諧波化的 fMRI matrix (F-1 修正)")
    parser.add_argument("--fmri_combat_dir", type=str, default=None,
                        help="覆寫 harmonized fMRI 目錄（搭配 --fmri_harmonized 使用）")
    # --- Data Augmentation (Gemini 推薦：FC-Mixup + DropEdge) ---
    parser.add_argument("--use_mixup", action="store_true", default=False,
                        help="在 training batch 上做 FC-Mixup（混合 fMRI feat + adj + sMRI + labels）")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                        help="Beta(alpha, alpha) for mixup lambda；越小越接近 one-hot")
    parser.add_argument("--mixup_prob", type=float, default=0.5,
                        help="每 batch 進行 mixup 的機率")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0,
                        help="DropEdge rate (GAT layers)，0 = 不啟用")
    parser.add_argument("--no_combat", action="store_true", default=False,
                        help="Skip ComBat harmonization (baseline)")
    parser.add_argument("--k_ratio", type=float, default=None,
                        help="Graph sparsity ratio (default: use module-level K_RATIO=0.20)")
    args = parser.parse_args()
    if args.k_ratio is not None:
        global K_RATIO
        K_RATIO = args.k_ratio
        print(f"[k_ratio override] K_RATIO set to {args.k_ratio} (k≈{int(116*args.k_ratio)})")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream"); RES_DIR = BASE_DIR / "results"
    # experiment tag: encodes split version + modality + combat
    mod_tag    = "" if args.modality == "both" else f"_{args.modality}"
    combat_tag = "_no_combat" if args.no_combat else ""
    if args.fmri_harmonized:
        if args.fmri_combat_dir and "nolabel" in args.fmri_combat_dir:
            fmri_tag = "_fmricombat_nolabel"
        elif args.fmri_combat_dir and args.fmri_combat_dir != "fmri_combat_v2":
            fmri_tag = "_fmricombat_taskspecific"
        else:
            fmri_tag = "_fmricombat"
    else:
        fmri_tag = ""
    # Augmentation suffix
    aug_parts = []
    if args.use_mixup:       aug_parts.append(f"mix{args.mixup_alpha}")
    if args.drop_edge_rate > 0: aug_parts.append(f"de{args.drop_edge_rate}")
    aug_tag = ("_aug_" + "_".join(aug_parts)) if aug_parts else ""
    # Seed suffix（給 ensemble 用，預設 seed=42 不加標）
    seed_tag = f"_s{args.seed}" if args.seed != 42 else ""
    exp_tag    = ("_v2" if args.split_v2 else "") + mod_tag + combat_tag + fmri_tag + aug_tag + seed_tag
    ckpt_dir   = args.ckpt_dir + exp_tag
    CKPT_DIR   = BASE_DIR / ckpt_dir; CKPT_DIR.mkdir(parents=True, exist_ok=True)
    
    TASK_CFG = {
        'NC_vs_AD':  {'classes': [0, 2], 'pos': 2},
        'NC_vs_MCI': {'classes': [0, 1], 'pos': 1},
        'MCI_vs_AD': {'classes': [1, 2], 'pos': 2},
    }
    cfg = TASK_CFG[args.task]

    # 1. Load Alignment Meta with Site info
    sfx = "_v2" if args.split_v2 else ""
    df_tr_meta = pd.read_csv(BASE_DIR / f"kd_train_aligned{sfx}.csv")
    df_te_meta = pd.read_csv(BASE_DIR / f"kd_test_aligned{sfx}.csv")
    df_tr_pcag = pd.read_csv(BASE_DIR / f"pcag_train_aligned{sfx}.csv")
    df_te_pcag = pd.read_csv(BASE_DIR / f"pcag_test_aligned{sfx}.csv")
    
    df_tr = df_tr_pcag.merge(df_tr_meta[['subject_id', 'source']], on='subject_id', how='inner')
    df_te = df_te_pcag.merge(df_te_meta[['subject_id', 'source']], on='subject_id', how='inner')
    df_tr = df_tr[df_tr['label'].isin(cfg['classes'])].reset_index(drop=True)
    df_tr['bin_label'] = (df_tr['label'] == cfg['pos']).astype(int)
    df_te = df_te[df_te['label'].isin(cfg['classes'])].reset_index(drop=True)
    df_te['bin_label'] = (df_te['label'] == cfg['pos']).astype(int)

    # F-1 修正：若指定 --fmri_harmonized，將 matrix_path 改成 ComBat 諧波化版本
    if args.fmri_harmonized:
        fmri_dir = BASE_DIR / (args.fmri_combat_dir or "fmri_combat_v2")
        def _swap_path(sid):
            p = fmri_dir / f"{sid}_combat.npy"
            if not p.exists():
                raise FileNotFoundError(f"Harmonized fMRI not found: {p}")
            return str(p)
        df_tr['matrix_path'] = df_tr['subject_id'].apply(_swap_path)
        df_te['matrix_path'] = df_te['subject_id'].apply(_swap_path)
        print(f"  [fMRI harmonized] using {fmri_dir} for all matrix_path")

    # 2. Load sMRI Features
    smri_cols = [f"Feature_{i}" for i in range(768)]
    smri_tr_all = pd.read_csv(BASE_DIR / f"brainiac_features_train{sfx}.csv")
    smri_te_all = pd.read_csv(BASE_DIR / f"brainiac_features_combined_test{sfx}.csv")
    smri_tr_raw = smri_tr_all.iloc[df_tr['smri_feat_row']][smri_cols].values
    smri_te_raw = smri_te_all.iloc[df_te['smri_feat_row']][smri_cols].values

    # 3. ComBat Harmonization (Fixing Leakage: Fit ONLY on train)
    # Skip ComBat for fmri_only (no sMRI used) and no_combat baselines
    skip_combat = args.no_combat or (args.modality == 'fmri_only')
    print(f"\n📦 ComBat Harmonization [{'' if not skip_combat else 'SKIPPED — '}modality={args.modality}, no_combat={args.no_combat}]")
    
    if skip_combat:
        smri_tr, smri_te = smri_tr_raw.copy(), smri_te_raw.copy()
        estimates = None
        print(f"  ComBat skipped. Raw sMRI: Train Mean={smri_tr.mean():.4f}, Test Mean={smri_te.mean():.4f}")
    else:
        # 3-1. Prepare site mapping
        all_sites = sorted(list(set(df_tr['source'].unique()) | set(df_te['source'].unique())))
        site_map = {s: i for i, s in enumerate(all_sites)}
        df_tr['site_idx'] = df_tr['source'].map(site_map)
        df_te['site_idx'] = df_te['source'].map(site_map)

        # 3-2. Fit ComBat on Train ONLY
        print(f"  Before ComBat (Train): Mean={smri_tr_raw.mean():.4f}, Std={smri_tr_raw.std():.4f}")
        combat_out = neuroCombat(
            dat=smri_tr_raw.T,
            covars=df_tr[['site_idx', 'bin_label']],
            batch_col='site_idx',
            categorical_cols=['bin_label']
        )
        smri_tr = combat_out['data'].T
        estimates = combat_out['estimates']
        print(f"  After ComBat (Train):  Mean={smri_tr.mean():.4f}, Std={smri_tr.std():.4f}")

        # 3-3. Apply ComBat to Test (leak-free)
        def apply_combat_local(dat, site_indices, estimates):
            s_mean = estimates['stand.mean']
            g_mean = s_mean.mean(axis=1) if (s_mean.ndim == 2 and s_mean.shape[1] > 1) else s_mean.flatten()
            v_pooled = estimates['var.pooled'].flatten()
            gamma = estimates['gamma.star'][site_indices]
            delta = estimates['delta.star'][site_indices]
            dat_std = (dat - g_mean) / (np.sqrt(v_pooled) + 1e-8)
            dat_adj = (dat_std - gamma) / (np.sqrt(delta) + 1e-8)
            return dat_adj * np.sqrt(v_pooled) + g_mean

        smri_te = apply_combat_local(smri_te_raw, df_te['site_idx'].values, estimates)
        print(f"  After ComBat (Test):   Mean={smri_te.mean():.4f}, Std={smri_te.std():.4f}")

    print(f"Task: {args.task} | Train: {len(df_tr)} | Test: {len(df_te)}")

    # Save ComBat parameters (only when ComBat is used)
    if not skip_combat:
        combat_params = {
            'site_map':   site_map,
            'stand_mean': estimates['stand.mean'].tolist(),
            'var_pooled': estimates['var.pooled'].flatten().tolist(),
            'gamma_star': estimates['gamma.star'].tolist(),
            'delta_star': estimates['delta.star'].tolist(),
        }
        combat_path = CKPT_DIR / f"combat_params_{args.task}.json"
        with open(combat_path, 'w') as f: json.dump(combat_params, f)
        print(f"[SAVED] ComBat params → {combat_path}")

    # 4. 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros(len(df_tr)); test_probs_folds = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(df_tr, df_tr['bin_label'])):
        print(f"\n--- Fold {fold+1}/5 ---")
        tr_paths, tr_smri, tr_labels = df_tr.iloc[tr_idx]['matrix_path'].tolist(), smri_tr[tr_idx], df_tr.iloc[tr_idx]['bin_label'].values
        val_paths, val_smri, val_labels = df_tr.iloc[val_idx]['matrix_path'].tolist(), smri_tr[val_idx], df_tr.iloc[val_idx]['bin_label'].values
        
        counts = np.bincount(tr_labels)
        weights = 1.0 / counts
        sampler = WeightedRandomSampler(weights[tr_labels], len(tr_labels))
        
        tr_loader = DataLoader(PCAGDataset(tr_paths, tr_smri, tr_labels), batch_size=args.batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(PCAGDataset(val_paths, val_smri, val_labels), batch_size=args.batch_size, shuffle=False)
        te_loader = DataLoader(PCAGDataset(df_te['matrix_path'].tolist(), smri_te, df_te['bin_label'].values), batch_size=args.batch_size, shuffle=False)
        
        model = PCAGModel(fusion_dim=args.fusion_dim, modality=args.modality).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
        criterion = nn.CrossEntropyLoss()
        
        # 啟用 DropEdge（global flag）
        set_drop_edge_rate(args.drop_edge_rate)
        if args.use_mixup or args.drop_edge_rate > 0:
            print(f"  [Augmentation] use_mixup={args.use_mixup} (alpha={args.mixup_alpha}, p={args.mixup_prob}), drop_edge={args.drop_edge_rate}")

        best_auc, patience, patience_counter, best_state = 0, 40, 0, None
        for epoch in range(args.epochs):
            model.train()
            for x, adj, smri, y in tr_loader:
                x, adj, smri, y = x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                # FC-Mixup augmentation (training only)
                if args.use_mixup and np.random.rand() < args.mixup_prob and x.size(0) > 1:
                    lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                    perm = torch.randperm(x.size(0), device=x.device)
                    x_mix    = lam * x    + (1 - lam) * x[perm]
                    adj_mix  = lam * adj  + (1 - lam) * adj[perm]
                    smri_mix = lam * smri + (1 - lam) * smri[perm]
                    y_b = y[perm]
                    out = model(x_mix, adj_mix, smri_mix)
                    loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y_b)
                else:
                    out = model(x, adj, smri)
                    loss = criterion(out, y)
                loss.backward(); optimizer.step()
            
            model.eval(); val_p = []
            with torch.no_grad():
                for x, adj, smri, y in val_loader:
                    out = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE)); val_p.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            
            auc = roc_auc_score(val_labels, val_p) if len(set(val_labels)) > 1 else 0.5
            if auc > best_auc: 
                best_auc = auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience: break
        
        print(f"Best Val AUC: {best_auc:.4f}")
        # Save fold checkpoint
        fold_ckpt = CKPT_DIR / f"pcag_combat_{args.task}_fold{fold}.pt"
        torch.save({'model_state': best_state, 'fold': fold, 'task': args.task,
                    'fusion_dim': args.fusion_dim, 'val_auc': best_auc}, fold_ckpt)
        print(f"[SAVED] {fold_ckpt}")
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            fold_p = []; te_p = []
            for x, adj, smri, y in val_loader:
                out = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE)); fold_p.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            oof_probs[val_idx] = fold_p
            for x, adj, smri, y in te_loader:
                out = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE)); te_p.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            test_probs_folds.append(te_p)

    # 5. Final Evaluation
    oof_auc = roc_auc_score(df_tr['bin_label'], oof_probs)
    print(f"\nFinal OOF AUC: {oof_auc:.4f}")
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(df_tr['bin_label'], oof_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    
    avg_test_p = np.mean(test_probs_folds, axis=0)
    test_auc = roc_auc_score(df_te['bin_label'], avg_test_p)
    test_pred = (avg_test_p >= best_thresh).astype(int)
    cm = confusion_matrix(df_te['bin_label'], test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "auc": float(test_auc), "threshold": float(best_thresh),
        "sens": float(tp / (tp + fn)) if (tp+fn)>0 else 0,
        "spec": float(tn / (tn + fp)) if (tn+fp)>0 else 0,
        "f1": float(f1_score(df_te['bin_label'], test_pred)),
        "acc": float(accuracy_score(df_te['bin_label'], test_pred)),
        "cm": cm.tolist()
    }
    print(f"\nCombat-Harmonized PCAG Test Metrics ({args.task}):")
    for k, v in metrics.items(): print(f"  {k}: {v}")
    
    res_suffix = exp_tag
    out_file = RES_DIR / f"pcag_combat_{args.task}_results{res_suffix}.json"
    with open(out_file, "w") as f: json.dump({"oof_auc": oof_auc, "test_metrics": metrics}, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Save probs for hybrid system
    prob_path = RES_DIR / f"pcag_combat_{args.task}_probs{res_suffix}.npz"
    # Also save modality tag in checkpoint name
    # (fold checkpoint already uses CKPT_DIR which encodes exp_tag)
    np.savez(str(prob_path),
             test_probs=avg_test_p,
             test_labels=df_te['bin_label'].values,
             test_subject_ids=df_te['subject_id'].values,
             oof_probs=oof_probs,
             oof_labels=df_tr['bin_label'].values)
    print(f"[SAVED] {prob_path}")

if __name__ == "__main__":
    main()
