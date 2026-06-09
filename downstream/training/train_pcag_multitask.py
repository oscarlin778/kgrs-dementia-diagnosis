"""
Phase-3: Multi-task Joint Training (NC_vs_AD + NC_vs_MCI + MCI_vs_AD)
Shared fMRI GAT encoder + 3 task-specific PCAG fusion heads.

Usage:
  python train_pcag_multitask.py [--epochs 200] [--seed 42] [--loss_mciad 1.5]

After training, saves:
  checkpoints/pcag_multitask_v1/  (full multi-task checkpoint per fold)
  checkpoints/pcag_multitask_v1_NC_vs_AD/  (task-specific, compatible with inference pipeline)
  checkpoints/pcag_multitask_v1_NC_vs_MCI/
  checkpoints/pcag_multitask_v1_MCI_vs_AD/
"""

import os, sys, argparse, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
from neuroCombat import neuroCombat

# ─── Constants (identical to train_pcag_combat_fusion.py) ───────────────────
NETWORK_MAP = {
    'DMN':   [34,35,66,67,64,65,22,23,24,25],
    'SMN':   [0,1,56,57,68,69],
    'VN':    [42,43,44,45,46,47,48,49,50,51,52,53],
    'SN':    [28,29,30,31,32,33],
    'FPN':   [6,7,58,59,60,61],
    'LN':    [36,37,38,39,40,41],
    'VAN':   [10,11,14,15],
    'BGN':   [70,71,72,73,74,75,76,77],
    'CereN': list(range(90,116)),
}
roi_to_net = {}
net_list = list(NETWORK_MAP.keys())
for i, (net, nodes) in enumerate(NETWORK_MAP.items()):
    for node in nodes: roi_to_net[node] = i

N_ROIS = 116
POOLING_MAT = torch.zeros(N_ROIS, len(net_list))
for i, net in enumerate(net_list):
    for node_idx in NETWORK_MAP[net]: POOLING_MAT[node_idx, i] = 1.0

HIDDEN_DIM      = 128
DROPOUT         = 0.4
K_RATIO         = 0.20
NODE_FEAT_DIM   = 116 + 5 + 2 + 2   # 125
N_NETWORKS      = 9
FMRI_EMBED_DIM  = N_NETWORKS * HIDDEN_DIM + HIDDEN_DIM  # 1280
_DROP_EDGE_RATE = 0.0

TASK_CFG = {
    'NC_vs_AD':  {'classes': [0, 2], 'pos': 2},
    'NC_vs_MCI': {'classes': [0, 1], 'pos': 1},
    'MCI_vs_AD': {'classes': [1, 2], 'pos': 2},
}

# ─── Feature Helpers ─────────────────────────────────────────────────────────
def extract_node_features(adj_z):
    N = adj_z.shape[0]; adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO); adj_bin = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]; adj_bin[i, top_idx] = 1.0
    adj_bin = np.maximum(adj_bin, adj_bin.T); degree = adj_bin.sum(axis=1)
    cc = np.diag(adj_bin @ adj_bin @ adj_bin) / (degree*(degree-1)+1e-8)
    adj_abs_thresh = adj_abs * adj_bin; pc = np.zeros(N, dtype=np.float32)
    for i in range(N):
        ki = adj_abs_thresh[i].sum()
        if ki > 1e-8:
            pc_i = 1.0
            for nv in NETWORK_MAP.values():
                kim = adj_abs_thresh[i, list(nv)].sum(); pc_i -= (kim/ki)**2
            pc[i] = float(np.clip(pc_i, 0.0, 1.0))
    features = []
    for i in range(N):
        row = adj_z[i].copy(); row[i] = 0
        stat_feat = np.array([row.mean(), row.std(), (row>0).mean(), (row<0).mean(), (np.abs(row)>0.1).sum()], dtype=np.float32)
        ni = roi_to_net.get(i, -1)
        if ni >= 0:
            wn = [r for r in NETWORK_MAP[net_list[ni]] if r != i]
            bn = [r for r in range(N) if r != i and roi_to_net.get(r,-1) != ni]
            w_fc = float(np.mean([row[r] for r in wn])) if wn else 0.0
            b_fc = float(np.mean([row[r] for r in bn])) if bn else 0.0
        else: w_fc, b_fc = 0.0, 0.0
        features.append(np.concatenate([row.astype(np.float32), stat_feat, np.array([w_fc, b_fc]), np.array([cc[i], pc[i]], dtype=np.float32)]))
    return np.stack(features, axis=0).astype(np.float32)

def build_adj(adj_z):
    N = adj_z.shape[0]; adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO); adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]; adj[i, top_idx] = adj_z[i, top_idx]
    return np.maximum(adj, adj.T)

# ─── Model Components ────────────────────────────────────────────────────────
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
        if self.training and _DROP_EDGE_RATE > 0:
            mask = (torch.rand_like(adj) > _DROP_EDGE_RATE).float()
            mask = ((mask + mask.transpose(-1,-2)) > 0).float(); adj = adj * mask
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
        self.vn_update = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ELU())
        self.register_buffer('pooling_mat', POOLING_MAT)
        self.net_attn = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.net_ln = nn.LayerNorm(N_NETWORKS*hidden_dim + hidden_dim)
        nn.init.normal_(self.virtual_node_emb, std=0.02)
    def forward(self, x, adj):
        B, N, _ = x.shape; h = self.bn_input(self.node_encoder(x).reshape(B*N,-1)).reshape(B, N, -1)
        vn = self.virtual_node_emb.expand(B,-1,-1) + h.mean(dim=1, keepdim=True)
        h = self.gat1(h, adj); vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)],-1)); h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat2(h, adj) + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)],-1)); h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat3(h, adj) + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)],-1))
        pooled = torch.matmul(h.transpose(1,2), self.pooling_mat).transpose(1,2)
        pooled = pooled * torch.softmax(self.net_attn(pooled), dim=1)
        return self.net_ln(torch.cat([pooled.reshape(B,-1), vn.squeeze(1)], dim=1))

class PCAGFusion(nn.Module):
    def __init__(self, fmri_dim=1280, smri_dim=768, fusion_dim=20, num_classes=2):
        super().__init__()
        self.fmri_proj   = nn.Linear(fmri_dim, fusion_dim)
        self.smri_proj_k = nn.Linear(smri_dim, fusion_dim)
        self.smri_proj_v = nn.Linear(smri_dim, fusion_dim)
        self.W_e  = nn.Linear(fusion_dim, fusion_dim); self.W_g1 = nn.Linear(fusion_dim, fusion_dim)
        self.W_g2 = nn.Linear(fusion_dim, fusion_dim)
        self.ln_e = nn.LayerNorm(fusion_dim); self.ln_g = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim, num_classes))
    def forward(self, fmri_emb, smri_feat):
        Q = self.fmri_proj(fmri_emb); K = self.smri_proj_k(smri_feat); V = self.smri_proj_v(smri_feat)
        P = (torch.tanh(Q)*torch.tanh(K)+1)/2; S = torch.sigmoid((Q*K)*P); V_hat = S * V
        E = F.relu(self.W_e(V_hat)); G = F.relu(self.W_g1(V_hat)+self.W_g2(Q)); C = self.ln_e(E)*self.ln_g(G)
        return self.classifier(C + F.relu(Q))

class PCAGMultiTaskModel(nn.Module):
    """Shared fMRI encoder + 3 independent PCAG fusion heads, one per OVO task."""
    def __init__(self, fusion_dim=20):
        super().__init__()
        self.shared_encoder = FMRIEncoder()
        self.head_nc_ad  = PCAGFusion(fusion_dim=fusion_dim)
        self.head_nc_mci = PCAGFusion(fusion_dim=fusion_dim)
        self.head_mci_ad = PCAGFusion(fusion_dim=fusion_dim)
        self._heads = [self.head_nc_ad, self.head_nc_mci, self.head_mci_ad]
        self._task_names = ['NC_vs_AD', 'NC_vs_MCI', 'MCI_vs_AD']

    def encode(self, x_node, adj):
        return self.shared_encoder(x_node, adj)

    def predict_task(self, fmri_emb, smri_feat, task_id: int):
        return self._heads[task_id](fmri_emb, smri_feat)

    def forward(self, x_node, adj, smri_feat, task_id: int):
        return self.predict_task(self.encode(x_node, adj), smri_feat, task_id)

    def export_single_task(self, task_id: int) -> dict:
        """Return state dict compatible with PCAGModel (fmri_encoder + pcag)."""
        head = self._heads[task_id]
        state = {}
        for k, v in self.shared_encoder.state_dict().items():
            state[f'fmri_encoder.{k}'] = v
        for k, v in head.state_dict().items():
            state[f'pcag.{k}'] = v
        return state

# ─── Dataset ─────────────────────────────────────────────────────────────────
class PCAGDataset(Dataset):
    def __init__(self, matrix_paths, smri_feats, labels):
        self.paths = matrix_paths; self.smri_feats = smri_feats.astype(np.float32)
        self.labels = labels; self._cache = {}
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        if path not in self._cache:
            mat = np.load(path); self._cache[path] = (extract_node_features(mat), build_adj(mat))
        feat, adj = self._cache[path]
        return (torch.tensor(feat), torch.tensor(adj),
                torch.tensor(self.smri_feats[i]), torch.tensor(self.labels[i], dtype=torch.long))

# ─── ComBat Helpers ───────────────────────────────────────────────────────────
def fit_apply_combat(smri_tr_raw, smri_te_raw, df_tr_task, df_te_task):
    all_sites = sorted(list(set(df_tr_task['source'].unique()) | set(df_te_task['source'].unique())))
    site_map  = {s: i for i, s in enumerate(all_sites)}
    df_tr_task = df_tr_task.copy(); df_te_task = df_te_task.copy()
    df_tr_task['site_idx'] = df_tr_task['source'].map(site_map)
    df_te_task['site_idx']  = df_te_task['source'].map(site_map)
    out = neuroCombat(dat=smri_tr_raw.T,
                      covars=df_tr_task[['site_idx', 'bin_label']],
                      batch_col='site_idx', categorical_cols=['bin_label'])
    smri_tr = out['data'].T
    est = out['estimates']
    s_mean  = est['stand.mean']
    g_mean  = s_mean.mean(axis=1) if (s_mean.ndim == 2 and s_mean.shape[1] > 1) else s_mean.flatten()
    v_pool  = est['var.pooled'].flatten()
    gamma   = est['gamma.star'][df_te_task['site_idx'].values]
    delta   = est['delta.star'][df_te_task['site_idx'].values]
    dat_std = (smri_te_raw - g_mean) / (np.sqrt(v_pool) + 1e-8)
    smri_te = (dat_std - gamma) / (np.sqrt(delta) + 1e-8) * np.sqrt(v_pool) + g_mean
    return smri_tr, smri_te, est, site_map

# ─── Per-task dataset builder ─────────────────────────────────────────────────
def build_task_datasets(task, tr_idx, val_idx, df_tr_all, df_te_all,
                        smri_tr_all, smri_te_all, fmri_dir):
    cfg = TASK_CFG[task]
    # Filter by task classes (using GLOBAL indices into df_tr_all)
    tr_mask  = df_tr_all.iloc[tr_idx]['label'].isin(cfg['classes'])
    val_mask = df_tr_all.iloc[val_idx]['label'].isin(cfg['classes'])
    te_mask  = df_te_all['label'].isin(cfg['classes'])

    # Absolute row indices in df_tr_all
    tr_rows  = np.array(tr_idx)[tr_mask.values]
    val_rows = np.array(val_idx)[val_mask.values]
    df_tr_task  = df_tr_all.iloc[tr_rows].copy().reset_index(drop=True)
    df_val_task = df_tr_all.iloc[val_rows].copy().reset_index(drop=True)
    df_te_task  = df_te_all[te_mask].copy().reset_index(drop=True)

    for df_ in [df_tr_task, df_val_task, df_te_task]:
        df_['bin_label'] = (df_['label'] == cfg['pos']).astype(int)

    # Reroute to harmonized fMRI
    def swap(sid): return str(fmri_dir / f"{sid}_combat.npy")
    df_tr_task['matrix_path']  = df_tr_task['subject_id'].apply(swap)
    df_val_task['matrix_path'] = df_val_task['subject_id'].apply(swap)
    df_te_task['matrix_path']  = df_te_task['subject_id'].apply(swap)

    # ComBat per task
    smri_tr_raw  = smri_tr_all[tr_rows]
    smri_val_raw = smri_tr_all[val_rows]
    smri_te_idx  = df_te_all[te_mask].index.tolist()
    smri_te_raw  = smri_te_all[smri_te_idx]

    smri_cols = [f"Feature_{i}" for i in range(768)]
    smri_tr_raw  = smri_tr_raw  if isinstance(smri_tr_raw, np.ndarray) else smri_tr_raw.values
    smri_te_raw  = smri_te_raw  if isinstance(smri_te_raw, np.ndarray) else smri_te_raw.values
    smri_val_raw = smri_val_raw if isinstance(smri_val_raw, np.ndarray) else smri_val_raw.values

    # Fit ComBat on TRAIN for this task, apply to val + test
    smri_tr_h, smri_te_h, est, site_map = fit_apply_combat(
        smri_tr_raw, smri_te_raw, df_tr_task, df_te_task)
    # Apply same transform to val
    s_mean = est['stand.mean']
    g_mean = s_mean.mean(axis=1) if (s_mean.ndim == 2 and s_mean.shape[1] > 1) else s_mean.flatten()
    v_pool = est['var.pooled'].flatten()
    df_val_task['site_idx'] = df_val_task['source'].map(site_map).fillna(0).astype(int)
    gamma_v = est['gamma.star'][df_val_task['site_idx'].values]
    delta_v = est['delta.star'][df_val_task['site_idx'].values]
    dat_std_v = (smri_val_raw - g_mean) / (np.sqrt(v_pool) + 1e-8)
    smri_val_h = (dat_std_v - gamma_v) / (np.sqrt(delta_v) + 1e-8) * np.sqrt(v_pool) + g_mean

    tr_labels  = df_tr_task['bin_label'].values
    val_labels = df_val_task['bin_label'].values
    te_labels  = df_te_task['bin_label'].values

    tr_ds  = PCAGDataset(df_tr_task['matrix_path'].tolist(),  smri_tr_h,  tr_labels)
    val_ds = PCAGDataset(df_val_task['matrix_path'].tolist(), smri_val_h, val_labels)
    te_ds  = PCAGDataset(df_te_task['matrix_path'].tolist(),  smri_te_h,  te_labels)
    return tr_ds, val_ds, te_ds, val_labels, te_labels

# ─── Evaluation Helper ────────────────────────────────────────────────────────
def eval_auc(model, loader, task_id, true_labels, DEVICE):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, adj, smri, _ in loader:
            logits = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE), task_id)
            probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
    if len(set(true_labels)) < 2: return 0.5
    return float(roc_auc_score(true_labels, probs)), np.array(probs)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--fusion_dim',  type=int,   default=20)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--patience',    type=int,   default=40)
    # Task loss weights
    parser.add_argument('--w_ncad',  type=float, default=1.0,
                        help='Loss weight for NC_vs_AD')
    parser.add_argument('--w_ncmci', type=float, default=0.5,
                        help='Loss weight for NC_vs_MCI (easier task)')
    parser.add_argument('--w_mciad', type=float, default=1.5,
                        help='Loss weight for MCI_vs_AD (hardest task)')
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}  |  seed={args.seed}  |  epochs={args.epochs}')
    print(f'Loss weights — NC_vs_AD:{args.w_ncad}  NC_vs_MCI:{args.w_ncmci}  MCI_vs_AD:{args.w_mciad}')

    BASE_DIR  = Path('/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream')
    FMRI_DIR  = BASE_DIR / 'fmri_combat_v2_nolabel'
    CKPT_BASE = BASE_DIR / 'checkpoints'

    seed_tag = f'_s{args.seed}' if args.seed != 42 else ''
    MT_CKPT  = CKPT_BASE / f'pcag_multitask_v1{seed_tag}'
    MT_CKPT.mkdir(parents=True, exist_ok=True)
    TASK_CKPT = {t: CKPT_BASE / f'pcag_multitask_v1_{t}{seed_tag}' for t in TASK_CFG}
    for d in TASK_CKPT.values(): d.mkdir(parents=True, exist_ok=True)

    # Load data
    smri_cols = [f'Feature_{i}' for i in range(768)]
    df_tr_all  = pd.read_csv(BASE_DIR / 'pcag_train_aligned_v2.csv')
    df_te_all  = pd.read_csv(BASE_DIR / 'pcag_test_aligned_v2.csv')
    df_tr_meta = pd.read_csv(BASE_DIR / 'kd_train_aligned_v2.csv')
    df_te_meta = pd.read_csv(BASE_DIR / 'kd_test_aligned_v2.csv')
    df_tr_all  = df_tr_all.merge(df_tr_meta[['subject_id','source']], on='subject_id', how='inner')
    df_te_all  = df_te_all.merge(df_te_meta[['subject_id','source']], on='subject_id', how='inner')

    smri_tr_raw_full = pd.read_csv(BASE_DIR / 'brainiac_features_train_v2.csv')[smri_cols].values
    smri_te_raw_full = pd.read_csv(BASE_DIR / 'brainiac_features_combined_test_v2.csv')[smri_cols].values
    # Align to df_tr_all row ordering via smri_feat_row
    smri_tr_all = smri_tr_raw_full[df_tr_all['smri_feat_row'].values]
    smri_te_all = smri_te_raw_full[df_te_all['smri_feat_row'].values]

    print(f'\nAll subjects — Train: {len(df_tr_all)}  Test: {len(df_te_all)}')
    print(f'Label dist (train): {dict(df_tr_all["label"].value_counts().sort_index())}')

    # 5-fold CV (stratified on 3-class label → consistent split across all tasks)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    TASKS = ['NC_vs_AD', 'NC_vs_MCI', 'MCI_vs_AD']
    TASK_IDS = {t: i for i, t in enumerate(TASKS)}
    WEIGHTS = [args.w_ncad, args.w_ncmci, args.w_mciad]

    oof_probs  = {t: np.zeros(len(df_tr_all)) for t in TASKS}
    oof_masks  = {t: np.zeros(len(df_tr_all), dtype=bool) for t in TASKS}
    test_probs_folds = {t: [] for t in TASKS}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(df_tr_all, df_tr_all['label'])):
        print(f'\n{"="*60}')
        print(f'FOLD {fold+1}/5')
        print(f'{"="*60}')

        # Build per-task datasets
        task_datasets = {}
        val_labels_all = {}; te_labels_all = {}
        for task in TASKS:
            tr_ds, val_ds, te_ds, val_lbl, te_lbl = build_task_datasets(
                task, tr_idx, val_idx, df_tr_all, df_te_all,
                smri_tr_all, smri_te_all, FMRI_DIR)
            task_datasets[task] = (tr_ds, val_ds, te_ds)
            val_labels_all[task] = val_lbl
            te_labels_all[task]  = te_lbl
            print(f'  {task}: train={len(tr_ds)}, val={len(val_ds)}, test={len(te_ds)}  pos_rate={val_lbl.mean():.2f}')

        # Build loaders
        def make_loader(ds, shuffle=False, sampler=None):
            return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=2, pin_memory=True)

        tr_loaders  = {}; val_loaders = {}; te_loaders = {}
        for task in TASKS:
            tr_ds, val_ds, te_ds = task_datasets[task]
            labels = np.array([tr_ds[i][3].item() for i in range(len(tr_ds))])
            counts = np.bincount(labels)
            samp_w = 1.0 / counts
            sampler = WeightedRandomSampler(samp_w[labels], len(labels))
            tr_loaders[task]  = make_loader(tr_ds, sampler=sampler)
            val_loaders[task] = make_loader(val_ds)
            te_loaders[task]  = make_loader(te_ds)

        # Model + optimizer
        model = PCAGMultiTaskModel(fusion_dim=args.fusion_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
        criterion = nn.CrossEntropyLoss()

        best_avg_auc = 0.0
        patience_counter = 0
        best_state = None

        # Training loop
        for epoch in range(args.epochs):
            model.train()

            # Round-robin: iterate over all 3 task loaders simultaneously
            iters = {t: iter(tr_loaders[t]) for t in TASKS}
            steps = max(len(tr_loaders[t]) for t in TASKS)

            for _ in range(steps):
                # Collect one batch from each available task
                batches = {}
                for task in TASKS:
                    try: batches[task] = next(iters[task])
                    except StopIteration:
                        iters[task] = iter(tr_loaders[task])
                        batches[task] = next(iters[task])

                # Heterogeneous forward: concatenate all tasks, shared encoder once
                all_x = []; all_adj = []; task_sizes = []
                smri_per_task = {}; y_per_task = {}
                for task in TASKS:
                    x, adj, smri, y = batches[task]
                    all_x.append(x); all_adj.append(adj)
                    task_sizes.append(len(y))
                    smri_per_task[task] = smri.to(DEVICE)
                    y_per_task[task]    = y.to(DEVICE)

                x_cat   = torch.cat(all_x).to(DEVICE)
                adj_cat = torch.cat(all_adj).to(DEVICE)

                optimizer.zero_grad()
                fmri_emb_cat = model.shared_encoder(x_cat, adj_cat)
                embs = torch.split(fmri_emb_cat, task_sizes)

                total_loss = torch.tensor(0.0, device=DEVICE)
                for task, emb, w in zip(TASKS, embs, WEIGHTS):
                    logits = model.predict_task(emb, smri_per_task[task], TASK_IDS[task])
                    total_loss = total_loss + w * criterion(logits, y_per_task[task])
                total_loss.backward()
                optimizer.step()

            # Validation
            aucs = {}
            for task in TASKS:
                auc, _ = eval_auc(model, val_loaders[task], TASK_IDS[task],
                                  val_labels_all[task], DEVICE)
                aucs[task] = auc

            # Weighted average AUC (same weights as loss)
            w_sum = sum(WEIGHTS)
            avg_auc = sum(WEIGHTS[i]*aucs[t] for i, t in enumerate(TASKS)) / w_sum

            if (epoch + 1) % 20 == 0:
                print(f'  Epoch {epoch+1:3d} | NC_vs_AD={aucs["NC_vs_AD"]:.3f}  '
                      f'NC_vs_MCI={aucs["NC_vs_MCI"]:.3f}  MCI_vs_AD={aucs["MCI_vs_AD"]:.3f}  '
                      f'avg={avg_auc:.3f}')

            if avg_auc > best_avg_auc:
                best_avg_auc  = avg_auc
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_val_aucs = aucs.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience: break

        print(f'\n  Best val AUCs — NC_vs_AD={best_val_aucs["NC_vs_AD"]:.4f}  '
              f'NC_vs_MCI={best_val_aucs["NC_vs_MCI"]:.4f}  MCI_vs_AD={best_val_aucs["MCI_vs_AD"]:.4f}')

        # Save full multi-task checkpoint
        mt_ckpt_path = MT_CKPT / f'multitask_fold{fold}.pt'
        torch.save({
            'model_state':  best_state,
            'fold':         fold,
            'val_aucs':     best_val_aucs,
            'fusion_dim':   args.fusion_dim,
            'model_type':   'multitask_v1',
        }, mt_ckpt_path)
        print(f'  [SAVED] {mt_ckpt_path}')

        # Export task-specific checkpoints (compatible with inference_pipeline_v2.py)
        model.load_state_dict(best_state)
        for task in TASKS:
            single_state = model.export_single_task(TASK_IDS[task])
            single_ckpt  = TASK_CKPT[task] / f'pcag_combat_{task}_fold{fold}.pt'
            torch.save({
                'model_state':  single_state,
                'fold':         fold,
                'task':         task,
                'fusion_dim':   args.fusion_dim,
                'val_auc':      best_val_aucs[task],
                'model_version': 'multitask_v1',
            }, single_ckpt)
            print(f'  [SAVED] {single_ckpt}')

        # OOF + test predictions
        model.load_state_dict(best_state); model.eval()
        for task in TASKS:
            task_id = TASK_IDS[task]
            # val predictions (mapped back to original df_tr_all indices)
            cfg = TASK_CFG[task]
            tr_all_mask  = df_tr_all['label'].isin(cfg['classes'])
            val_task_idx = [i for i in val_idx if df_tr_all.iloc[i]['label'] in cfg['classes']]

            _, val_p = eval_auc(model, val_loaders[task], task_id, val_labels_all[task], DEVICE)
            for orig_idx, prob in zip(val_task_idx, val_p):
                oof_probs[task][orig_idx] = prob
                oof_masks[task][orig_idx] = True

            _, te_p = eval_auc(model, te_loaders[task], task_id, te_labels_all[task], DEVICE)
            test_probs_folds[task].append(te_p)

    # ─── Final Evaluation ────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('MULTI-TASK FINAL RESULTS')
    print('='*60)
    results = {}
    for task in TASKS:
        cfg = TASK_CFG[task]
        te_mask  = df_te_all['label'].isin(cfg['classes'])
        te_labels = (df_te_all[te_mask]['label'] == cfg['pos']).astype(int).values
        avg_te_p = np.mean(test_probs_folds[task], axis=0)
        test_auc = roc_auc_score(te_labels, avg_te_p)

        tr_mask  = df_tr_all['label'].isin(cfg['classes'])
        oof_valid = oof_masks[task]
        oof_auc = roc_auc_score(
            (df_tr_all['label'] == cfg['pos']).astype(int).values[oof_valid],
            oof_probs[task][oof_valid])

        print(f'  {task}: OOF AUC={oof_auc:.4f}  Test AUC={test_auc:.4f}')
        results[task] = {'oof_auc': float(oof_auc), 'test_auc': float(test_auc)}

    # Baseline comparison
    print('\nBaseline comparison (single-task 5-seed ensemble):')
    print('  NC_vs_AD:  0.791  →', f'{results["NC_vs_AD"]["test_auc"]:.4f}')
    print('  NC_vs_MCI: 0.686  →', f'{results["NC_vs_MCI"]["test_auc"]:.4f}')
    print('  MCI_vs_AD: 0.672  →', f'{results["MCI_vs_AD"]["test_auc"]:.4f}')

    out_file = BASE_DIR / 'results' / f'pcag_multitask_v1{seed_tag}_results.json'
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump({'seed': args.seed, 'weights': WEIGHTS, 'tasks': results}, f, indent=2)
    print(f'\n[SAVED] {out_file}')

if __name__ == '__main__':
    main()
