import os
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/wei-chi/Data/script')
import save_experiment_results as ser

# ===============================================================
# Settings & Hyperparameters
# ===============================================================
CSV_PATHS = [
    "/home/wei-chi/Model/_dataset_mapping.csv",
    "/home/wei-chi/Data/dataset_index_116_clean_old.csv",
    "/home/wei-chi/Data/adni_dataset_index_116.csv"
]
MATRIX_DIR = "/home/wei-chi/Model/processed_116_matrices"
TEACHER_PROBS_DIR = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints"
UNIFIED_SPLIT_PATH = "/home/wei-chi/Data/script/unified_subject_split.json"

HIDDEN_DIM      = 128
DROPOUT         = 0.4
LR              = 3e-4
WEIGHT_DECAY    = 5e-3
EPOCHS          = 200
BATCH_SIZE      = 16
SEED            = 42
K_RATIO         = 0.20
PATIENCE        = 40

LAMBDA_CE       = 1.0
LAMBDA_KD       = 0.5
LAMBDA_CONTRA   = 0.0
CONTRA_TEMP     = 0.5

# MTL Weights (NC_vs_MCI is the focus)
ALPHA, BETA, GAMMA = 0.5, 1.0, 0.5 
LAMBDA_DOMAIN = 0.3

SEEDS = [42, 123, 456]

# ===============================================================
# 1. Network map & node features (Unchanged)
# ===============================================================
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
N_NETWORKS = len(NETWORK_MAP)
POOLING_MAT = torch.zeros(116, N_NETWORKS)
for i, net in enumerate(NETWORK_MAP):
    for node_idx in NETWORK_MAP[net]: POOLING_MAT[node_idx, i] = 1.0

def extract_node_features(adj_z: np.ndarray) -> np.ndarray:
    N = adj_z.shape[0]; net_list = list(NETWORK_MAP.keys())
    roi_to_net = {roi: i for i, net in enumerate(net_list) for roi in NETWORK_MAP[net]}
    adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0); k = int(N * K_RATIO)
    adj_bin = np.zeros((N, N), dtype=np.float32)
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
        features.append(np.concatenate([fc_feat, stat_feat, np.array([w_fc, b_fc], dtype=np.float32), topo_feat]))
    return np.stack(features, axis=0).astype(np.float32)

NODE_FEAT_DIM = 116 + 5 + 2 + 2

# ===============================================================
# 2. E8 Decoupled MTL Model Architecture
# ===============================================================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H, self.d, self.out_dim = num_heads, out_dim // num_heads, out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(self.d, 1, bias=False); self.a_dst = nn.Linear(self.d, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_dim); self.dropout = nn.Dropout(dropout)
    def forward(self, h, adj):
        B, N, _ = h.shape; Wh_flat = self.W(h); Wh = Wh_flat.view(B, N, self.H, self.d)
        e = F.leaky_relu(self.a_src(Wh).squeeze(-1).unsqueeze(2) + self.a_dst(Wh).squeeze(-1).unsqueeze(1), negative_slope=0.2)
        e = e + adj.unsqueeze(-1) * 0.5; e = e.masked_fill((adj.abs() < 1e-6).unsqueeze(-1), -1e9)
        alpha = self.dropout(F.softmax(e, dim=2))
        alpha_t = alpha.permute(0, 3, 1, 2).reshape(B * self.H, N, N); Wh_t = Wh.permute(0, 2, 1, 3).reshape(B * self.H, N, self.d)
        out = torch.bmm(alpha_t, Wh_t).reshape(B, self.H, N, self.d).permute(0, 2, 1, 3).reshape(B, N, self.out_dim)
        out = self.bn(out.reshape(B * N, -1)).reshape(B, N, -1)
        return F.elu(self.dropout(out)) + Wh_flat

class TaskAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ELU(), nn.Dropout(0.2), nn.Linear(256, dim))
    def forward(self, x): return x + self.net(x)

class FNPGNNv8_E8(nn.Module):
    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(0.2))
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.gat1 = GATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=0.2)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=0.2)
        self.gat3 = GATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=0.2)
        self.vn_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU())
        self.register_buffer('pooling_mat', POOLING_MAT)
        self.net_attn = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.net_ln = nn.LayerNorm(N_NETWORKS * hidden_dim + hidden_dim)
        
        # E8 Adapters
        head_dim = N_NETWORKS * hidden_dim + hidden_dim
        self.adapter_nc_ad = TaskAdapter(head_dim)
        self.adapter_nc_mci = TaskAdapter(head_dim)
        self.adapter_mci_ad = TaskAdapter(head_dim)
        
        def make_head(): return nn.Sequential(nn.Dropout(dropout), nn.Linear(head_dim, 256), nn.ELU(), nn.Dropout(dropout/2), nn.Linear(256, 2))
        self.head_nc_ad = make_head()
        self.head_nc_mci = make_head()
        self.head_mci_ad = make_head()
        nn.init.normal_(self.virtual_node_emb, std=0.02)

    def forward(self, x, adj):
        B, N, _ = x.shape; h = self.bn_input(self.node_encoder(x).reshape(B * N, -1)).reshape(B, N, -1)
        vn = self.virtual_node_emb.expand(B, -1, -1) + h.mean(dim=1, keepdim=True)
        h = self.gat1(h, adj); vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1)); h = h + vn.expand(-1, N, -1) * 0.1
        h_new = self.gat2(h, adj); h = h_new + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1)); h = h + vn.expand(-1, N, -1) * 0.1
        h_new = self.gat3(h, adj); h = h_new + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))
        pooled = torch.matmul(h.transpose(1, 2), self.pooling_mat).transpose(1, 2); pooled = pooled * torch.softmax(self.net_attn(pooled), dim=1)
        flat = self.net_ln(torch.cat([pooled.reshape(B, -1), vn.squeeze(1)], dim=1))
        
        # Branch through adapters
        out_nc_ad = self.head_nc_ad(self.adapter_nc_ad(flat))
        out_nc_mci = self.head_nc_mci(self.adapter_nc_mci(flat))
        out_mci_ad = self.head_mci_ad(self.adapter_mci_ad(flat))
        return out_nc_ad, out_nc_mci, out_mci_ad, flat

# ===============================================================
# 3. DANN Component (Global GRL)
# ===============================================================
class DomainClassifier(nn.Module):
    def __init__(self, input_dim=N_NETWORKS * HIDDEN_DIM + HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(self, x, alpha=1.0):
        x = x.view_as(x); x = x.detach() + (x - x.detach()) * (-alpha) # GRL
        return self.net(x)

# ===============================================================
# 4. Dataset & Balanced Sampler
# ===============================================================
def get_subject_id(path_str):
    basename = os.path.basename(str(path_str))
    clean = re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy|_T1_MNI\.nii\.gz|_T1\.nii\.gz|\.nii\.gz)$', '', basename)
    clean = re.sub(r'^(sub-|sub_|old_dswau)', '', clean); return clean.strip()

class MultiTaskDataset_E8(Dataset):
    def __init__(self, dataframe, teacher_probs_all=None):
        self.data_cache = []
        for _, row in dataframe.iterrows():
            adj_raw = np.load(row['matrix_path']); subj_id = get_subject_id(row['matrix_path'])
            diag = str(row['diagnosis']).upper(); src = row.get('source', 'TPMIC'); domain_label = 1 if str(src).upper() == 'ADNI' else 0
            labels = {'nc_ad': -1, 'nc_mci': -1, 'mci_ad': -1}
            if diag == 'NC':  labels['nc_ad'] = 0; labels['nc_mci'] = 0; diag_type = 0
            elif diag == 'MCI': labels['nc_mci'] = 1; labels['mci_ad'] = 0; diag_type = 1
            elif diag == 'AD':  labels['nc_ad'] = 1; labels['mci_ad'] = 1; diag_type = 2
            else: diag_type = -1
            soft, has_soft = {'nc_ad': torch.zeros(2), 'nc_mci': torch.zeros(2), 'mci_ad': torch.zeros(2)}, {'nc_ad': False, 'nc_mci': False, 'mci_ad': False}
            for task in labels:
                if labels[task] != -1 and teacher_probs_all and task in teacher_probs_all:
                    if subj_id in teacher_probs_all[task]:
                        soft[task] = torch.FloatTensor(teacher_probs_all[task][subj_id]).flip(0); has_soft[task] = True
            adj_z = np.arctanh(np.clip(adj_raw, -0.999, 0.999)); x_feat = extract_node_features(adj_z); adj_abs = np.abs(adj_z); np.fill_diagonal(adj_abs, 0); k = int(116 * K_RATIO)
            adj_mask = np.zeros_like(adj_z)
            for i in range(116): top_idx = np.argsort(adj_abs[i])[-k:]; adj_mask[i, top_idx] = adj_z[i, top_idx]
            adj_mask = np.maximum(adj_mask, adj_mask.T); np.fill_diagonal(adj_mask, 1.0)
            rowsum = np.abs(adj_mask).sum(1); rowsum[rowsum == 0] = 1e-10
            d_mat = np.diag(np.power(rowsum, -0.5)); adj_norm = d_mat @ adj_mask @ d_mat
            self.data_cache.append({'x': torch.FloatTensor(x_feat), 'adj': torch.FloatTensor(adj_norm), 'labels': labels, 'soft': soft, 'has_soft': has_soft, 'subj_id': subj_id, 'domain_label': torch.tensor(domain_label, dtype=torch.float32), 'diag_type': diag_type})
    def __len__(self): return len(self.data_cache)
    def __getitem__(self, idx): return self.data_cache[idx]

class BalancedTriClassSampler(Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.diag_to_idx = {0: [], 1: [], 2: []}
        for i in self.indices:
            dt = dataset.data_cache[i]['diag_type']
            if dt in self.diag_to_idx: self.diag_to_idx[dt].append(i)
        self.num_samples = max(len(v) for v in self.diag_to_idx.values()) * 3
    def __iter__(self):
        res = []
        for _ in range(self.num_samples // 3):
            for dt in [0, 1, 2]:
                res.append(np.random.choice(self.diag_to_idx[dt]))
        return iter(res)
    def __len__(self): return self.num_samples

# ===============================================================
# 5. Training Loop
# ===============================================================
def run_e8_seed(df_full, teacher_probs_all, device, seed, ckpt_dir=None):
    torch.manual_seed(seed); np.random.seed(seed)
    with open(UNIFIED_SPLIT_PATH, 'r') as f: unified_split = json.load(f)
    oof_results = {task: [None]*len(df_full) for task in ['nc_ad', 'nc_mci', 'mci_ad']}
    oof_true = {task: [None]*len(df_full) for task in ['nc_ad', 'nc_mci', 'mci_ad']}
    for fold in range(5):
        print(f"    Fold {fold+1}/5", end="  ")
        val_subjs = set(unified_split[f"fold_{fold}"])
        train_df = df_full[~df_full['matrix_path'].apply(get_subject_id).isin(val_subjs)].reset_index(drop=True)
        val_df   = df_full[df_full['matrix_path'].apply(get_subject_id).isin(val_subjs)].reset_index(drop=True)
        train_ds = MultiTaskDataset_E8(train_df, teacher_probs_all); val_ds = MultiTaskDataset_E8(val_df, teacher_probs_all)
        sampler = BalancedTriClassSampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True); val_loader = DataLoader(val_ds, batch_size=1)
        model = FNPGNNv8_E8().to(device); domain_clf = DomainClassifier().to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(domain_clf.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
        ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        best_val_auc_sum, best_state, patience_cnt = 0.0, None, 0
        for epoch in range(EPOCHS):
            model.train(); domain_clf.train(); alpha = float(2.0 / (1.0 + np.exp(-10.0 * (epoch / EPOCHS))) - 1.0)
            for b in train_loader:
                x, adj, dom_lbl = b['x'].to(device), b['adj'].to(device), b['domain_label'].to(device); dt = b['diag_type'].to(device)
                out_nc_ad, out_nc_mci, out_mci_ad, flat = model(x, adj)
                loss = torch.tensor(0.0, device=device); tasks = [('nc_ad', out_nc_ad, ALPHA), ('nc_mci', out_nc_mci, BETA), ('mci_ad', out_mci_ad, GAMMA)]
                for t_name, t_out, t_w in tasks:
                    t_lbl = b['labels'][t_name].to(device); mask = (t_lbl != -1)
                    if mask.any():
                        loss_ce = ce_loss_fn(t_out[mask], t_lbl[mask]); loss += t_w * loss_ce
                        t_soft = b['soft'][t_name].to(device); t_has_soft = b['has_soft'][t_name].to(device)
                        valid_kd = mask & t_has_soft
                        if valid_kd.any(): loss += t_w * LAMBDA_KD * nn.KLDivLoss(reduction="batchmean")(F.log_softmax(t_out[valid_kd], dim=1), t_soft[valid_kd])
                # E8: NC-only Domain Alignment
                loss_domain = torch.tensor(0.0, device=device); is_nc = (dt == 0)
                if is_nc.any(): loss_domain = F.binary_cross_entropy_with_logits(domain_clf(flat[is_nc], alpha).squeeze(1), dom_lbl[is_nc])
                loss += LAMBDA_DOMAIN * loss_domain
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            # Validation
            model.eval(); v_probs, v_true = {t: [] for t in oof_results}, {t: [] for t in oof_true}
            with torch.no_grad():
                for b_v in val_loader:
                    v_out = model(b_v['x'].to(device), b_v['adj'].to(device))
                    for i, t in enumerate(['nc_ad', 'nc_mci', 'mci_ad']):
                        t_lbl = b_v['labels'][t].item()
                        if t_lbl != -1: v_probs[t].append(F.softmax(v_out[i], dim=1).cpu().numpy()[0, 1]); v_true[t].append(t_lbl)
            auc_sum = 0
            for t in v_probs:
                if len(set(v_true[t])) > 1: auc_sum += roc_auc_score(v_true[t], v_probs[t])
            if auc_sum > best_val_auc_sum: best_val_auc_sum, patience_cnt, best_state = auc_sum, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else: patience_cnt += 1
            if patience_cnt >= PATIENCE: break
        model.load_state_dict(best_state); model.eval(); val_subj_to_idx = {get_subject_id(p): i for i, p in enumerate(df_full['matrix_path'])}
        with torch.no_grad():
            for b_v in val_loader:
                v_out = model(b_v['x'].to(device), b_v['adj'].to(device)); sid = b_v['subj_id'][0]; idx = val_subj_to_idx[sid]
                for i, t in enumerate(['nc_ad', 'nc_mci', 'mci_ad']):
                    t_lbl = b_v['labels'][t].item()
                    if t_lbl != -1: oof_results[t][idx] = F.softmax(v_out[i], dim=1).cpu().numpy()[0]; oof_true[t][idx] = t_lbl
        print(f"best val AUC sum: {best_val_auc_sum:.3f}")
    return oof_results, oof_true

def analyze_data():
    valid_data, seen = [], set()
    for path in CSV_PATHS:
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m_path = (row.get('matrix_path') or (os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy") if pd.notna(row.get('new_id_base')) else None) or (os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy") if pd.notna(row.get('Subject')) else None))
            if not (m_path and os.path.exists(m_path)) or m_path in seen: continue
            try:
                if np.load(m_path).shape != (116, 116): continue
                diag = str(row.get('diagnosis', '')).upper()
                if diag in ['NC', 'MCI', 'AD']:
                    src = 'ADNI' if ('adni' in m_path.lower() or 'old_dswau' in m_path.lower()) else 'TPMIC'
                    valid_data.append({'matrix_path': m_path, 'diagnosis': diag, 'source': src}); seen.add(m_path)
            except: continue
    return valid_data

def main():
    df_full = pd.DataFrame(analyze_data()); teacher_probs_all = {}
    for task in [('NC','AD'), ('NC','MCI'), ('MCI','AD')]:
        safe = f"{task[0]}_vs_{task[1]}"
        p = os.path.join(TEACHER_PROBS_DIR, f"teacher_logits_{safe}.npy")
        if os.path.exists(p): teacher_probs_all[safe.lower()] = np.load(p, allow_pickle=True).item()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"E8 MTL Decoupled: ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA}")
    seed_probs = {t: [] for t in ['nc_ad', 'nc_mci', 'mci_ad']}
    for seed in SEEDS:
        print(f"\n  [Seed {seed}]")
        res, trues = run_e8_seed(df_full, teacher_probs_all, device, seed)
        for t in res: seed_probs[t].append(np.array([p if p is not None else [0.5, 0.5] for p in res[t]]))
    exp_metrics = {}
    for t in seed_probs:
        avg_p = np.mean(seed_probs[t], axis=0); t_true = [v for v in trues[t] if v is not None]
        t_prob = [avg_p[i, 1] for i, v in enumerate(trues[t]) if v is not None]
        auc = roc_auc_score(t_true, t_prob); acc = accuracy_score(t_true, (np.array(t_prob)>0.5).astype(int))
        print(f"E8 Task {t}: ACC={acc:.3f}, AUC={auc:.3f}")
        exp_metrics[t] = {"auc": float(auc), "acc": float(acc)}
    ser.save_metrics("E8_Decoupled_MTL", exp_metrics); ser.update_comparison_chart()

if __name__ == "__main__": main()
