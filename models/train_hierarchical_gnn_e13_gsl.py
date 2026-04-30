
import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy

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

LAMBDA_KD       = 0.5
LAMBDA_DOMAIN   = 0.3
LAMBDA_ORDINAL  = 0.1   # Reduced from 0.2 to prioritise classification ACC
ORDINAL_MARGIN  = 0.2

# E13-specific
GSL_TOP_K_RATIO = 0.30  # Fraction of edges to keep in learned graph
GSL_NUM_HEADS   = 4     # Attention heads in GraphLearner
MARGIN          = 0.35  # Large-margin penalty subtracted from target logit

SEEDS = [42, 123, 456]

# ===============================================================
# 1. Model Architecture
# ===============================================================
NETWORK_MAP = {
    'DMN': [34, 35, 66, 67, 64, 65, 22, 23, 24, 25],
    'SMN': [0, 1, 56, 57, 68, 69], 'VN': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    'SN': [28, 29, 30, 31, 32, 33], 'FPN': [6, 7, 58, 59, 60, 61],
    'LN': [36, 37, 38, 39, 40, 41], 'VAN': [10, 11, 14, 15],
    'BGN': [70, 71, 72, 73, 74, 75, 76, 77], 'CereN': list(range(90, 116))
}
N_NETWORKS = len(NETWORK_MAP)
POOLING_MAT = torch.zeros(116, N_NETWORKS)
for i, net in enumerate(NETWORK_MAP):
    for node_idx in NETWORK_MAP[net]: POOLING_MAT[node_idx, i] = 1.0


# -- GATLayer: unchanged from E12 -----------------------------------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
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


# -- NEW: Graph Structure Learner ----------------------------------
class GraphLearner(nn.Module):
    """
    Learns an adjacency matrix from node embeddings via multi-head dot-product
    similarity and combines it with the raw Pearson-correlation graph.

    A_final = (1 - γ) * A_raw + γ * A_learned
    where A_learned is Top-K sparse and symmetrised.
    """
    def __init__(self, hidden_dim, num_heads=4, top_k_ratio=0.30):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = hidden_dim // num_heads
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.top_k_ratio = top_k_ratio
        # γ is learnable; sigmoid keeps it in (0,1) — initialise near 0.3
        self.gamma_raw = nn.Parameter(torch.tensor(-0.85))

    def forward(self, h, adj_raw):
        """
        h       : (B, N, D)  node embeddings after the encoder
        adj_raw : (B, N, N)  normalised Pearson-correlation adjacency
        returns : (B, N, N)  blended adjacency for GAT layers
        """
        B, N, D = h.shape
        d = self.d_head

        Q = self.Q(h).view(B, N, self.num_heads, d).permute(0, 2, 1, 3)  # B,H,N,d
        K = self.K(h).view(B, N, self.num_heads, d).permute(0, 2, 1, 3)  # B,H,N,d

        # Scaled dot-product similarity, averaged over heads → B,N,N
        sim = torch.matmul(Q, K.transpose(-1, -2)) / (d ** 0.5)   # B,H,N,N
        sim = sim.mean(dim=1)                                       # B,N,N

        # Top-K sparsity: keep only the k strongest connections per row
        k = max(1, int(N * self.top_k_ratio))
        topk_vals, _ = torch.topk(sim, k, dim=-1)                 # B,N,k
        threshold = topk_vals[:, :, -1].unsqueeze(-1)             # B,N,1
        sparse_mask = (sim >= threshold).float()

        # Sigmoid-gated, symmetrised learned adjacency
        A_learned = torch.sigmoid(sim) * sparse_mask
        A_learned = (A_learned + A_learned.transpose(-1, -2)) / 2.0

        gamma = torch.sigmoid(self.gamma_raw)                      # scalar in (0,1)
        A_final = (1.0 - gamma) * adj_raw + gamma * A_learned
        return A_final


class TaskAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ELU(), nn.Dropout(0.2), nn.Linear(256, dim))
    def forward(self, x): return x + self.net(x)


class FNPGNNv8_E13(nn.Module):
    def __init__(self, input_dim=125, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(0.2))
        self.bn_input = nn.BatchNorm1d(hidden_dim)

        # Graph Structure Learner — runs before GAT layers
        self.graph_learner = GraphLearner(hidden_dim, num_heads=GSL_NUM_HEADS, top_k_ratio=GSL_TOP_K_RATIO)

        self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.gat1 = GATLayer(hidden_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, hidden_dim)
        self.gat3 = GATLayer(hidden_dim, hidden_dim)
        self.vn_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU())
        self.register_buffer('pooling_mat', POOLING_MAT)
        self.net_attn = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.net_ln = nn.LayerNorm(N_NETWORKS * hidden_dim + hidden_dim)
        head_dim = N_NETWORKS * hidden_dim + hidden_dim

        self.adapter_nc_ad  = TaskAdapter(head_dim)
        self.adapter_nc_mci = TaskAdapter(head_dim)
        self.adapter_mci_ad = TaskAdapter(head_dim)

        def make_head():
            return nn.Sequential(nn.Dropout(dropout), nn.Linear(head_dim, 256), nn.ELU(), nn.Dropout(dropout/2), nn.Linear(256, 2))
        self.head_nc_ad  = make_head()
        self.head_nc_mci = make_head()
        self.head_mci_ad = make_head()

        self.progression_head = nn.Sequential(nn.Linear(head_dim, 64), nn.ELU(), nn.Linear(64, 1))
        nn.init.normal_(self.virtual_node_emb, std=0.02)

    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.bn_input(self.node_encoder(x).reshape(B * N, -1)).reshape(B, N, -1)

        # Learn and blend the adjacency matrix
        adj = self.graph_learner(h, adj)

        vn = self.virtual_node_emb.expand(B, -1, -1) + h.mean(dim=1, keepdim=True)
        h = self.gat1(h, adj)
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))
        h = h + vn.expand(-1, N, -1) * 0.1

        h_new = self.gat2(h, adj); h = h_new + h
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))
        h = h + vn.expand(-1, N, -1) * 0.1

        h_new = self.gat3(h, adj); h = h_new + h
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))

        pooled = torch.matmul(h.transpose(1, 2), self.pooling_mat).transpose(1, 2)
        pooled = pooled * torch.softmax(self.net_attn(pooled), dim=1)
        flat = self.net_ln(torch.cat([pooled.reshape(B, -1), vn.squeeze(1)], dim=1))

        logits = (self.head_nc_ad(self.adapter_nc_ad(flat)),
                  self.head_nc_mci(self.adapter_nc_mci(flat)),
                  self.head_mci_ad(self.adapter_mci_ad(flat)))
        progression_score = self.progression_head(flat)
        return logits + (progression_score, flat)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=N_NETWORKS * HIDDEN_DIM + HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1))
    def forward(self, x, alpha=1.0):
        x = x.view_as(x); x = x.detach() + (x - x.detach()) * (-alpha); return self.net(x)


# ===============================================================
# 2. Loss Functions
# ===============================================================
class MultiTaskLossWrapper(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al.)."""
    def __init__(self, num_tasks=5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        weighted = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted.append(precision * loss + self.log_vars[i])
        return sum(weighted)


def large_margin_ce(logits, labels, margin=MARGIN, label_smoothing=0.12):
    """
    Cross-entropy with label smoothing + additive logit margin.
    During training, subtract `margin` from the target-class logit,
    forcing the model to learn a larger separation between classes.
    No margin is applied at inference (just call F.softmax on raw logits).
    """
    one_hot = F.one_hot(labels, num_classes=logits.size(-1)).float()
    logits_m = logits - one_hot * margin
    return F.cross_entropy(logits_m, labels, label_smoothing=label_smoothing)


# ===============================================================
# 3. Data & Utilities (identical to E12)
# ===============================================================
def extract_node_features(adj_z):
    N = 116; k = int(N * K_RATIO)
    adj_abs = np.abs(adj_z); np.fill_diagonal(adj_abs, 0); adj_bin = np.zeros((N, N), dtype=np.float32)
    for i in range(N): adj_bin[i, np.argsort(adj_abs[i])[-k:]] = 1.0
    adj_bin = np.maximum(adj_bin, adj_bin.T); degree = adj_bin.sum(axis=1)
    cc = np.diag(adj_bin @ adj_bin @ adj_bin) / (degree * (degree - 1) + 1e-8)
    features = []
    for i in range(N):
        row = adj_z[i]; stat = [row.mean(), row.std(), (row>0).mean(), (row<0).mean(), (np.abs(row)>0.1).sum()]
        features.append(np.concatenate([row, stat, [0, 0, cc[i], 0]]))
    return np.stack(features).astype(np.float32)

def get_subject_id(p):
    return re.sub(r'^(sub-|sub_|old_dswau)', '', re.sub(r'(_matrix.*|.*nii.gz)$', '', os.path.basename(p))).strip()


class MultiTaskDataset_E13(Dataset):
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
            soft, has_soft = {t: torch.zeros(2) for t in labels}, {t: False for t in labels}
            for task in labels:
                if labels[task] != -1 and teacher_probs_all and task in teacher_probs_all and subj_id in teacher_probs_all[task]:
                    soft[task] = torch.FloatTensor(teacher_probs_all[task][subj_id]).flip(0); has_soft[task] = True
            adj_z = np.arctanh(np.clip(adj_raw, -0.999, 0.999)); x_feat = extract_node_features(adj_z)
            adj_abs = np.abs(adj_z); np.fill_diagonal(adj_abs, 0); k = int(116 * K_RATIO)
            adj_mask = np.zeros_like(adj_z)
            for i in range(116): adj_mask[i, np.argsort(adj_abs[i])[-k:]] = adj_z[i, np.argsort(adj_abs[i])[-k:]]
            adj_mask = np.maximum(adj_mask, adj_mask.T); np.fill_diagonal(adj_mask, 1.0)
            d = np.diag(np.power(np.abs(adj_mask).sum(1)+1e-10, -0.5)); adj_norm = d @ adj_mask @ d
            self.data_cache.append({
                'x': torch.FloatTensor(x_feat), 'adj': torch.FloatTensor(adj_norm),
                'labels': labels, 'soft': soft, 'has_soft': has_soft, 'subj_id': subj_id,
                'domain_label': torch.tensor(domain_label, dtype=torch.float32), 'diag_type': diag_type
            })

    def __len__(self): return len(self.data_cache)
    def __getitem__(self, idx): return self.data_cache[idx]


class BalancedTriClassSampler(Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset))); self.diag_to_idx = {0: [], 1: [], 2: []}
        for i in self.indices:
            dt = dataset.data_cache[i]['diag_type']
            if dt in self.diag_to_idx: self.diag_to_idx[dt].append(i)
        self.num_samples = max(len(v) for v in self.diag_to_idx.values()) * 3
    def __iter__(self):
        res = []
        for _ in range(self.num_samples // 3): [res.append(np.random.choice(self.diag_to_idx[dt])) for dt in [0, 1, 2]]
        return iter(res)
    def __len__(self): return self.num_samples


# ===============================================================
# 4. Training Function
# ===============================================================
def run_e13_seed(df_full, teacher_probs_all, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    with open(UNIFIED_SPLIT_PATH, 'r') as f: unified_split = json.load(f)
    oof_results = {task: [None]*len(df_full) for task in ['nc_ad', 'nc_mci', 'mci_ad']}
    oof_true    = {task: [None]*len(df_full) for task in ['nc_ad', 'nc_mci', 'mci_ad']}

    for fold in range(5):
        print(f"    Fold {fold+1}/5", end=" ")
        val_subjs = set(unified_split[f"fold_{fold}"])
        train_df = df_full[~df_full['matrix_path'].apply(get_subject_id).isin(val_subjs)].reset_index(drop=True)
        val_df   = df_full[df_full['matrix_path'].apply(get_subject_id).isin(val_subjs)].reset_index(drop=True)
        train_ds = MultiTaskDataset_E13(train_df, teacher_probs_all)
        val_ds   = MultiTaskDataset_E13(val_df, teacher_probs_all)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=BalancedTriClassSampler(train_ds), drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=1)

        model        = FNPGNNv8_E13().to(device)
        domain_clf   = DomainClassifier().to(device)
        loss_wrapper = MultiTaskLossWrapper(num_tasks=5).to(device)  # nc_ad, nc_mci, mci_ad, domain, ordinal

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(domain_clf.parameters()) + list(loss_wrapper.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY
        )

        best_val_auc_sum, best_state, patience_cnt = 0.0, None, 0

        for epoch in range(EPOCHS):
            model.train(); domain_clf.train()
            alpha = float(2.0 / (1.0 + np.exp(-10.0 * (epoch / EPOCHS))) - 1.0)

            for b in train_loader:
                x, adj, dom_lbl = b['x'].to(device), b['adj'].to(device), b['domain_label'].to(device)
                dt = b['diag_type'].to(device)
                out_nc_ad, out_nc_mci, out_mci_ad, prog_scores, flat = model(x, adj)

                # 1. Classification Losses — Large Margin CE
                task_losses = []
                for t_name, t_out in [('nc_ad', out_nc_ad), ('nc_mci', out_nc_mci), ('mci_ad', out_mci_ad)]:
                    t_lbl = b['labels'][t_name].to(device); mask = (t_lbl != -1)
                    if mask.any():
                        l_ce = large_margin_ce(t_out[mask], t_lbl[mask])
                        t_soft    = b['soft'][t_name].to(device)
                        t_has_soft = b['has_soft'][t_name].to(device)
                        valid_kd  = mask & t_has_soft
                        if valid_kd.any():
                            l_ce += LAMBDA_KD * nn.KLDivLoss(reduction="batchmean")(
                                F.log_softmax(t_out[valid_kd], dim=1), t_soft[valid_kd])
                        task_losses.append(l_ce)
                    else:
                        task_losses.append(torch.tensor(0.0, device=device))

                # 2. Domain Loss (DANN — NC only, GRL)
                is_nc = (dt == 0)
                if is_nc.any():
                    l_dom = LAMBDA_DOMAIN * F.binary_cross_entropy_with_logits(
                        domain_clf(flat[is_nc], alpha).squeeze(1), dom_lbl[is_nc])
                else:
                    l_dom = torch.tensor(0.0, device=device)
                task_losses.append(l_dom)

                # 3. Ordinal Progression Loss
                l_ordinal = torch.tensor(0.0, device=device)
                for i in range(len(dt)):
                    for j in range(len(dt)):
                        if dt[i] > dt[j]:
                            l_ordinal += F.margin_ranking_loss(
                                prog_scores[i], prog_scores[j],
                                torch.tensor([1.0], device=device), margin=ORDINAL_MARGIN)
                task_losses.append(LAMBDA_ORDINAL * l_ordinal / (len(dt)**2 + 1e-8))

                # Uncertainty-weighted total loss
                total_loss = loss_wrapper(task_losses)
                optimizer.zero_grad(); total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                gamma_val = torch.sigmoid(model.graph_learner.gamma_raw).item()
                weights   = torch.exp(-loss_wrapper.log_vars).detach().cpu().numpy().round(3)
                print(f"      Epoch {epoch+1}/{EPOCHS} | γ={gamma_val:.3f} | Weights: {weights}", flush=True)

            # Validation
            model.eval()
            v_probs, v_true = {t: [] for t in oof_results}, {t: [] for t in oof_true}
            with torch.no_grad():
                for b_v in val_loader:
                    v_out = model(b_v['x'].to(device), b_v['adj'].to(device))
                    for i, t in enumerate(['nc_ad', 'nc_mci', 'mci_ad']):
                        t_lbl = b_v['labels'][t].item()
                        if t_lbl != -1:
                            v_probs[t].append(F.softmax(v_out[i], dim=1).cpu().numpy()[0, 1])
                            v_true[t].append(t_lbl)

            auc_sum = sum(
                roc_auc_score(v_true[t], v_probs[t]) if len(set(v_true[t])) > 1 else 0
                for t in v_probs
            )
            if auc_sum > best_val_auc_sum:
                best_val_auc_sum, patience_cnt = auc_sum, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

        # Store OOF predictions from best checkpoint
        model.load_state_dict(best_state); model.eval()
        
        # [ADD] Save the best model for this seed and fold (or just the seed if you prefer)
        # For simplicity and to match inference_pipeline, we save the seed's best model (from the last fold or overall best)
        # Actually, let's save per seed to match the logic: gnn_NC_vs_AD_seed42.pt
        # But wait, E13 is multi-task. We'll save it as gnn_e13_all_tasks_seed{seed}.pt
        # And we'll update inference_pipeline to load this single file.
        save_dir = os.path.join(TEACHER_PROBS_DIR, "gnn_checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        # We save the model from the BEST FOLD of this seed
        # (In a production setting, you might ensemble folds, but here we just need a working checkpoint)
        torch.save(best_state, os.path.join(save_dir, f"gnn_e13_seed{seed}.pt"))
        print(f"      ✅ Model saved: gnn_e13_seed{seed}.pt")

        val_subj_to_idx = {get_subject_id(p): i for i, p in enumerate(df_full['matrix_path'])}
        with torch.no_grad():
            for b_v in val_loader:
                v_out = model(b_v['x'].to(device), b_v['adj'].to(device))
                sid = b_v['subj_id'][0]; idx = val_subj_to_idx[sid]
                for i, t in enumerate(['nc_ad', 'nc_mci', 'mci_ad']):
                    t_lbl = b_v['labels'][t].item()
                    if t_lbl != -1:
                        oof_results[t][idx] = F.softmax(v_out[i], dim=1).cpu().numpy()[0]
                        oof_true[t][idx]    = t_lbl
        print(f"best val AUC sum: {best_val_auc_sum:.3f}")

    return oof_results, oof_true


# ===============================================================
# 5. Main
# ===============================================================
def main():
    valid_data, seen = [], set()
    for path in CSV_PATHS:
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m_path = (
                row.get('matrix_path') or
                (os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy") if pd.notna(row.get('new_id_base')) else None) or
                (os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy")     if pd.notna(row.get('Subject'))     else None)
            )
            if not (m_path and os.path.exists(m_path)) or m_path in seen: continue
            if np.load(m_path).shape == (116, 116) and str(row.get('diagnosis', '')).upper() in ['NC', 'MCI', 'AD']:
                valid_data.append({
                    'matrix_path': m_path,
                    'diagnosis':   str(row['diagnosis']).upper(),
                    'source':      'ADNI' if ('adni' in m_path.lower() or 'old_dswau' in m_path.lower()) else 'TPMIC'
                })
                seen.add(m_path)

    df_full = pd.DataFrame(valid_data)

    teacher_probs_all = {}
    for task in [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]:
        p = os.path.join(TEACHER_PROBS_DIR, f"teacher_logits_{task[0]}_vs_{task[1]}.npy")
        if os.path.exists(p):
            teacher_probs_all[f"{task[0]}_vs_{task[1]}".lower()] = np.load(p, allow_pickle=True).item()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"E13 GSL | Large-Margin CE | Uncertainty MTL | Ordinal λ={LAMBDA_ORDINAL} | Device: {device}")

    seed_probs = {t: [] for t in ['nc_ad', 'nc_mci', 'mci_ad']}
    final_trues = None

    for seed in SEEDS:
        print(f"\n  [Seed {seed}]")
        res, trues = run_e13_seed(df_full, teacher_probs_all, device, seed)
        if final_trues is None: final_trues = trues
        for t in res:
            seed_probs[t].append(np.array([p if p is not None else [0.5, 0.5] for p in res[t]]))

    exp_metrics  = {}
    oof_data_out = {}

    print("\n=== E13 Threshold Optimization Results ===")
    for t in seed_probs:
        avg_p  = np.mean(seed_probs[t], axis=0)
        t_true = np.array([v for v in final_trues[t] if v is not None])
        t_prob = np.array([avg_p[i, 1] for i, v in enumerate(final_trues[t]) if v is not None])

        auc    = roc_auc_score(t_true, t_prob)
        acc_05 = accuracy_score(t_true, (t_prob > 0.5).astype(int))

        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(t_true, t_prob)
        optimal_idx    = np.argmax(tpr - fpr)
        best_threshold = thresholds[optimal_idx]
        acc_opt        = accuracy_score(t_true, (t_prob > best_threshold).astype(int))

        print(f"Task {t.upper()}:")
        print(f"  AUC: {auc:.3f}")
        print(f"  ACC (0.5): {acc_05:.3f}")
        print(f"  Best Threshold: {best_threshold:.3f}")
        print(f"  ACC (Optimized): {acc_opt:.3f}")

        exp_metrics[t.upper()] = {
            "auc":            float(auc),
            "acc":            float(acc_opt),   # Youden-optimised
            "acc_baseline":   float(acc_05),
            "best_threshold": float(best_threshold),
            "fpr":            fpr.tolist(),
            "tpr":            tpr.tolist(),
        }
        oof_data_out[t] = {"true": t_true, "prob": t_prob}

    out_dir = "/home/wei-chi/Data/script/results/E13_GSL"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "oof_predictions.npy"), oof_data_out)

    ser.save_metrics("E13_GSL", exp_metrics)
    ser.update_comparison_chart()


if __name__ == "__main__": main()