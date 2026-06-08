import os
import re
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ===============================================================
# Constants & Model (Copied from pretraining script)
# ===============================================================
HIDDEN_DIM = 128
DROPOUT = 0.4
K_RATIO = 0.20
MARGIN = 0.35
ORDINAL_MARGIN = 0.2
LAMBDA_ORDINAL = 0.1

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

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, top_k_ratio=0.30):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = hidden_dim // num_heads
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.top_k_ratio = top_k_ratio
        self.gamma_raw = nn.Parameter(torch.tensor(-0.85))

    def forward(self, h, adj_raw):
        B, N, D = h.shape
        d = self.d_head
        Q = self.Q(h).view(B, N, self.num_heads, d).permute(0, 2, 1, 3)
        K = self.K(h).view(B, N, self.num_heads, d).permute(0, 2, 1, 3)
        sim = torch.matmul(Q, K.transpose(-1, -2)) / (d ** 0.5)
        sim = sim.mean(dim=1)
        k = max(1, int(N * self.top_k_ratio))
        topk_vals, _ = torch.topk(sim, k, dim=-1)
        threshold = topk_vals[:, :, -1].unsqueeze(-1)
        sparse_mask = (sim >= threshold).float()
        A_learned = torch.sigmoid(sim) * sparse_mask
        A_learned = (A_learned + A_learned.transpose(-1, -2)) / 2.0
        gamma = torch.sigmoid(self.gamma_raw)
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
        self.graph_learner = GraphLearner(hidden_dim, num_heads=4, top_k_ratio=0.30)
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

# ===============================================================
# Utilities
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

class MultiTaskDataset_Finetune(Dataset):
    def __init__(self, dataframe):
        self.data_cache = []
        for _, row in dataframe.iterrows():
            adj_raw = np.load(row['matrix_path'])
            diag = str(row['diagnosis']).upper()
            labels = {'nc_ad': -1, 'nc_mci': -1, 'mci_ad': -1}
            if diag == 'NC':  labels['nc_ad'] = 0; labels['nc_mci'] = 0; diag_type = 0
            elif diag == 'MCI': labels['nc_mci'] = 1; labels['mci_ad'] = 0; diag_type = 1
            elif diag == 'AD':  labels['nc_ad'] = 1; labels['mci_ad'] = 1; diag_type = 2
            else: diag_type = -1
            adj_z = np.arctanh(np.clip(adj_raw, -0.999, 0.999)); x_feat = extract_node_features(adj_z)
            adj_abs = np.abs(adj_z); np.fill_diagonal(adj_abs, 0); k = int(116 * K_RATIO)
            adj_mask = np.zeros_like(adj_z)
            for i in range(116): adj_mask[i, np.argsort(adj_abs[i])[-k:]] = adj_z[i, np.argsort(adj_abs[i])[-k:]]
            adj_mask = np.maximum(adj_mask, adj_mask.T); np.fill_diagonal(adj_mask, 1.0)
            d = np.diag(np.power(np.abs(adj_mask).sum(1)+1e-10, -0.5)); adj_norm = d @ adj_mask @ d
            self.data_cache.append({
                'x': torch.FloatTensor(x_feat), 'adj': torch.FloatTensor(adj_norm),
                'labels': labels, 'diag_type': diag_type
            })
    def __len__(self): return len(self.data_cache)
    def __getitem__(self, idx): return self.data_cache[idx]

class BalancedTriClassSampler(Sampler):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset))); self.diag_to_idx = {0: [], 1: [], 2: []}
        for i in self.indices:
            dt = dataset.data_cache[i]['diag_type']
            if dt in self.diag_to_idx: self.diag_to_idx[dt].append(i)
        self.num_samples = max(len(v) for v in self.diag_to_idx.values() if len(v) > 0) * 3
    def __iter__(self):
        res = []
        avail_types = [dt for dt in [0, 1, 2] if len(self.diag_to_idx[dt]) > 0]
        for _ in range(self.num_samples // 3): 
            for dt in avail_types:
                res.append(np.random.choice(self.diag_to_idx[dt]))
        return iter(res)
    def __len__(self): return self.num_samples

def large_margin_ce(logits, labels, margin=MARGIN, label_smoothing=0.12):
    one_hot = F.one_hot(labels, num_classes=logits.size(-1)).float()
    logits_m = logits - one_hot * margin
    return F.cross_entropy(logits_m, labels, label_smoothing=label_smoothing)

# ===============================================================
# Main Fine-tuning Loop
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-ckpt-dir", type=str, required=True)
    parser.add_argument("--tpmic-train-csv", type=str, required=True)
    parser.add_argument("--tpmic-fraction", type=float, default=1.0)
    parser.add_argument("--output-ckpt-dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_ckpt_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data
    df_train_full = pd.read_csv(args.tpmic_train_csv)
    if args.tpmic_fraction < 1.0:
        df_train, _ = pd.read_csv(args.tpmic_train_csv).pipe(
            lambda d: (d.groupby('diagnosis', group_keys=False)
                        .apply(lambda x: x.sample(frac=args.tpmic_fraction, random_state=args.seed)), None)
        )
        print(f"Sampled {len(df_train)} subjects ({args.tpmic_fraction*100:.0f}%) from TPMIC train.")
    else:
        df_train = df_train_full

    train_ds = MultiTaskDataset_Finetune(df_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=BalancedTriClassSampler(train_ds), drop_last=True)

    # 2. Fine-tune for each task if specified, or just once for all tasks
    # The requirement says: "for each OVO task, load the matching .pt file... fine-tune on TPMIC"
    # and "Output checkpoint naming: {task}_finetune_frac{tpmic_fraction:.2f}.pt"
    
    tasks = ['NC_vs_AD', 'NC_vs_MCI', 'MCI_vs_AD']
    for task in tasks:
        print(f"\nFine-tuning task: {task}")
        model = FNPGNNv8_E13().to(device)
        
        # Load pretrained weights
        ckpt_path = os.path.join(args.pretrain_ckpt_dir, f"{task}.pt")
        if not os.path.exists(ckpt_path):
            # Fallback to general checkpoint if specific task one doesn't exist
            ckpt_path = os.path.join(args.pretrain_ckpt_dir, "gnn_e13_seed42.pt")
        
        if os.path.exists(ckpt_path):
            print(f"Loading weights from {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"Warning: No checkpoint found at {ckpt_path}. Starting from scratch.")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
        
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            for b in train_loader:
                x, adj = b['x'].to(device), b['adj'].to(device)
                dt = b['diag_type'].to(device)
                out_nc_ad, out_nc_mci, out_mci_ad, prog_scores, _ = model(x, adj)
                
                # We fine-tune the whole model but focus on the OVO tasks
                task_losses = []
                for t_name, t_out in [('nc_ad', out_nc_ad), ('nc_mci', out_nc_mci), ('mci_ad', out_mci_ad)]:
                    t_lbl = b['labels'][t_name].to(device); mask = (t_lbl != -1)
                    if mask.any():
                        task_losses.append(large_margin_ce(t_out[mask], t_lbl[mask]))
                
                # Ordinal loss
                l_ordinal = torch.tensor(0.0, device=device)
                for i in range(len(dt)):
                    for j in range(len(dt)):
                        if dt[i] > dt[j]:
                            l_ordinal += F.margin_ranking_loss(
                                prog_scores[i], prog_scores[j],
                                torch.tensor([1.0], device=device), margin=ORDINAL_MARGIN)
                l_total = sum(task_losses) + LAMBDA_ORDINAL * l_ordinal / (len(dt)**2 + 1e-8)
                
                optimizer.zero_grad(); l_total.backward()
                optimizer.step()
                epoch_loss += l_total.item()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

        # Save fine-tuned checkpoint
        out_name = f"{task}_finetune_frac{args.tpmic_fraction:.2f}.pt"
        torch.save(model.state_dict(), os.path.join(args.output_ckpt_dir, out_name))
        print(f"Saved: {out_name}")

if __name__ == "__main__": main()
