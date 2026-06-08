import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# --- Reuse definitions from extract_tpmic_gnn_embeddings.py ---
HIDDEN_DIM = 128
DROPOUT    = 0.4
K_RATIO    = 0.20
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
N_NETWORKS  = len(NETWORK_MAP)
POOLING_MAT = torch.zeros(116, N_NETWORKS)
for i, net in enumerate(NETWORK_MAP):
    for node_idx in NETWORK_MAP[net]:
        POOLING_MAT[node_idx, i] = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H = num_heads; self.d = out_dim // num_heads; self.out_dim = out_dim
        self.W      = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src  = nn.Linear(self.d, 1, bias=False)
        self.a_dst  = nn.Linear(self.d, 1, bias=False)
        self.bn     = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, h, adj):
        B, N, _ = h.shape
        Wh_flat = self.W(h); Wh = Wh_flat.view(B, N, self.H, self.d)
        e = F.leaky_relu(self.a_src(Wh).squeeze(-1).unsqueeze(2) + self.a_dst(Wh).squeeze(-1).unsqueeze(1), negative_slope=0.2)
        e = e + adj.unsqueeze(-1) * 0.5
        e = e.masked_fill((adj.abs() < 1e-6).unsqueeze(-1), -1e9)
        alpha = self.dropout(F.softmax(e, dim=2))
        alpha_t = alpha.permute(0,3,1,2).reshape(B*self.H, N, N)
        Wh_t    = Wh.permute(0,2,1,3).reshape(B*self.H, N, self.d)
        out = torch.bmm(alpha_t, Wh_t).reshape(B, self.H, N, self.d)
        out = out.permute(0,2,1,3).reshape(B, N, self.out_dim)
        out = self.bn(out.reshape(B*N,-1)).reshape(B, N, -1)
        return F.elu(self.dropout(out)) + Wh_flat

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, top_k_ratio=0.30):
        super().__init__()
        self.Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.top_k = max(1, int(116 * top_k_ratio))
        self.gamma_raw = nn.Parameter(torch.zeros([]))
    def forward(self, h, adj_raw):
        gamma = torch.sigmoid(self.gamma_raw)
        B, N, _ = h.shape
        Q = self.Q(h); K = self.K(h)
        scores = torch.bmm(Q, K.transpose(1,2)) / (h.shape[-1] ** 0.5)
        topk   = scores.topk(self.top_k, dim=-1)
        mask   = torch.zeros_like(scores).scatter_(-1, topk.indices, 1.0)
        adj_learned = F.softmax(scores * mask + (1 - mask) * -1e9, dim=-1)
        return gamma * adj_learned + (1 - gamma) * adj_raw

class TaskAdapter(nn.Module):
    def __init__(self, dim, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, bottleneck), nn.ELU(), nn.Dropout(0.1), nn.Linear(bottleneck, dim))
    def forward(self, x): return x + self.net(x)

class FNPGNNv8_E13(nn.Module):
    def __init__(self, input_dim=125):
        super().__init__()
        H = HIDDEN_DIM
        self.node_encoder = nn.Sequential(nn.Linear(input_dim, H), nn.ELU(), nn.Dropout(0.2))
        self.bn_input     = nn.BatchNorm1d(H)
        self.graph_learner = GraphLearner(H, num_heads=4, top_k_ratio=0.30)
        self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, H))
        self.gat1 = GATLayer(H, H); self.gat2 = GATLayer(H, H); self.gat3 = GATLayer(H, H)
        self.vn_update = nn.Sequential(nn.Linear(H*2, H), nn.ELU())
        self.register_buffer('pooling_mat', POOLING_MAT)
        self.net_attn = nn.Sequential(nn.Linear(H, 32), nn.Tanh(), nn.Linear(32, 1))
        head_dim = N_NETWORKS * H + H
        self.net_ln = nn.LayerNorm(head_dim)
        self.adapter_nc_ad  = TaskAdapter(head_dim)
        self.adapter_nc_mci = TaskAdapter(head_dim)
        self.adapter_mci_ad = TaskAdapter(head_dim)
        nn.init.normal_(self.virtual_node_emb, std=0.02)
    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.bn_input(self.node_encoder(x).reshape(B*N,-1)).reshape(B, N, -1)
        adj = self.graph_learner(h, adj)
        vn = self.virtual_node_emb.expand(B,-1,-1) + h.mean(dim=1, keepdim=True)
        h = self.gat1(h, adj)
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        h = h + vn.expand(-1,N,-1) * 0.1
        h_new = self.gat2(h, adj); h = h_new + h
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        h = h + vn.expand(-1,N,-1) * 0.1
        h_new = self.gat3(h, adj); h = h_new + h
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        pooled = torch.matmul(h.transpose(1,2), self.pooling_mat).transpose(1,2)
        pooled = pooled * torch.softmax(self.net_attn(pooled), dim=1)
        flat = self.net_ln(torch.cat([pooled.reshape(B,-1), vn.squeeze(1)], dim=1))
        return flat

def extract_node_features(adj_z):
    N = 116
    adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO)
    adj_bin = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]; adj_bin[i, top_idx] = 1.0
    adj_bin = np.maximum(adj_bin, adj_bin.T)
    degree = adj_bin.sum(axis=1)
    cc = np.diag(adj_bin @ adj_bin @ adj_bin) / (degree * (degree - 1) + 1e-8)
    adj_abs_thresh = adj_abs * adj_bin
    pc = np.zeros(N, dtype=np.float32)
    roi_to_net = {roi: i for i, nodes in enumerate(NETWORK_MAP.values()) for roi in nodes}
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
            w_nodes = [r for r in list(NETWORK_MAP.values())[net_i] if r != i]
            b_nodes = [r for r in range(N) if r != i and roi_to_net.get(r,-1) != net_i]
            w_fc = float(np.mean([row[r] for r in w_nodes])) if w_nodes else 0.0
            b_fc = float(np.mean([row[r] for r in b_nodes])) if b_nodes else 0.0
        else: w_fc, b_fc = 0.0, 0.0
        features.append(np.concatenate([fc_feat, stat_feat, np.array([w_fc, b_fc]), np.array([cc[i], pc[i]])]))
    return np.stack(features).astype(np.float32)

def build_adj(adj_z):
    N = 116; adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO); adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]; adj[i, top_idx] = adj_z[i, top_idx]
    return np.maximum(adj, adj.T)

class SimpleDataset(Dataset):
    def __init__(self, df):
        self.ids = df['subject_id'].tolist(); self.paths = df['matrix_path'].tolist(); self.labels = df['label'].tolist()
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        mat = np.load(self.paths[i]); feat = extract_node_features(mat); adj = build_adj(mat)
        return self.ids[i], torch.tensor(feat), torch.tensor(adj), torch.tensor(self.labels[i])

def main():
    BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
    CKPT_BASE = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/models/checkpoints/finetune_tpmic_full")
    df_tr = pd.read_csv(BASE_DIR / "kd_train_aligned.csv")
    df_te = pd.read_csv(BASE_DIR / "kd_test_aligned.csv")
    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    df_adni = df_all[df_all['source'] == 'ADNI_new'].reset_index(drop=True)
    
    tasks = {
        'NC_vs_AD':  {'classes': [0, 2], 'pos': 2},
        'NC_vs_MCI': {'classes': [0, 1], 'pos': 1},
        'MCI_vs_AD': {'classes': [1, 2], 'pos': 2},
    }
    model = FNPGNNv8_E13().to(DEVICE)
    for task, cfg in tasks.items():
        print(f"\nTask: {task}")
        ckpt = CKPT_BASE / f"{task}_finetune_frac1.00.pt"
        if not ckpt.exists(): continue
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False); model.eval()
        df_task = df_adni[df_adni['label'].isin(cfg['classes'])].reset_index(drop=True)
        ds = SimpleDataset(df_task); dl = DataLoader(ds, batch_size=1, num_workers=4)
        sids, embs, labels, bin_labels = [], [], [], []
        with torch.no_grad():
            for sid, feat, adj, label in dl:
                flat = model(feat.to(DEVICE), adj.to(DEVICE))
                sids.append(sid[0]); embs.append(flat.cpu().numpy()[0]); labels.append(label.item()); bin_labels.append(1 if label.item()==cfg['pos'] else 0)
        out = BASE_DIR / "embeddings" / f"gnn_embeddings_adni_{task}.npz"
        out.parent.mkdir(exist_ok=True)
        np.savez(out, subject_ids=np.array(sids), embeddings=np.array(embs), labels=np.array(labels), bin_labels=np.array(bin_labels))
        print(f"  [SAVED] {out} ({len(sids)} subjects)")

if __name__ == "__main__": main()
