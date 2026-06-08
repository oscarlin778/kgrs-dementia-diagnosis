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

# --- Constants & ROI Map (Directly from train_kd_fusion.py) ---
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
    N = adj_z.shape[0]
    adj_abs = np.abs(adj_z.copy())
    np.fill_diagonal(adj_abs, 0)
    
    # Binary graph for CC/PC
    k = int(N * K_RATIO)
    adj_bin = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]
        adj_bin[i, top_idx] = 1.0
    adj_bin = np.maximum(adj_bin, adj_bin.T)
    
    degree = adj_bin.sum(axis=1)
    cc = np.diag(adj_bin @ adj_bin @ adj_bin) / (degree * (degree - 1) + 1e-8)
    
    adj_abs_thresh = adj_abs * adj_bin
    pc = np.zeros(N, dtype=np.float32)
    for i in range(N):
        ki = adj_abs_thresh[i].sum()
        if ki > 1e-8:
            pc_i = 1.0
            for net_nodes in NETWORK_MAP.values():
                kim = adj_abs_thresh[i, list(net_nodes)].sum()
                pc_i -= (kim / ki) ** 2
            pc[i] = float(np.clip(pc_i, 0.0, 1.0))
            
    features = []
    for i in range(N):
        row = adj_z[i].copy(); row[i] = 0
        fc_feat = row.astype(np.float32)
        stat_feat = np.array([row.mean(), row.std(), (row>0).mean(), (row<0).mean(), (np.abs(row)>0.1).sum()], dtype=np.float32)
        
        net_i = roi_to_net.get(i, -1)
        if net_i >= 0:
            w_nodes = [r for r in NETWORK_MAP[net_list[net_i]] if r != i]
            b_nodes = [r for r in range(N) if r != i and roi_to_net.get(r, -1) != net_i]
            w_fc = float(np.mean([row[r] for r in w_nodes])) if w_nodes else 0.0
            b_fc = float(np.mean([row[r] for r in b_nodes])) if b_nodes else 0.0
        else:
            w_fc, b_fc = 0.0, 0.0
            
        topo_feat = np.array([cc[i], pc[i]], dtype=np.float32)
        features.append(np.concatenate([fc_feat, stat_feat, np.array([w_fc, b_fc]), topo_feat]))
    return np.stack(features, axis=0).astype(np.float32)

def build_adj(adj_z):
    N = adj_z.shape[0]
    adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]
        adj[i, top_idx] = adj_z[i, top_idx]
    return np.maximum(adj, adj.T)

# --- Model Components ---
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H = num_heads; self.d = out_dim // num_heads; self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(self.d, 1, bias=False)
        self.a_dst = nn.Linear(self.d, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_dim); self.dropout = nn.Dropout(dropout)
    def forward(self, h, adj):
        B, N, _ = h.shape
        Wh_flat = self.W(h); Wh = Wh_flat.view(B, N, self.H, self.d)
        e = F.leaky_relu(self.a_src(Wh).squeeze(-1).unsqueeze(2) + self.a_dst(Wh).squeeze(-1).unsqueeze(1), negative_slope=0.2)
        e = e + adj.unsqueeze(-1) * 0.5
        e = e.masked_fill((adj.abs() < 1e-6).unsqueeze(-1), -1e9)
        alpha = self.dropout(F.softmax(e, dim=2))
        alpha_t = alpha.permute(0,3,1,2).reshape(B*self.H, N, N)
        Wh_t = Wh.permute(0,2,1,3).reshape(B*self.H, N, self.d)
        out = torch.bmm(alpha_t, Wh_t).reshape(B, self.H, N, self.d)
        out = out.permute(0,2,1,3).reshape(B, N, self.out_dim)
        out = self.bn(out.reshape(B*N, -1)).reshape(B, N, -1)
        return F.elu(self.dropout(out)) + Wh_flat

class FMRIEncoder(nn.Module):
    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(0.2))
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.gat1, self.gat2, self.gat3 = [GATLayer(hidden_dim, hidden_dim) for _ in range(3)]
        self.vn_update = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ELU())
        self.register_buffer('pooling_mat', POOLING_MAT)
        self.net_attn = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.net_ln = nn.LayerNorm(N_NETWORKS * hidden_dim + hidden_dim)
        nn.init.normal_(self.virtual_node_emb, std=0.02)
        
    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.bn_input(self.node_encoder(x).reshape(B*N,-1)).reshape(B, N, -1)
        vn = self.virtual_node_emb.expand(B,-1,-1) + h.mean(dim=1, keepdim=True)
        h = self.gat1(h, adj)
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat2(h, adj) + h
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat3(h, adj) + h
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        pooled = torch.matmul(h.transpose(1,2), self.pooling_mat).transpose(1,2)
        pooled = pooled * torch.softmax(self.net_attn(pooled), dim=1)
        flat = self.net_ln(torch.cat([pooled.reshape(B,-1), vn.squeeze(1)], dim=1))
        return flat

class PCAGFusion(nn.Module):
    def __init__(self, fmri_dim=1280, smri_dim=768, fusion_dim=20, num_classes=2):
        super().__init__()
        self.fmri_proj = nn.Linear(fmri_dim, fusion_dim)
        self.smri_proj_k = nn.Linear(smri_dim, fusion_dim)
        self.smri_proj_v = nn.Linear(smri_dim, fusion_dim)
        
        self.W_e = nn.Linear(fusion_dim, fusion_dim)
        self.W_g1 = nn.Linear(fusion_dim, fusion_dim)
        self.W_g2 = nn.Linear(fusion_dim, fusion_dim)
        self.ln_e = nn.LayerNorm(fusion_dim)
        self.ln_g = nn.LayerNorm(fusion_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, fmri_emb, smri_feat):
        Q = self.fmri_proj(fmri_emb)
        K = self.smri_proj_k(smri_feat)
        V = self.smri_proj_v(smri_feat)
        
        P = (torch.tanh(Q) * torch.tanh(K) + 1) / 2
        A_gated = (Q * K) * P
        S = torch.sigmoid(A_gated)
        V_hat = S * V
        
        E = F.relu(self.W_e(V_hat))
        G = F.relu(self.W_g1(V_hat) + self.W_g2(Q))
        C = self.ln_e(E) * self.ln_g(G)
        
        fused = C + F.relu(Q)
        return self.classifier(fused)

class PCAGModel(nn.Module):
    def __init__(self, fusion_dim=20):
        super().__init__()
        self.fmri_encoder = FMRIEncoder()
        self.pcag = PCAGFusion(fusion_dim=fusion_dim)
    def forward(self, x_node, adj, smri_feat):
        fmri_emb = self.fmri_encoder(x_node, adj)
        return self.pcag(fmri_emb, smri_feat)

# --- Dataset ---
class PCAGDataset(Dataset):
    def __init__(self, matrix_paths, smri_feats, labels):
        self.paths = matrix_paths
        self.smri_feats = smri_feats.astype(np.float32)
        self.labels = labels
        self._cache = {}
        
    def __len__(self): return len(self.paths)
    
    def __getitem__(self, i):
        path = self.paths[i]
        if path not in self._cache:
            mat = np.load(path)
            feat = extract_node_features(mat)
            adj = build_adj(mat)
            self._cache[path] = (feat, adj)
        feat, adj = self._cache[path]
        return (torch.tensor(feat), torch.tensor(adj), 
                torch.tensor(self.smri_feats[i]), torch.tensor(self.labels[i], dtype=torch.long))

# --- Training Script ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="NC_vs_MCI", choices=["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fusion_dim", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
    RES_DIR  = BASE_DIR / "results"
    RES_DIR.mkdir(exist_ok=True)

    TASK_CFG = {
        'NC_vs_AD':  {'classes': [0, 2], 'pos': 2},
        'NC_vs_MCI': {'classes': [0, 1], 'pos': 1},
        'MCI_vs_AD': {'classes': [1, 2], 'pos': 2},
    }
    cfg = TASK_CFG[args.task]

    # Load Alignments
    df_tr = pd.read_csv(BASE_DIR / "pcag_train_aligned.csv")
    df_te = pd.read_csv(BASE_DIR / "pcag_test_aligned.csv")

    # Filter by task
    df_tr = df_tr[df_tr['label'].isin(cfg['classes'])].reset_index(drop=True)
    df_tr['bin_label'] = (df_tr['label'] == cfg['pos']).astype(int)
    
    df_te = df_te[df_te['label'].isin(cfg['classes'])].reset_index(drop=True)
    df_te['bin_label'] = (df_te['label'] == cfg['pos']).astype(int)

    # Load sMRI Features
    smri_cols = [f"Feature_{i}" for i in range(768)]
    smri_tr_all = pd.read_csv(BASE_DIR / "brainiac_features_train.csv")
    smri_te_all = pd.read_csv(BASE_DIR / "brainiac_features_combined_test.csv")
    
    # Use smri_feat_row to index correctly
    smri_tr = smri_tr_all.iloc[df_tr['smri_feat_row']][smri_cols].values
    smri_te = smri_te_all.iloc[df_te['smri_feat_row']][smri_cols].values

    print(f"Task: {args.task} | Train: {len(df_tr)} | Test: {len(df_te)}")

    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros(len(df_tr))
    test_probs_folds = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(df_tr, df_tr['bin_label'])):
        print(f"\n--- Fold {fold+1}/5 ---")
        
        # Split
        tr_paths, tr_smri, tr_labels = df_tr.iloc[tr_idx]['matrix_path'].tolist(), smri_tr[tr_idx], df_tr.iloc[tr_idx]['bin_label'].values
        val_paths, val_smri, val_labels = df_tr.iloc[val_idx]['matrix_path'].tolist(), smri_tr[val_idx], df_tr.iloc[val_idx]['bin_label'].values
        
        # Sampler
        counts = np.bincount(tr_labels)
        weights = 1.0 / counts
        sample_weights = weights[tr_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        tr_ds = PCAGDataset(tr_paths, tr_smri, tr_labels)
        val_ds = PCAGDataset(val_paths, val_smri, val_labels)
        te_ds = PCAGDataset(df_te['matrix_path'].tolist(), smri_te, df_te['bin_label'].values)
        
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)
        
        model = PCAGModel(fusion_dim=args.fusion_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
        criterion = nn.CrossEntropyLoss()
        
        best_auc = 0
        patience = 40
        patience_counter = 0
        best_state = None
        
        for epoch in range(args.epochs):
            model.train()
            for x, adj, smri, y in tr_loader:
                x, adj, smri, y = x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x, adj, smri)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            
            # Val
            model.eval()
            val_p = []
            with torch.no_grad():
                for x, adj, smri, y in val_loader:
                    out = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE))
                    val_p.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            
            auc = roc_auc_score(val_labels, val_p)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience: break
        
        print(f"Best Val AUC: {best_auc:.4f}")
        model.load_state_dict(best_state)
        
        # OOF
        model.eval()
        with torch.no_grad():
            fold_p = []
            for x, adj, smri, y in val_loader:
                out = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE))
                fold_p.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            oof_probs[val_idx] = fold_p
            
            # Test inference for this fold
            te_p = []
            for x, adj, smri, y in te_loader:
                out = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE))
                te_p.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            test_probs_folds.append(te_p)

    # --- Final Evaluation ---
    oof_auc = roc_auc_score(df_tr['bin_label'], oof_probs)
    print(f"\nFinal OOF AUC: {oof_auc:.4f}")
    
    # Youden's J for Threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(df_tr['bin_label'], oof_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    
    # Average Test Probs
    avg_test_p = np.mean(test_probs_folds, axis=0)
    test_auc = roc_auc_score(df_te['bin_label'], avg_test_p)
    test_y = df_te['bin_label'].values
    test_pred = (avg_test_p >= best_thresh).astype(int)
    
    cm = confusion_matrix(test_y, test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        "auc": float(test_auc),
        "threshold": float(best_thresh),
        "sens": float(tp / (tp + fn)) if (tp+fn)>0 else 0,
        "spec": float(tn / (tn + fp)) if (tn+fp)>0 else 0,
        "prec": float(tp / (tp + fp)) if (tp+fp)>0 else 0,
        "f1": float(f1_score(test_y, test_pred)),
        "npv": float(tn / (tn + fn)) if (tn+fn)>0 else 0,
        "acc": float(accuracy_score(test_y, test_pred)),
        "cm": cm.tolist()
    }
    
    print("\nPCAG Test Metrics:")
    for k, v in metrics.items(): print(f"  {k}: {v}")
    
    # Compare with Baseline
    try:
        with open(BASE_DIR / "results/kd_comprehensive_metrics.json", "r") as f:
            baseline = json.load(f)[args.task]
        print("\nComparison with Baseline:")
        print(f"  PCAG AUC: {test_auc:.4f}")
        print(f"  KD   AUC: {baseline['kd']['auc']:.4f}")
        print(f"  E5   AUC: {baseline['e5']['auc']:.4f}")
    except:
        print("\nBaseline metrics not found.")

    # Save
    out_file = RES_DIR / f"pcag_{args.task}_results.json"
    with open(out_file, "w") as f:
        json.dump({"oof_auc": oof_auc, "test_metrics": metrics}, f, indent=2)
    print(f"\nResults saved to {out_file}")

    prob_path = RES_DIR / f"pcag_{args.task}_probs.npz"
    np.savez(str(prob_path),
             test_probs=avg_test_p,
             test_labels=df_te['bin_label'].values,
             test_subject_ids=df_te['subject_id'].values,
             oof_probs=oof_probs,
             oof_labels=df_tr['bin_label'].values)
    print(f"[SAVED] {prob_path}")

if __name__ == "__main__":
    main()
