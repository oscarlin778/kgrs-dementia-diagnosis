import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
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
TEACHER_DIR = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints"

HIDDEN_DIM      = 128
DROPOUT         = 0.4
LR              = 3e-4
WEIGHT_DECAY    = 5e-3
EPOCHS          = 200
BATCH_SIZE      = 16
N_FOLDS         = 5
K_RATIO         = 0.20
PATIENCE        = 40
ALIGN_DIM       = 128
SEEDS           = [42, 123, 456]

# 我們統一目標維度為 116
TARGET_ROI = 116
NODE_FEAT_DIM = TARGET_ROI + 9 # 116 FC + 5 Stats + 2 Dist + 2 Topo

# ===============================================================
# 1. 網路圖譜與特徵萃取 
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

POOLING_MAT = torch.zeros(TARGET_ROI, N_NETWORKS)
for i, net in enumerate(NETWORK_MAP):
    for node_idx in NETWORK_MAP[net]:
        if node_idx < TARGET_ROI:
            POOLING_MAT[node_idx, i] = 1.0

def extract_node_features(adj_z: np.ndarray) -> np.ndarray:
    N = adj_z.shape[0]
    net_list = list(NETWORK_MAP.keys())
    roi_to_net = {roi: i for i, net in enumerate(net_list) for roi in NETWORK_MAP[net]}
    adj_abs = np.abs(adj_z.copy())
    np.fill_diagonal(adj_abs, 0)
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
                kim = adj_abs_thresh[i, [n for n in net_nodes if n < N]].sum()
                pc_i -= (kim / ki) ** 2
            pc[i] = float(np.clip(pc_i, 0.0, 1.0))
    features = []
    for i in range(N):
        row = adj_z[i].copy(); row[i] = 0
        stat_feat = np.array([row.mean(), row.std(), (row>0).mean(), (row<0).mean(), (np.abs(row)>0.1).sum()], dtype=np.float32)
        net_i = roi_to_net.get(i, -1)
        if net_i >= 0:
            w_nodes = [r for r in NETWORK_MAP[net_list[net_i]] if r != i and r < N]
            b_nodes = [r for r in range(N) if r != i and roi_to_net.get(r, -1) != net_i]
            w_fc = float(np.mean([row[r] for r in w_nodes])) if w_nodes else 0.0
            b_fc = float(np.mean([row[r] for r in b_nodes])) if b_nodes else 0.0
        else:
            w_fc, b_fc = 0.0, 0.0
            
        # 關鍵修復：這裡的 row 可能只有 112 維，但我們需要填充到 TARGET_ROI (116) 維
        fc_padded = np.zeros(TARGET_ROI, dtype=np.float32)
        fc_padded[:N] = row.astype(np.float32)
        
        features.append(np.concatenate([fc_padded, stat_feat, np.array([w_fc, b_fc, cc[i], pc[i]], dtype=np.float32)]))
    
    # 最終的 features 是 (N, 116+9)
    # 我們需要將其垂直填充到 (116, 116+9)
    final_features = np.zeros((TARGET_ROI, NODE_FEAT_DIM), dtype=np.float32)
    final_features[:N, :] = np.stack(features, axis=0)
    return final_features

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.H, self.d = num_heads, out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src, self.a_dst = nn.Linear(self.d, 1, bias=False), nn.Linear(self.d, 1, bias=False)
        self.bn, self.dropout = nn.BatchNorm1d(out_dim), nn.Dropout(dropout)
    def forward(self, h, adj, return_attn=False):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.H, self.d)
        e = F.leaky_relu(self.a_src(Wh).transpose(1, 2) + self.a_dst(Wh).transpose(1, 2).transpose(2, 3), negative_slope=0.2).permute(0, 2, 3, 1)
        e = e + adj.unsqueeze(-1) * 0.5
        e = e.masked_fill((adj.abs() < 1e-6).unsqueeze(-1), -1e9)
        alpha = self.dropout(F.softmax(e, dim=2))
        out = torch.bmm(alpha.permute(0, 3, 1, 2).reshape(B*self.H, N, N), Wh.permute(0, 2, 1, 3).reshape(B*self.H, N, self.d)).view(B, self.H, N, self.d).permute(0, 2, 1, 3).reshape(B, N, -1)
        out = F.elu(self.dropout(self.bn(out.reshape(B*N, -1)).reshape(B, N, -1))) + self.W(h)
        return (out, alpha.sum(dim=1).mean(dim=-1).detach().cpu()) if return_attn else out

class FNPGNN_Ablation(nn.Module):
    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=128, dropout=0.4):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(0.2))
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.gat1 = GATLayer(hidden_dim, hidden_dim); self.gat2 = GATLayer(hidden_dim, hidden_dim); self.gat3 = GATLayer(hidden_dim, hidden_dim)
        self.vn_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ELU())
        self.register_buffer('pooling_mat', POOLING_MAT)
        self.net_attn = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.net_ln = nn.LayerNorm(N_NETWORKS * hidden_dim + hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(N_NETWORKS * hidden_dim + hidden_dim, 256), nn.ELU(), nn.Dropout(dropout/2), nn.Linear(256, 2))
        self.projector = nn.Sequential(nn.Linear(N_NETWORKS * hidden_dim + hidden_dim, 128), nn.ELU(), nn.Linear(128, 64))
        self.gnn_align_proj = nn.Sequential(nn.Linear(N_NETWORKS * hidden_dim + hidden_dim, 256), nn.ELU(), nn.Linear(256, ALIGN_DIM))
        self.resnet_align_proj = nn.Sequential(nn.Linear(512, 256), nn.ELU(), nn.Linear(256, ALIGN_DIM))
        nn.init.normal_(self.virtual_node_emb, std=0.02)
    def forward(self, x, adj, return_attn=False):
        B, N, _ = x.shape
        h = self.bn_input(self.node_encoder(x).reshape(B * N, -1)).reshape(B, N, -1)
        vn = self.virtual_node_emb.expand(B, -1, -1) + h.mean(dim=1, keepdim=True)
        if return_attn: h, imp1 = self.gat1(h, adj, True)
        else: h = self.gat1(h, adj)
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1)); h = h + vn.expand(-1, N, -1) * 0.1
        if return_attn: h, imp2 = self.gat2(h, adj, True)
        else: h = self.gat2(h, adj)
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1)); h = h + vn.expand(-1, N, -1) * 0.1
        if return_attn: h, imp3 = self.gat3(h, adj, True)
        else: h = self.gat3(h, adj)
        pooled = torch.matmul(h.transpose(1, 2), self.pooling_mat).transpose(1, 2)
        flat = self.net_ln(torch.cat([(pooled * torch.softmax(self.net_attn(pooled), dim=1)).reshape(B, -1), vn.squeeze(1)], dim=1))
        return (self.classifier(flat), flat, torch.stack([imp1, imp2, imp3], dim=0).mean(dim=0)) if return_attn else (self.classifier(flat), flat)

def distillation_loss(student_logits, teacher_probs):
    return nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_logits, dim=1), teacher_probs)

class SupConLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp
    def forward(self, features, labels):
        B = features.shape[0]
        if B < 2: return torch.tensor(0.0, device=features.device)
        sim = torch.matmul(features, features.T) / self.temp
        pos_mask = (labels.view(-1, 1) == labels.view(1, -1)).float(); pos_mask.fill_diagonal_(0)
        if pos_mask.sum() == 0: return torch.tensor(0.0, device=features.device)
        log_denom = torch.log((torch.exp(sim) * (1 - torch.eye(B, device=features.device))).sum(dim=1, keepdim=True) + 1e-8)
        return (-(pos_mask * (sim - log_denom)).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)).mean()

def get_subject_id(path_str):
    return re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy|_T1_MNI\.nii\.gz|_T1\.nii\.gz|\.nii\.gz)$', '', os.path.basename(str(path_str))).replace('sub-', '').replace('sub_', '').replace('old_dswau', '').strip()

class fMRIDataset_Ablation(Dataset):
    def __init__(self, dataframe, teacher_probs=None, teacher_embeds=None):
        self.data_cache = []
        for _, row in dataframe.iterrows():
            adj_raw = np.load(row['matrix_path']); label = row['current_task_label']; subj_id = get_subject_id(row['matrix_path'])
            soft = teacher_probs.get(subj_id, None); embed = teacher_embeds.get(subj_id, None)
            adj_z = np.arctanh(np.clip(adj_raw, -0.999, 0.999))
            N = adj_z.shape[0]
            
            # 特徵萃取 (已包含 padding 到 TARGET_ROI)
            x_feat = extract_node_features(adj_z)
            
            # 鄰接矩陣處理
            adj_abs = np.abs(adj_z); np.fill_diagonal(adj_abs, 0); k = int(N * K_RATIO); adj_mask = np.zeros_like(adj_z)
            for i in range(N):
                top_idx = np.argsort(adj_abs[i])[-k:]
                adj_mask[i, top_idx] = adj_z[i, top_idx]
            adj_mask = np.maximum(adj_mask, adj_mask.T); np.fill_diagonal(adj_mask, 1.0)
            rowsum = np.abs(adj_mask).sum(1); rowsum[rowsum == 0] = 1e-10; d_inv = np.diag(np.power(rowsum, -0.5))
            adj_norm_small = d_inv @ adj_mask @ d_inv
            
            # 鄰接矩陣 Padding 到 TARGET_ROI x TARGET_ROI
            adj_norm = np.zeros((TARGET_ROI, TARGET_ROI), dtype=np.float32)
            adj_norm[:N, :N] = adj_norm_small
            # 沒用到的 ROI 也要有自環
            for i in range(N, TARGET_ROI): adj_norm[i, i] = 1.0
            
            self.data_cache.append({
                'x': torch.FloatTensor(x_feat), 'adj': torch.FloatTensor(adj_norm), 'label': torch.tensor(label, dtype=torch.long),
                'soft_label': torch.FloatTensor(soft).flip(0) if soft is not None else torch.zeros(2), 'has_soft': torch.tensor(soft is not None, dtype=torch.bool),
                'teacher_embed': torch.FloatTensor(embed) if embed is not None else torch.zeros(512), 'has_embed': torch.tensor(embed is not None, dtype=torch.bool),
                'teacher_conf': torch.tensor(np.max(soft) if soft is not None else 1.0, dtype=torch.float32)
            })
    def __len__(self): return len(self.data_cache)
    def __getitem__(self, idx): return self.data_cache[idx]

def run_task(df_task, task_pair, teacher_probs, teacher_embeds, device, seed, args):
    torch.manual_seed(seed); np.random.seed(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    contra_loss_fn = SupConLoss(); ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    all_prob, all_true = [None]*len(df_task), [None]*len(df_task)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_task, df_task['current_task_label'])):
        print(f"    Fold {fold+1}", end=" ")
        train_ds = fMRIDataset_Ablation(df_task.iloc[train_idx], teacher_probs, teacher_embeds)
        val_ds = fMRIDataset_Ablation(df_task.iloc[val_idx], teacher_probs, teacher_embeds)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=WeightedRandomSampler([1.0/np.bincount(df_task.iloc[train_idx]['current_task_label'])[l] for l in df_task.iloc[train_idx]['current_task_label']], len(train_idx)), drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1)

        model = FNPGNN_Ablation().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, eta_min=1e-6)

        best_acc, best_state, patience = 0.0, None, 0
        for epoch in range(EPOCHS):
            model.train()
            for b in train_loader:
                x, adj, lbl = b['x'].to(device), b['adj'].to(device), b['label'].to(device)
                soft_lbl, has_soft = b['soft_label'].to(device), b['has_soft'].to(device)
                t_embed, has_embed, t_conf = b['teacher_embed'].to(device), b['has_embed'].to(device), b['teacher_conf'].to(device)
                logits, flat = model(x, adj)
                loss_ce = ce_loss_fn(logits, lbl)
                loss_kd = distillation_loss(logits[has_soft], soft_lbl[has_soft]) if has_soft.any() else torch.tensor(0.0, device=device)
                loss_contra = contra_loss_fn(F.normalize(model.projector(flat), dim=-1), lbl)
                loss_align = torch.tensor(0.0, device=device)
                if has_embed.any():
                    mses = F.mse_loss(model.gnn_align_proj(flat[has_embed]), model.resnet_align_proj(t_embed[has_embed]), reduction='none').mean(dim=1)
                    if args.dynamic_mci: mses = mses * t_conf[has_embed]
                    loss_align = mses.mean()
                loss = (1.0 * loss_ce) + (0.5 * loss_kd) + (args.lambda_contra * loss_contra) + (args.lambda_align * loss_align)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            scheduler.step()
            model.eval(); vt, vp = [], []
            with torch.no_grad():
                for b in val_loader:
                    out, _ = model(b['x'].to(device), b['adj'].to(device))
                    vp.append(out.argmax(1).item()); vt.append(b['label'].item())
            acc = balanced_accuracy_score(vt, vp)
            if acc > best_acc: best_acc, best_state, patience = acc, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
            else: patience += 1
            if patience >= PATIENCE: break
        
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            for idx, b in zip(val_idx, val_loader):
                out, _ = model(b['x'].to(device), b['adj'].to(device))
                all_prob[idx] = torch.softmax(out, 1).cpu().numpy()[0]; all_true[idx] = b['label'].item()
        print(f"Acc: {best_acc*100:.1f}%")
    return np.stack(all_prob), np.array(all_true)

def main():
    print("🚀 SCRIPT STARTING...", flush=True)
    sys.stdout.flush()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_align", type=float, default=0.1)
    parser.add_argument("--lambda_contra", type=float, default=0.1)
    parser.add_argument("--dynamic_mci", action="store_true")
    parser.add_argument("--exp_suffix", type=str, default="")
    args = parser.parse_args()
    exp_name = f"E2_align_{args.lambda_align}_contra_{args.lambda_contra}"
    if args.dynamic_mci: exp_name += "_dynamic"
    if args.exp_suffix: exp_name = f"E2_{args.exp_suffix}"
    
    valid_data = []
    for path in CSV_PATHS:
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m_path = row.get('matrix_path') or (os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy") if pd.notna(row.get('new_id_base')) else None) or (os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy") if pd.notna(row.get('Subject')) else None)
            if m_path and os.path.exists(m_path):
                diag = str(row.get('diagnosis', '')).upper()
                if diag in ['NC', 'AD', 'MCI']: valid_data.append({'matrix_path': m_path, 'diagnosis': diag})
    df_full = pd.DataFrame(valid_data).drop_duplicates('matrix_path').reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    exp_metrics = {}
    for task in tasks:
        print(f"\nTask: {task[0]} vs {task[1]}")
        df_task = df_full[df_full['diagnosis'].isin(task)].copy()
        df_task['current_task_label'] = df_task['diagnosis'].map({task[0]: 0, task[1]: 1})
        df_task = df_task.reset_index(drop=True)
        safe = f"{task[0]}_vs_{task[1]}"
        probs = np.load(os.path.join(TEACHER_DIR, f"teacher_logits_{safe}.npy"), allow_pickle=True).item()
        embeds = np.load(os.path.join(TEACHER_DIR, f"teacher_embeddings_{safe}.npy"), allow_pickle=True).item()
        seed_probs = []
        for seed in SEEDS:
            print(f"  Seed {seed}")
            p, t = run_task(df_task, task, probs, embeds, device, seed, args)
            seed_probs.append(p)
        avg_p = np.mean(seed_probs, axis=0); acc = accuracy_score(t, avg_p.argmax(1)); auc = roc_auc_score(t, avg_p[:, 1])
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(t, avg_p[:, 1])
        exp_metrics[safe] = {"auc": float(auc), "acc": float(acc), "fpr": fpr.tolist(), "tpr": tpr.tolist()}
    ser.save_metrics(f"E2_feature_alignment/{exp_name}", exp_metrics)
    ser.update_comparison_chart()

if __name__ == "__main__": main()
