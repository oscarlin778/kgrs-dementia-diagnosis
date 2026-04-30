import os
import re
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
# Settings & Hyperparameters (E2: Feature Alignment)
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
SEED            = 42
K_RATIO         = 0.20
PATIENCE        = 40

# 🌟 E2 Loss Configuration
LAMBDA_CE       = 1.0    # Label loss
LAMBDA_KD       = 0.5    # Knowledge Distillation (Logits)
LAMBDA_CONTRA   = 0.1    # Contrastive Learning (Enabled in E2)
LAMBDA_ALIGN    = 0.1    # Feature Alignment (New in E2)
CONTRA_TEMP     = 0.5
ALIGN_DIM       = 128    # Projection dimension for alignment

# Multi-run Ensemble
SEEDS = [42, 123, 456]

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

POOLING_MAT = torch.zeros(116, N_NETWORKS)
for i, net in enumerate(NETWORK_MAP):
    for node_idx in NETWORK_MAP[net]:
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
    cc = cc.astype(np.float32)
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
        row = adj_z[i].copy()
        row[i] = 0
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
        features.append(np.concatenate([fc_feat, stat_feat, np.array([w_fc, b_fc], dtype=np.float32), topo_feat]))
    return np.stack(features, axis=0).astype(np.float32)

NODE_FEAT_DIM = 116 + 5 + 2 + 2

# ===============================================================
# 2. 模型架構 (E2: 加入 Alignment Projectors)
# ===============================================================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H = num_heads
        self.d = out_dim // num_heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(self.d, 1, bias=False)
        self.a_dst = nn.Linear(self.d, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj, return_attn=False):
        B, N, _ = h.shape
        Wh_flat = self.W(h)
        Wh = Wh_flat.view(B, N, self.H, self.d)
        e = F.leaky_relu(
            self.a_src(Wh).squeeze(-1).unsqueeze(2) +
            self.a_dst(Wh).squeeze(-1).unsqueeze(1),
            negative_slope=0.2
        )
        e = e + adj.unsqueeze(-1) * 0.5
        e = e.masked_fill((adj.abs() < 1e-6).unsqueeze(-1), -1e9)
        alpha_raw = F.softmax(e, dim=2)
        alpha = self.dropout(alpha_raw)
        alpha_t = alpha.permute(0, 3, 1, 2).reshape(B * self.H, N, N)
        Wh_t    = Wh.permute(0, 2, 1, 3).reshape(B * self.H, N, self.d)
        out = torch.bmm(alpha_t, Wh_t).reshape(B, self.H, N, self.d)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.out_dim)
        out = self.bn(out.reshape(B * N, -1)).reshape(B, N, -1)
        result = F.elu(self.dropout(out)) + Wh_flat
        if return_attn:
            node_imp = alpha_raw.sum(dim=1).mean(dim=-1).detach().cpu()
            return result, node_imp
        return result

class FNPGNNv8_E2(nn.Module):
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
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(N_NETWORKS * hidden_dim + hidden_dim, 256),
            nn.ELU(), nn.Dropout(dropout / 2), nn.Linear(256, 2)
        )
        
        # Projector for SupConLoss (1280 -> 128 -> 64)
        self.projector = nn.Sequential(nn.Linear(N_NETWORKS * hidden_dim + hidden_dim, 128), nn.ELU(), nn.Linear(128, 64))
        
        # 🌟 New in E2: Projectors for Feature Alignment (Target: ResNet 512-d)
        self.gnn_align_proj = nn.Sequential(
            nn.Linear(N_NETWORKS * hidden_dim + hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, ALIGN_DIM)
        )
        self.resnet_align_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, ALIGN_DIM)
        )
        
        nn.init.normal_(self.virtual_node_emb, std=0.02)

    def forward(self, x, adj, return_embedding=False, return_attn=False):
        B, N, _ = x.shape
        h = self.bn_input(self.node_encoder(x).reshape(B * N, -1)).reshape(B, N, -1)
        vn = self.virtual_node_emb.expand(B, -1, -1) + h.mean(dim=1, keepdim=True)

        if return_attn: h, imp1 = self.gat1(h, adj, return_attn=True)
        else: h = self.gat1(h, adj)
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))
        h = h + vn.expand(-1, N, -1) * 0.1

        if return_attn: h_new, imp2 = self.gat2(h, adj, return_attn=True)
        else: h_new = self.gat2(h, adj)
        h = h_new + h
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))
        h = h + vn.expand(-1, N, -1) * 0.1

        if return_attn: h_new, imp3 = self.gat3(h, adj, return_attn=True)
        else: h_new = self.gat3(h, adj)
        h = h_new + h

        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))
        pooled = torch.matmul(h.transpose(1, 2), self.pooling_mat).transpose(1, 2)
        pooled = pooled * torch.softmax(self.net_attn(pooled), dim=1)
        flat = self.net_ln(torch.cat([pooled.reshape(B, -1), vn.squeeze(1)], dim=1))

        if return_embedding: return F.normalize(self.projector(flat), dim=-1)
        logits = self.classifier(flat)
        if return_attn:
            node_imp = torch.stack([imp1, imp2, imp3], dim=0).mean(dim=0)
            return logits, flat, node_imp
        return logits, flat

# ===============================================================
# 3. 蒸餾與對比損失函數
# ===============================================================
def distillation_loss(student_logits, teacher_probs):
    student_log_probs = F.log_softmax(student_logits, dim=1)
    return nn.KLDivLoss(reduction="batchmean")(student_log_probs, teacher_probs)

class SupConLoss(nn.Module):
    def __init__(self, temperature=CONTRA_TEMP):
        super().__init__()
        self.temp = temperature
    def forward(self, features, labels):
        B = features.shape[0]
        if B < 2: return torch.tensor(0.0, device=features.device)
        sim = torch.matmul(features, features.T) / self.temp
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)
        if pos_mask.sum() == 0: return torch.tensor(0.0, device=features.device)
        diag_mask = (~torch.eye(B, dtype=torch.bool, device=features.device)).float()
        log_denom = torch.log((torch.exp(sim) * diag_mask).sum(dim=1, keepdim=True) + 1e-8)
        loss = -(pos_mask * (sim - log_denom)).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        return loss.mean()

# ===============================================================
# 4. KD + Alignment 資料集
# ===============================================================
def get_subject_id(path_str):
    basename = os.path.basename(str(path_str))
    clean = re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy|_T1_MNI\.nii\.gz|_T1\.nii\.gz|\.nii\.gz)$', '', basename)
    clean = re.sub(r'^(sub-|sub_|old_dswau)', '', clean)
    return clean.strip()

class fMRIDataset_E2(Dataset):
    def __init__(self, dataframe, teacher_probs=None, teacher_embeds=None):
        self.data_cache = []
        self.has_soft_count = 0
        self.has_embed_count = 0
        
        for _, row in dataframe.iterrows():
            adj_raw = np.load(row['matrix_path'])
            label = row['current_task_label']
            subj_id = get_subject_id(row['matrix_path'])
            
            soft_label = teacher_probs.get(subj_id, None) if teacher_probs else None
            teacher_embed = teacher_embeds.get(subj_id, None) if teacher_embeds else None
            
            has_soft = soft_label is not None
            has_embed = teacher_embed is not None
            
            if has_soft: self.has_soft_count += 1
            if has_embed: self.has_embed_count += 1

            adj_z = np.arctanh(np.clip(adj_raw, -0.999, 0.999))
            x_feat = extract_node_features(adj_z)
            adj_abs = np.abs(adj_z)
            np.fill_diagonal(adj_abs, 0)
            k = int(116 * K_RATIO)
            adj_mask = np.zeros_like(adj_z)
            for i in range(116):
                top_idx = np.argsort(adj_abs[i])[-k:]
                adj_mask[i, top_idx] = adj_z[i, top_idx]
            adj_mask = np.maximum(adj_mask, adj_mask.T)
            np.fill_diagonal(adj_mask, 1.0)
            rowsum = np.abs(adj_mask).sum(1)
            rowsum[rowsum == 0] = 1e-10
            d_mat = np.diag(np.power(rowsum, -0.5))
            adj_norm = d_mat @ adj_mask @ d_mat

            self.data_cache.append({
                'x': torch.FloatTensor(x_feat),
                'adj': torch.FloatTensor(adj_norm),
                'label': torch.tensor(label, dtype=torch.long),
                'soft_label': torch.FloatTensor(soft_label).flip(0) if has_soft else torch.zeros(2),
                'has_soft': torch.tensor(has_soft, dtype=torch.bool),
                'teacher_embed': torch.FloatTensor(teacher_embed) if has_embed else torch.zeros(512),
                'has_embed': torch.tensor(has_embed, dtype=torch.bool),
                'subj_id': subj_id 
            })

    def __len__(self): return len(self.data_cache)
    def __getitem__(self, idx): return self.data_cache[idx]

# ===============================================================
# 5. 訓練任務迴圈
# ===============================================================
def load_teacher_data(task_pair):
    safe_name = f"{task_pair[0]}_vs_{task_pair[1]}"
    
    # 1. Load Logits (Probabilities)
    prob_path = os.path.join(TEACHER_DIR, f"teacher_logits_{safe_name}.npy")
    probs = np.load(prob_path, allow_pickle=True).item() if os.path.exists(prob_path) else {}
    
    # 2. Load Embeddings
    embed_path = os.path.join(TEACHER_DIR, f"teacher_embeddings_{safe_name}.npy")
    embeds = np.load(embed_path, allow_pickle=True).item() if os.path.exists(embed_path) else {}
    
    print(f"  📚 載入老師 ({safe_name}) 資料: {len(probs)} 筆機率, {len(embeds)} 筆 Embeddings。")
    return probs, embeds

def run_task_e2(df_task, task_pair, teacher_probs, teacher_embeds, device, seed, ckpt_dir=None):
    class_a, class_b = task_pair
    labels_arr = df_task['current_task_label'].values
    strata_arr = (
        labels_arr.astype(str) + '_' + df_task['source'].fillna('TPMIC').values
    ) if 'source' in df_task.columns else labels_arr

    torch.manual_seed(seed)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    contra_loss_fn = SupConLoss(temperature=CONTRA_TEMP)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    all_val_prob, all_val_true = [None] * len(df_task), [None] * len(df_task)
    all_val_attn = [None] * len(df_task)
    all_val_flat = [None] * len(df_task)

    seed_best_acc, seed_best_state = 0.0, None

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_task, strata_arr)):
        print(f"    Fold {fold+1}/{N_FOLDS}", end="  ")
        train_df = df_task.iloc[train_idx].reset_index(drop=True)
        val_df = df_task.iloc[val_idx].reset_index(drop=True)

        train_ds = fMRIDataset_E2(train_df, teacher_probs, teacher_embeds)
        val_ds = fMRIDataset_E2(val_df, teacher_probs, teacher_embeds)

        class_counts = np.bincount(train_df['current_task_label'].values)
        weights = [1.0 / class_counts[l] for l in train_df['current_task_label'].values]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1)

        model = FNPGNNv8_E2().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

        best_val_acc, best_state, patience_cnt = 0.0, None, 0

        for epoch in range(EPOCHS):
            model.train()
            for b in train_loader:
                x, adj, lbl = b['x'].to(device), b['adj'].to(device), b['label'].to(device)
                soft_lbl, has_soft = b['soft_label'].to(device), b['has_soft'].to(device)
                t_embed, has_embed = b['teacher_embed'].to(device), b['has_embed'].to(device)

                logits, flat = model(x, adj)
                
                # 1. Classification Loss
                loss_ce = ce_loss_fn(logits, lbl)

                # 2. Knowledge Distillation Loss (Logits)
                loss_kd = torch.tensor(0.0, device=device)
                if has_soft.any():
                    loss_kd = distillation_loss(logits[has_soft], soft_lbl[has_soft])

                # 3. Contrastive Loss (Enabled)
                proj = F.normalize(model.projector(flat), dim=-1)
                loss_contra = contra_loss_fn(proj, lbl)

                # 4. Feature Alignment Loss (New)
                loss_align = torch.tensor(0.0, device=device)
                if has_embed.any():
                    gnn_proj = model.gnn_align_proj(flat[has_embed])
                    res_proj = model.resnet_align_proj(t_embed[has_embed])
                    loss_align = F.mse_loss(gnn_proj, res_proj)

                # Total Loss
                loss = (LAMBDA_CE * loss_ce) + \
                       (LAMBDA_KD * loss_kd) + \
                       (LAMBDA_CONTRA * loss_contra) + \
                       (LAMBDA_ALIGN * loss_align)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            model.eval()
            vt, vp = [], []
            with torch.no_grad():
                for b in val_loader:
                    out, _ = model(b['x'].to(device), b['adj'].to(device))
                    vp.append(out.argmax(dim=1).item())
                    vt.append(b['label'].item())
            val_acc = balanced_accuracy_score(vt, vp)

            if val_acc > best_val_acc:
                best_val_acc, patience_cnt = val_acc, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1

            if patience_cnt >= PATIENCE: break

        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        with torch.no_grad():
            for sample_i, b in zip(val_idx, val_loader):
                out, flat, node_imp = model(b['x'].to(device), b['adj'].to(device), return_attn=True)
                all_val_prob[sample_i] = torch.softmax(out, dim=1).cpu().numpy()[0]
                all_val_true[sample_i] = b['label'].item()
                all_val_attn[sample_i] = node_imp[0].numpy()
                all_val_flat[sample_i] = flat[0].detach().cpu().numpy()

        print(f"best val balanced_acc: {best_val_acc*100:.1f}%")

        if best_val_acc > seed_best_acc:
            seed_best_acc = best_val_acc
            seed_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if ckpt_dir is not None and seed_best_state is not None:
        safe_name = f"{task_pair[0]}_vs_{task_pair[1]}"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"gnn_e2_align_{safe_name}_seed{seed}.pt")
        torch.save(seed_best_state, ckpt_path)
        print(f"  💾 checkpoint → {ckpt_path}")

    return np.stack(all_val_prob, axis=0), all_val_true, np.stack(all_val_attn, axis=0), np.stack(all_val_flat, axis=0)

def run_task_multi_seed(df_full, task_pair, device):
    class_a, class_b = task_pair
    task_name = f"{class_a} vs {class_b}"
    print(f"\n{'='*60}\n  Task: {task_name} (E2 Feature Alignment)\n{'='*60}")

    df_task = df_full[df_full['diagnosis'].isin([class_a, class_b])].copy()
    df_task['current_task_label'] = df_task['diagnosis'].map({class_a: 0, class_b: 1})
    df_task = df_task.reset_index(drop=True)

    teacher_probs, teacher_embeds = load_teacher_data(task_pair)
    tmp_ds = fMRIDataset_E2(df_task, teacher_probs, teacher_embeds)
    print(f"  🔗 共 {len(tmp_ds)} 人，KD 配對 {tmp_ds.has_soft_count} 人，Alignment 配對 {tmp_ds.has_embed_count} 人。")
    del tmp_ds

    ckpt_dir = os.path.join(TEACHER_DIR, "gnn_e2_checkpoints")
    seed_probs, seed_attns, seed_flats = [], [], []
    for seed in SEEDS:
        print(f"\n  [Seed {seed}]")
        probs, all_true, attns, flats = run_task_e2(df_task, task_pair, teacher_probs, teacher_embeds, device, seed, ckpt_dir=ckpt_dir)
        seed_probs.append(probs)
        seed_attns.append(attns)
        seed_flats.append(flats)

    avg_probs = np.mean(seed_probs, axis=0)
    avg_attn  = np.mean(seed_attns, axis=0)
    avg_flats = np.mean(seed_flats, axis=0)
    all_pred = avg_probs.argmax(axis=1).tolist()
    acc = accuracy_score(all_true, all_pred)
    try: auc = roc_auc_score(all_true, avg_probs[:, 1])
    except: auc = float('nan')
    print(f"\n  Multi-seed E2 avg OOF: {acc*100:.1f}% / AUC {auc:.3f}")

    from sklearn.metrics import roc_curve as _roc_curve
    try: _fpr, _tpr, _ = _roc_curve(all_true, avg_probs[:, 1])
    except: _fpr, _tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # Ensemble with Teacher
    subj_ids = df_task['matrix_path'].apply(get_subject_id).tolist()
    best_ens_acc, best_ens_auc, best_w = acc, auc, 1.0
    for gnn_w in [0.9, 0.8, 0.7, 0.6, 0.5]:
        ens_prob = avg_probs.copy()
        for i, sid in enumerate(subj_ids):
            if sid in teacher_probs:
                ens_prob[i] = gnn_w * avg_probs[i] + (1 - gnn_w) * teacher_probs[sid]
        ens_acc = accuracy_score(all_true, ens_prob.argmax(axis=1))
        try: ens_auc = roc_auc_score(all_true, ens_prob[:, 1])
        except: ens_auc = float('nan')
        if ens_acc > best_ens_acc:
            best_ens_acc, best_ens_auc, best_w = ens_acc, ens_auc, gnn_w
    
    # Save Results
    safe_name = f"{class_a}_vs_{class_b}"
    attn_dict = {get_subject_id(df_task.iloc[i]['matrix_path']): {'importance': avg_attn[i], 'label': int(all_true[i]), 'prob': avg_probs[i]} for i in range(len(df_task))}
    np.save(os.path.join(TEACHER_DIR, f"gnn_e2_attention_{safe_name}.npy"), attn_dict, allow_pickle=True)
    
    emb_dict = {get_subject_id(df_task.iloc[i]['matrix_path']): avg_flats[i].astype(np.float32) for i in range(len(df_task))}
    np.save(os.path.join(TEACHER_DIR, f"gnn_e2_embeddings_{safe_name}.npy"), emb_dict, allow_pickle=True)

    return (acc, auc, confusion_matrix(all_true, all_pred), best_ens_acc, best_ens_auc, best_w,
            avg_probs, np.array(all_true), df_task['matrix_path'].tolist(), _fpr, _tpr)

def load_data():
    valid_data, seen_paths = [], set()
    for path in CSV_PATHS:
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m_path = row.get('matrix_path') or (os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy") if pd.notna(row.get('new_id_base')) else None) or (os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy") if pd.notna(row.get('Subject')) else None)
            if not (m_path and os.path.exists(m_path)) or m_path in seen_paths: continue
            try:
                if np.load(m_path).shape != (116, 116): continue
                diag = str(row.get('diagnosis', '')).upper()
                if diag:
                    src = 'ADNI' if ('adni' in m_path.lower() or 'old_dswau' in m_path.lower()) else 'TPMIC'
                    valid_data.append({'matrix_path': m_path, 'diagnosis': diag, 'source': src})
                    seen_paths.add(m_path)
            except: continue
    return pd.DataFrame(valid_data)

def main():
    print("🚀 FNP-GNN E2 (Feature Alignment + Contrastive + KD)")
    print(f"   Loss: CE(1.0) + KD(0.5) + Contra(0.1) + Align(0.1, dim={ALIGN_DIM})")
    
    df_full = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    EXP_NAME = "E2_feature_alignment"
    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    results, exp_metrics, all_oof = {}, {}, {}

    for task in tasks:
        res = run_task_multi_seed(df_full, task, device)
        (acc, auc, cm, ens_acc, ens_auc, best_w, avg_probs, all_true, matrix_paths, fpr, tpr) = res
        results[f"{task[0]} vs {task[1]}"] = (acc, auc, cm, ens_acc, ens_auc, best_w)
        safe = f"{task[0]}_vs_{task[1]}"
        exp_metrics[safe] = {"auc": float(auc), "acc": float(acc), "fpr": fpr.tolist(), "tpr": tpr.tolist()}
        all_oof[safe] = (avg_probs, all_true, matrix_paths)

    # Logging and Saving
    print("\n" + "="*60 + "\n🏆 E2 最終效能總榜單\n" + "="*60)
    for t, (acc, auc, _, e_acc, e_auc, w) in results.items():
        print(f"  {t:<16} GNN: {acc*100:>5.1f}% ({auc:.3f}) | Ens: {e_acc*100:>5.1f}% ({e_auc:.3f}) | w_GNN: {w:.1f}")

    ser.save_metrics(EXP_NAME, exp_metrics)
    ser.update_comparison_chart()

if __name__ == "__main__":
    main()
