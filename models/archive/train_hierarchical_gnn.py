import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# ===============================================================
# Settings
# ===============================================================
CSV_PATHS = [
    "/home/wei-chi/Model/_dataset_mapping.csv",
    "/home/wei-chi/Data/dataset_index_116_clean_old.csv",
    "/home/wei-chi/Data/adni_dataset_index_116.csv"
]
MATRIX_DIR = "/home/wei-chi/Model/processed_116_matrices"

HIDDEN_DIM      = 128
N_HEADS         = 4
DROPOUT         = 0.4
LR              = 3e-4
WEIGHT_DECAY    = 5e-3
EPOCHS          = 200
BATCH_SIZE      = 16
N_FOLDS         = 5
SEED            = 42
K_RATIO         = 0.20
PATIENCE        = 40

# Loss weights
LAMBDA_CE       = 1.0    # Cross-entropy loss weight
LAMBDA_CONTRA   = 0.5    # Contrastive loss weight
MIXUP_ALPHA     = 0.3    # Mixup interpolation strength (0 = off)
CONTRA_TEMP     = 0.5    # Contrastive learning temperature

# ===============================================================
# 1. Network map
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

# ===============================================================
# 2. Node feature extraction (123-dim)
# ===============================================================
def extract_node_features(adj_z: np.ndarray) -> np.ndarray:
    N = adj_z.shape[0]
    net_list = list(NETWORK_MAP.keys())
    roi_to_net = {}
    for net_i, net in enumerate(net_list):
        for roi in NETWORK_MAP[net]:
            roi_to_net[roi] = net_i

    features = []
    for i in range(N):
        row = adj_z[i].copy()
        row[i] = 0

        fc_feat    = row.astype(np.float32)
        mean_fc    = float(row.mean())
        std_fc     = float(row.std())
        pos_ratio  = float((row > 0).mean())
        neg_ratio  = float((row < 0).mean())
        degree     = float((np.abs(row) > 0.1).sum())
        stat_feat  = np.array([mean_fc, std_fc, pos_ratio, neg_ratio, degree],
                              dtype=np.float32)

        net_i = roi_to_net.get(i, -1)
        if net_i >= 0:
            within_nodes  = [r for r in NETWORK_MAP[net_list[net_i]] if r != i]
            between_nodes = [r for r in range(N)
                             if r != i and roi_to_net.get(r, -1) != net_i]
            within_fc  = float(np.mean([row[r] for r in within_nodes])) if within_nodes else 0.0
            between_fc = float(np.mean([row[r] for r in between_nodes])) if between_nodes else 0.0
        else:
            within_fc, between_fc = 0.0, 0.0
        net_feat = np.array([within_fc, between_fc], dtype=np.float32)

        features.append(np.concatenate([fc_feat, stat_feat, net_feat]))

    return np.stack(features, axis=0).astype(np.float32)

NODE_FEAT_DIM = 116 + 5 + 2  # 123

# ===============================================================
# 3. Graph Mixup
#    在兩個樣本之間插值，製造虛擬訓練樣本
# ===============================================================
def graph_mixup(x1, adj1, x2, adj2, lam):
    """
    lam ~ Beta(alpha, alpha) 控制插值比例
    x   : (B, N, F)
    adj : (B, N, N)
    """
    x_mix   = lam * x1   + (1 - lam) * x2
    adj_mix = lam * adj1 + (1 - lam) * adj2
    return x_mix, adj_mix

# ===============================================================
# 4. Graph Augmentation
# ===============================================================
def augment_graph(adj, drop_edge_prob=0.10, jitter_std=0.05):
    adj  = adj.clone()
    mask = torch.bernoulli(
        torch.full(adj.shape, 1 - drop_edge_prob, device=adj.device)
    )
    adj  = adj * mask
    noise = torch.randn_like(adj) * jitter_std
    adj  = adj + noise * (adj != 0).float()
    idx  = torch.arange(adj.shape[-1], device=adj.device)
    adj[:, idx, idx] = 1.0
    return adj

# ===============================================================
# 5. GAT Layer
# ===============================================================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.dropout  = nn.Dropout(dropout)
        self.W  = nn.Linear(in_dim, out_dim, bias=False)
        self.a  = nn.Parameter(torch.zeros(1, n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, h, adj):
        B, N, _ = h.shape
        Wh   = self.W(h).view(B, N, self.n_heads, self.head_dim)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1, -1)
        cat  = torch.cat([Wh_i, Wh_j], dim=-1)
        e    = (cat * self.a.unsqueeze(0).unsqueeze(0)).sum(-1)
        e    = F.leaky_relu(e, 0.2)
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e    = e.masked_fill(mask, float('-inf'))
        attn = F.softmax(e, dim=2)
        attn = self.dropout(attn)
        out  = (attn.unsqueeze(-1) * Wh_j).sum(2).reshape(B, N, -1)
        out  = self.bn(out.reshape(B * N, -1)).reshape(B, N, -1)
        return F.elu(out) + self.W(h)

# ===============================================================
# 6. FNP-GNN v4
#    新增：Virtual Node + Projection Head for Contrastive Learning
# ===============================================================
class FNPGNNv4(nn.Module):
    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
        )
        self.bn_input = nn.BatchNorm1d(hidden_dim)

        # Virtual node embedding（全局腦網路整合節點）
        # virtual node 會與所有真實節點雙向連接
        self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.virtual_node_emb.view(1, -1).unsqueeze(0))

        # GAT layers（真實節點）
        self.gat1 = GATLayer(hidden_dim, hidden_dim, n_heads, dropout=0.2)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, n_heads, dropout=0.2)

        # Virtual node update（從所有節點聚合到 virtual node）
        self.vn_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
        )

        # Hierarchical pooling
        self.register_buffer('pooling_mat', POOLING_MAT)

        # Network attention
        self.net_attn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Layer norm — 維度 = network pooling + virtual node = K*D + D
        self.net_ln = nn.LayerNorm(N_NETWORKS * hidden_dim + hidden_dim)

        # Classifier
        # input = network pooling + virtual node
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(N_NETWORKS * hidden_dim + hidden_dim, 256),
            nn.ELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 2)
        )

        # Projection head for contrastive learning
        # 將 embedding 投影到更低維空間做對比學習
        self.projector = nn.Sequential(
            nn.Linear(N_NETWORKS * hidden_dim + hidden_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64)
        )

        self.apply(self._init_weights)
        # virtual node 單獨初始化
        nn.init.normal_(self.virtual_node_emb, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, adj, augment=False, return_embedding=False):
        B, N, _ = x.shape

        if augment:
            adj = augment_graph(adj, drop_edge_prob=0.10, jitter_std=0.05)

        # Node encoding
        h = self.node_encoder(x)
        h = self.bn_input(h.reshape(B * N, -1)).reshape(B, N, -1)

        # Virtual node 初始值：所有節點平均
        vn = self.virtual_node_emb.expand(B, -1, -1)  # (B, 1, D)
        vn = vn + h.mean(dim=1, keepdim=True)

        # GAT layer 1 + virtual node update
        h  = self.gat1(h, adj)
        # Virtual node 聚合所有節點的訊息
        vn_new = self.vn_update(
            torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1)
        )
        # Virtual node 的資訊廣播回所有節點
        h  = h + vn_new.expand(-1, N, -1) * 0.1

        # GAT layer 2 + virtual node update
        h2 = self.gat2(h, adj)
        h  = h2 + h   # skip

        vn = self.vn_update(
            torch.cat([vn_new, h.mean(dim=1, keepdim=True)], dim=-1)
        )

        # Hierarchical pooling
        pooled = torch.matmul(
            h.transpose(1, 2), self.pooling_mat
        ).transpose(1, 2)  # (B, K, D)

        # Network attention
        net_w  = torch.softmax(self.net_attn(pooled), dim=1)
        pooled = pooled * net_w

        # Flatten + concat virtual node
        flat = torch.cat([
            pooled.reshape(B, -1),   # (B, K*D)
            vn.squeeze(1)             # (B, D)
        ], dim=1)                     # (B, K*D + D)

        flat = self.net_ln(flat)

        if return_embedding:
            proj = F.normalize(self.projector(flat), dim=-1)
            return proj

        logits = self.classifier(flat)
        return logits, flat

# ===============================================================
# 7. Supervised Contrastive Loss
#    同類別 embedding 靠近，不同類別推遠
# ===============================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=CONTRA_TEMP):
        super().__init__()
        self.temp = temperature

    def forward(self, features, labels):
        """
        features : (B, D) L2-normalized embeddings
        labels   : (B,)
        """
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temp  # (B, B)

        # Mask: same class pairs (excluding diagonal)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)

        # 若沒有正樣本對（batch 內只有一個類別），跳過
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # Exclude diagonal from denominator
        diag_mask = (~torch.eye(B, dtype=torch.bool, device=features.device)).float()

        exp_sim = torch.exp(sim) * diag_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        return loss.mean()

# ===============================================================
# 8. Dataset
# ===============================================================
class fMRIDataset(Dataset):
    def __init__(self, dataframe):
        self.data_cache = []
        for _, row in dataframe.iterrows():
            adj_raw = np.load(row['matrix_path'])
            label   = row['current_task_label']

            adj_z  = np.arctanh(np.clip(adj_raw, -0.999, 0.999))
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
            rowsum   = np.abs(adj_mask).sum(1)
            rowsum[rowsum == 0] = 1e-10
            d_mat    = np.diag(np.power(rowsum, -0.5))
            adj_norm = d_mat @ adj_mask @ d_mat

            self.data_cache.append({
                'x':     torch.FloatTensor(x_feat),
                'adj':   torch.FloatTensor(adj_norm),
                'label': torch.tensor(label, dtype=torch.long)
            })

    def __len__(self): return len(self.data_cache)
    def __getitem__(self, idx): return self.data_cache[idx]

# ===============================================================
# 9. Feature importance (net_attn visualization)
# ===============================================================
def plot_feature_importance(model, val_loader, task_pair, task_name, device):
    try:
        networks  = list(NETWORK_MAP.keys())
        class_a, class_b = task_pair
        attn_class0, attn_class1 = [], []

        model.eval()
        with torch.no_grad():
            for b in val_loader:
                x   = b['x'].to(device)
                adj = b['adj'].to(device)
                lbl = b['label'].item()

                h   = model.node_encoder(x)
                B, N, _ = h.shape
                h   = model.bn_input(h.reshape(B * N, -1)).reshape(B, N, -1)
                vn  = model.virtual_node_emb.expand(B, -1, -1)
                vn  = vn + h.mean(dim=1, keepdim=True)
                h   = model.gat1(h, adj)
                vn  = model.vn_update(
                    torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1)
                )
                h   = h + vn.expand(-1, N, -1) * 0.1
                h2  = model.gat2(h, adj)
                h   = h2 + h
                pooled = torch.matmul(
                    h.transpose(1, 2), model.pooling_mat
                ).transpose(1, 2)
                net_w = torch.softmax(
                    model.net_attn(pooled), dim=1
                ).squeeze().cpu().numpy()

                if lbl == 0:
                    attn_class0.append(net_w)
                else:
                    attn_class1.append(net_w)

        mean0 = np.mean(attn_class0, axis=0) if attn_class0 else np.zeros(N_NETWORKS)
        mean1 = np.mean(attn_class1, axis=0) if attn_class1 else np.zeros(N_NETWORKS)
        diff  = np.abs(mean1 - mean0)

        x_pos = np.arange(N_NETWORKS)
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        bars0 = ax.bar(x_pos - width/2, mean0, width,
                       label=class_a, color='#5DADE2', edgecolor='black')
        bars1 = ax.bar(x_pos + width/2, mean1, width,
                       label=class_b, color='#E74C3C', edgecolor='black')
        for i in np.argsort(diff)[-2:]:
            bars0[i].set_edgecolor('gold'); bars0[i].set_linewidth(2.5)
            bars1[i].set_edgecolor('gold'); bars1[i].set_linewidth(2.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(networks, fontsize=11)
        ax.set_ylabel('Mean Network Attention Weight', fontsize=12)
        ax.set_xlabel('Functional Networks', fontsize=12)
        ax.set_title(
            f'FNP-GNN v4 Network Attention ({task_name})\n'
            f'(Gold border = top-2 networks with largest class difference)',
            fontsize=13, fontweight='bold'
        )
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        print(f"\n  [Feature Importance] Networks ranked by class difference:")
        for i in np.argsort(diff)[::-1]:
            print(f"    {networks[i]:6s}  {class_a}={mean0[i]:.4f}  "
                  f"{class_b}={mean1[i]:.4f}  diff={diff[i]:.4f}")

        safe = task_name.replace(" ", "_")
        plt.tight_layout()
        plt.savefig(f'gnn_v4_feature_importance_{safe}.png', dpi=300)
        plt.close()
        print(f"  Saved: gnn_v4_feature_importance_{safe}.png")

    except Exception as e:
        import traceback
        print(f"  Feature importance failed: {e}")
        traceback.print_exc()

# ===============================================================
# 10. Task runner (5-Fold + Mixup + Contrastive + Ensemble)
# ===============================================================
def run_task(df_full, task_pair, device):
    class_a, class_b = task_pair
    task_name = f"{class_a} vs {class_b}"
    print(f"\n{'='*60}")
    print(f"  Task: {task_name}")
    print(f"{'='*60}")

    df_task = df_full[df_full['diagnosis'].isin([class_a, class_b])].copy()
    df_task['current_task_label'] = df_task['diagnosis'].map({class_a: 0, class_b: 1})
    df_task = df_task.reset_index(drop=True)

    labels_arr = df_task['current_task_label'].values
    for cls, cnt in zip([class_a, class_b], np.bincount(labels_arr)):
        print(f"  {cls}: {cnt}")
    print(f"  Total: {len(df_task)} | node_feat_dim: {NODE_FEAT_DIM}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    contra_loss_fn = SupConLoss(temperature=CONTRA_TEMP)

    # Ensemble: 每個 fold 保留最佳模型，最後 soft-vote
    fold_models   = []
    fold_val_data = []   # (val_dataset, val_indices)
    all_val_idx   = []
    all_val_prob  = [None] * len(df_task)
    all_val_true  = [None] * len(df_task)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_task, labels_arr)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}")
        train_df = df_task.iloc[train_idx].reset_index(drop=True)
        val_df   = df_task.iloc[val_idx].reset_index(drop=True)

        train_labels   = train_df['current_task_label'].values
        class_counts   = np.bincount(train_labels)
        sample_weights = [1.0 / class_counts[l] for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                        replacement=True)

        train_ds = fMRIDataset(train_df)
        val_ds   = fMRIDataset(val_df)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=1)

        model     = FNPGNNv4().to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        cw = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
        ce_loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

        best_val_acc = 0.0
        best_state   = None
        patience_cnt = 0

        for epoch in range(EPOCHS):
            model.train()
            for b in train_loader:
                x   = b['x'].to(device)
                adj = b['adj'].to(device)
                lbl = b['label'].to(device)
                B   = x.shape[0]

                # -- Graph Mixup --
                if MIXUP_ALPHA > 0 and B >= 2:
                    lam   = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    perm  = torch.randperm(B, device=device)
                    x_mix, adj_mix = graph_mixup(x, adj, x[perm], adj[perm], lam)
                    lbl_a, lbl_b   = lbl, lbl[perm]
                else:
                    x_mix, adj_mix = x, adj
                    lbl_a, lbl_b, lam = lbl, lbl, 1.0

                # Forward
                logits, _ = model(x_mix, adj_mix, augment=True)

                # Mixup CE loss
                loss_ce = (lam * ce_loss_fn(logits, lbl_a) +
                           (1 - lam) * ce_loss_fn(logits, lbl_b))

                # Contrastive loss（用原始未 mixup 的樣本）
                proj = model(x, adj, augment=False, return_embedding=True)
                loss_contra = contra_loss_fn(proj, lbl)

                loss = LAMBDA_CE * loss_ce + LAMBDA_CONTRA * loss_contra

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # Validation
            model.eval()
            vt, vp = [], []
            with torch.no_grad():
                for b in val_loader:
                    out, _ = model(b['x'].to(device), b['adj'].to(device),
                                   augment=False)
                    vp.append(out.argmax(dim=1).item())
                    vt.append(b['label'].item())
            val_acc = accuracy_score(vt, vp)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_cnt = 0
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1

            if (epoch + 1) % 40 == 0:
                print(f"    Epoch {epoch+1:3d}/{EPOCHS} | "
                      f"val: {val_acc*100:.1f}% | best: {best_val_acc*100:.1f}%")

            if patience_cnt >= PATIENCE:
                print(f"    Early stop @ epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        # Collect soft probabilities for ensemble
        with torch.no_grad():
            for sample_i, b in zip(val_idx, val_loader):
                out, _ = model(b['x'].to(device), b['adj'].to(device),
                               augment=False)
                prob = torch.softmax(out, dim=1).cpu().numpy()[0]
                all_val_prob[sample_i] = prob
                all_val_true[sample_i] = b['label'].item()

        fold_models.append(model)
        fold_val_data.append((val_loader, val_idx))
        print(f"  Fold {fold+1} best val acc: {best_val_acc*100:.1f}%")

    # -- Ensemble soft voting --
    all_true = [all_val_true[i] for i in range(len(df_task))]
    all_prob_arr = np.stack(all_val_prob, axis=0)          # (N, 2)
    all_pred = all_prob_arr.argmax(axis=1).tolist()
    all_prob1 = all_prob_arr[:, 1].tolist()

    acc = accuracy_score(all_true, all_pred)
    cm  = confusion_matrix(all_true, all_pred)
    try:
        auc = roc_auc_score(all_true, all_prob1)
    except ValueError:
        auc = float('nan')

    # Feature importance from last fold
    last_val_loader, _ = fold_val_data[-1]
    plot_feature_importance(fold_models[-1], last_val_loader,
                            task_pair, task_name, device)

    return acc, auc, cm

# ===============================================================
# 11. Data loader
# ===============================================================
def load_data():
    valid_data, seen_paths = [], set()
    for path in CSV_PATHS:
        if not os.path.exists(path):
            print(f"  Skipping: {path}")
            continue
        print(f"  Reading: {os.path.basename(path)}")
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m_path = None
            if 'matrix_path' in row and pd.notna(row['matrix_path']):
                m_path = row['matrix_path']
            elif 'new_id_base' in row and pd.notna(row['new_id_base']):
                m_path = os.path.join(MATRIX_DIR,
                                      f"{row['new_id_base']}_matrix_116.npy")
            elif 'Subject' in row and pd.notna(row['Subject']):
                m_path = os.path.join(MATRIX_DIR,
                                      f"{row['Subject']}_matrix_116.npy")
            if not (m_path and os.path.exists(m_path)): continue
            if m_path in seen_paths: continue
            try:
                mat = np.load(m_path)
                if mat.shape != (116, 116): continue
                diag = str(row.get('diagnosis', '')).upper()
                if not diag: continue
                valid_data.append({'matrix_path': m_path, 'diagnosis': diag})
                seen_paths.add(m_path)
            except Exception:
                continue
    return pd.DataFrame(valid_data)

# ===============================================================
# 12. Main
# ===============================================================
def main():
    print("FNP-GNN v4")
    print("  New: Virtual Node + Supervised Contrastive Loss + Graph Mixup + Ensemble")
    print(f"  Hidden={HIDDEN_DIM} | Heads={N_HEADS} | Epochs={EPOCHS}")
    print(f"  Lambda_CE={LAMBDA_CE} | Lambda_Contra={LAMBDA_CONTRA} | "
          f"Mixup_alpha={MIXUP_ALPHA}\n")

    df_full = load_data()
    print(f"\nLoaded {len(df_full)} valid samples")
    if len(df_full) == 0:
        print("No data found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    tasks   = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('FNP-GNN v4 (fMRI) - Confusion Matrices',
                 fontsize=16, fontweight='bold')

    for idx, task in enumerate(tasks):
        acc, auc, cm = run_task(df_full, task, device)
        key = f"{task[0]} vs {task[1]}"
        results[key] = (acc, auc, cm)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=[task[0], task[1]],
                    yticklabels=[task[0], task[1]],
                    annot_kws={"size": 16})
        axes[idx].set_title(
            f"{task[0]} vs {task[1]}\nAcc: {acc*100:.1f}%  AUC: {auc:.3f}",
            fontsize=14)
        axes[idx].set_ylabel('True Label', fontsize=12)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)

    print("\n" + "="*60)
    print("FNP-GNN v4 Results")
    print("="*60)
    print(f"{'Task':<16} {'Accuracy':>9} {'AUC':>8}")
    print("-"*36)
    for task_name, (acc, auc, _) in results.items():
        flag = "OK" if acc >= 0.80 else "--"
        print(f"[{flag}] {task_name:<14} {acc*100:>8.1f}%  {auc:>7.3f}")

    plt.tight_layout()
    plt.savefig('fmri_gnn_v4_confusion_matrices.png', dpi=300)
    print("\nSaved: fmri_gnn_v4_confusion_matrices.png")


if __name__ == "__main__":
    main()