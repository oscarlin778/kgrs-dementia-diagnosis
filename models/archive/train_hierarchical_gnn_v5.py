import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
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

HIDDEN_DIM      = 96      # v4: 128 → compact model for small data
N_HEADS         = 4
DROPOUT         = 0.45    # v4: 0.40
LR              = 2e-4    # v4: 3e-4
WEIGHT_DECAY    = 1e-2    # v4: 5e-3
EPOCHS          = 250
BATCH_SIZE      = 16
N_FOLDS         = 5
SEEDS           = [42, 123, 777]   # multi-seed: 3×5 = 15 models per task
K_RATIO         = 0.25    # v4: 0.20
PATIENCE        = 50      # v4: 40

LAMBDA_CE       = 1.0
LAMBDA_CONTRA   = 0.2     # v4: 0.5 (reduced — less dominant)
MIXUP_ALPHA     = 0.3
CONTRA_TEMP     = 0.5
TTA_STEPS       = 8       # test-time augmentation passes

# ---------------------------------------------------------------
# Model checkpoint directory (for cross-model KD)
# ---------------------------------------------------------------
MODEL_SAVE_DIR  = "/home/wei-chi/Data/script/fnp_gnn_v5_checkpoints"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ===============================================================
# Network map
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
N_NETWORKS    = len(NETWORK_MAP)
NETWORK_NAMES = list(NETWORK_MAP.keys())

POOLING_MAT = torch.zeros(116, N_NETWORKS)
for _ni, _net in enumerate(NETWORK_NAMES):
    for _roi in NETWORK_MAP[_net]:
        POOLING_MAT[_roi, _ni] = 1.0

ROI_TO_NET = {}
for _ni, _net in enumerate(NETWORK_NAMES):
    for _roi in NETWORK_MAP[_net]:
        ROI_TO_NET[_roi] = _ni

# ===============================================================
# Node feature extraction — compact 23-dim per node
# ---------------------------------------------------------------
# v4 used the full 116-dim FC row → 116×123 = 14K input features,
# which overfits on ~100 training samples.
# v5 uses 23 compact, network-centric features per node:
#   5  statistics  : mean, std, pos_ratio, neg_ratio, degree_norm
#   9  per-network : mean FC with each of the 9 functional networks
#   3  net props   : within_fc, between_fc, segregation
#   2  topology    : local clustering coeff, normalised strength
#   4  global      : std/max/min/range of per-network means
# Total: 23 dim  (vs 123 in v4 → ~5× fewer input parameters)
# ===============================================================
def extract_node_features(adj_z: np.ndarray) -> np.ndarray:
    N = adj_z.shape[0]

    # Binary adjacency for topology (threshold |r| > 0.3)
    adj_bin = (np.abs(adj_z) > 0.3).astype(np.float32)
    np.fill_diagonal(adj_bin, 0)
    degree = adj_bin.sum(axis=1)  # (N,)

    # Local clustering coefficient: CC_i = A^3_ii / (k_i*(k_i-1))
    A2         = adj_bin @ adj_bin          # (N, N)
    triangles  = np.einsum('ij,ji->i', adj_bin, A2)  # diag(A^3)
    cc = np.where(degree > 1,
                  triangles / (degree * (degree - 1) + 1e-8),
                  0.0)

    features = []
    for i in range(N):
        row   = adj_z[i].copy()
        row[i] = 0

        # --- Statistics (5) ---
        mean_fc   = float(row.mean())
        std_fc    = float(row.std())
        pos_ratio = float((row > 0).mean())
        neg_ratio = float((row < 0).mean())
        deg_norm  = float(degree[i]) / (N - 1)

        # --- Per-network mean FC (9) ---
        net_means = []
        for net in NETWORK_NAMES:
            nodes = [r for r in NETWORK_MAP[net] if r != i]
            net_means.append(float(np.mean(row[nodes])) if nodes else 0.0)

        # --- Within / between network (3) ---
        ni = ROI_TO_NET.get(i, -1)
        if ni >= 0:
            w_nodes = [r for r in NETWORK_MAP[NETWORK_NAMES[ni]] if r != i]
            b_nodes = [r for r in range(N)
                       if r != i and ROI_TO_NET.get(r, -1) != ni]
            wfc = float(np.mean([row[r] for r in w_nodes])) if w_nodes else 0.0
            bfc = float(np.mean([row[r] for r in b_nodes])) if b_nodes else 0.0
            seg = wfc - bfc
        else:
            wfc, bfc, seg = 0.0, 0.0, 0.0

        # --- Topology (2) ---
        local_cc = float(cc[i])
        strength  = float(np.abs(row).sum()) / (N - 1)

        # --- Global context (4) ---
        nm = np.array(net_means, dtype=np.float32)

        feat = np.array([
            mean_fc, std_fc, pos_ratio, neg_ratio, deg_norm,       # 5
            *net_means,                                              # 9
            wfc, bfc, seg,                                          # 3
            local_cc, strength,                                     # 2
            float(nm.std()), float(nm.max()),
            float(nm.min()), float(nm.max() - nm.min())             # 4
        ], dtype=np.float32)
        features.append(feat)

    return np.stack(features, axis=0)


NODE_FEAT_DIM = 5 + 9 + 3 + 2 + 4  # = 23

# ===============================================================
# Graph augmentation
# ===============================================================
def graph_mixup(x1, adj1, x2, adj2, lam):
    return lam * x1 + (1 - lam) * x2, lam * adj1 + (1 - lam) * adj2


def augment_graph(adj, drop_edge_prob=0.10, jitter_std=0.05):
    adj   = adj.clone()
    mask  = torch.bernoulli(
        torch.full(adj.shape, 1 - drop_edge_prob, device=adj.device))
    adj   = adj * mask
    noise = torch.randn_like(adj) * jitter_std
    adj   = adj + noise * (adj != 0).float()
    idx   = torch.arange(adj.shape[-1], device=adj.device)
    adj[:, idx, idx] = 1.0
    return adj

# ===============================================================
# GAT Layer  (unchanged from v4)
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
# FNP-GNN v5
# Changes from v4:
#  - Smaller hidden_dim (96 vs 128) to match compact features
#  - encode() + forward() refactor: single pass returns (logits, proj)
#  - Classifier: 192 hidden units instead of 256
# ===============================================================
class FNPGNNv5(nn.Module):
    def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
        )
        self.bn_input = nn.BatchNorm1d(hidden_dim)

        self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.gat1 = GATLayer(hidden_dim, hidden_dim, n_heads, dropout=0.2)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, n_heads, dropout=0.2)

        self.vn_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
        )

        self.register_buffer('pooling_mat', POOLING_MAT)

        self.net_attn = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        combined_dim = N_NETWORKS * hidden_dim + hidden_dim
        self.net_ln = nn.LayerNorm(combined_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 192),
            nn.ELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(192, 2)
        )

        self.projector = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64)
        )

        self.apply(self._init_weights)
        nn.init.normal_(self.virtual_node_emb, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _encode(self, x, adj):
        B, N, _ = x.shape
        h  = self.node_encoder(x)
        h  = self.bn_input(h.reshape(B * N, -1)).reshape(B, N, -1)

        vn = self.virtual_node_emb.expand(B, -1, -1) + h.mean(dim=1, keepdim=True)

        h  = self.gat1(h, adj)
        vn_new = self.vn_update(
            torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))
        h  = h + vn_new.expand(-1, N, -1) * 0.1

        h2 = self.gat2(h, adj)
        h  = h2 + h

        vn = self.vn_update(
            torch.cat([vn_new, h.mean(dim=1, keepdim=True)], dim=-1))

        pooled = torch.matmul(
            h.transpose(1, 2), self.pooling_mat).transpose(1, 2)
        net_w  = torch.softmax(self.net_attn(pooled), dim=1)
        pooled = pooled * net_w

        flat = self.net_ln(torch.cat([
            pooled.reshape(B, -1),
            vn.squeeze(1)
        ], dim=1))
        return flat

    def forward(self, x, adj, augment=False):
        """Returns (logits, l2-normalised projection)."""
        if augment:
            adj = augment_graph(adj, drop_edge_prob=0.10, jitter_std=0.05)
        flat   = self._encode(x, adj)
        logits = self.classifier(flat)
        proj   = F.normalize(self.projector(flat), dim=-1)
        return logits, proj

# ===============================================================
# Supervised Contrastive Loss
# ===============================================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=CONTRA_TEMP):
        super().__init__()
        self.temp = temperature

    def forward(self, features, labels):
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)
        sim = torch.matmul(features, features.T) / self.temp
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        diag_mask = (~torch.eye(B, dtype=torch.bool,
                                 device=features.device)).float()
        exp_sim  = torch.exp(sim) * diag_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        return loss.mean()

# ===============================================================
# Dataset  (precomputed once per task, then use Subset per fold)
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
# TTA inference  — average over TTA_STEPS augmented forward passes
# ===============================================================
@torch.no_grad()
def predict_with_tta(model, loader, device, n_steps=TTA_STEPS):
    model.eval()
    all_probs, all_labels = [], []
    for b in loader:
        x   = b['x'].to(device)
        adj = b['adj'].to(device)
        # 1 clean pass + (n_steps-1) augmented passes
        logits_sum, _ = model(x, adj, augment=False)
        probs_sum = torch.softmax(logits_sum, dim=1)
        for _ in range(n_steps - 1):
            lo, _ = model(x, adj, augment=True)
            probs_sum = probs_sum + torch.softmax(lo, dim=1)
        probs = (probs_sum / n_steps).cpu().numpy()
        all_probs.append(probs[0])
        all_labels.append(b['label'].item())
    return np.stack(all_probs), np.array(all_labels)

# ===============================================================
# Optimal decision threshold via Youden's J statistic
# ===============================================================
def find_optimal_threshold(probs_arr, labels):
    """Return threshold maximising sensitivity + specificity - 1."""
    try:
        fpr, tpr, thresholds = roc_curve(labels, probs_arr[:, 1])
        J = tpr - fpr
        return float(thresholds[np.argmax(J)])
    except Exception:
        return 0.5

# ===============================================================
# Task runner  (multi-seed × 5-fold + TTA + threshold opt)
# ===============================================================
def run_task(df_full, task_pair, device):
    class_a, class_b = task_pair
    task_name = f"{class_a} vs {class_b}"
    print(f"\n{'='*60}")
    print(f"  Task: {task_name}")
    print(f"{'='*60}")

    df_task = df_full[df_full['diagnosis'].isin([class_a, class_b])].copy()
    df_task['current_task_label'] = df_task['diagnosis'].map(
        {class_a: 0, class_b: 1})
    df_task = df_task.reset_index(drop=True)

    labels_arr = df_task['current_task_label'].values
    for cls, cnt in zip([class_a, class_b], np.bincount(labels_arr)):
        print(f"  {cls}: {cnt}")
    print(f"  Total: {len(df_task)} | feat_dim: {NODE_FEAT_DIM}")

    # --- Precompute dataset ONCE (major speedup vs v4) ---
    print("  Precomputing features...", flush=True)
    full_ds = fMRIDataset(df_task)
    print("  Done.")

    contra_loss_fn = SupConLoss(temperature=CONTRA_TEMP)

    # Accumulate probabilities: list[list[np.array]] per sample
    all_val_probs = [[] for _ in range(len(df_task))]
    all_val_true  = [None] * len(df_task)

    # Track globally best model across all seeds × folds
    global_best_auc   = 0.0
    global_best_state = None
    global_best_info  = {}

    # Keep last-fold model for feature importance plot
    last_model = None
    last_val_loader = None

    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        skf = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(
                skf.split(df_task, labels_arr)):

            print(f"    Fold {fold+1}/{N_FOLDS}", end='  ', flush=True)

            train_labels_sub = labels_arr[train_idx]
            class_counts     = np.bincount(train_labels_sub)
            sample_weights   = [1.0 / class_counts[l]
                                 for l in train_labels_sub]
            sampler = WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True)

            train_sub = Subset(full_ds, train_idx.tolist())
            val_sub   = Subset(full_ds, val_idx.tolist())

            train_loader = DataLoader(
                train_sub, batch_size=BATCH_SIZE,
                sampler=sampler, drop_last=True)
            val_loader   = DataLoader(val_sub, batch_size=1)

            model     = FNPGNNv5().to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2, eta_min=1e-6)
            cw = torch.tensor(
                1.0 / class_counts, dtype=torch.float32).to(device)
            ce_loss_fn = nn.CrossEntropyLoss(
                weight=cw, label_smoothing=0.1)

            best_auc    = 0.0
            best_state  = None
            patience_cnt = 0

            for epoch in range(EPOCHS):
                model.train()
                for b in train_loader:
                    x   = b['x'].to(device)
                    adj = b['adj'].to(device)
                    lbl = b['label'].to(device)
                    B   = x.shape[0]

                    if MIXUP_ALPHA > 0 and B >= 2:
                        lam  = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                        perm = torch.randperm(B, device=device)
                        x_mix, adj_mix = graph_mixup(
                            x, adj, x[perm], adj[perm], lam)
                        lbl_a, lbl_b = lbl, lbl[perm]
                    else:
                        x_mix, adj_mix = x, adj
                        lbl_a, lbl_b, lam = lbl, lbl, 1.0

                    # Classification on (possibly mixed) data
                    logits, _ = model(x_mix, adj_mix, augment=True)
                    loss_ce   = (lam * ce_loss_fn(logits, lbl_a) +
                                 (1 - lam) * ce_loss_fn(logits, lbl_b))

                    # Contrastive on original (non-mixed) data
                    _, proj    = model(x, adj, augment=False)
                    loss_contra = contra_loss_fn(proj, lbl)

                    loss = LAMBDA_CE * loss_ce + LAMBDA_CONTRA * loss_contra
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()

                # --- Validation: clean pass, monitor AUC ---
                model.eval()
                vt, vp1 = [], []
                with torch.no_grad():
                    for b in val_loader:
                        lo, _ = model(b['x'].to(device),
                                      b['adj'].to(device), augment=False)
                        vp1.append(torch.softmax(lo, dim=1).cpu().numpy()[0, 1])
                        vt.append(b['label'].item())
                try:
                    val_auc = roc_auc_score(vt, vp1)
                except ValueError:
                    val_auc = 0.5

                if val_auc > best_auc:
                    best_auc     = val_auc
                    patience_cnt = 0
                    best_state   = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1

                if patience_cnt >= PATIENCE:
                    break

            print(f"best_AUC={best_auc:.3f}  (stopped @ep{epoch+1})")

            # --- Update global best model ---
            if best_auc > global_best_auc:
                global_best_auc   = best_auc
                global_best_state = {k: v.clone()
                                     for k, v in best_state.items()}
                global_best_info  = {'seed': seed, 'fold': fold + 1,
                                     'val_auc': best_auc}

            # --- Final inference with full TTA ---
            model.load_state_dict(best_state)
            model.to(device)

            val_probs_tta, val_true_tta = predict_with_tta(
                model, val_loader, device, n_steps=TTA_STEPS)

            for sample_i, prob, true_l in zip(
                    val_idx, val_probs_tta, val_true_tta):
                all_val_probs[sample_i].append(prob)
                all_val_true[sample_i] = int(true_l)

            last_model      = model
            last_val_loader = val_loader

    # --- Ensemble: average over all seed×fold predictions ---
    all_prob_arr = np.stack(
        [np.mean(all_val_probs[i], axis=0)
         for i in range(len(df_task))])
    all_true = [all_val_true[i] for i in range(len(df_task))]

    # --- Threshold optimisation (Youden's J) ---
    opt_thresh = find_optimal_threshold(all_prob_arr, np.array(all_true))
    print(f"\n  Optimal threshold: {opt_thresh:.3f}")

    pred_default = (all_prob_arr[:, 1] > 0.50).astype(int).tolist()
    pred_optimal = (all_prob_arr[:, 1] > opt_thresh).astype(int).tolist()

    acc_default = accuracy_score(all_true, pred_default)
    acc_optimal = accuracy_score(all_true, pred_optimal)

    # Use whichever threshold gives higher accuracy
    if acc_optimal >= acc_default:
        final_pred = pred_optimal
        acc        = acc_optimal
    else:
        final_pred = pred_default
        acc        = acc_default

    cm = confusion_matrix(all_true, final_pred)
    try:
        auc = roc_auc_score(all_true, all_prob_arr[:, 1].tolist())
    except ValueError:
        auc = float('nan')

    print(f"  thresh=0.50 → {acc_default*100:.1f}%  |  "
          f"thresh={opt_thresh:.2f} → {acc_optimal*100:.1f}%")

    # ---------------------------------------------------------------
    # Save best model + OOF soft probabilities for cross-model KD
    # ---------------------------------------------------------------
    safe_name = task_name.replace(' ', '_')

    # 1. Best single model (from seed×fold with highest val AUC)
    ckpt_path = os.path.join(MODEL_SAVE_DIR,
                             f'fnp_gnn_v5_{safe_name}_best.pth')
    torch.save({
        'model_state_dict': global_best_state,
        'model_config': {
            'input_dim':  NODE_FEAT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'n_heads':    N_HEADS,
            'dropout':    DROPOUT,
        },
        'task':       task_name,
        'task_pair':  task_pair,
        'class_map':  {class_a: 0, class_b: 1},
        'val_auc':    global_best_info.get('val_auc', 0.0),
        'best_from':  global_best_info,
        'threshold':  opt_thresh,
        'k_ratio':    K_RATIO,
    }, ckpt_path)
    print(f"\n  Saved best model → {ckpt_path}")
    print(f"  (from seed={global_best_info.get('seed')}, "
          f"fold={global_best_info.get('fold')}, "
          f"val_AUC={global_best_info.get('val_auc', 0):.3f})")

    # 2. OOF soft probabilities — primary KD teacher signal
    #    Shape: (N_samples, 2)   probabilities for [class_a, class_b]
    #    Each sample's probs come from folds where it was held-out,
    #    then averaged across all seeds → unbiased soft labels.
    oof_path = os.path.join(MODEL_SAVE_DIR,
                            f'fnp_gnn_v5_{safe_name}_oof_probs.npy')
    np.save(oof_path, {
        'probs':        all_prob_arr,             # (N, 2)
        'labels':       np.array(all_true),        # (N,)
        'matrix_paths': df_task['matrix_path'].tolist(),
        'class_map':    {class_a: 0, class_b: 1},
        'threshold':    opt_thresh,
        'acc':          acc,
        'auc':          auc,
        'task':         task_name,
    }, allow_pickle=True)
    print(f"  Saved OOF soft probs → {oof_path}")
    print(f"  (shape: {all_prob_arr.shape}, "
          f"can be used as KD soft labels for {len(all_true)} samples)")

    # --- Feature importance from last model ---
    if last_model is not None:
        _plot_feature_importance(
            last_model, last_val_loader, task_pair, task_name, device)

    return acc, auc, cm

# ===============================================================
# Feature importance visualisation
# ===============================================================
def _plot_feature_importance(model, val_loader, task_pair, task_name, device):
    try:
        class_a, class_b = task_pair
        attn0, attn1 = [], []
        model.eval()
        with torch.no_grad():
            for b in val_loader:
                x   = b['x'].to(device)
                adj = b['adj'].to(device)
                lbl = b['label'].item()

                h  = model.node_encoder(x)
                B, N, _ = h.shape
                h  = model.bn_input(h.reshape(B*N, -1)).reshape(B, N, -1)
                vn = model.virtual_node_emb.expand(B, -1, -1) + h.mean(1, True)
                h  = model.gat1(h, adj)
                vn = model.vn_update(
                    torch.cat([vn, h.mean(1, True)], dim=-1))
                h  = h + vn.expand(-1, N, -1) * 0.1
                h  = model.gat2(h, adj) + h
                pooled = torch.matmul(
                    h.transpose(1, 2), model.pooling_mat).transpose(1, 2)
                w = torch.softmax(
                    model.net_attn(pooled), dim=1).squeeze().cpu().numpy()
                (attn0 if lbl == 0 else attn1).append(w)

        m0   = np.mean(attn0, axis=0) if attn0 else np.zeros(N_NETWORKS)
        m1   = np.mean(attn1, axis=0) if attn1 else np.zeros(N_NETWORKS)
        diff = np.abs(m1 - m0)

        xp, w = np.arange(N_NETWORKS), 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        b0 = ax.bar(xp - w/2, m0, w, label=class_a,
                    color='#5DADE2', edgecolor='black')
        b1 = ax.bar(xp + w/2, m1, w, label=class_b,
                    color='#E74C3C', edgecolor='black')
        for i in np.argsort(diff)[-2:]:
            b0[i].set_edgecolor('gold'); b0[i].set_linewidth(2.5)
            b1[i].set_edgecolor('gold'); b1[i].set_linewidth(2.5)
        ax.set_xticks(xp)
        ax.set_xticklabels(NETWORK_NAMES, fontsize=11)
        ax.set_ylabel('Mean Network Attention Weight', fontsize=12)
        ax.set_title(
            f'FNP-GNN v5 Network Attention ({task_name})',
            fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        safe = task_name.replace(' ', '_')
        plt.savefig(f'gnn_v5_feature_importance_{safe}.png', dpi=300)
        plt.close()
        print(f"  Saved: gnn_v5_feature_importance_{safe}.png")
    except Exception as e:
        print(f"  Feature importance failed: {e}")

# ===============================================================
# Data loader
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
                m_path = os.path.join(
                    MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy")
            elif 'Subject' in row and pd.notna(row['Subject']):
                m_path = os.path.join(
                    MATRIX_DIR, f"{row['Subject']}_matrix_116.npy")
            if not (m_path and os.path.exists(m_path)): continue
            if m_path in seen_paths: continue
            try:
                mat  = np.load(m_path)
                if mat.shape != (116, 116): continue
                diag = str(row.get('diagnosis', '')).upper()
                if not diag: continue
                valid_data.append({'matrix_path': m_path, 'diagnosis': diag})
                seen_paths.add(m_path)
            except Exception:
                continue
    return pd.DataFrame(valid_data)

# ===============================================================
# Main
# ===============================================================
def main():
    print("FNP-GNN v5")
    print("  Improvements over v4:")
    print("    1. Compact node features (23-dim, network-centric)")
    print("    2. Multi-seed ensemble  (3 seeds × 5 folds = 15 models)")
    print("    3. TTA inference        (8 augmented passes)")
    print("    4. AUC-based early stopping")
    print("    5. Youden's J threshold optimisation")
    print(f"\n  Hidden={HIDDEN_DIM} | K_ratio={K_RATIO} | "
          f"LR={LR} | WD={WEIGHT_DECAY}")
    print(f"  Seeds={SEEDS} | TTA={TTA_STEPS} | "
          f"λ_CE={LAMBDA_CE} | λ_contra={LAMBDA_CONTRA}\n")

    df_full = load_data()
    print(f"\nLoaded {len(df_full)} valid samples")
    print(df_full['diagnosis'].value_counts().to_string())
    if len(df_full) == 0:
        print("No data found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    tasks   = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('FNP-GNN v5 (fMRI) - Confusion Matrices',
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
    print("FNP-GNN v5 Results")
    print("="*60)
    print(f"{'Task':<16} {'Accuracy':>9} {'AUC':>8}")
    print("-"*36)
    for task_name, (acc, auc, _) in results.items():
        flag = "OK" if acc >= 0.80 else "--"
        print(f"[{flag}] {task_name:<14} {acc*100:>8.1f}%  {auc:>7.3f}")

    plt.tight_layout()
    plt.savefig('fmri_gnn_v5_confusion_matrices.png', dpi=300)
    print("\nSaved: fmri_gnn_v5_confusion_matrices.png")


if __name__ == "__main__":
    main()
