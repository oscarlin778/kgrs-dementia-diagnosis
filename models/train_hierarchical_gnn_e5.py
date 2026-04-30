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
# Settings & Hyperparameters (same as E3 base)
# ===============================================================
CSV_PATHS = [
    "/home/wei-chi/Model/_dataset_mapping.csv",
    "/home/wei-chi/Data/dataset_index_116_clean_old.csv",
    "/home/wei-chi/Data/adni_dataset_index_116.csv"
]
MATRIX_DIR = "/home/wei-chi/Model/processed_116_matrices"
TEACHER_PROBS_DIR = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints"

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

LAMBDA_CE       = 1.0
LAMBDA_KD       = 0.5
LAMBDA_CONTRA   = 0.0
CONTRA_TEMP     = 0.5

SEEDS = [42, 123, 456]

# ── E5: hybrid per-task domain adversarial weight ───────────────
# NC_vs_MCI: revert to 0.3 (λ=0.5 in E4 over-aligned, E3=0.3 was better)
# MCI_vs_AD: keep 0.1 from E4 (+0.011 AUC confirmed)
# NC_vs_AD: unchanged at 0.3
LAMBDA_DOMAIN_PER_TASK = {
    ('NC', 'AD'):  0.3,
    ('NC', 'MCI'): 0.3,
    ('MCI', 'AD'): 0.1,
}

# ===============================================================
# 1. Network map & node features
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
# 2. Model Architecture
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


class FNPGNNv8_KD(nn.Module):
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
        self.projector = nn.Sequential(nn.Linear(N_NETWORKS * hidden_dim + hidden_dim, 128), nn.ELU(), nn.Linear(128, 64))
        nn.init.normal_(self.virtual_node_emb, std=0.02)

    def forward(self, x, adj, return_embedding=False, return_attn=False):
        B, N, _ = x.shape
        h = self.bn_input(self.node_encoder(x).reshape(B * N, -1)).reshape(B, N, -1)
        vn = self.virtual_node_emb.expand(B, -1, -1) + h.mean(dim=1, keepdim=True)

        if return_attn:
            h, imp1 = self.gat1(h, adj, return_attn=True)
        else:
            h = self.gat1(h, adj)
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))
        h = h + vn.expand(-1, N, -1) * 0.1

        if return_attn:
            h_new, imp2 = self.gat2(h, adj, return_attn=True)
        else:
            h_new = self.gat2(h, adj)
        h = h_new + h
        vn = self.vn_update(torch.cat([vn, h.mean(dim=1, keepdim=True)], dim=-1))
        h = h + vn.expand(-1, N, -1) * 0.1

        if return_attn:
            h_new, imp3 = self.gat3(h, adj, return_attn=True)
        else:
            h_new = self.gat3(h, adj)
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
# 3. DANN Components
# ===============================================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)


class DomainClassifier(nn.Module):
    """Binary domain classifier (TPMIC=0, ADNI=1) applied after GRL."""
    def __init__(self, input_dim=N_NETWORKS * HIDDEN_DIM + HIDDEN_DIM):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, flat, alpha=1.0):
        return self.net(self.grl(flat, alpha))


def get_dann_alpha(epoch, total_epochs):
    """Standard DANN alpha schedule: ramps from 0 to 1 over training."""
    p = epoch / total_epochs
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ===============================================================
# 4. Loss functions
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
# 5. Dataset (adds domain_label and is_nc_or_ad for DANN)
# ===============================================================
def get_subject_id(path_str):
    basename = os.path.basename(str(path_str))
    clean = re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy|_T1_MNI\.nii\.gz|_T1\.nii\.gz|\.nii\.gz)$', '', basename)
    clean = re.sub(r'^(sub-|sub_|old_dswau)', '', clean)
    return clean.strip()


class fMRIDataset_KD(Dataset):
    def __init__(self, dataframe, teacher_dict=None):
        self.data_cache = []
        self.has_teacher_count = 0

        for _, row in dataframe.iterrows():
            adj_raw = np.load(row['matrix_path'])
            label = row['current_task_label']
            subj_id = get_subject_id(row['matrix_path'])

            teacher_prob = teacher_dict.get(subj_id, None) if teacher_dict else None
            has_soft = teacher_prob is not None
            if has_soft: self.has_teacher_count += 1

            src = row.get('source', 'TPMIC') if hasattr(row, 'get') else row['source'] if 'source' in row.index else 'TPMIC'
            domain_label = 1 if str(src).upper() == 'ADNI' else 0

            diag = str(row.get('diagnosis', '') if hasattr(row, 'get') else row['diagnosis']).upper()
            is_nc_or_ad = diag in ('NC', 'AD')

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
                'soft_label': torch.FloatTensor(teacher_prob).flip(0) if has_soft else torch.zeros(2),
                'has_soft': torch.tensor(has_soft, dtype=torch.bool),
                'subj_id': subj_id,
                'domain_label': torch.tensor(domain_label, dtype=torch.float32),
                'is_nc_or_ad': torch.tensor(is_nc_or_ad, dtype=torch.bool),
            })

    def __len__(self): return len(self.data_cache)
    def __getitem__(self, idx): return self.data_cache[idx]


# ===============================================================
# 6. Training loop
# ===============================================================
def load_teacher_probs(task_pair):
    safe_name = f"{task_pair[0]}_vs_{task_pair[1]}"
    npy_path = os.path.join(TEACHER_PROBS_DIR, f"teacher_logits_{safe_name}.npy")
    if os.path.exists(npy_path):
        teacher_dict = np.load(npy_path, allow_pickle=True).item()
        print(f"  Teacher probs loaded ({safe_name}): {len(teacher_dict)} subjects")
        return teacher_dict
    else:
        print(f"  Teacher probs not found: {npy_path}, using CE only")
        return {}


def run_task_dann(df_task, task_pair, teacher_dict, device, seed,
                  lambda_domain=0.3, ckpt_dir=None):
    """DANN method=2: domain loss for all tasks; MCI subjects masked out."""
    class_a, class_b = task_pair
    labels_arr = df_task['current_task_label'].values
    strata_arr = (
        labels_arr.astype(str) + '_' + df_task['source'].fillna('TPMIC').values
    ) if 'source' in df_task.columns else labels_arr

    torch.manual_seed(seed)
    np.random.seed(seed)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    contra_loss_fn = SupConLoss(temperature=CONTRA_TEMP)

    all_val_prob = [None] * len(df_task)
    all_val_true = [None] * len(df_task)
    all_val_attn = [None] * len(df_task)
    all_val_flat = [None] * len(df_task)

    seed_best_acc, seed_best_state = 0.0, None

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_task, strata_arr)):
        print(f"    Fold {fold+1}/{N_FOLDS}", end="  ")
        train_df = df_task.iloc[train_idx].reset_index(drop=True)
        val_df   = df_task.iloc[val_idx].reset_index(drop=True)

        train_ds = fMRIDataset_KD(train_df, teacher_dict)
        val_ds   = fMRIDataset_KD(val_df, teacher_dict)

        class_counts = np.bincount(train_df['current_task_label'].values)
        weights = [1.0 / class_counts[l] for l in train_df['current_task_label'].values]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=1)

        model = FNPGNNv8_KD().to(device)
        domain_clf = DomainClassifier().to(device)

        all_params = list(model.parameters()) + list(domain_clf.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
        ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc, best_state, patience_cnt = 0.0, None, 0

        for epoch in range(EPOCHS):
            model.train()
            domain_clf.train()
            alpha = get_dann_alpha(epoch, EPOCHS)

            for b in train_loader:
                x       = b['x'].to(device)
                adj     = b['adj'].to(device)
                lbl     = b['label'].to(device)
                soft_lbl  = b['soft_label'].to(device)
                has_soft  = b['has_soft'].to(device)
                dom_lbl   = b['domain_label'].to(device)
                is_ncad   = b['is_nc_or_ad'].to(device)

                logits, flat = model(x, adj)
                proj = F.normalize(model.projector(flat), dim=-1)

                loss_ce = ce_loss_fn(logits, lbl)

                loss_kd = torch.tensor(0.0, device=device)
                valid_kd = has_soft == True
                if valid_kd.any():
                    loss_kd = distillation_loss(logits[valid_kd], soft_lbl[valid_kd])

                loss_contra = contra_loss_fn(proj, lbl)

                # DANN method=2: only NC/AD subjects contribute to domain loss
                loss_domain = torch.tensor(0.0, device=device)
                if is_ncad.any():
                    domain_logit = domain_clf(flat[is_ncad], alpha)
                    loss_domain = F.binary_cross_entropy_with_logits(
                        domain_logit.squeeze(1), dom_lbl[is_ncad])

                loss = (LAMBDA_CE * loss_ce + LAMBDA_KD * loss_kd
                        + LAMBDA_CONTRA * loss_contra + lambda_domain * loss_domain)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(domain_clf.parameters(), 1.0)
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
        ckpt_path = os.path.join(ckpt_dir, f"e5_{safe_name}_seed{seed}.pt")
        torch.save(seed_best_state, ckpt_path)

    return (np.stack(all_val_prob, axis=0), all_val_true,
            np.stack(all_val_attn, axis=0), np.stack(all_val_flat, axis=0))


def run_task_multi_seed(df_full, task_pair, device):
    class_a, class_b = task_pair
    task_name = f"{class_a} vs {class_b}"
    lambda_domain = LAMBDA_DOMAIN_PER_TASK[task_pair]
    print(f"\n{'='*60}\n  Task: {task_name}  [E5 DANN method=2, λ_domain={lambda_domain}]\n{'='*60}")

    df_task = df_full[df_full['diagnosis'].isin([class_a, class_b])].copy()
    df_task['current_task_label'] = df_task['diagnosis'].map({class_a: 0, class_b: 1})
    df_task = df_task.reset_index(drop=True)

    teacher_dict = load_teacher_probs(task_pair)
    tmp_ds = fMRIDataset_KD(df_task, teacher_dict)
    tpmic_n = sum(1 for d in tmp_ds.data_cache if d['domain_label'].item() == 0)
    adni_n  = sum(1 for d in tmp_ds.data_cache if d['domain_label'].item() == 1)
    print(f"  Subjects: {len(tmp_ds)} total  (TPMIC={tpmic_n}, ADNI={adni_n})")
    print(f"  Teacher labels matched: {tmp_ds.has_teacher_count}")
    del tmp_ds

    ckpt_dir = os.path.join(TEACHER_PROBS_DIR, "e5_checkpoints")
    seed_probs, seed_attns, seed_flats = [], [], []
    for seed in SEEDS:
        print(f"\n  [Seed {seed}]")
        probs, all_true, attns, flats = run_task_dann(
            df_task, task_pair, teacher_dict, device, seed,
            lambda_domain=lambda_domain, ckpt_dir=ckpt_dir)
        seed_probs.append(probs)
        seed_attns.append(attns)
        seed_flats.append(flats)
        s_acc = accuracy_score(all_true, probs.argmax(axis=1))
        print(f"    Seed {seed} OOF acc: {s_acc*100:.1f}%")

    avg_probs = np.mean(seed_probs, axis=0)
    avg_attn  = np.mean(seed_attns, axis=0)
    avg_flats = np.mean(seed_flats, axis=0)
    all_pred  = avg_probs.argmax(axis=1).tolist()
    acc = accuracy_score(all_true, all_pred)
    cm  = confusion_matrix(all_true, all_pred)
    try:   auc = roc_auc_score(all_true, avg_probs[:, 1])
    except: auc = float('nan')
    print(f"\n  Multi-seed avg OOF: {acc*100:.1f}% / AUC {auc:.3f}")

    from sklearn.metrics import roc_curve as _roc_curve
    try:
        _fpr, _tpr, _ = _roc_curve(all_true, avg_probs[:, 1])
    except Exception:
        _fpr, _tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    subj_ids = df_task['matrix_path'].apply(get_subject_id).tolist()
    best_ens_acc, best_ens_auc, best_w = acc, auc, 1.0
    for gnn_w in [0.9, 0.8, 0.7, 0.6, 0.5]:
        ens_prob = avg_probs.copy()
        for i, sid in enumerate(subj_ids):
            if sid in teacher_dict:
                ens_prob[i] = gnn_w * avg_probs[i] + (1 - gnn_w) * teacher_dict[sid]
        ens_acc = accuracy_score(all_true, ens_prob.argmax(axis=1))
        try: ens_auc = roc_auc_score(all_true, ens_prob[:, 1])
        except: ens_auc = float('nan')
        if ens_acc > best_ens_acc:
            best_ens_acc, best_ens_auc, best_w = ens_acc, ens_auc, gnn_w
    if best_w < 1.0:
        print(f"  Ensemble: GNN×{best_w:.1f} + Teacher×{1-best_w:.1f} → {best_ens_acc*100:.1f}% / AUC {best_ens_auc:.3f}")

    return (acc, auc, cm, best_ens_acc, best_ens_auc, best_w,
            avg_probs, np.array(all_true), df_task['matrix_path'].tolist(), _fpr, _tpr)


# ===============================================================
# 7. Data Loader
# ===============================================================
def load_data():
    valid_data, seen_paths = [], set()
    for path in CSV_PATHS:
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            m_path = (row.get('matrix_path')
                      or (os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy") if pd.notna(row.get('new_id_base')) else None)
                      or (os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy") if pd.notna(row.get('Subject')) else None))
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


# ===============================================================
# 8. Main
# ===============================================================
def run_experiment(df_full, device, exp_name="E5_hybrid_lambda"):
    print(f"\n{'#'*70}")
    print(f"  E5 DANN — Hybrid λ_domain (best-of-series per task)")
    print(f"  NC_vs_AD: λ={LAMBDA_DOMAIN_PER_TASK[('NC','AD')]}  "
          f"NC_vs_MCI: λ={LAMBDA_DOMAIN_PER_TASK[('NC','MCI')]}  "
          f"MCI_vs_AD: λ={LAMBDA_DOMAIN_PER_TASK[('MCI','AD')]}")
    print(f"{'#'*70}")

    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    results    = {}
    exp_metrics = {}
    all_oof    = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('E5 Hybrid λ_domain - Confusion Matrices', fontsize=16, fontweight='bold')

    for idx, task in enumerate(tasks):
        (acc, auc, cm, ens_acc, ens_auc, best_w,
         avg_probs, all_true, matrix_paths, fpr, tpr) = run_task_multi_seed(
            df_full, task, device)

        results[f"{task[0]} vs {task[1]}"] = (acc, auc, cm, ens_acc, ens_auc, best_w)
        safe = f"{task[0]}_vs_{task[1]}"
        exp_metrics[safe] = {"auc": float(auc), "acc": float(acc),
                              "fpr": fpr.tolist(), "tpr": tpr.tolist()}
        all_oof[safe] = (avg_probs, all_true, matrix_paths)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=[task[0], task[1]], yticklabels=[task[0], task[1]],
                    annot_kws={"size": 16})
        axes[idx].set_title(f"{task[0]} vs {task[1]}\nAcc: {acc*100:.1f}%  AUC: {auc:.3f}", fontsize=14)

    print(f"\n{'='*60}")
    print(f"  Saving results → results/{exp_name}/")
    print(f"{'='*60}")

    baseline_path = os.path.join(ser.RESULTS_DIR, "baseline", "metrics.json")
    baseline_metrics = None
    if os.path.exists(baseline_path):
        import json
        with open(baseline_path) as f:
            baseline_metrics = json.load(f)

    ser.save_metrics(exp_name, exp_metrics)
    ser.plot_roc_comparison(exp_name, exp_metrics, baseline_metrics=baseline_metrics)

    for safe, (avg_probs, all_true, matrix_paths) in all_oof.items():
        y_pred = avg_probs.argmax(axis=1)
        ser.plot_confusion_matrix(exp_name, all_true, y_pred, safe)

    ser.update_comparison_chart()

    plt.tight_layout()
    out_dir = os.path.join(ser.RESULTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()

    print(f"\n{'='*60}")
    print(f"  Final results — {exp_name}")
    print(f"{'='*60}")
    print(f"{'Task':<16} {'ACC':>8} {'AUC':>8}   {'Ens ACC':>8} {'Ens AUC':>8}")
    print("-"*60)
    for task_name, (acc, auc, _, ens_acc, ens_auc, best_w) in results.items():
        print(f"  {task_name:<14} {acc*100:>7.1f}%  {auc:>7.3f}   {ens_acc*100:>7.1f}%  {ens_auc:>7.3f}")

    return exp_metrics


def main():
    print("E5 DANN — Hybrid λ_domain (best-of-series per task)")
    print(f"LAMBDA_DOMAIN_PER_TASK={LAMBDA_DOMAIN_PER_TASK}, seeds={SEEDS}")

    df_full = load_data()
    print(f"\nDataset: {len(df_full)} subjects total")
    print(df_full['source'].value_counts().to_string())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    exp_metrics = run_experiment(df_full, device, exp_name="E5_hybrid_lambda")

    # Compare E3 method2 vs E5
    import json
    e3_path = os.path.join(ser.RESULTS_DIR, "E3_domain_adversarial", "E3_method2", "metrics.json")
    if os.path.exists(e3_path):
        with open(e3_path) as f:
            e3_metrics = json.load(f)
        print(f"\n{'='*70}")
        print("  E3 Method2 vs E5 Comparison")
        print(f"{'='*70}")
        print(f"{'Task':<16} {'E3 AUC':>8} {'E5 AUC':>8}  {'Δ AUC':>8}  {'Winner':>8}")
        print("-"*60)
        for task in ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]:
            a3 = e3_metrics.get(task, {}).get('auc', float('nan'))
            a5 = exp_metrics.get(task, {}).get('auc', float('nan'))
            delta = a5 - a3
            winner = "E5" if delta > 0 else ("E3" if delta < 0 else "Tie")
            print(f"  {task:<14} {a3:>7.3f}  {a5:>7.3f}  {delta:>+7.3f}   {winner}")


if __name__ == "__main__":
    main()
