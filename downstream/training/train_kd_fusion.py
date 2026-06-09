import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
train_kd_fusion.py
==================
Cross-Modal Knowledge Distillation
  Teacher : sMRI Ensemble (BrainIAC light-FT + ResNet50) — pre-computed soft labels
  Student : fMRI GNN (FNPGNNv8_KD)

Training data : splits/combined_train.csv (Full fMRI training set, 193 subjects)
Test data     : splits/combined_test.csv  (fMRI-only, 49 subjects)
                kd_test_aligned.csv       (for teacher comparison only)

Usage:
  python train_kd_fusion.py [--task NC_vs_MCI] [--alpha 0.3] [--temperature 3.0]
"""

import os, sys, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path

sys.path.insert(0, '/home/wei-chi/Alzheimers_Project/external_data/scripts')

# ── CLI Args ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",        default="NC_vs_MCI",
                        choices=["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"])
    parser.add_argument("--alpha",       type=float, default=0.3,
                        help="Weight for CE loss (1-alpha = KD loss weight)")
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ── Paths ────────────────────────────────────────────────────────────
    FMRI_TRAIN_CSV = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits/combined_train.csv"
    FMRI_TEST_CSV  = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits/combined_test.csv"
    KD_TEST_CSV    = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/kd_test_aligned.csv"
    PROB_DIR       = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/teacher_probs")
    OUT_DIR        = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results")
    OUT_DIR.mkdir(exist_ok=True)

    # ── Task config ──────────────────────────────────────────────────────
    TASK_CFG = {
        "NC_vs_AD":  {"classes": [0, 2], "pos": 2},
        "NC_vs_MCI": {"classes": [0, 1], "pos": 1},
        "MCI_vs_AD": {"classes": [1, 2], "pos": 2},
    }
    cfg = TASK_CFG[args.task]

    # ── GNN hyperparams (copy from train_hierarchical_gnn_kd.py) ────────
    HIDDEN_DIM = 128
    DROPOUT    = 0.4
    K_RATIO    = 0.20
    N_FOLDS    = 5
    PATIENCE   = 40
    WEIGHT_DECAY = 5e-3

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
    N_NETWORKS   = len(NETWORK_MAP)
    POOLING_MAT  = torch.zeros(116, N_NETWORKS)
    for i, net in enumerate(NETWORK_MAP):
        for node_idx in NETWORK_MAP[net]:
            POOLING_MAT[node_idx, i] = 1.0

    NODE_FEAT_DIM = 116 + 5 + 2 + 2

    # ── Node feature extraction (exact copy from existing GNN scripts) ───
    def extract_node_features(adj_z: np.ndarray) -> np.ndarray:
        N = adj_z.shape[0]
        net_list = list(NETWORK_MAP.keys())
        roi_to_net = {roi: i for i, net in enumerate(net_list) for roi in NETWORK_MAP[net]}
        adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
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

    def build_adj(adj_z: np.ndarray) -> np.ndarray:
        N = adj_z.shape[0]
        adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
        k = int(N * K_RATIO)
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            top_idx = np.argsort(adj_abs[i])[-k:]
            adj[i, top_idx] = adj_z[i, top_idx]
        return np.maximum(adj, adj.T)

    # ── GNN Architecture (copied from train_hierarchical_gnn_kd.py) ──────
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
            e = F.leaky_relu(
                self.a_src(Wh).squeeze(-1).unsqueeze(2) +
                self.a_dst(Wh).squeeze(-1).unsqueeze(1), negative_slope=0.2)
            e = e + adj.unsqueeze(-1) * 0.5
            e = e.masked_fill((adj.abs() < 1e-6).unsqueeze(-1), -1e9)
            alpha = self.dropout(F.softmax(e, dim=2))
            alpha_t = alpha.permute(0,3,1,2).reshape(B*self.H, N, N)
            Wh_t    = Wh.permute(0,2,1,3).reshape(B*self.H, N, self.d)
            out = torch.bmm(alpha_t, Wh_t).reshape(B, self.H, N, self.d)
            out = out.permute(0,2,1,3).reshape(B, N, self.out_dim)
            out = self.bn(out.reshape(B*N,-1)).reshape(B, N, -1)
            return F.elu(self.dropout(out)) + Wh_flat

    class StudentGNN(nn.Module):
        def __init__(self, input_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
            super().__init__()
            self.node_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Dropout(0.2))
            self.bn_input = nn.BatchNorm1d(hidden_dim)
            self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.gat1 = GATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=0.2)
            self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=0.2)
            self.gat3 = GATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=0.2)
            self.vn_update = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ELU())
            self.register_buffer('pooling_mat', POOLING_MAT)
            self.net_attn = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1))
            self.net_ln = nn.LayerNorm(N_NETWORKS * hidden_dim + hidden_dim)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(N_NETWORKS*hidden_dim+hidden_dim, 256),
                nn.ELU(), nn.Dropout(dropout/2), nn.Linear(256, 2))
            nn.init.normal_(self.virtual_node_emb, std=0.02)
        def forward(self, x, adj):
            B, N, _ = x.shape
            h = self.bn_input(self.node_encoder(x).reshape(B*N,-1)).reshape(B, N, -1)
            vn = self.virtual_node_emb.expand(B,-1,-1) + h.mean(dim=1, keepdim=True)
            h = self.gat1(h, adj)
            vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
            h = h + vn.expand(-1,N,-1)*0.1
            h_new = self.gat2(h, adj); h = h_new + h
            vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
            h = h + vn.expand(-1,N,-1)*0.1
            h_new = self.gat3(h, adj); h = h_new + h
            vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
            pooled = torch.matmul(h.transpose(1,2), self.pooling_mat).transpose(1,2)
            pooled = pooled * torch.softmax(self.net_attn(pooled), dim=1)
            flat = self.net_ln(torch.cat([pooled.reshape(B,-1), vn.squeeze(1)], dim=1))
            return self.classifier(flat)

    # ── Dataset ───────────────────────────────────────────────────────────
    class FMRIKDDataset(Dataset):
        """
        fMRI dataset. teacher_prob is None for subjects without sMRI alignment.
        """
        def __init__(self, matrix_paths, labels, teacher_probs=None):
            # teacher_probs: numpy (N,2) or None (will fill with -1 sentinel)
            self.paths   = matrix_paths
            self.labels  = labels
            # use -1 as sentinel: no teacher signal for this subject
            if teacher_probs is not None:
                self.teacher = teacher_probs.astype(np.float32)
            else:
                self.teacher = np.full((len(labels), 2), -1.0, dtype=np.float32)
            self._cache  = {}

        def _load(self, path):
            if path not in self._cache:
                mat = np.load(path).astype(np.float32)
                self._cache[path] = (extract_node_features(mat), build_adj(mat))
            return self._cache[path]

        def __len__(self): return len(self.paths)

        def __getitem__(self, i):
            feat, adj = self._load(self.paths[i])
            return (torch.tensor(feat), torch.tensor(adj),
                    torch.tensor(self.labels[i], dtype=torch.long),
                    torch.tensor(self.teacher[i], dtype=torch.float32))

    # ── Loss ──────────────────────────────────────────────────────────────
    def kd_loss(student_logits, teacher_probs_tensor, labels, alpha, T):
        """
        teacher_probs_tensor: already [0,1] probabilities (N,2), on correct device
        T is only applied to student logits, not teacher probs
        (because we only have teacher probs, not logits)
        """
        ce = F.cross_entropy(student_logits, labels)
        q_student = F.log_softmax(student_logits / T, dim=1)
        # teacher probs used directly as soft targets (no T scaling)
        kl = F.kl_div(q_student, teacher_probs_tensor.detach(), reduction='batchmean')
        return alpha * ce + (1.0 - alpha) * kl

    # ── Load teacher soft labels ──────────────────────────────────────────
    def load_teacher_data(task_safe, split="train"):
        probs_path = PROB_DIR / f"{task_safe}_{split}_teacher_probs.npy"
        ids_path   = PROB_DIR / f"{task_safe}_{split}_subject_ids.npy"
        lbl_path   = PROB_DIR / f"{task_safe}_{split}_labels.npy"
        if not probs_path.exists():
            raise FileNotFoundError(f"Missing: {probs_path}. Run compute_kd_teacher_probs.py first.")
        return (np.load(probs_path),           # (N,2)
                np.load(ids_path, allow_pickle=True),  # (N,)
                np.load(lbl_path))             # (N,)

    def main():
        print(f"\n{'='*60}")
        print(f"Cross-Modal KD  |  Task: {args.task}")
        print(f"  α={args.alpha}  T={args.temperature}  seed={args.seed}")
        print(f"{'='*60}")

        # ── Load full fMRI training data (all 193 subjects) ───────────────
        fmri_train_df = pd.read_csv(FMRI_TRAIN_CSV)
        fmri_train_df = fmri_train_df[fmri_train_df["label"].isin(cfg["classes"])].reset_index(drop=True)
        fmri_train_df["bin_label"] = (fmri_train_df["label"] == cfg["pos"]).astype(int)

        # ── Load teacher soft labels ──────────────────────────────────────
        # For MCI_vs_AD, teacher is broken (AUC~0.58) → skip KD, use CE only
        if args.task == "MCI_vs_AD":
            print("[INFO] MCI_vs_AD: teacher quality insufficient (AUC<0.6), using CE-only mode")
            effective_alpha = 1.0
            teacher_map = {}
        else:
            effective_alpha = args.alpha
            try:
                tr_teacher_probs, tr_teacher_ids, _ = load_teacher_data(args.task, "train")
                teacher_map = {sid: probs for sid, probs in zip(tr_teacher_ids, tr_teacher_probs)}
            except FileNotFoundError:
                print(f"[WARN] Teacher data not found for {args.task}, using CE-only mode")
                effective_alpha = 1.0
                teacher_map = {}

        matrix_paths = fmri_train_df["matrix_path"].tolist()
        bin_labels   = fmri_train_df["bin_label"].tolist()
        subject_ids  = fmri_train_df["subject_id"].tolist()

        # Build per-subject teacher probs array (None → sentinel -1 in Dataset)
        n = len(bin_labels)
        all_teacher_probs = np.full((n, 2), -1.0, dtype=np.float32)
        matched = 0
        if teacher_map:
            for i, sid in enumerate(subject_ids):
                if sid in teacher_map:
                    all_teacher_probs[i] = teacher_map[sid]
                    matched += 1
        print(f"Training subjects: {n}  with teacher signal: {matched}  CE-only: {n-matched}")
        print(f"Task={args.task}  n={n}  pos={sum(bin_labels)}  alpha={effective_alpha}  T={args.temperature}")

        # ── 5-Fold CV on full training data ──────────────────────────────
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=args.seed)
        oof_probs  = np.zeros(len(bin_labels))
        fold_models = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(matrix_paths, bin_labels)):
            print(f"\n── Fold {fold+1}/{N_FOLDS} ──")
            tr_paths  = [matrix_paths[i] for i in tr_idx]
            tr_labels = [bin_labels[i]   for i in tr_idx]
            tr_tprobs = all_teacher_probs[tr_idx]

            val_paths  = [matrix_paths[i] for i in val_idx]
            val_labels = [bin_labels[i]   for i in val_idx]
            val_tprobs = all_teacher_probs[val_idx]

            # Class balancing sampler
            cc = np.bincount(tr_labels)
            sw = [1.0/cc[l] for l in tr_labels]
            sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

            tr_ds  = FMRIKDDataset(tr_paths,  tr_labels, tr_tprobs)
            val_ds = FMRIKDDataset(val_paths, val_labels, val_tprobs)
            tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, sampler=sampler, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

            model = StudentGNN().to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6)

            best_val_auc, patience_cnt, best_state = 0.0, 0, None
            for epoch in range(args.epochs):
                model.train()
                for feats, adjs, labels, t_probs in tr_loader:
                    feats, adjs, labels = feats.to(DEVICE), adjs.to(DEVICE), labels.to(DEVICE)
                    t_probs = t_probs.to(DEVICE)
                    
                    optimizer.zero_grad()
                    logits = model(feats, adjs)
                    
                    # Always compute CE on all samples
                    ce_loss = F.cross_entropy(logits, labels)
                    
                    # KD loss only on samples that HAVE teacher probs (sentinel check)
                    has_teacher = (t_probs[:, 0] >= 0)  # True where teacher is available
                    if has_teacher.any() and effective_alpha < 1.0:
                        loss = kd_loss(logits[has_teacher], t_probs[has_teacher], labels[has_teacher], effective_alpha, args.temperature)
                        # Note: we need to weight the CE part by all samples or just the has_teacher ones?
                        # The requested implementation was:
                        # q_student = F.log_softmax(logits[has_teacher] / args.temperature, dim=1)
                        # kl = F.kl_div(q_student, t_probs[has_teacher].detach(), reduction='batchmean')
                        # loss = args.alpha * ce_loss + (1.0 - args.alpha) * kl
                        
                        # Re-implementing exactly as requested in "改動 3" hint:
                        q_student = F.log_softmax(logits[has_teacher] / args.temperature, dim=1)
                        kl = F.kl_div(q_student, t_probs[has_teacher].detach(), reduction='batchmean')
                        loss = effective_alpha * ce_loss + (1.0 - effective_alpha) * kl
                    else:
                        loss = ce_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                # Val AUC
                model.eval()
                val_probs, val_true = [], []
                with torch.no_grad():
                    for feats, adjs, labels, _ in val_loader:
                        logits = model(feats.to(DEVICE), adjs.to(DEVICE))
                        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        val_probs.extend(probs); val_true.extend(labels.numpy())
                val_auc = roc_auc_score(val_true, val_probs) if len(set(val_true)) > 1 else 0.5
                if val_auc > best_val_auc:
                    best_val_auc, patience_cnt = val_auc, 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE: break

            model.load_state_dict(best_state); model.eval()
            fold_models.append(model)

            # OOF probs
            val_ds_clean = FMRIKDDataset(val_paths, val_labels, None)
            val_loader_clean = DataLoader(val_ds_clean, batch_size=1, shuffle=False, num_workers=0)
            with torch.no_grad():
                oof_probs_fold = []
                for feats, adjs, _, _ in val_loader_clean:
                    logits = model(feats.to(DEVICE), adjs.to(DEVICE))
                    oof_probs_fold.append(F.softmax(logits, dim=1)[:, 1].cpu().item())
            for j, idx in enumerate(val_idx):
                oof_probs[idx] = oof_probs_fold[j]
            print(f"   Best Val AUC = {best_val_auc:.4f}")

        oof_auc = roc_auc_score(bin_labels, oof_probs)
        oof_acc = accuracy_score(bin_labels, (oof_probs >= 0.5).astype(int))
        print(f"\nStudent OOF  AUC={oof_auc:.4f}  ACC={oof_acc:.4f}")

        # ── Test on full fMRI-only combined_test.csv ──────────────────────
        fmri_test_df = pd.read_csv(FMRI_TEST_CSV)
        fmri_test_df = fmri_test_df[fmri_test_df["label"].isin(cfg["classes"])].reset_index(drop=True)
        fmri_test_df["bin_label"] = (fmri_test_df["label"] == cfg["pos"]).astype(int)
        te_paths  = fmri_test_df["matrix_path"].tolist()
        te_labels = fmri_test_df["bin_label"].tolist()

        te_ds     = FMRIKDDataset(te_paths, te_labels, None)
        te_loader = DataLoader(te_ds, batch_size=1, shuffle=False, num_workers=0)

        # 5-fold ensemble for test
        all_fold_probs = []
        for m in fold_models:
            fold_p = []
            with torch.no_grad():
                for feats, adjs, _, _ in te_loader:
                    logits = m(feats.to(DEVICE), adjs.to(DEVICE))
                    fold_p.append(F.softmax(logits, dim=1)[:, 1].cpu().item())
            all_fold_probs.append(fold_p)
        te_probs  = np.mean(all_fold_probs, axis=0)
        te_auc    = roc_auc_score(te_labels, te_probs) if len(set(te_labels)) > 1 else 0.5
        te_acc   = accuracy_score(te_labels, (te_probs >= 0.5).astype(int))
        print(f"Student Test AUC={te_auc:.4f}  ACC={te_acc:.4f}  (n={len(te_labels)})")

        # ── Teacher test AUC (for aligned subjects) ───────────────────────
        try:
            te_teacher_probs, _, te_teacher_labels = load_teacher_data(args.task, "test")
            if len(set(te_teacher_labels)) > 1:
                teacher_test_auc = roc_auc_score(te_teacher_labels, te_teacher_probs[:, 1])
                print(f"Teacher Test AUC={teacher_test_auc:.4f}  (n={len(te_teacher_labels)}, aligned only)")
        except FileNotFoundError:
            teacher_test_auc = None

        # ── Summary Table ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS  —  Task: {args.task}")
        print(f"{'='*60}")
        print(f"{'Model':<35} {'OOF AUC':>9} {'Test AUC':>9}")
        print(f"{'-'*55}")
        print(f"{'sMRI Teacher Ensemble (test only)':35} {'N/A':>9} {teacher_test_auc:>9.4f}" if teacher_test_auc else "")
        print(f"{'fMRI GNN Student (KD-trained)':35} {oof_auc:>9.4f} {te_auc:>9.4f}")
        print(f"{'='*60}")

        # ── Save results ─────────────────────────────────────────────────
        result = {
            "task": args.task,
            "alpha": effective_alpha,
            "temperature": args.temperature,
            "student_oof": {"auc": round(oof_auc, 4), "acc": round(oof_acc, 4), "n": len(bin_labels)},
            "student_test": {"auc": round(te_auc, 4), "acc": round(te_acc, 4), "n": len(te_labels)},
            "teacher_test_auc": round(teacher_test_auc, 4) if teacher_test_auc else None,
        }
        out_path = OUT_DIR / f"kd_{args.task}_a{int(effective_alpha*10)}_T{int(args.temperature)}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[SAVED] {out_path}")

        prob_path = OUT_DIR / f"kd_{args.task}_a{int(effective_alpha*10)}_T{int(args.temperature)}_probs.npz"
        np.savez(str(prob_path),
                 test_probs=te_probs,          test_labels=np.array(te_labels),
                 oof_probs=oof_probs,           oof_labels=np.array(bin_labels))
        print(f"[SAVED] {prob_path}")

    main()
