"""
eval_finetune_fraction_curve.py
================================
Direct GNN evaluation across fine-tuning data fractions (0, 0.2, 0.4, 0.6, 0.8, 1.0).
No API server required.

Outputs:
  results/finetune_fraction_curve.json
  results/figures/Finetune_fraction_AUC_curve.png
"""

import sys, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream')

# ── Constants (same as train_finetune_gnn_tpmic.py) ──────────────────
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

# ── Model (copied from train_finetune_gnn_tpmic.py) ──────────────────
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
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck), nn.ELU(),
            nn.Dropout(0.1), nn.Linear(bottleneck, dim))
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
        def make_head():
            return nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(head_dim, 256),
                                  nn.ELU(), nn.Dropout(DROPOUT/2), nn.Linear(256, 2))
        self.head_nc_ad  = make_head()
        self.head_nc_mci = make_head()
        self.head_mci_ad = make_head()
        self.progression_head = nn.Sequential(nn.Linear(head_dim, 64), nn.ELU(), nn.Linear(64, 1))
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
        return (self.head_nc_ad(self.adapter_nc_ad(flat)),
                self.head_nc_mci(self.adapter_nc_mci(flat)),
                self.head_mci_ad(self.adapter_mci_ad(flat)),
                self.progression_head(flat), flat)

# ── Node feature extraction (same as training) ────────────────────────
def extract_node_features(adj_z):
    N = 116
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
        fc_feat   = row.astype(np.float32)
        stat_feat = np.array([row.mean(), row.std(), (row>0).mean(),
                               (row<0).mean(), (np.abs(row)>0.1).sum()], dtype=np.float32)
        net_i = roi_to_net.get(i, -1)
        if net_i >= 0:
            w_nodes = [r for r in NETWORK_MAP[net_list[net_i]] if r != i]
            b_nodes = [r for r in range(N) if r != i and roi_to_net.get(r,-1) != net_i]
            w_fc = float(np.mean([row[r] for r in w_nodes])) if w_nodes else 0.0
            b_fc = float(np.mean([row[r] for r in b_nodes])) if b_nodes else 0.0
        else:
            w_fc, b_fc = 0.0, 0.0
        topo_feat = np.array([cc[i], pc[i]], dtype=np.float32)
        features.append(np.concatenate([fc_feat, stat_feat, np.array([w_fc, b_fc]), topo_feat]))
    return np.stack(features, axis=0).astype(np.float32)

def build_adj(adj_z):
    N = 116
    adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]
        adj[i, top_idx] = adj_z[i, top_idx]
    return np.maximum(adj, adj.T)

# ── Dataset ───────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths  = paths
        self.labels = labels
        self._cache = {}
    def _load(self, path):
        if path not in self._cache:
            mat = np.load(path).astype(np.float32)
            self._cache[path] = (extract_node_features(mat), build_adj(mat))
        return self._cache[path]
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        feat, adj = self._load(self.paths[i])
        return torch.tensor(feat), torch.tensor(adj), torch.tensor(self.labels[i], dtype=torch.long)

def collate(batch):
    feats, adjs, labels = zip(*batch)
    return torch.stack(feats), torch.stack(adjs), torch.stack(labels)

# ── Evaluation ────────────────────────────────────────────────────────
TASK_CFG = {
    "NC_vs_AD":  {"classes": [0,2], "pos": 2, "head_idx": 0},
    "NC_vs_MCI": {"classes": [0,1], "pos": 1, "head_idx": 1},
    "MCI_vs_AD": {"classes": [1,2], "pos": 2, "head_idx": 2},
}

@torch.no_grad()
def evaluate(model, loader, head_idx):
    model.eval()
    all_probs, all_labels = [], []
    for feat, adj, label in loader:
        feat, adj = feat.to(DEVICE), adj.to(DEVICE)
        logits_tuple = model(feat, adj)
        logits = logits_tuple[head_idx]
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(label.numpy().tolist())
    if len(set(all_labels)) < 2:
        return float('nan')
    return roc_auc_score(all_labels, all_probs)

# ── Checkpoint map ────────────────────────────────────────────────────
CKPT_BASE = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/checkpoints")
CKPT_MAP  = {
    0.0: {task: str(CKPT_BASE / "combined_gnn/gnn_checkpoints" / f"{task}.pt") for task in TASK_CFG},
    0.2: {task: str(CKPT_BASE / f"finetune_tpmic_frac02/{task}_finetune_frac0.20.pt") for task in TASK_CFG},
    0.4: {task: str(CKPT_BASE / f"finetune_tpmic_frac04/{task}_finetune_frac0.40.pt") for task in TASK_CFG},
    0.6: {task: str(CKPT_BASE / f"finetune_tpmic_frac06/{task}_finetune_frac0.60.pt") for task in TASK_CFG},
    0.8: {task: str(CKPT_BASE / f"finetune_tpmic_frac08/{task}_finetune_frac0.80.pt") for task in TASK_CFG},
    1.0: {task: str(Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/models/checkpoints/finetune_tpmic_full") / f"{task}_finetune_frac1.00.pt") for task in TASK_CFG},
}

# Full-fraction fallback (some may be named differently)
FULL_FALLBACKS = {
    task: str(CKPT_BASE / f"finetune_tpmic_full/{task}.pt") for task in TASK_CFG
}

# ── Load test data ────────────────────────────────────────────────────
TEST_CSV = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits/combined_test.csv"
test_df  = pd.read_csv(TEST_CSV)
print(f"Test set: {len(test_df)} subjects  source: {test_df['source'].value_counts().to_dict()}")

# Build per-task loaders (cache node features once)
task_loaders = {}
for task, cfg in TASK_CFG.items():
    sub = test_df[test_df["label"].isin(cfg["classes"])].reset_index(drop=True)
    sub["bin_label"] = (sub["label"] == cfg["pos"]).astype(int)
    ds = TestDataset(sub["matrix_path"].tolist(), sub["bin_label"].tolist())
    task_loaders[task] = (DataLoader(ds, batch_size=8, collate_fn=collate), cfg["head_idx"])
    print(f"  {task}: {len(sub)} subjects  pos={sub['bin_label'].sum()}")

# ── Main evaluation loop ──────────────────────────────────────────────
results = {}   # fraction → task → auc
FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for frac in FRACTIONS:
    results[frac] = {}
    print(f"\n── Fraction {frac:.1f} ──────────────────────────────")
    for task, cfg in TASK_CFG.items():
        ckpt_path = CKPT_MAP[frac][task]
        # Fallback for full fraction
        if not Path(ckpt_path).exists() and frac == 1.0:
            ckpt_path = FULL_FALLBACKS[task]
        if not Path(ckpt_path).exists():
            print(f"  {task}: checkpoint NOT FOUND ({ckpt_path})")
            results[frac][task] = None
            continue

        model = FNPGNNv8_E13().to(DEVICE)
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(sd, strict=True)

        loader, head_idx = task_loaders[task]
        auc = evaluate(model, loader, head_idx)
        results[frac][task] = round(auc, 4) if not np.isnan(auc) else None
        print(f"  {task}: AUC={auc:.4f}  ({ckpt_path.split('/')[-1]})")

# ── Save JSON ─────────────────────────────────────────────────────────
OUT_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results")
out_json = OUT_DIR / "finetune_fraction_curve.json"
with open(out_json, "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print(f"\n[SAVED] {out_json}")

# ── Plot ──────────────────────────────────────────────────────────────
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

TASK_COLORS = {"NC_vs_AD": "#E07B7B", "NC_vs_MCI": "#5B9BD5", "MCI_vs_AD": "#70AD47"}
TASK_LABELS = {"NC_vs_AD": "NC vs AD", "NC_vs_MCI": "NC vs MCI", "MCI_vs_AD": "MCI vs AD"}

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("fMRI GNN AUC vs TPMIC Fine-tuning Data Fraction",
             fontsize=13, fontweight="bold")

for task in TASK_CFG:
    aucs  = [results[f][task] for f in FRACTIONS]
    valid = [(f, a) for f, a in zip(FRACTIONS, aucs) if a is not None]
    if not valid: continue
    xs, ys = zip(*valid)
    ax.plot(xs, ys, marker="o", markersize=7, linewidth=2.2,
            color=TASK_COLORS[task], label=TASK_LABELS[task])
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5,
                    color=TASK_COLORS[task])

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Random (0.5)")
ax.set_xlabel("TPMIC Labeled Data Fraction", fontsize=11)
ax.set_ylabel("Test AUC (combined_test)", fontsize=11)
ax.set_xticks(FRACTIONS)
ax.set_xticklabels(["0\n(zero-shot)", "0.2", "0.4", "0.6", "0.8", "1.0\n(full FT)"])
ax.set_ylim(0.35, 1.05)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_fig = FIG_DIR / "Finetune_fraction_AUC_curve.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"[SAVED] {out_fig}")
plt.close()

# ── Console table ─────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"{'Fraction':<10} {'NC_vs_AD':>10} {'NC_vs_MCI':>10} {'MCI_vs_AD':>10}")
print(f"{'-'*55}")
for frac in FRACTIONS:
    row = results[frac]
    def fmt(v): return f"{v:.4f}" if v is not None else "  N/A  "
    print(f"{frac:<10.1f} {fmt(row.get('NC_vs_AD')):>10} "
          f"{fmt(row.get('NC_vs_MCI')):>10} {fmt(row.get('MCI_vs_AD')):>10}")
