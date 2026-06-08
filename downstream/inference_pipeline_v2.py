"""
inference_pipeline_v2.py
========================
End-to-end inference using valid (leak-free) models:
  NC/AD  : PCAG-ComBat ensemble (5 folds)
  NC/MCI : PCAG-ComBat ensemble (5 folds)
  MCI/AD : KD GNN ensemble (3 seeds)

Usage:
    python inference_pipeline_v2.py \\
        --matrix /path/to/sub_matrix_116.npy \\
        --t1     /path/to/T1.nii.gz \\
        --site   ADNI \\
        --subject_id sub-001

    # Benchmark on synthetic data (no real scan needed):
    python inference_pipeline_v2.py --benchmark
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
SCRIPTS    = os.path.dirname(BASE)

PCAG_CKPT_DIR       = os.path.join(BASE, "checkpoints", "pcag_combat_v2")
PCAG_FMRI_ONLY_DIR  = os.path.join(BASE, "checkpoints", "pcag_combat_v2_fmri_only_fmricombat_nolabel")
PCAG_SMRI_ONLY_DIR  = os.path.join(BASE, "checkpoints", "pcag_combat_v2_smri_only_fmricombat_nolabel")

# MCI_vs_AD 5-seed noaug ensemble（論文 median 策略，所有 seed 都已訓練完畢）
_NOAUG_BASE = os.path.join(BASE, "checkpoints", "pcag_ensemble_noaug_v2_fmricombat_nolabel")
PCAG_MCIAD_ENSEMBLE_DIRS = [
    _NOAUG_BASE,
    _NOAUG_BASE + "_s123",
    _NOAUG_BASE + "_s456",
    _NOAUG_BASE + "_s789",
    _NOAUG_BASE + "_s2024",
]
KD_CKPT_DIR         = os.path.join(SCRIPTS, "checkpoints", "resnet_checkpoints", "gnn_checkpoints")
VIT_CKPT       = os.path.join(SCRIPTS, "models", "checkpoints",
                               "finetune_tpmic_full", "vit_mci.ckpt")
AAL_ATLAS_PATH = os.path.join(os.path.dirname(SCRIPTS), "datasets", "nilearn_data",
                               "aal_SPM12", "aal", "atlas", "AAL.nii")
SMRI_SAL_DIR   = os.path.join(BASE, "results", "smri_saliency")
os.makedirs(SMRI_SAL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants (must match training) ──────────────────────────────────────────
NETWORK_MAP = {
    'DMN':   [34,35,66,67,64,65,22,23,24,25],
    'SMN':   [0,1,56,57,68,69],
    'VN':    [42,43,44,45,46,47,48,49,50,51,52,53],
    'SN':    [28,29,30,31,32,33],
    'FPN':   [6,7,58,59,60,61],
    'LN':    [36,37,38,39,40,41],
    'VAN':   [10,11,14,15],
    'BGN':   [70,71,72,73,74,75,76,77],
    'CereN': list(range(90,116)),
}
N_ROIS, N_NETWORKS, HIDDEN_DIM = 116, 9, 128
NODE_FEAT_DIM = N_ROIS + 5 + 2 + 2   # 125
FMRI_EMBED_DIM = N_NETWORKS * HIDDEN_DIM + HIDDEN_DIM  # 1280
K_RATIO = 0.20
net_list = list(NETWORK_MAP.keys())
roi_to_net = {node: i for i, (net, nodes) in enumerate(NETWORK_MAP.items()) for node in nodes}
POOLING_MAT = torch.zeros(N_ROIS, N_NETWORKS)
for i, net in enumerate(net_list):
    for node in NETWORK_MAP[net]:
        POOLING_MAT[node, i] = 1.0


# ── Feature extraction helpers ────────────────────────────────────────────────
def extract_node_features(adj_z):
    N = adj_z.shape[0]; adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO); adj_bin = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]; adj_bin[i, top_idx] = 1.0
    adj_bin = np.maximum(adj_bin, adj_bin.T); degree = adj_bin.sum(axis=1)
    cc = np.diag(adj_bin @ adj_bin @ adj_bin) / (degree * (degree - 1) + 1e-8)
    adj_abs_thresh = adj_abs * adj_bin; pc = np.zeros(N, dtype=np.float32)
    for i in range(N):
        ki = adj_abs_thresh[i].sum()
        if ki > 1e-8:
            pc_i = 1.0
            for net_nodes in NETWORK_MAP.values():
                kim = adj_abs_thresh[i, list(net_nodes)].sum(); pc_i -= (kim / ki) ** 2
            pc[i] = float(np.clip(pc_i, 0.0, 1.0))
    features = []
    for i in range(N):
        row = adj_z[i].copy(); row[i] = 0
        stat_feat = np.array([row.mean(), row.std(), (row>0).mean(), (row<0).mean(),
                              (np.abs(row)>0.1).sum()], dtype=np.float32)
        net_i = roi_to_net.get(i, -1)
        if net_i >= 0:
            w_nodes = [r for r in NETWORK_MAP[net_list[net_i]] if r != i]
            b_nodes = [r for r in range(N) if r != i and roi_to_net.get(r,-1) != net_i]
            w_fc = float(np.mean([row[r] for r in w_nodes])) if w_nodes else 0.0
            b_fc = float(np.mean([row[r] for r in b_nodes])) if b_nodes else 0.0
        else: w_fc, b_fc = 0.0, 0.0
        features.append(np.concatenate([row.astype(np.float32), stat_feat,
                                         np.array([w_fc, b_fc, cc[i], pc[i]], dtype=np.float32)]))
    return np.stack(features).astype(np.float32)

def build_adj(adj_z):
    N = adj_z.shape[0]; adj_abs = np.abs(adj_z.copy()); np.fill_diagonal(adj_abs, 0)
    k = int(N * K_RATIO); adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        top_idx = np.argsort(adj_abs[i])[-k:]; adj[i, top_idx] = adj_z[i, top_idx]
    return np.maximum(adj, adj.T)


# ── Model definitions (identical to training scripts) ─────────────────────────
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.H = num_heads; self.d = out_dim // num_heads; self.out_dim = out_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(self.d, 1, bias=False); self.a_dst = nn.Linear(self.d, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_dim); self.dropout = nn.Dropout(dropout)
        self._last_alpha = None   # 推論時抓 attention 用
    def forward(self, h, adj):
        B, N, _ = h.shape; Wh_flat = self.W(h); Wh = Wh_flat.view(B, N, self.H, self.d)
        e = F.leaky_relu(self.a_src(Wh).squeeze(-1).unsqueeze(2) +
                         self.a_dst(Wh).squeeze(-1).unsqueeze(1), negative_slope=0.2)
        e = e + adj.unsqueeze(-1) * 0.5
        e = e.masked_fill((adj.abs() < 1e-6).unsqueeze(-1), -1e9)
        alpha = self.dropout(F.softmax(e, dim=2))
        # 儲存 attention：shape (B, N, N, H)；inference 時方便 rollout
        if not self.training:
            self._last_alpha = alpha.detach()
        alpha_t = alpha.permute(0,3,1,2).reshape(B*self.H, N, N)
        Wh_t = Wh.permute(0,2,1,3).reshape(B*self.H, N, self.d)
        out = torch.bmm(alpha_t, Wh_t).reshape(B, self.H, N, self.d).permute(0,2,1,3).reshape(B, N, self.out_dim)
        out = self.bn(out.reshape(B*N, -1)).reshape(B, N, -1)
        return F.elu(self.dropout(out)) + Wh_flat

class FMRIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM, HIDDEN_DIM), nn.ELU(), nn.Dropout(0.2))
        self.bn_input = nn.BatchNorm1d(HIDDEN_DIM)
        self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.gat1 = GATLayer(HIDDEN_DIM, HIDDEN_DIM)
        self.gat2 = GATLayer(HIDDEN_DIM, HIDDEN_DIM)
        self.gat3 = GATLayer(HIDDEN_DIM, HIDDEN_DIM)
        self.vn_update = nn.Sequential(nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), nn.ELU())
        self.register_buffer('pooling_mat', POOLING_MAT)
        self.net_attn = nn.Sequential(nn.Linear(HIDDEN_DIM, 32), nn.Tanh(), nn.Linear(32, 1))
        self.net_ln = nn.LayerNorm(N_NETWORKS * HIDDEN_DIM + HIDDEN_DIM)
        nn.init.normal_(self.virtual_node_emb, std=0.02)
    def forward(self, x, adj):
        B, N, _ = x.shape
        h = self.bn_input(self.node_encoder(x).reshape(B*N,-1)).reshape(B, N, -1)
        vn = self.virtual_node_emb.expand(B,-1,-1) + h.mean(dim=1, keepdim=True)
        h = self.gat1(h, adj); vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat2(h, adj) + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat3(h, adj) + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        pooled = torch.matmul(h.transpose(1,2), self.pooling_mat).transpose(1,2)
        net_attn_weights = torch.softmax(self.net_attn(pooled), dim=1)  # (B, 9, 1)
        pooled = pooled * net_attn_weights
        # 推論時記錄 net attention 給 R-3 抓
        if not self.training:
            self._last_net_attn = net_attn_weights.detach().squeeze(-1)  # (B, 9)
        return self.net_ln(torch.cat([pooled.reshape(B,-1), vn.squeeze(1)], dim=1))

class PCAGFusion(nn.Module):
    def __init__(self, fmri_dim=1280, smri_dim=768, fusion_dim=20, num_classes=2):
        super().__init__()
        self.fmri_proj = nn.Linear(fmri_dim, fusion_dim); self.smri_proj_k = nn.Linear(smri_dim, fusion_dim)
        self.smri_proj_v = nn.Linear(smri_dim, fusion_dim); self.W_e = nn.Linear(fusion_dim, fusion_dim)
        self.W_g1 = nn.Linear(fusion_dim, fusion_dim); self.W_g2 = nn.Linear(fusion_dim, fusion_dim)
        self.ln_e = nn.LayerNorm(fusion_dim); self.ln_g = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim, num_classes))
    def forward(self, fmri_emb, smri_feat, return_attn: bool = False):
        Q = self.fmri_proj(fmri_emb); K = self.smri_proj_k(smri_feat); V = self.smri_proj_v(smri_feat)
        P = (torch.tanh(Q) * torch.tanh(K) + 1) / 2
        A_gated = (Q * K) * P; S = torch.sigmoid(A_gated); V_hat = S * V
        E = F.relu(self.W_e(V_hat)); G = F.relu(self.W_g1(V_hat) + self.W_g2(Q))
        C = self.ln_e(E) * self.ln_g(G)
        logits = self.classifier(C + F.relu(Q))
        if return_attn:
            return logits, {"S": S.detach(), "A_gated": A_gated.detach(),
                            "Q": Q.detach(), "K": K.detach()}
        return logits

class BidirectionalPCAGFusion(nn.Module):
    """Phase-1 upgrade: bidirectional cross-attention (fMRI↔sMRI), param-efficient."""
    def __init__(self, fmri_dim=1280, smri_dim=768, fusion_dim=20, num_classes=2):
        super().__init__()
        self.fmri_proj   = nn.Linear(fmri_dim, fusion_dim)
        self.smri_proj_k = nn.Linear(smri_dim,  fusion_dim)
        self.smri_proj_v = nn.Linear(smri_dim,  fusion_dim)
        self.W_e1  = nn.Linear(fusion_dim, fusion_dim)
        self.W_g11 = nn.Linear(fusion_dim, fusion_dim)
        self.W_g12 = nn.Linear(fusion_dim, fusion_dim)
        self.ln_e1 = nn.LayerNorm(fusion_dim)
        self.ln_g1 = nn.LayerNorm(fusion_dim)
        self.smri_proj   = nn.Linear(smri_dim,  fusion_dim)
        self.fmri_proj_k = nn.Linear(fmri_dim,  fusion_dim)
        self.fmri_proj_v = nn.Linear(fmri_dim,  fusion_dim)
        self.W_e2  = nn.Linear(fusion_dim, fusion_dim)
        self.W_g21 = nn.Linear(fusion_dim, fusion_dim)
        self.W_g22 = nn.Linear(fusion_dim, fusion_dim)
        self.ln_e2 = nn.LayerNorm(fusion_dim)
        self.ln_g2 = nn.LayerNorm(fusion_dim)
        self.merge_w = nn.Parameter(torch.tensor(0.5))
        self.ln_out  = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim, num_classes))

    def _pcag_attn(self, Q, K, V, W_e, W_g1, W_g2, ln_e, ln_g):
        P = (torch.tanh(Q) * torch.tanh(K) + 1) / 2
        V_hat = torch.sigmoid((Q * K) * P) * V
        E = F.relu(W_e(V_hat)); G = F.relu(W_g1(V_hat) + W_g2(Q))
        return ln_e(E) * ln_g(G)

    def forward(self, fmri_emb, smri_feat, return_attn: bool = False):
        Q_f = self.fmri_proj(fmri_emb)
        Q_s = self.smri_proj(smri_feat)
        C1 = self._pcag_attn(Q_f, self.smri_proj_k(smri_feat), self.smri_proj_v(smri_feat),
                              self.W_e1, self.W_g11, self.W_g12, self.ln_e1, self.ln_g1) + F.relu(Q_f)
        C2 = self._pcag_attn(Q_s, self.fmri_proj_k(fmri_emb), self.fmri_proj_v(fmri_emb),
                              self.W_e2, self.W_g21, self.W_g22, self.ln_e2, self.ln_g2) + F.relu(Q_s)
        w = torch.sigmoid(self.merge_w)
        out = self.ln_out(w * C1 + (1 - w) * C2)
        logits = self.classifier(out)
        if return_attn:
            return logits, {"Q_f": Q_f.detach(), "Q_s": Q_s.detach(),
                            "C1": C1.detach(), "C2": C2.detach(),
                            "merge_w": w.item()}
        return logits


class PCAGModel(nn.Module):
    def __init__(self, fusion_dim=20, bidir_fusion=False):
        super().__init__()
        self.fmri_encoder = FMRIEncoder()
        if bidir_fusion:
            self.pcag = BidirectionalPCAGFusion(fusion_dim=fusion_dim)
        else:
            self.pcag = PCAGFusion(fusion_dim=fusion_dim)

    def forward(self, x_node, adj, smri_feat, return_attn: bool = False):
        fmri_emb = self.fmri_encoder(x_node, adj)
        if return_attn:
            logits, attn = self.pcag(fmri_emb, smri_feat, return_attn=True)
            attn["fmri_emb"] = fmri_emb.detach()
            return logits, attn
        return self.pcag(fmri_emb, smri_feat)

class KDGNNModel(nn.Module):
    """FNPGNNv8_KD stripped to encoder + MCI/AD head for inference."""
    def __init__(self):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(NODE_FEAT_DIM, HIDDEN_DIM), nn.ELU(), nn.Dropout(0.2))
        self.bn_input = nn.BatchNorm1d(HIDDEN_DIM)
        self.virtual_node_emb = nn.Parameter(torch.zeros(1, 1, HIDDEN_DIM))
        self.gat1 = GATLayer(HIDDEN_DIM, HIDDEN_DIM)
        self.gat2 = GATLayer(HIDDEN_DIM, HIDDEN_DIM)
        self.gat3 = GATLayer(HIDDEN_DIM, HIDDEN_DIM)
        self.vn_update = nn.Sequential(nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), nn.ELU())
        self.register_buffer('pooling_mat', POOLING_MAT)
        self.net_attn = nn.Sequential(nn.Linear(HIDDEN_DIM, 32), nn.Tanh(), nn.Linear(32, 1))
        self.net_ln = nn.LayerNorm(N_NETWORKS * HIDDEN_DIM + HIDDEN_DIM)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(FMRI_EMBED_DIM, 256),
            nn.ELU(), nn.Dropout(0.2), nn.Linear(256, 2)
        )
        nn.init.normal_(self.virtual_node_emb, std=0.02)
    def _encode(self, x, adj):
        B, N, _ = x.shape
        h = self.bn_input(self.node_encoder(x).reshape(B*N,-1)).reshape(B, N, -1)
        vn = self.virtual_node_emb.expand(B,-1,-1) + h.mean(dim=1, keepdim=True)
        h = self.gat1(h, adj); vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat2(h, adj) + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        h = h + vn.expand(-1,N,-1)*0.1
        h = self.gat3(h, adj) + h; vn = self.vn_update(torch.cat([vn, h.mean(dim=1,keepdim=True)], dim=-1))
        pooled = torch.matmul(h.transpose(1,2), self.pooling_mat).transpose(1,2)
        pooled = pooled * torch.softmax(self.net_attn(pooled), dim=1)
        return self.net_ln(torch.cat([pooled.reshape(B,-1), vn.squeeze(1)], dim=1))
    def forward(self, x, adj):
        return self.classifier(self._encode(x, adj))


# ── BrainIAC feature extractor ────────────────────────────────────────────────
class BrainIACExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        from monai.networks.nets import ViT
        self.vit = ViT(
            in_channels=1, img_size=(96,96,96), patch_size=(16,16,16),
            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
            classification=False, qkv_bias=False, save_attn=True,  # R-Fix-4
        )
        raw = torch.load(VIT_CKPT, map_location='cpu', weights_only=False)
        sd  = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
        strips = ['model.backbone.backbone.', 'backbone.backbone.', 'model.backbone.', 'backbone.']
        best_p, best_c = '', 0
        for p in strips:
            c = sum(1 for k in sd if k.startswith(p))
            if c > best_c: best_c, best_p = c, p
        cleaned = {k[len(best_p):]: v for k, v in sd.items() if k.startswith(best_p)}
        self.vit.load_state_dict(cleaned, strict=False)
        for p in self.vit.parameters(): p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        seq_out, _ = self.vit(x)
        return seq_out.mean(dim=1)   # (B, 768)

    @torch.no_grad()
    def attention_rollout(self, x) -> np.ndarray:
        """Compute attention rollout → (96,96,96) saliency map."""
        _ = self.vit(x)   # triggers save_attn into block.attn.att_mat
        n_patches = 216   # 6×6×6
        rollout = torch.eye(n_patches).unsqueeze(0).to(x.device)
        for block in self.vit.blocks:
            att = getattr(block.attn, "att_mat", None)
            if att is None:
                continue
            att = att.detach().float()
            att = att.mean(dim=1)                              # avg heads → (B,216,216)
            att_r = 0.5 * att + 0.5 * torch.eye(n_patches, device=x.device).unsqueeze(0)
            att_r = att_r / att_r.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            rollout = torch.bmm(att_r, rollout)
        # Use max over heads (not mean) for sharper maps
        importance = rollout[0].max(dim=0).values.cpu().numpy()  # (216,)
        sal_3d = torch.tensor(importance.reshape(6,6,6)).unsqueeze(0).unsqueeze(0).float()
        sal_96 = F.interpolate(sal_3d, size=(96,96,96), mode='trilinear',
                               align_corners=False).squeeze().numpy()
        # Keep only top 10% activations — ViT rollout is diffuse, need aggressive threshold
        threshold = np.percentile(sal_96, 90)
        sal_96 = np.where(sal_96 >= threshold, sal_96 - threshold, 0.0)
        vmax = sal_96.max()
        if vmax > 0:
            sal_96 = sal_96 / vmax
        return sal_96.astype(np.float32)


# ── ComBat single-sample application ─────────────────────────────────────────
def load_combat_params(task: str, ckpt_dir: str = None) -> dict:
    base = ckpt_dir or PCAG_CKPT_DIR
    path = os.path.join(base, f"combat_params_{task}.json")
    with open(path) as f:
        p = json.load(f)
    return {
        'site_map':   p['site_map'],
        'stand_mean': np.array(p['stand_mean']),   # (768, n_samples)
        'var_pooled': np.array(p['var_pooled']),   # (768,)
        'gamma_star': np.array(p['gamma_star']),   # (n_sites, 768)
        'delta_star': np.array(p['delta_star']),   # (n_sites, 768)
    }

def apply_combat(x: np.ndarray, site: str, params: dict) -> np.ndarray:
    """Apply ComBat harmonization to a single 768-d sMRI feature vector."""
    site_idx = params['site_map'].get(site, 0)
    grand_mean = params['stand_mean'].mean(axis=1)   # (768,)
    var_pooled  = params['var_pooled']
    gamma = params['gamma_star'][site_idx]            # (768,)
    delta = params['delta_star'][site_idx]            # (768,)
    x_std = (x - grand_mean) / (np.sqrt(var_pooled) + 1e-8)
    x_adj = (x_std - gamma) / (np.sqrt(delta) + 1e-8)
    return x_adj * np.sqrt(var_pooled) + grand_mean


# ── Model loader ──────────────────────────────────────────────────────────────
def load_pcag_models(task: str, ckpt_dir: str = None):
    base = ckpt_dir or PCAG_CKPT_DIR
    models = []
    for fold in range(5):
        path = os.path.join(base, f"pcag_combat_{task}_fold{fold}.pt")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        is_bidir = ckpt.get('model_version', 'v1') == 'bidir_v1'
        m = PCAGModel(fusion_dim=ckpt.get('fusion_dim', 20),
                      bidir_fusion=is_bidir).to(DEVICE)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        models.append(m)
    return models

def load_pcag_models_multi_seed(task: str, seed_dirs: list):
    """從多個 seed 目錄載入模型，用於 5-seed ensemble（如 MCI_vs_AD）。"""
    all_models = []
    for d in seed_dirs:
        try:
            all_models.extend(load_pcag_models(task, d))
        except Exception as e:
            print(f"  [warn] 跳過 {d}: {e}")
    return all_models

def load_kd_models():
    models = []
    for seed in [42, 123, 456]:
        path = os.path.join(KD_CKPT_DIR, f"gnn_MCI_vs_AD_seed{seed}.pt")
        sd = torch.load(path, map_location='cpu', weights_only=False)
        m = KDGNNModel().to(DEVICE)
        # Load only matching keys (KD checkpoint has extra heads we don't use)
        own_keys = set(m.state_dict().keys())
        filtered = {k: v for k, v in sd.items() if k in own_keys}
        m.load_state_dict(filtered, strict=False)
        m.eval()
        models.append(m)
    return models


# ── sMRI preprocessing ────────────────────────────────────────────────────────
def get_smri_transform():
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst,
        ResizeWithPadOrCrop, NormalizeIntensity, EnsureType,
    )
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ResizeWithPadOrCrop(spatial_size=(96,96,96), mode='constant'),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        EnsureType(data_type='tensor', dtype=torch.float32),
    ])


# ── Main inference class ──────────────────────────────────────────────────────
class InferencePipelineV2:
    def __init__(self):
        print(f"Loading models on {DEVICE}...")
        t0 = time.perf_counter()

        self.brainiac   = BrainIACExtractor().to(DEVICE).eval()
        self.smri_tfm   = get_smri_transform()

        # Dual-modal (fusion) models
        # NC_vs_AD: seed=42 aug（5 folds）
        # NC_vs_MCI: seed=456 noaug（5 folds，論文 single_best 策略）
        # MCI_vs_AD: 5-seed noaug median ensemble（25 models，論文 median 策略）
        self.pcag_nc_ad   = load_pcag_models('NC_vs_AD')
        self.pcag_nc_mci  = load_pcag_models('NC_vs_MCI')
        self.pcag_mci_ad  = load_pcag_models_multi_seed('MCI_vs_AD', PCAG_MCIAD_ENSEMBLE_DIRS)
        self.kd_mci_ad    = load_kd_models()

        # fMRI-only ablation models (all 3 tasks)
        self.pcag_fmri_nc_ad  = load_pcag_models('NC_vs_AD',  PCAG_FMRI_ONLY_DIR)
        self.pcag_fmri_nc_mci = load_pcag_models('NC_vs_MCI', PCAG_FMRI_ONLY_DIR)
        self.pcag_fmri_mci_ad = load_pcag_models('MCI_vs_AD', PCAG_FMRI_ONLY_DIR)

        # sMRI-only ablation models (NC_vs_AD and MCI_vs_AD only; NC_vs_MCI uses BrainIAC head)
        self.pcag_smri_nc_ad  = load_pcag_models('NC_vs_AD',  PCAG_SMRI_ONLY_DIR)
        self.pcag_smri_mci_ad = load_pcag_models('MCI_vs_AD', PCAG_SMRI_ONLY_DIR)

        # BrainIAC fine-tuned NC vs MCI classifier head (vit_mci.ckpt → model.classifier.fc)
        self.brainiac_nc_mci_fc = self._load_brainiac_classifier_head()

        self.combat_nc_ad   = load_combat_params('NC_vs_AD')
        self.combat_nc_mci  = load_combat_params('NC_vs_MCI')
        self.combat_mci_ad  = load_combat_params('MCI_vs_AD')

        print(f"  Models loaded in {time.perf_counter()-t0:.2f}s")

    @torch.no_grad()
    def _encode_fmri(self, matrix_path: str):
        mat = np.load(matrix_path)
        if mat.ndim == 3:
            mat = mat[0]
        if mat.shape != (N_ROIS, N_ROIS):
            n = mat.shape[0]
            if n < N_ROIS:
                # Zero-pad missing ROIs to 116×116
                padded = np.zeros((N_ROIS, N_ROIS), dtype=mat.dtype)
                padded[:n, :n] = mat
                mat = padded
                print(f"  [warn] fMRI matrix {mat.shape} padded {n}→{N_ROIS}")
            else:
                raise ValueError(
                    f"fMRI matrix shape {mat.shape} != ({N_ROIS},{N_ROIS}): {matrix_path}"
                )
        feat  = extract_node_features(mat)
        adj   = build_adj(mat)
        x     = torch.tensor(feat).unsqueeze(0).to(DEVICE)
        adj_t = torch.tensor(adj).unsqueeze(0).to(DEVICE)
        return x, adj_t

    @torch.no_grad()
    def _encode_smri(self, t1_path: str):
        vol = self.smri_tfm(t1_path).unsqueeze(0).to(DEVICE)
        return self.brainiac(vol).squeeze(0).cpu().numpy()   # (768,)

    @torch.no_grad()
    def _pcag_predict(self, models, x, adj, smri_harmonized):
        smri_t = torch.tensor(smri_harmonized, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        probs  = []
        for m in models:
            result = m(x, adj, smri_t)
            # PCAGModel.forward() 回傳 logits（inference 版不含 align_loss）
            logits = result[0] if isinstance(result, tuple) else result
            out = F.softmax(logits, dim=1)
            probs.append(out[0, 1].item())
        return float(np.median(probs) if len(probs) > 5 else np.mean(probs))

    @torch.no_grad()
    def _kd_predict(self, models, x, adj):
        probs = []
        for m in models:
            out = F.softmax(m(x, adj), dim=1)
            probs.append(out[0, 1].item())
        return float(np.mean(probs))

    def predict(self, matrix_path: str, t1_path: str, site: str = 'ADNI',
                subject_id: str = 'unknown', verbose: bool = True):
        timings = {}

        # 1. fMRI preprocessing
        t = time.perf_counter()
        x, adj = self._encode_fmri(matrix_path)
        timings['fmri_preprocess'] = time.perf_counter() - t

        # 2. sMRI feature extraction (BrainIAC ViT)
        t = time.perf_counter()
        smri_feat = self._encode_smri(t1_path)
        timings['smri_brainiac'] = time.perf_counter() - t

        # 3. ComBat harmonization
        t = time.perf_counter()
        smri_nc_ad  = apply_combat(smri_feat, site, self.combat_nc_ad)
        smri_nc_mci = apply_combat(smri_feat, site, self.combat_nc_mci)
        smri_mci_ad = apply_combat(smri_feat, site, self.combat_mci_ad)
        timings['combat'] = time.perf_counter() - t

        # 4. Task predictions (all PCAG dual-modal)
        t = time.perf_counter()
        p_nc_ad  = self._pcag_predict(self.pcag_nc_ad,  x, adj, smri_nc_ad)
        timings['pcag_nc_ad'] = time.perf_counter() - t

        t = time.perf_counter()
        p_nc_mci = self._pcag_predict(self.pcag_nc_mci, x, adj, smri_nc_mci)
        timings['pcag_nc_mci'] = time.perf_counter() - t

        t = time.perf_counter()
        p_mci_ad = self._pcag_predict(self.pcag_mci_ad, x, adj, smri_mci_ad)
        timings['pcag_mci_ad'] = time.perf_counter() - t

        # 5. OVO → 3-class
        votes = {'NC': 0, 'MCI': 0, 'AD': 0}
        if p_nc_ad  >= 0.5: votes['AD']  += 1
        else:                votes['NC']  += 1
        if p_nc_mci >= 0.5: votes['MCI'] += 1
        else:                votes['NC']  += 1
        if p_mci_ad >= 0.5: votes['AD']  += 1
        else:                votes['MCI'] += 1
        prediction = max(votes, key=votes.get)
        timings['total'] = sum(timings.values())

        if verbose:
            print(f"\n{'='*55}")
            print(f"  Subject: {subject_id}  |  Site: {site}")
            print(f"{'='*55}")
            print(f"  OVO probabilities:")
            print(f"    NC vs AD  → P(AD)  = {p_nc_ad:.3f}")
            print(f"    NC vs MCI → P(MCI) = {p_nc_mci:.3f}")
            print(f"    MCI vs AD → P(AD)  = {p_mci_ad:.3f}")
            print(f"  Votes: {votes}")
            print(f"  Prediction: {prediction}")
            print(f"\n  Timings:")
            for k, v in timings.items():
                label = f"    {k:<22}"
                print(f"{label}: {v*1000:7.1f} ms")

        return {
            'subject_id': subject_id,
            'prediction': prediction,
            'votes': votes,
            'probs': {'NC_vs_AD': p_nc_ad, 'NC_vs_MCI': p_nc_mci, 'MCI_vs_AD': p_mci_ad},
            'timings_ms': {k: round(v*1000, 1) for k, v in timings.items()},
        }


    def _load_brainiac_classifier_head(self):
        """Load the fine-tuned NC vs MCI linear head from vit_mci.ckpt."""
        try:
            raw = torch.load(VIT_CKPT, map_location='cpu', weights_only=False)
            sd  = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
            w   = sd.get('model.classifier.fc.weight', sd.get('classifier.fc.weight'))
            b   = sd.get('model.classifier.fc.bias',   sd.get('classifier.fc.bias'))
            if w is None:
                return None
            fc = nn.Linear(w.shape[1], w.shape[0])
            fc.weight = nn.Parameter(w.float())
            fc.bias   = nn.Parameter(b.float())
            return fc.to(DEVICE).eval()
        except Exception as e:
            print(f"  [warning] BrainIAC classifier head load failed: {e}")
            return None

    @torch.no_grad()
    def _brainiac_nc_mci_predict(self, smri_embedding: np.ndarray) -> float:
        """Use BrainIAC fine-tuned head for NC vs MCI. Returns P(MCI)."""
        if self.brainiac_nc_mci_fc is None:
            return 0.5
        t = torch.tensor(smri_embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logit = self.brainiac_nc_mci_fc(t)
        return float(torch.sigmoid(logit).squeeze().cpu())

    def predict_fmri_only(self, matrix_path: str, site: str = 'ADNI'):
        """Run all 3 tasks using fMRI-only ablation models (zero sMRI features)."""
        x, adj = self._encode_fmri(matrix_path)
        smri_zero = np.zeros(768, dtype=np.float32)
        p_nc_ad  = self._pcag_predict(self.pcag_fmri_nc_ad,  x, adj, smri_zero)
        p_nc_mci = self._pcag_predict(self.pcag_fmri_nc_mci, x, adj, smri_zero)
        p_mci_ad = self._pcag_predict(self.pcag_fmri_mci_ad, x, adj, smri_zero)
        return {'probs': {'NC_vs_AD': p_nc_ad, 'NC_vs_MCI': p_nc_mci, 'MCI_vs_AD': p_mci_ad}}

    def predict_smri_only(self, t1_path: str, site: str = 'ADNI'):
        """
        sMRI-only inference:
          NC vs MCI  → BrainIAC fine-tuned head (vit_mci.ckpt) — best accuracy
          NC vs AD   → PCAG sMRI-only ablation model
          MCI vs AD  → PCAG sMRI-only ablation model
        """
        smri_feat   = self._encode_smri(t1_path)
        smri_nc_ad  = apply_combat(smri_feat, site, self.combat_nc_ad)
        smri_mci_ad = apply_combat(smri_feat, site, self.combat_mci_ad)
        # NC vs MCI: BrainIAC fine-tuned head (raw embedding, no combat needed)
        p_nc_mci = self._brainiac_nc_mci_predict(smri_feat)
        # NC vs AD / MCI vs AD: PCAG sMRI-only (zero fMRI graph)
        mat_zero  = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
        x_z   = torch.tensor(extract_node_features(mat_zero)).unsqueeze(0).to(DEVICE)
        adj_z = torch.tensor(build_adj(mat_zero)).unsqueeze(0).to(DEVICE)
        p_nc_ad  = self._pcag_predict(self.pcag_smri_nc_ad,  x_z, adj_z, smri_nc_ad)
        p_mci_ad = self._pcag_predict(self.pcag_smri_mci_ad, x_z, adj_z, smri_mci_ad)
        return {'probs': {'NC_vs_AD': p_nc_ad, 'NC_vs_MCI': p_nc_mci, 'MCI_vs_AD': p_mci_ad}}


# ── Benchmark (synthetic data) ────────────────────────────────────────────────
def run_benchmark(n_runs: int = 5):
    import tempfile, nibabel as nib

    print(f"Benchmark mode: {n_runs} runs on synthetic data")
    pipe = InferencePipelineV2()

    # Create synthetic data
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        mat = np.random.randn(116, 116).astype(np.float32)
        mat = (mat + mat.T) / 2; np.fill_diagonal(mat, 0)
        np.save(f.name, mat); matrix_path = f.name

    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
        vol = np.random.randn(96, 96, 96).astype(np.float32)
        img = nib.Nifti1Image(vol, np.eye(4)); nib.save(img, f.name)
        t1_path = f.name

    print(f"\nWarming up...")
    pipe.predict(matrix_path, t1_path, site='ADNI', subject_id='warmup', verbose=False)

    all_timings = []
    for i in range(n_runs):
        result = pipe.predict(matrix_path, t1_path, site='ADNI',
                              subject_id=f'synthetic-{i+1}', verbose=False)
        all_timings.append(result['timings_ms'])
        print(f"  Run {i+1}: total={result['timings_ms']['total']:.0f}ms  pred={result['prediction']}")

    print(f"\n{'='*55}")
    print(f"  Benchmark Results ({n_runs} runs, avg ± std)")
    print(f"{'='*55}")
    keys = list(all_timings[0].keys())
    for k in keys:
        vals = [t[k] for t in all_timings]
        print(f"  {k:<24}: {np.mean(vals):7.1f} ± {np.std(vals):5.1f} ms")

    os.unlink(matrix_path); os.unlink(t1_path)


# ── Compatibility interface for api_server.py ─────────────────────────────────
import re as _re, glob as _glob
from dataclasses import dataclass
from typing import Optional as _Optional
from dotenv import load_dotenv as _load_dotenv

_load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

sys.path.insert(0, os.path.dirname(__file__))
from config import SMRI_ROOT, TPMIC_SMRI_ROOT

_NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
_NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "nm6131027")

TASKS = [("NC", "AD"), ("NC", "MCI"), ("MCI", "AD")]

INFERENCE_THRESHOLDS = {
    "fmri":        {"NC_vs_AD": 0.500, "NC_vs_MCI": 0.500, "MCI_vs_AD": 0.500},
    "smri":        {"NC_vs_AD": 0.541, "NC_vs_MCI": 0.510, "MCI_vs_AD": 0.527},
    "fusion":      {"NC_vs_AD": 0.500, "NC_vs_MCI": 0.500, "MCI_vs_AD": 0.500},
    "pcag_combat": {"NC_vs_AD": 0.500, "NC_vs_MCI": 0.500, "MCI_vs_AD": 0.500},
    "kd_gnn":      {"NC_vs_AD": 0.500, "NC_vs_MCI": 0.500, "MCI_vs_AD": 0.500},
}

AAL116_NAMES = [
    'Precentral_L','Precentral_R','Frontal_Sup_L','Frontal_Sup_R',
    'Frontal_Sup_Orb_L','Frontal_Sup_Orb_R','Frontal_Mid_L','Frontal_Mid_R',
    'Frontal_Mid_Orb_L','Frontal_Mid_Orb_R','Frontal_Inf_Oper_L','Frontal_Inf_Oper_R',
    'Frontal_Inf_Tri_L','Frontal_Inf_Tri_R','Frontal_Inf_Orb_L','Frontal_Inf_Orb_R',
    'Rolandic_Oper_L','Rolandic_Oper_R','Supp_Motor_Area_L','Supp_Motor_Area_R',
    'Olfactory_L','Olfactory_R','Frontal_Sup_Med_L','Frontal_Sup_Med_R',
    'Frontal_Med_Orb_L','Frontal_Med_Orb_R','Rectus_L','Rectus_R',
    'Insula_L','Insula_R','Cingulum_Ant_L','Cingulum_Ant_R',
    'Cingulum_Mid_L','Cingulum_Mid_R','Cingulum_Post_L','Cingulum_Post_R',
    'Hippocampus_L','Hippocampus_R','ParaHippocampal_L','ParaHippocampal_R',
    'Amygdala_L','Amygdala_R','Calcarine_L','Calcarine_R',
    'Cuneus_L','Cuneus_R','Lingual_L','Lingual_R',
    'Occipital_Sup_L','Occipital_Sup_R','Occipital_Mid_L','Occipital_Mid_R',
    'Occipital_Inf_L','Occipital_Inf_R','Fusiform_L','Fusiform_R',
    'Postcentral_L','Postcentral_R','Parietal_Sup_L','Parietal_Sup_R',
    'Parietal_Inf_L','Parietal_Inf_R','SupraMarginal_L','SupraMarginal_R',
    'Angular_L','Angular_R','Precuneus_L','Precuneus_R',
    'Paracentral_Lob_L','Paracentral_Lob_R','Caudate_L','Caudate_R',
    'Putamen_L','Putamen_R','Pallidum_L','Pallidum_R',
    'Thalamus_L','Thalamus_R','Heschl_L','Heschl_R',
    'Temporal_Sup_L','Temporal_Sup_R','Temporal_Pole_Sup_L','Temporal_Pole_Sup_R',
    'Temporal_Mid_L','Temporal_Mid_R','Temporal_Pole_Mid_L','Temporal_Pole_Mid_R',
    'Temporal_Inf_L','Temporal_Inf_R','Cerebelum_Crus1_L','Cerebelum_Crus1_R',
    'Cerebelum_Crus2_L','Cerebelum_Crus2_R','Cerebelum_3_L','Cerebelum_3_R',
    'Cerebelum_4_5_L','Cerebelum_4_5_R','Cerebelum_6_L','Cerebelum_6_R',
    'Cerebelum_7b_L','Cerebelum_7b_R','Cerebelum_8_L','Cerebelum_8_R',
    'Cerebelum_9_L','Cerebelum_9_R','Cerebelum_10_L','Cerebelum_10_R',
    'Vermis_1_2','Vermis_3','Vermis_4_5','Vermis_6',
    'Vermis_7','Vermis_8','Vermis_9','Vermis_10',
]


@dataclass
class ModalityInput:
    matrix_path: _Optional[str]
    t1_path:     _Optional[str]
    subject_id:  str


def _build_t1_lookup(adni_root: str, tpmic_root: str) -> dict:
    lookup = {}
    if os.path.isdir(adni_root):
        for path in _glob.glob(os.path.join(adni_root, "**", "*_T1_MNI.nii.gz"), recursive=True):
            m = _re.match(r"(\d+_S_\d+)_T1_MNI\.nii\.gz", os.path.basename(path))
            if m:
                lookup[m.group(1)] = path
    if os.path.isdir(tpmic_root):
        for path in _glob.glob(os.path.join(tpmic_root, "**", "sub_*_T1.nii.gz"), recursive=True):
            m = _re.match(r"(sub_\d+)_T1\.nii\.gz", os.path.basename(path))
            if m:
                lookup[m.group(1)] = path
    return lookup

_T1_LOOKUP: dict = _build_t1_lookup(SMRI_ROOT, TPMIC_SMRI_ROOT)
if _T1_LOOKUP:
    print(f"[init] T1 lookup 建立完成，共 {len(_T1_LOOKUP)} 筆 sMRI 影像。")


def find_t1_path(subject_id: str) -> _Optional[str]:
    m = _re.search(r"(\d{3}_S_\d{4})", subject_id)
    if m:
        return _T1_LOOKUP.get(m.group(1))
    m = _re.search(r"(sub_\d+)", subject_id)
    if m:
        return _T1_LOOKUP.get(m.group(1))
    return None


def query_knowledge_graph(top_roi_names: list, task_pair: tuple) -> dict:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return {"error": "neo4j 套件未安裝"}
    try:
        driver = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
    except Exception as e:
        return {"error": str(e)}

    class_a, class_b = task_pair
    task_str = f"{class_a}_vs_{class_b}"
    results = {"roi_details": [], "network_summary": []}

    with driver.session() as session:
        for roi_name in top_roi_names:
            rec = session.run("""
                MATCH (r:ROI {name: $name})
                OPTIONAL MATCH (r)-[:BELONGS_TO]->(n:BrainNetwork)
                OPTIONAL MATCH (n)-[a:ATTENTION_DIFF {task: $task}]->(d:DiseaseStage)
                RETURN r.function AS func, r.ad_relevance AS relevance, r.notes AS notes,
                       n.abbr AS network, n.fullname AS net_fullname,
                       a.diff_value AS attn_diff, a.interpretation AS interpretation
            """, name=roi_name, task=task_str).single()
            if rec:
                results["roi_details"].append({
                    "roi": roi_name, "function": rec["func"],
                    "ad_relevance": rec["relevance"], "notes": rec["notes"],
                    "network": rec["network"], "net_fullname": rec["net_fullname"],
                    "attn_diff": rec["attn_diff"], "interpretation": rec["interpretation"],
                })

        net_recs = session.run("""
            MATCH (n:BrainNetwork)-[a:ATTENTION_DIFF {task: $task}]->(d:DiseaseStage)
            RETURN n.abbr AS network, n.fullname AS fullname,
                   a.diff_value AS diff, a.interpretation AS interp
            ORDER BY abs(a.diff_value) DESC
        """, task=task_str)
        for r in net_recs:
            results["network_summary"].append({
                "network": r["network"], "fullname": r["fullname"],
                "diff": r["diff"], "interp": r["interp"],
            })

    driver.close()
    return results


def _detect_site(subject_id: str) -> str:
    if _re.search(r'\d{3}_S_\d{4}', subject_id):
        return 'ADNI'
    return 'TPMIC'


def _compute_fmri_findings(matrix_path: str) -> dict:
    """FC connectivity-based ROI saliency (mean absolute strength per node)."""
    mat = np.load(matrix_path)
    if mat.ndim == 3:
        mat = mat[0]
    mat = np.arctanh(np.clip(mat, -0.999, 0.999))
    np.fill_diagonal(mat, 0)
    sal = np.abs(mat).mean(axis=1)
    sal_mean = sal.mean()
    sal = np.clip((sal - sal_mean) / (sal.max() - sal_mean + 1e-8), 0, 1)
    top_idx = np.argsort(sal)[-10:][::-1]
    return {
        "top_regions": [{"name": AAL116_NAMES[i], "saliency": float(sal[i])} for i in top_idx],
        "saliency_116": sal.tolist(),
    }


def _compute_smri_saliency(pipe, t1_path: str, subject_id: str) -> dict:
    """
    Run BrainIAC attention rollout, save NIfTI, compute per-ROI scores.
    Returns {"saliency_path": str, "saliency_url_path": str, "top_regions": list}
    """
    try:
        import nibabel as nib
        from scipy.ndimage import zoom as ndimage_zoom

        # Derive safe_sid from T1 basename to match batch-precomputed filenames
        t1_base  = os.path.basename(t1_path)
        for suffix in ("_T1_MNI.nii.gz", "_T1.nii.gz"):
            if t1_base.endswith(suffix):
                t1_base = t1_base[:-len(suffix)]
                break
        safe_sid = t1_base.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(SMRI_SAL_DIR, f"{safe_sid}_brainiac_rollout.nii.gz")

        # Fast path: NIfTI already exists — skip rollout, just read ROI scores
        if os.path.exists(out_path):
            sal_out    = nib.load(out_path).get_fdata().astype(np.float32)
            top_regions = []
            if os.path.exists(AAL_ATLAS_PATH):
                atlas_data = np.asarray(nib.load(AAL_ATLAS_PATH).get_fdata(), dtype=np.int32)
                if atlas_data.shape != sal_out.shape:
                    from scipy.ndimage import zoom as _zoom
                    scale_a    = [sal_out.shape[i] / atlas_data.shape[i] for i in range(3)]
                    atlas_data = _zoom(atlas_data, scale_a, order=0).astype(np.int32)
                unique_labels = np.sort(np.unique(atlas_data))
                unique_labels = unique_labels[unique_labels > 0]
                for idx_0, lbl in enumerate(unique_labels):
                    if idx_0 >= len(AAL116_NAMES): break
                    mask = atlas_data == lbl
                    if mask.sum() == 0: continue
                    top_regions.append({"name": AAL116_NAMES[idx_0], "saliency": float(sal_out[mask].mean())})
                top_regions.sort(key=lambda x: x["saliency"], reverse=True)
                top_regions = top_regions[:10]
            return {"saliency_path": out_path, "top_regions": top_regions}

        vol = pipe.smri_tfm(t1_path).unsqueeze(0).to(DEVICE)
        sal_96 = pipe.brainiac.attention_rollout(vol)   # (96,96,96) float32

        # ── Save as NIfTI in patient's T1 space ──────────────────────────────
        ref_nii   = nib.load(t1_path)
        ref_shape = tuple(ref_nii.shape[:3])
        affine    = ref_nii.affine

        if ref_shape != (96, 96, 96):
            scale   = [ref_shape[i] / 96.0 for i in range(3)]
            sal_out = ndimage_zoom(sal_96, scale, order=1).astype(np.float32)
        else:
            sal_out = sal_96

        nib.save(nib.Nifti1Image(sal_out, affine), out_path)

        # ── Per-ROI scores via AAL116 ─────────────────────────────────────────
        top_regions = []
        if os.path.exists(AAL_ATLAS_PATH):
            atlas_nii  = nib.load(AAL_ATLAS_PATH)
            atlas_data = np.asarray(atlas_nii.get_fdata(), dtype=np.int32)

            # Resample atlas to saliency output shape
            if atlas_data.shape != sal_out.shape:
                scale_a    = [sal_out.shape[i] / atlas_data.shape[i] for i in range(3)]
                atlas_data = ndimage_zoom(atlas_data, scale_a, order=0).astype(np.int32)

            # Map sorted AAL labels → sequential AAL116_NAMES index
            unique_labels = np.sort(np.unique(atlas_data))
            unique_labels = unique_labels[unique_labels > 0]  # skip background

            for idx_0, lbl in enumerate(unique_labels):
                if idx_0 >= len(AAL116_NAMES):
                    break
                mask = atlas_data == lbl
                if mask.sum() == 0:
                    continue
                mean_sal = float(sal_out[mask].mean())
                top_regions.append({"name": AAL116_NAMES[idx_0], "saliency": mean_sal})

            top_regions.sort(key=lambda x: x["saliency"], reverse=True)
            top_regions = top_regions[:10]

        return {"saliency_path": out_path, "top_regions": top_regions}

    except Exception as e:
        print(f"  [smri_saliency] failed for {subject_id}: {e}")
        return {"saliency_path": None, "top_regions": []}


_PIPELINE_SINGLETON: _Optional[InferencePipelineV2] = None

def _get_pipeline() -> InferencePipelineV2:
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = InferencePipelineV2()
    return _PIPELINE_SINGLETON


def run_multimodal_inference(modality_input: ModalityInput, device: torch.device) -> dict:
    """Drop-in replacement for api_server.py — uses PCAG-ComBat + KD GNN."""
    pipe = _get_pipeline()
    has_fmri = bool(modality_input.matrix_path)
    has_smri = bool(modality_input.t1_path)
    site = _detect_site(modality_input.subject_id)

    fmri_findings = None
    if has_fmri:
        try:
            fmri_findings = _compute_fmri_findings(modality_input.matrix_path)
        except Exception as e:
            print(f"  [v2] fMRI findings 計算失敗：{e}")

    smri_saliency = {"saliency_path": None, "top_regions": []}
    if has_smri:
        smri_saliency = _compute_smri_saliency(pipe, modality_input.t1_path,
                                                modality_input.subject_id)

    results = {}

    if has_fmri and has_smri:
        try:
            v2 = pipe.predict(
                modality_input.matrix_path, modality_input.t1_path,
                site=site, subject_id=modality_input.subject_id, verbose=False,
            )
        except Exception as e:
            print(f"  [v2] predict 失敗：{e}")
            return {}

        # Run single-modal models for concordance analysis (results NOT used for OVO)
        v_fmri_only = None
        v_smri_only = None
        try:
            v_fmri_only = pipe.predict_fmri_only(modality_input.matrix_path, site=site)
        except Exception as e:
            print(f"  [v2] fMRI-only concordance 失敗：{e}")
        try:
            v_smri_only = pipe.predict_smri_only(modality_input.t1_path, site=site)
        except Exception as e:
            print(f"  [v2] sMRI-only concordance 失敗：{e}")

        for task_name, class_a, class_b, reason in [
            ("NC_vs_AD",  "NC", "AD",  "pcag_combat"),
            ("NC_vs_MCI", "NC", "MCI", "pcag_combat"),
            ("MCI_vs_AD", "MCI", "AD", "pcag_combat"),
        ]:
            prob = v2['probs'][task_name]
            pred = 1 if prob >= 0.5 else 0
            margin = abs(prob - 0.5)
            confidence = "high" if margin > 0.2 else "medium" if margin > 0.1 else "low"

            # Single-modal class labels (for report concordance only)
            if v_fmri_only:
                fp = v_fmri_only['probs'][task_name]
                fmri_pred_label = class_b if fp >= 0.5 else class_a
            else:
                fmri_pred_label = None
            if v_smri_only:
                sp = v_smri_only['probs'][task_name]
                smri_pred_label = class_b if sp >= 0.5 else class_a
            else:
                smri_pred_label = None

            results[task_name] = {
                "prediction":    pred,
                "prob_positive": prob,
                "prob_fused":    prob,
                "conf_smri":     0.0,
                "conf_fmri":     0.0,
                "modality_used": reason,
                "fusion_reason": reason,
                "fmri_findings": fmri_findings,
                "smri_findings": {"top_regions": smri_saliency["top_regions"]},
                "smri_saliency_path": smri_saliency["saliency_path"],
                "fmri_pred":     fmri_pred_label,
                "smri_pred":     smri_pred_label,
                "confidence":    confidence,
                "class_a":       class_a,
                "class_b":       class_b,
            }

    elif has_fmri and not has_smri:
        # fMRI-only：use fMRI-only ablation models for all 3 tasks
        try:
            v_fmri = pipe.predict_fmri_only(modality_input.matrix_path, site=site)
        except Exception as e:
            print(f"  [v2] fMRI-only predict 失敗：{e}")
            return {}
        for task_name, class_a, class_b in [
            ("NC_vs_AD",  "NC", "AD"),
            ("NC_vs_MCI", "NC", "MCI"),
            ("MCI_vs_AD", "MCI", "AD"),
        ]:
            prob   = v_fmri['probs'][task_name]
            pred   = 1 if prob >= 0.5 else 0
            margin = abs(prob - 0.5)
            results[task_name] = {
                "prediction":    pred,
                "prob_positive": prob,
                "prob_fused":    prob,
                "conf_smri":     0.0,
                "conf_fmri":     round(margin, 4),
                "modality_used": "fmri_only",
                "fusion_reason": "pcag_fmri_only",
                "fmri_findings": fmri_findings,
                "smri_findings": None,
                "fmri_pred":     pred,
                "smri_pred":     None,
                "confidence":    "high" if margin > 0.2 else "medium" if margin > 0.1 else "low",
                "class_a":       class_a,
                "class_b":       class_b,
        }
    elif not has_fmri and has_smri:
        # sMRI-only：use sMRI-only ablation models for all 3 tasks
        try:
            v_smri = pipe.predict_smri_only(modality_input.t1_path, site=site)
        except Exception as e:
            print(f"  [v2] sMRI-only predict 失敗：{e}")
            return {}
        for task_name, class_a, class_b in [
            ("NC_vs_AD",  "NC", "AD"),
            ("NC_vs_MCI", "NC", "MCI"),
            ("MCI_vs_AD", "MCI", "AD"),
        ]:
            prob   = v_smri['probs'][task_name]
            pred   = 1 if prob >= 0.5 else 0
            margin = abs(prob - 0.5)
            results[task_name] = {
                "prediction":    pred,
                "prob_positive": prob,
                "prob_fused":    prob,
                "conf_smri":     round(margin, 4),
                "conf_fmri":     0.0,
                "modality_used": "smri_only",
                "fusion_reason": "pcag_smri_only",
                "fmri_findings": None,
                "smri_findings": {"top_regions": smri_saliency["top_regions"]},
                "smri_saliency_path": smri_saliency["saliency_path"],
                "fmri_pred":     None,
                "smri_pred":     pred,
                "confidence":    "high" if margin > 0.2 else "medium" if margin > 0.1 else "low",
                "class_a":       class_a,
                "class_b":       class_b,
            }

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix',     type=str, help='Path to fMRI FC matrix .npy')
    parser.add_argument('--t1',         type=str, help='Path to T1 .nii.gz')
    parser.add_argument('--site',       type=str, default='ADNI', choices=['ADNI','TPMIC'])
    parser.add_argument('--subject_id', type=str, default='unknown')
    parser.add_argument('--benchmark',  action='store_true', help='Run benchmark on synthetic data')
    parser.add_argument('--n_runs',     type=int, default=5)
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(n_runs=args.n_runs)
    elif args.matrix and args.t1:
        pipe = InferencePipelineV2()
        pipe.predict(args.matrix, args.t1, args.site, args.subject_id)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
