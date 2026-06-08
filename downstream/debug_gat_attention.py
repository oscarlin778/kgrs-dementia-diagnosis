"""
debug_gat_attention.py
======================
診斷 GAT attention 為什麼這麼平均：
  1. 看 raw alpha 的分布（最大值、entropy）
  2. 比較 NC vs AD 病人的 attention 模式
  3. 看 final layer alpha 是否有明顯 peak
"""
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path

from inference_pipeline_v2 import (
    load_pcag_models, load_combat_params, apply_combat,
    extract_node_features, build_adj,
    AAL116_NAMES, NETWORK_MAP, net_list, DEVICE,
)

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "pcag_test_aligned_v2.csv"
SMRI_PKL = BASE_DIR / "sid_to_smri_feat.pkl"


def detect_site(sid):
    import re
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', sid) else 'TPMIC'


def diagnose_alpha(alpha, name):
    """alpha: (B, N, N, H)"""
    a_np = alpha.cpu().numpy()
    # Average heads
    a = a_np.mean(axis=-1)[0]  # (N, N)
    # Row entropy
    row_entropy = -(a * np.log(a + 1e-9)).sum(axis=1)
    print(f"  {name}: shape={a.shape}, mean={a.mean():.4f}, max={a.max():.4f}, "
          f"99% percentile={np.percentile(a, 99):.4f}")
    print(f"    row entropy: mean={row_entropy.mean():.3f}, min={row_entropy.min():.3f}, "
          f"max={row_entropy.max():.3f}, max possible={np.log(116):.3f}")
    # How peaked: ratio of top-1 to uniform
    top1_per_row = a.max(axis=1)
    print(f"    per-row top-1 alpha: mean={top1_per_row.mean():.3f} (uniform would be ~{1/23:.3f})")
    return a


def main():
    df = pd.read_csv(CSV_PATH)
    with open(SMRI_PKL, "rb") as f:
        sid_to_feat = pickle.load(f)

    # 選一個 NC 和一個 AD 病患
    nc_sid = df[df['label']==0]['subject_id'].iloc[0]
    ad_sid = df[df['label']==2]['subject_id'].iloc[0]

    models = load_pcag_models('NC_vs_AD')
    combat_params = load_combat_params('NC_vs_AD')
    m = models[0]  # 第 1 fold

    for label_name, sid in [("NC", nc_sid), ("AD", ad_sid)]:
        print(f"\n{'='*60}")
        print(f"Patient: {sid}  (label = {label_name})")
        print('='*60)
        row = df[df['subject_id']==sid].iloc[0]
        mat = np.load(row['matrix_path'])
        if mat.ndim == 3:
            mat = mat[0]
        feat = extract_node_features(mat)
        adj = build_adj(mat)
        x = torch.tensor(feat).unsqueeze(0).to(DEVICE)
        adj_t = torch.tensor(adj).unsqueeze(0).to(DEVICE)
        smri_feat = sid_to_feat[sid]
        site = detect_site(sid)
        smri_adj = apply_combat(smri_feat, site, combat_params)
        smri_t = torch.tensor(smri_adj, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _ = m(x, adj_t, smri_t)

        enc = m.fmri_encoder
        print("\nRaw GAT attention diagnostics:")
        diagnose_alpha(enc.gat1._last_alpha, "gat1")
        diagnose_alpha(enc.gat2._last_alpha, "gat2")
        a3 = diagnose_alpha(enc.gat3._last_alpha, "gat3")

        # net_attn diagnostics
        net_w = enc._last_net_attn[0].cpu().numpy()
        print(f"\nNet attention (softmax over 9 networks):")
        for n, v in zip(net_list, net_w):
            print(f"  {n}: {v*100:.2f}%  (uniform would be {100/9:.2f}%)")

        # GAT3 alpha 最 peaked 的 ROI（看 column sum，即 "誰被注意到")
        col_attention = a3.sum(axis=0)  # (116,)
        col_norm = col_attention / col_attention.sum()
        top10 = np.argsort(col_norm)[-10:][::-1]
        print(f"\nGAT3 top-10 most-attended ROIs (column sum normalized):")
        for r in top10:
            net = next((n for n, idxs in NETWORK_MAP.items() if r in idxs), "Other")
            print(f"  {AAL116_NAMES[r]:30s}  ({net})  attn={col_norm[r]:.4f}  "
                  f"(uniform={1/116:.4f})")


if __name__ == "__main__":
    main()
