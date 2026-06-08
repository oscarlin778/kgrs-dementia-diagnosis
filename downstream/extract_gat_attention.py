"""
extract_gat_attention.py
========================
對測試集每位病患抓取 fMRI GNN 的 attention。

⚠️ 設計決策（debug 之後）：
  - 跨 5 個 fold 平均會把 attention 訊號洗成 uniform。
    Raw alpha 的 max 可以到 0.39（vs uniform 0.043），但平均 5 folds 後變成 ~0.01
  - 改用 **fold 0** 為代表，並用 raw GAT3 column attention（不做 rollout）
  - Net attention 也用 fold 0 raw 值

輸出：
  results/gat_attention_v2.npz
    keys:
      subject_ids        : (n,)
      tasks              : (3,)
      roi_importance     : (n, 3, 116)  — fold-0 GAT3 column attention normalised
      net_attention      : (n, 3, 9)    — fold-0 net_attn softmax (no avg)
      top_rois           : (n, 3, 10)
      top_roi_scores     : (n, 3, 10)
      fold_variability   : (n, 3)       — std of top-1 network across folds (sanity)
"""
import time
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path

from inference_pipeline_v2 import (
    load_pcag_models, load_combat_params, apply_combat,
    extract_node_features, build_adj,
    AAL116_NAMES, NETWORK_MAP, net_list,
    N_ROIS, N_NETWORKS, DEVICE,
)

BASE_DIR  = Path(__file__).parent
RES_DIR   = BASE_DIR / "results"
CSV_PATH  = BASE_DIR / "pcag_test_aligned_v2.csv"
SMRI_PKL  = BASE_DIR / "sid_to_smri_feat.pkl"

TASKS = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]


def detect_site(sid: str) -> str:
    import re
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', sid) else 'TPMIC'


def compute_attention_rollout(alphas: list) -> np.ndarray:
    """
    alphas: list of (B, N, N, H) tensors from each GAT layer
    回傳 (N, N) attention rollout matrix（已 average B=1 + head H）
    """
    rolled = None
    for alpha in alphas:
        # alpha: (B=1, N, N, H) → average over heads → (N, N)
        a = alpha.mean(dim=-1)[0].cpu().numpy()
        # 加上 skip connection
        a = a + np.eye(a.shape[0])
        # normalize row
        a = a / (a.sum(axis=1, keepdims=True) + 1e-8)
        if rolled is None:
            rolled = a
        else:
            rolled = a @ rolled  # later layers attend to earlier
    return rolled


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--fmri_dir", type=str, default=None)
    parser.add_argument("--out_suffix", type=str, default="")
    args = parser.parse_args()

    print(f"Loading PCAG models (ckpt_dir={args.ckpt_dir or 'default'})...")
    ckpt_dir_abs = None
    if args.ckpt_dir:
        ckpt_dir_abs = str(BASE_DIR / args.ckpt_dir) if not args.ckpt_dir.startswith("/") else args.ckpt_dir
    models_per_task = {t: load_pcag_models(t, ckpt_dir=ckpt_dir_abs) for t in TASKS}
    combat_per_task = {t: load_combat_params(t, ckpt_dir=ckpt_dir_abs) for t in TASKS}

    print("Loading BrainIAC sMRI features...")
    with open(SMRI_PKL, "rb") as f:
        sid_to_feat = pickle.load(f)

    df = pd.read_csv(CSV_PATH)

    # 諧波化 matrix 路徑替換
    if args.fmri_dir:
        fmri_root = BASE_DIR / args.fmri_dir
        def _swap(sid):
            p = fmri_root / f"{sid}_combat.npy"
            return str(p) if p.exists() else None
        new_paths = df["subject_id"].apply(_swap)
        df["matrix_path"] = new_paths
        df = df[df["matrix_path"].notna()].reset_index(drop=True)
        print(f"  using harmonized fMRI from {fmri_root}")

    print(f"Test set: {len(df)} patients × {len(TASKS)} tasks")

    n = len(df)
    out = {
        "subject_ids":     df["subject_id"].values,
        "labels":          df["label"].values,
        "tasks":           np.array(TASKS),
        "roi_importance":  np.zeros((n, 3, N_ROIS),     dtype=np.float32),
        "net_attention":   np.zeros((n, 3, N_NETWORKS), dtype=np.float32),
        "top_rois":        np.zeros((n, 3, 10),         dtype=np.int32),
        "top_roi_scores":  np.zeros((n, 3, 10),         dtype=np.float32),
        "fold_variability":np.zeros((n, 3),             dtype=np.float32),
        "network_labels":  np.array(net_list),
    }

    t0 = time.time()
    for i, row in df.iterrows():
        sid = row["subject_id"]
        if i % 10 == 0:
            print(f"[{i+1}/{n}] {sid}  ({time.time()-t0:.0f}s)")

        if sid not in sid_to_feat:
            print(f"  [WARN] no sMRI feat for {sid}, skipping")
            continue

        # 預處理 fMRI
        mat = np.load(row["matrix_path"])
        if mat.ndim == 3:
            mat = mat[0]
        feat = extract_node_features(mat)
        adj  = build_adj(mat)
        x     = torch.tensor(feat).unsqueeze(0).to(DEVICE)
        adj_t = torch.tensor(adj).unsqueeze(0).to(DEVICE)
        site  = detect_site(sid)
        smri_feat = sid_to_feat[sid]

        for j, task in enumerate(TASKS):
            smri_adj = apply_combat(smri_feat, site, combat_per_task[task])
            smri_t   = torch.tensor(smri_adj, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # 用 fold 0 為代表（避免跨 fold 平均洗掉訊號）
            # 同時收集所有 fold 的「top-1 network」做 sanity-check
            top_nets_per_fold = []
            chosen_roi_imp = None
            chosen_net_attn = None
            with torch.no_grad():
                for fold_idx, m in enumerate(models_per_task[task]):
                    _ = m(x, adj_t, smri_t)
                    enc = m.fmri_encoder
                    if enc.gat3._last_alpha is None:
                        continue
                    # GAT3 column attention（沒做 rollout）
                    a3 = enc.gat3._last_alpha.mean(dim=-1)[0].cpu().numpy()  # (N, N)
                    col = a3.sum(axis=0)  # 對 column 求和 = ROI 被注意的總量
                    col_norm = col / (col.sum() + 1e-8)
                    net_attn = enc._last_net_attn[0].cpu().numpy()
                    net_attn = net_attn / (net_attn.sum() + 1e-8)
                    top_nets_per_fold.append(int(np.argmax(net_attn)))
                    if fold_idx == 0:
                        chosen_roi_imp = col_norm
                        chosen_net_attn = net_attn

            if chosen_roi_imp is None:
                continue

            # fold variability：5 folds 的 top-1 network 是否一致
            # std=0 表示所有 fold 都同意；變大表示 fold 之間不同調
            fold_var = float(np.std(top_nets_per_fold)) if top_nets_per_fold else 0.0

            top_idx = np.argsort(chosen_roi_imp)[-10:][::-1]
            out["roi_importance"][i, j]   = chosen_roi_imp
            out["net_attention"][i, j]    = chosen_net_attn
            out["top_rois"][i, j]         = top_idx
            out["top_roi_scores"][i, j]   = chosen_roi_imp[top_idx]
            out["fold_variability"][i, j] = fold_var

    out_path = RES_DIR / f"gat_attention_v2{args.out_suffix}.npz"
    np.savez(out_path, **out)
    print(f"\n[SAVED] {out_path}")

    # Sanity check
    print("\n=== Sample patient ===")
    sample_idx = list(out["subject_ids"]).index("003_S_6264")
    sid = out["subject_ids"][sample_idx]
    print(f"Patient {sid} (label={out['labels'][sample_idx]} = AD)")
    for j, task in enumerate(TASKS):
        print(f"  {task}:")
        net_str = ", ".join(f"{n}={v*100:.1f}%" for n, v in
                            zip(out["network_labels"], out["net_attention"][sample_idx, j]))
        print(f"    Networks: {net_str}")
        print(f"    Top ROIs:")
        for r, s in zip(out["top_rois"][sample_idx, j][:5],
                        out["top_roi_scores"][sample_idx, j][:5]):
            print(f"      {AAL116_NAMES[r]:30s} imp={s:.4f}")


if __name__ == "__main__":
    main()
