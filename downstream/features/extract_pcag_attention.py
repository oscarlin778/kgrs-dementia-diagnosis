import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
extract_pcag_attention.py
=========================
對測試集所有病患抓取 PCAG cross-attention：
  - 對每個 task (NC_vs_AD, NC_vs_MCI, MCI_vs_AD) 跑 5 折模型，平均 attention
  - 計算每個 fusion dim 的 gating S (sigmoid output, range [0,1])
  - 用 fmri_proj 權重靜態分析「fusion dim → 腦網路」的對應
  - 結合得到「per-patient PCAG attention」per task

輸出：
  results/pcag_attention_v2.npz
    keys:
      subject_ids    : array of subject IDs
      tasks          : ['NC_vs_AD', 'NC_vs_MCI', 'MCI_vs_AD']
      gating_S       : shape (n_subj, n_tasks, 20)  — per-dim activation
      net_attention  : shape (n_subj, n_tasks, 9)   — aggregated to 9 networks
      entropy        : shape (n_subj, n_tasks)      — attention entropy
      top_active_dims: shape (n_subj, n_tasks, 5)   — top 5 most-active dim indices

執行：
  conda activate AD
  python extract_pcag_attention.py
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from inference_pipeline_v2 import (
    PCAGModel, load_pcag_models, load_combat_params, apply_combat,
    extract_node_features, build_adj, N_ROIS, DEVICE,
    NETWORK_MAP, net_list,
)

BASE_DIR  = Path(__file__).parent
RES_DIR   = BASE_DIR / "results"
CSV_PATH  = BASE_DIR / "pcag_test_aligned_v2.csv"
TPMIC_SMRI_FEAT = BASE_DIR / "sid_to_smri_feat.pkl"

TASKS = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]
FUSION_DIM = 20
N_NETWORKS = 9
HIDDEN_DIM = 128


def detect_site(subject_id: str) -> str:
    import re
    if re.search(r'\d{3}_S_\d{4}', subject_id):
        return 'ADNI'
    return 'TPMIC'


def get_smri_feat(sid: str, sid_to_feat: dict) -> np.ndarray | None:
    """取得 BrainIAC 768-d 特徵：優先從 pre-computed lookup 拿。"""
    if sid in sid_to_feat:
        return sid_to_feat[sid]
    return None


def compute_static_dim_to_network(model: PCAGModel) -> np.ndarray:
    """
    用 fmri_proj 權重靜態分析每個 fusion dim 主要 attend 到哪些網路。
    fmri_proj.weight: (20, 1280)，前 1152 (=9*128) 為網路特徵，最後 128 為 virtual node。
    回傳 (20, 9) 的 dim→network 重要度（已 normalize）。
    """
    W = model.pcag.fmri_proj.weight.detach().cpu().numpy()  # (20, 1280)
    W_net = W[:, :N_NETWORKS * HIDDEN_DIM].reshape(FUSION_DIM, N_NETWORKS, HIDDEN_DIM)
    # L2 norm per network → (20, 9)
    importance = np.linalg.norm(W_net, axis=2)
    # normalize per fusion dim
    importance = importance / (importance.sum(axis=1, keepdims=True) + 1e-8)
    return importance


def extract_attention_for_patient(subject_id: str, matrix_path: str,
                                  smri_feat: np.ndarray, site: str,
                                  models_per_task: dict, combat_per_task: dict,
                                  dim_net_per_task: dict) -> dict:
    """跑單一病患的所有 task，回傳每個 task 的 attention summary."""
    # fMRI 預處理
    mat = np.load(matrix_path)
    if mat.ndim == 3:
        mat = mat[0]
    feat = extract_node_features(mat)
    adj  = build_adj(mat)
    x     = torch.tensor(feat).unsqueeze(0).to(DEVICE)
    adj_t = torch.tensor(adj).unsqueeze(0).to(DEVICE)

    result = {}
    for task in TASKS:
        smri_adj = apply_combat(smri_feat, site, combat_per_task[task])
        smri_t   = torch.tensor(smri_adj, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # 5 折 attention 平均
        S_folds = []
        with torch.no_grad():
            for m in models_per_task[task]:
                _, attn = m(x, adj_t, smri_t, return_attn=True)
                S_folds.append(attn["S"].cpu().numpy()[0])  # (20,)
        S_mean = np.mean(S_folds, axis=0)  # (20,)

        # 網路層級：用 (S 加權) × (dim→net mapping) → 9 個網路活化度
        net_attn = S_mean @ dim_net_per_task[task]  # (9,)
        net_attn = net_attn / (net_attn.sum() + 1e-8)  # normalize

        # entropy
        p = S_mean / (S_mean.sum() + 1e-8)
        entropy = -float(np.sum(p * np.log(p + 1e-8)))

        # top active dims
        top_dims = np.argsort(S_mean)[-5:][::-1]

        result[task] = {
            "gating_S":         S_mean.tolist(),
            "net_attention":    net_attn.tolist(),
            "entropy":          entropy,
            "top_active_dims":  top_dims.tolist(),
            "top_active_dim_values": S_mean[top_dims].tolist(),
        }

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="PCAG checkpoint dir; defaults to inference_pipeline_v2 default")
    parser.add_argument("--fmri_dir", type=str, default=None,
                        help="Harmonized fMRI matrix dir; if set, swap matrix_path to it")
    parser.add_argument("--out_suffix", type=str, default="",
                        help="Output file suffix (e.g. '_nolabel')")
    args = parser.parse_args()

    print(f"Loading PCAG models (ckpt_dir={args.ckpt_dir or 'default'})...")
    ckpt_dir_abs = None
    if args.ckpt_dir:
        ckpt_dir_abs = str(BASE_DIR / args.ckpt_dir) if not args.ckpt_dir.startswith("/") else args.ckpt_dir
    models_per_task = {t: load_pcag_models(t, ckpt_dir=ckpt_dir_abs) for t in TASKS}
    combat_per_task = {t: load_combat_params(t, ckpt_dir=ckpt_dir_abs) for t in TASKS}

    print("Computing static dim→network mapping (per task, averaged over 5 folds)...")
    dim_net_per_task = {}
    for t in TASKS:
        importances = [compute_static_dim_to_network(m) for m in models_per_task[t]]
        dim_net_per_task[t] = np.mean(importances, axis=0)  # (20, 9)
        print(f"  {t}: dim→net shape={dim_net_per_task[t].shape}")

    # 載入 BrainIAC 預先算好的 sMRI 特徵
    print("Loading pre-computed BrainIAC sMRI features...")
    import pickle
    with open(TPMIC_SMRI_FEAT, "rb") as f:
        sid_to_feat = pickle.load(f)
    print(f"  loaded {len(sid_to_feat)} sMRI features")

    # 讀測試集
    df = pd.read_csv(CSV_PATH)
    print(f"Test set: {len(df)} patients")

    # 諧波化 matrix 路徑替換
    if args.fmri_dir:
        fmri_root = BASE_DIR / args.fmri_dir
        def _swap(sid):
            p = fmri_root / f"{sid}_combat.npy"
            return str(p) if p.exists() else None
        new_paths = df["subject_id"].apply(_swap)
        if new_paths.isna().any():
            missing = df.loc[new_paths.isna(), "subject_id"].tolist()
            print(f"  [WARN] missing harmonized fMRI for: {missing}")
        df["matrix_path"] = new_paths
        df = df[df["matrix_path"].notna()].reset_index(drop=True)
        print(f"  using harmonized fMRI from {fmri_root}, {len(df)} patients remain")

    results = []
    t0 = time.time()
    for idx, row in df.iterrows():
        sid = row["subject_id"]
        elapsed = time.time() - t0
        print(f"[{idx+1}/{len(df)}] {sid}  ({elapsed:.0f}s)")

        smri_feat = get_smri_feat(sid, sid_to_feat)
        if smri_feat is None:
            print(f"  [WARN] no sMRI feat for {sid}, skipping")
            continue

        site = detect_site(sid)
        try:
            attn = extract_attention_for_patient(
                sid, row["matrix_path"], smri_feat, site,
                models_per_task, combat_per_task, dim_net_per_task,
            )
        except Exception as e:
            print(f"  [WARN] failed: {e}")
            continue

        results.append({
            "subject_id": sid,
            "label":      int(row["label"]),
            "site":       site,
            "attention":  attn,
        })

    # 存成 npz（方便 eval_report_quality 載入）
    n = len(results)
    out = {
        "subject_ids":     np.array([r["subject_id"] for r in results]),
        "labels":          np.array([r["label"]      for r in results]),
        "sites":           np.array([r["site"]       for r in results]),
        "tasks":           np.array(TASKS),
        "gating_S":        np.zeros((n, 3, FUSION_DIM)),
        "net_attention":   np.zeros((n, 3, N_NETWORKS)),
        "entropy":         np.zeros((n, 3)),
        "top_active_dims": np.zeros((n, 3, 5), dtype=np.int32),
        "top_active_dim_values": np.zeros((n, 3, 5)),
        "network_labels":  np.array(net_list),
    }
    for i, r in enumerate(results):
        for j, task in enumerate(TASKS):
            a = r["attention"][task]
            out["gating_S"][i, j]              = a["gating_S"]
            out["net_attention"][i, j]         = a["net_attention"]
            out["entropy"][i, j]               = a["entropy"]
            out["top_active_dims"][i, j]       = a["top_active_dims"]
            out["top_active_dim_values"][i, j] = a["top_active_dim_values"]

    out_path = RES_DIR / f"pcag_attention_v2{args.out_suffix}.npz"
    out_path.parent.mkdir(exist_ok=True)
    np.savez(out_path, **out)
    print(f"\n[SAVED] {out_path}  ({n} patients × {len(TASKS)} tasks)")


if __name__ == "__main__":
    main()
