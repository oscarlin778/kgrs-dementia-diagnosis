"""
extract_full_predictions.py
============================
對 49 個測試病患都跑 3 個 task 的 PCAG-ComBat + fmri-only + smri-only 模型，
得到完整的 (patient × task) 機率矩陣（不限定 test set 是否包含該病患）。

這樣 LLM 報告可以對每位病患展示 3 個 task 的預測，而非只展示 test set 內的 task。

執行：
  conda activate AD
  python extract_full_predictions.py

輸出：
  results/full_predictions_v2_nolabel.npz
    keys:
      subject_ids : (49,)
      labels      : (49,)
      sites       : (49,)
      tasks       : ['NC_vs_AD', 'NC_vs_MCI', 'MCI_vs_AD']
      fused_probs : (49, 3)   PCAG-ComBat 機率 per task
      fmri_probs  : (49, 3)   fMRI-only 機率
      smri_probs  : (49, 3)   sMRI-only 機率
"""
import re, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from inference_pipeline_v2 import (
    PCAGModel, load_pcag_models, load_combat_params, apply_combat,
    extract_node_features, build_adj, DEVICE,
)

BASE_DIR = Path(__file__).parent
RES_DIR  = BASE_DIR / "results"
CSV_PATH = BASE_DIR / "pcag_test_aligned_v2.csv"
SMRI_PKL = BASE_DIR / "sid_to_smri_feat.pkl"
FMRI_DIR = BASE_DIR / "fmri_combat_v2_nolabel"

TASKS = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]

# 三種模型的 checkpoint dir
CKPT_DIRS = {
    "fused":     BASE_DIR / "checkpoints/pcag_fmricombat_nolabel_v2",
    "fmri_only": BASE_DIR / "checkpoints/pcag_combat_v2_fmri_only_fmricombat_nolabel",
    "smri_only": BASE_DIR / "checkpoints/pcag_combat_v2_smri_only_fmricombat_nolabel",
}


def detect_site(sid):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(sid)) else 'TPMIC'


def predict_one(models, x, adj, smri_t):
    """5 fold 平均的 PCAG 機率."""
    probs = []
    with torch.no_grad():
        for m in models:
            logits = m(x, adj, smri_t)
            probs.append(F.softmax(logits, dim=1)[0, 1].item())
    return float(np.mean(probs))


def main():
    print("Loading models...")
    models = {}
    combats = {}
    for variant, ckpt_dir in CKPT_DIRS.items():
        models[variant] = {t: load_pcag_models(t, ckpt_dir=str(ckpt_dir)) for t in TASKS}
        # fmri_only skips ComBat (no sMRI used)，故無 combat_params
        if variant == "fmri_only":
            combats[variant] = None
        else:
            combats[variant] = {t: load_combat_params(t, ckpt_dir=str(ckpt_dir)) for t in TASKS}
        print(f"  {variant} loaded")

    with open(SMRI_PKL, "rb") as f:
        sid_to_feat = pickle.load(f)
    print(f"Loaded {len(sid_to_feat)} sMRI features")

    df = pd.read_csv(CSV_PATH)
    print(f"Test set: {len(df)} patients × 3 tasks × 3 variants = {len(df)*9} predictions")

    n = len(df)
    out = {
        "subject_ids": df["subject_id"].values,
        "labels":      df["label"].values,
        "sites":       np.array([detect_site(s) for s in df["subject_id"]]),
        "tasks":       np.array(TASKS),
        "fused_probs": np.zeros((n, 3), dtype=np.float32),
        "fmri_probs":  np.zeros((n, 3), dtype=np.float32),
        "smri_probs":  np.zeros((n, 3), dtype=np.float32),
    }

    t0 = time.time()
    for i, row in df.iterrows():
        sid = row["subject_id"]
        if i % 10 == 0:
            print(f"[{i+1}/{n}] {sid}  ({time.time()-t0:.0f}s)")

        # 用 no-label ComBat 諧波化矩陣
        mat_path = FMRI_DIR / f"{sid}_combat.npy"
        mat = np.load(mat_path)
        feat = extract_node_features(mat); adj = build_adj(mat)
        x = torch.tensor(feat).unsqueeze(0).to(DEVICE)
        adj_t = torch.tensor(adj).unsqueeze(0).to(DEVICE)
        site = detect_site(sid)
        smri_feat = sid_to_feat[sid]

        for j, task in enumerate(TASKS):
            # Fused PCAG
            smri_adj = apply_combat(smri_feat, site, combats["fused"][task])
            smri_t = torch.tensor(smri_adj, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out["fused_probs"][i, j] = predict_one(models["fused"][task], x, adj_t, smri_t)

            # fMRI-only：sMRI feat = zeros（768-d zero tensor，跳過 ComBat）
            smri_zero = torch.zeros(1, 768, dtype=torch.float32).to(DEVICE)
            out["fmri_probs"][i, j] = predict_one(models["fmri_only"][task], x, adj_t, smri_zero)

            # sMRI-only (fMRI feat zeroed — but pipeline doesn't easily support; use combat'd smri + zero fmri)
            # 對 smri_only model 來說，fmri 的影響應該已在訓練時設為 0；inference 也傳 0
            smri_adj2 = apply_combat(smri_feat, site, combats["smri_only"][task])
            smri_t2 = torch.tensor(smri_adj2, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            x_zero = torch.zeros_like(x)  # zero fMRI input
            adj_zero = torch.zeros_like(adj_t)
            out["smri_probs"][i, j] = predict_one(models["smri_only"][task], x_zero, adj_zero, smri_t2)

    out_path = RES_DIR / "full_predictions_v2_nolabel.npz"
    np.savez(out_path, **out)
    print(f"\n[SAVED] {out_path}")

    # Sanity check
    print("\n=== Sanity check: 1 sample patient (003_S_6264 = AD) ===")
    idx = list(out["subject_ids"]).index("003_S_6264")
    print(f"Subject: {out['subject_ids'][idx]}, true_label={out['labels'][idx]}")
    for j, task in enumerate(TASKS):
        print(f"  {task}: fused={out['fused_probs'][idx,j]:.3f}, "
              f"fmri_only={out['fmri_probs'][idx,j]:.3f}, "
              f"smri_only={out['smri_probs'][idx,j]:.3f}")


if __name__ == "__main__":
    main()
