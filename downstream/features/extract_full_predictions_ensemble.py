import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
extract_full_predictions_ensemble.py
=====================================
跑 5 seeds × 3 tasks ensemble，對所有 49 個 test patient 輸出 3 個 task 的機率。
每個 task：載入 5 個 seed 的 5-fold 模型，全部 inference，平均 25 個結果。

Per-task best 設定（從前面實驗確定）：
  NC_vs_AD: augmentation (mixup 0.2 + DE 0.2)
  NC_vs_MCI / MCI_vs_AD: baseline (no aug)

輸出：
  results/full_predictions_v2_nolabel_ensemble.npz
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
CKPT_ROOT = BASE_DIR / "checkpoints"

TASKS = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]
SEEDS = [42, 123, 456, 789, 2024]

# 從 OOF-selected ensemble summary 載入每個 task 的 selected 策略
import json as _json
_SUMMARY_PATH = Path(__file__).parent / "results" / "ensemble_v2_nolabel_summary.json"
if _SUMMARY_PATH.exists():
    with open(_SUMMARY_PATH) as f:
        ENSEMBLE_CONFIG = _json.load(f)
    print(f"[Loaded] OOF-selected ensemble config:")
    for t, r in ENSEMBLE_CONFIG.items():
        print(f"  {t}: strategy={r['selected_strategy']}")
else:
    ENSEMBLE_CONFIG = None

# Per-task ckpt dir naming（須與 train script 的 exp_tag 對齊）
# seed=42 不加 _s 後綴；其他加 _s{seed}
def get_ckpt_dirs(task: str, variant: str) -> list:
    dirs = []
    for s in SEEDS:
        s_tag = "" if s == 42 else f"_s{s}"
        if variant == "aug":
            name = f"pcag_ensemble_aug_v2_fmricombat_nolabel_aug_mix0.2_de0.2{s_tag}"
        elif variant == "noaug":
            name = f"pcag_ensemble_noaug_v2_fmricombat_nolabel{s_tag}"
        else:
            raise ValueError(variant)
        dirs.append(CKPT_ROOT / name)
    return dirs


TASK_VARIANTS = {
    "NC_vs_AD":  "aug",     # augmentation 對此 task 有效
    "NC_vs_MCI": "noaug",   # baseline 最佳
    "MCI_vs_AD": "noaug",
}


def detect_site(sid):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(sid)) else 'TPMIC'


def predict_one(models, x, adj, smri_t):
    probs = []
    with torch.no_grad():
        for m in models:
            logits = m(x, adj, smri_t)
            probs.append(F.softmax(logits, dim=1)[0, 1].item())
    return float(np.mean(probs))


def main():
    print("Loading models for ensemble (5 seeds × 5 folds × 3 tasks)...")
    # 對每個 task 載入 5 個 seed 的 5 fold models (25 models per task)
    ensemble_models = {}  # task → list of [m_fold0, ...m_fold4] for each seed
    combat_params = {}    # task → list of combat params (per seed)
    for task in TASKS:
        variant = TASK_VARIANTS[task]
        ensemble_models[task] = []
        combat_params[task] = []
        for ckpt_dir in get_ckpt_dirs(task, variant):
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Missing ensemble ckpt dir: {ckpt_dir}")
            models = load_pcag_models(task, ckpt_dir=str(ckpt_dir))
            params = load_combat_params(task, ckpt_dir=str(ckpt_dir))
            ensemble_models[task].append(models)
            combat_params[task].append(params)
        print(f"  {task} ({variant}): {len(ensemble_models[task])} seeds × 5 folds = "
              f"{sum(len(ms) for ms in ensemble_models[task])} models")

    with open(SMRI_PKL, "rb") as f:
        sid_to_feat = pickle.load(f)

    df = pd.read_csv(CSV_PATH)
    print(f"Test set: {len(df)} patients")

    n = len(df)
    out = {
        "subject_ids": df["subject_id"].values,
        "labels":      df["label"].values,
        "sites":       np.array([detect_site(s) for s in df["subject_id"]]),
        "tasks":       np.array(TASKS),
        "fused_probs": np.zeros((n, 3), dtype=np.float32),
        "n_models":    np.array([sum(len(ms) for ms in ensemble_models[t]) for t in TASKS]),
    }

    t0 = time.time()
    for i, row in df.iterrows():
        sid = row["subject_id"]
        if i % 10 == 0:
            print(f"[{i+1}/{n}] {sid}  ({time.time()-t0:.0f}s)")

        mat_path = FMRI_DIR / f"{sid}_combat.npy"
        mat = np.load(mat_path)
        feat = extract_node_features(mat); adj = build_adj(mat)
        x = torch.tensor(feat).unsqueeze(0).to(DEVICE)
        adj_t = torch.tensor(adj).unsqueeze(0).to(DEVICE)
        site = detect_site(sid)
        smri_feat = sid_to_feat[sid]

        for j, task in enumerate(TASKS):
            seed_probs = []
            for seed_idx in range(len(SEEDS)):
                params = combat_params[task][seed_idx]
                smri_adj = apply_combat(smri_feat, site, params)
                smri_t = torch.tensor(smri_adj, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                prob = predict_one(ensemble_models[task][seed_idx], x, adj_t, smri_t)
                seed_probs.append(prob)
            # OOF-selected aggregation strategy per task
            if ENSEMBLE_CONFIG and task in ENSEMBLE_CONFIG:
                strategy = ENSEMBLE_CONFIG[task]["selected_strategy"]
                if strategy == "median":
                    out["fused_probs"][i, j] = float(np.median(seed_probs))
                elif strategy == "top3_oof":
                    top3_seeds = ENSEMBLE_CONFIG[task]["all_strategies"]["top3_oof"]["selected_seeds"]
                    top3_idx = [SEEDS.index(s) for s in top3_seeds]
                    out["fused_probs"][i, j] = float(np.mean([seed_probs[k] for k in top3_idx]))
                elif strategy == "single_best":
                    best_seed = ENSEMBLE_CONFIG[task]["all_strategies"]["single_best"]["seed"]
                    best_idx = SEEDS.index(best_seed)
                    out["fused_probs"][i, j] = float(seed_probs[best_idx])
                else:  # mean
                    out["fused_probs"][i, j] = float(np.mean(seed_probs))
            else:
                out["fused_probs"][i, j] = float(np.mean(seed_probs))

    out_path = RES_DIR / "full_predictions_v2_nolabel_ensemble.npz"
    np.savez(out_path, **out)
    print(f"\n[SAVED] {out_path}")

    # Sanity check
    print("\n=== Sanity check ===")
    sample_i = list(out["subject_ids"]).index("003_S_6264")
    print(f"Patient {out['subject_ids'][sample_i]} (label={out['labels'][sample_i]})")
    for j, task in enumerate(TASKS):
        print(f"  {task}: ensemble fused prob = {out['fused_probs'][sample_i, j]:.3f}")

    # Compute ensemble test AUC per task
    from sklearn.metrics import roc_auc_score
    print("\n=== Ensemble test AUC (per task, on official test subsets) ===")
    df_full = pd.read_csv(CSV_PATH)
    for j, task in enumerate(TASKS):
        cls_pair = {"NC_vs_AD": [0, 2], "NC_vs_MCI": [0, 1], "MCI_vs_AD": [1, 2]}[task]
        pos = cls_pair[1]
        mask = df_full["label"].isin(cls_pair).values
        bin_labels = (df_full["label"].values == pos)[mask].astype(int)
        probs = out["fused_probs"][mask, j]
        auc = roc_auc_score(bin_labels, probs)
        print(f"  {task} (n={mask.sum()}): AUC = {auc:.4f}")


if __name__ == "__main__":
    main()
