"""
eval_paper_metrics.py — Clean paper evaluation using PCAG for all 3 tasks.

Uses pcag_combat_v2 checkpoints (train-only ComBat, v2 split) to evaluate
on v2 test set. No KD GNN used here — avoids data leakage for MCI_vs_AD.

Outputs:
  - Per-task AUC + 95% bootstrap CI
  - 3-class confusion matrix + balanced accuracy
  - results/paper_metrics_v2.json

Usage:
    conda activate AD
    cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
    python eval_paper_metrics.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             balanced_accuracy_score, classification_report)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
CKPT = BASE / "checkpoints" / "pcag_combat_v2"
RES  = BASE / "results"

# ── Imports from training script (same model definitions) ────────────────────
sys.path.insert(0, str(BASE))
from train_pcag_combat_fusion import PCAGModel, extract_node_features, build_adj

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BOOT = 1000
SEED   = 42
rng    = np.random.default_rng(SEED)

TASK_CFG = {
    "NC_vs_AD":  {"classes": [0, 2], "pos": 2, "neg": "NC", "posi": "AD"},
    "NC_vs_MCI": {"classes": [0, 1], "pos": 1, "neg": "NC", "posi": "MCI"},
    "MCI_vs_AD": {"classes": [1, 2], "pos": 2, "neg": "MCI", "posi": "AD"},
}
LABEL_MAP = {0: "NC", 1: "MCI", 2: "AD"}

# ── Load v2 test set ─────────────────────────────────────────────────────────
pcag_te = pd.read_csv(BASE / "pcag_test_aligned_v2.csv")
smri_cols = [f"Feature_{i}" for i in range(768)]
feat_te = pd.read_csv(BASE / "brainiac_features_combined_test_v2.csv")

print(f"v2 test set: NC={sum(pcag_te.label==0)} MCI={sum(pcag_te.label==1)} AD={sum(pcag_te.label==2)}")
print(f"Device: {DEVICE}\n")

# ── ComBat helpers ───────────────────────────────────────────────────────────
def load_combat_params(task: str):
    path = CKPT / f"combat_params_{task}.json"
    with open(path) as f:
        p = json.load(f)
    return {
        "site_map":   p["site_map"],
        "stand_mean": np.array(p["stand_mean"]),
        "var_pooled": np.array(p["var_pooled"]),
        "gamma_star": np.array(p["gamma_star"]),
        "delta_star": np.array(p["delta_star"]),
    }

def apply_combat_batch(smri_raw: np.ndarray, site_indices: np.ndarray, params: dict) -> np.ndarray:
    grand_mean = params["stand_mean"].mean(axis=1)      # (768,)
    var_pooled  = params["var_pooled"]                   # (768,)
    gamma = params["gamma_star"][site_indices]           # (n, 768)
    delta = params["delta_star"][site_indices]           # (n, 768)
    x_std = (smri_raw - grand_mean) / (np.sqrt(var_pooled) + 1e-8)
    x_adj = (x_std - gamma) / (np.sqrt(delta) + 1e-8)
    return x_adj * np.sqrt(var_pooled) + grand_mean

# ── fMRI encoder ─────────────────────────────────────────────────────────────
_fmri_cache = {}

def encode_fmri(matrix_path: str):
    if matrix_path in _fmri_cache:
        return _fmri_cache[matrix_path]
    mat = np.load(matrix_path)
    if mat.ndim == 3:
        mat = mat[0]
    feat = extract_node_features(mat)
    adj  = build_adj(mat)
    x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    a    = torch.tensor(adj,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
    _fmri_cache[matrix_path] = (x, a)
    return x, a

# ── Bootstrap CI ────────────────────────────────────────────────────────────
def bootstrap_auc_ci(y_true, y_prob, n=N_BOOT):
    aucs = []
    n_s  = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_s, size=n_s)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

# ── Per-task evaluation ──────────────────────────────────────────────────────
task_results = {}

for task, cfg in TASK_CFG.items():
    print(f"--- {task} ---")

    # Check checkpoints exist
    ckpt_paths = [CKPT / f"pcag_combat_{task}_fold{i}.pt" for i in range(5)]
    missing = [p for p in ckpt_paths if not p.exists()]
    if missing:
        print(f"  [SKIP] missing checkpoints: {[p.name for p in missing]}")
        continue

    # Load models
    models = []
    for p in ckpt_paths:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        m = PCAGModel(fusion_dim=20).to(DEVICE)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)

    # Load combat params
    params = load_combat_params(task)

    # Filter test patients for this binary task
    df_task = pcag_te[pcag_te["label"].isin(cfg["classes"])].reset_index(drop=True)
    smri_raw = feat_te.loc[df_task["smri_feat_row"]][smri_cols].values.astype(np.float32)

    # Site indices
    def get_site(subject_id):
        if "_S_" in str(subject_id):
            return params["site_map"].get("ADNI_new", 0)
        return params["site_map"].get("TPMIC", 1)
    site_indices = np.array([get_site(sid) for sid in df_task["subject_id"]])

    # Apply ComBat
    smri_harm = apply_combat_batch(smri_raw, site_indices, params)

    # Predict
    y_true = (df_task["label"] == cfg["pos"]).astype(int).values
    y_prob = []

    with torch.no_grad():
        for i, (_, row) in enumerate(df_task.iterrows()):
            x, adj = encode_fmri(row["matrix_path"])
            smri_t = torch.tensor(smri_harm[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            fold_probs = []
            for m in models:
                out = F.softmax(m(x, adj, smri_t), dim=1)
                fold_probs.append(out[0, 1].item())
            y_prob.append(float(np.mean(fold_probs)))

    y_prob = np.array(y_prob)

    # Metrics
    auc = roc_auc_score(y_true, y_prob)
    ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_prob)
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"  n={len(y_true)} ({cfg['neg']}={sum(y_true==0)}, {cfg['posi']}={sum(y_true==1)})")
    print(f"  AUC = {auc:.3f} (95% CI: {ci_lo:.3f}–{ci_hi:.3f})")
    print(f"  Sensitivity = {sens:.3f}, Specificity = {spec:.3f}")
    print(f"  CM: {cm.tolist()}")
    print()

    task_results[task] = {
        "auc": round(auc, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
        "sens": round(sens, 4), "spec": round(spec, 4),
        "n": int(len(y_true)), "cm": cm.tolist(),
        "y_true": y_true.tolist(), "y_prob": y_prob.tolist(),
    }

# ── 3-class evaluation ───────────────────────────────────────────────────────
print("=== 3-class Hierarchical Evaluation ===")
if all(t in task_results for t in ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]):
    # Collect per-patient predictions
    y3_true, y3_pred = [], []
    for _, row in pcag_te.iterrows():
        sid  = row["subject_id"]
        true = int(row["label"])

        # Get probabilities for this patient from each task
        def get_prob(task):
            cfg   = TASK_CFG[task]
            df_t  = pcag_te[pcag_te["label"].isin(cfg["classes"])].reset_index(drop=True)
            idx   = df_t.index[df_t["subject_id"] == sid].tolist()
            if not idx:
                return None
            return task_results[task]["y_prob"][idx[0]]

        p_ad  = get_prob("NC_vs_AD")
        p_mci = get_prob("NC_vs_MCI")
        p_adg = get_prob("MCI_vs_AD")

        if p_ad is None and p_mci is None:
            continue  # NC patients not in MCI_vs_AD — handled below

        # Hierarchical decision
        is_disease = (p_ad is not None and p_ad >= 0.5) or (p_mci is not None and p_mci >= 0.5)
        if not is_disease:
            pred = 0  # NC
        else:
            pred = 2 if (p_adg is not None and p_adg >= 0.5) else 1  # AD or MCI

        y3_true.append(true)
        y3_pred.append(pred)

    y3_true = np.array(y3_true)
    y3_pred = np.array(y3_pred)
    bal_acc = balanced_accuracy_score(y3_true, y3_pred)
    cm3 = confusion_matrix(y3_true, y3_pred, labels=[0, 1, 2])

    print(f"  Patients evaluated: {len(y3_true)}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Confusion Matrix (NC/MCI/AD):")
    print(f"    {cm3}")
    print()
    print(classification_report(y3_true, y3_pred, target_names=["NC", "MCI", "AD"]))

    task_results["_3class"] = {
        "balanced_accuracy": round(bal_acc, 4),
        "cm": cm3.tolist(),
        "n_evaluated": int(len(y3_true)),
    }
else:
    print("  [SKIP] Not all 3 tasks completed yet.")

# ── Save ─────────────────────────────────────────────────────────────────────
RES.mkdir(exist_ok=True)
save = {k: {k2: v2 for k2, v2 in v.items() if k2 not in ("y_true", "y_prob")}
        for k, v in task_results.items()}
with open(RES / "paper_metrics_v2.json", "w") as f:
    json.dump(save, f, indent=2)
print(f"\n[SAVED] results/paper_metrics_v2.json")
