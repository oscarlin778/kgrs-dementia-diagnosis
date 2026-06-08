"""
run_loocv.py
============
Full-dataset Leave-One-Out Cross-Validation (LOOCV) for PCAG-ComBat.
Uses all 204 patients (train + test combined).

Strategy:
  - fMRI ComBat: globally pre-harmonized (fmri_combat_v2_nolabel/), not per-fold
    (justification: ComBat params are stable; fit on 155 vs 203 patients differs minimally)
  - sMRI ComBat: refitted per task per fold on N-1 training patients (task-specific labels)
  - Training: fixed 150 epochs, no inner val set (standard for small-sample LOOCV)
  - No data augmentation (reproducibility)

Each patient belongs to exactly 2 binary tasks:
  NC (0)  -> NC_vs_AD, NC_vs_MCI
  MCI (1) -> NC_vs_MCI, MCI_vs_AD
  AD (2)  -> NC_vs_AD, MCI_vs_AD

Output:
  results/loocv_results.npz  (per-task LOOCV probs + labels)
  results/loocv_summary.json (AUC, sensitivity, specificity, 95% CI)

Estimated runtime: ~5 hours (408 model trainings x ~40s each + sMRI ComBat)
"""
import sys, os, re, json, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_pcag_combat_fusion import (
    PCAGModel, extract_node_features, build_adj, set_drop_edge_rate,
)
from neuroCombat import neuroCombat, neuroCombatFromTraining

BASE_DIR  = Path(__file__).parent
RES_DIR   = BASE_DIR / "results"
FMRI_DIR  = BASE_DIR / "fmri_combat_v2_nolabel"

LOOCV_EPOCHS = 150
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMRI_COLS    = [f"Feature_{i}" for i in range(768)]

TASKS = {
    "NC_vs_AD":  {"classes": [0, 2], "pos": 2},
    "NC_vs_MCI": {"classes": [0, 1], "pos": 1},
    "MCI_vs_AD": {"classes": [1, 2], "pos": 2},
}

# Which tasks each label participates in
LABEL_TO_TASKS = {
    0: ["NC_vs_AD", "NC_vs_MCI"],  # NC
    1: ["NC_vs_MCI", "MCI_vs_AD"], # MCI
    2: ["NC_vs_AD", "MCI_vs_AD"],  # AD
}


def normalize_source(src):
    """Map site source strings to 'ADNI' or 'TPMIC'."""
    return "ADNI" if "ADNI" in str(src) else "TPMIC"


class InMemoryPCAGDataset(Dataset):
    def __init__(self, feats, adjs, smri_feats, labels):
        self.feats     = feats                          # list of (116, 125)
        self.adjs      = adjs                           # list of (116, 116)
        self.smri      = smri_feats.astype(np.float32) # (N, 768)
        self.labels    = labels.astype(np.int64)        # (N,)
    def __len__(self): return len(self.feats)
    def __getitem__(self, i):
        return (
            torch.tensor(self.feats[i], dtype=torch.float32),
            torch.tensor(self.adjs[i],  dtype=torch.float32),
            torch.tensor(self.smri[i],  dtype=torch.float32),
            torch.tensor(self.labels[i], dtype=torch.long),
        )


def fit_and_apply_smri_combat(smri_train, sites_train, bin_labels_train, smri_test, site_test):
    """Fit sMRI ComBat on training patients, apply to one test patient."""
    unique_sites = sorted(set(sites_train))
    site_map = {s: i for i, s in enumerate(unique_sites)}
    dat = smri_train.T  # (768, N_train)
    covars = pd.DataFrame({
        "batch":     [site_map[s] for s in sites_train],
        "bin_label": bin_labels_train.astype(int),
    })
    fit = neuroCombat(dat=dat, covars=covars, batch_col="batch", categorical_cols=["bin_label"])
    estimates = fit["estimates"]
    smri_train_harm = fit["data"].T  # (N_train, 768)

    # Apply to test patient
    test_dat = smri_test.reshape(-1, 1)  # (768, 1)
    test_site_idx = site_map.get(site_test, 0)
    apply_out = neuroCombatFromTraining(
        dat=test_dat, batch=np.array([test_site_idx]), estimates=estimates,
    )
    smri_test_harm = apply_out["data"][:, 0]  # (768,)
    return smri_train_harm, smri_test_harm


def train_loocv_model(feats_tr, adjs_tr, smri_tr, labels_tr, seed=SEED):
    """Train PCAG for fixed epochs (no val set)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    counts  = np.bincount(labels_tr, minlength=2)
    weights = 1.0 / np.maximum(counts, 1).astype(float)
    sample_w = weights[labels_tr]
    sampler  = WeightedRandomSampler(sample_w, len(labels_tr), replacement=True)

    ds     = InMemoryPCAGDataset(feats_tr, adjs_tr, smri_tr, labels_tr)
    loader = DataLoader(ds, batch_size=16, sampler=sampler, num_workers=0)

    model     = PCAGModel(fusion_dim=20).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-3)
    criterion = nn.CrossEntropyLoss()
    set_drop_edge_rate(0.0)  # no augmentation

    for _ in range(LOOCV_EPOCHS):
        model.train()
        for x, adj, smri, y in loader:
            x, adj, smri, y = x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x, adj, smri), y).backward()
            optimizer.step()
    return model


def predict_prob(model, feat, adj, smri):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        a = torch.tensor(adj,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
        s = torch.tensor(smri, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return float(F.softmax(model(x, a, s), dim=1)[0, 1].cpu())


def bootstrap_auc(probs, labels, n_boot=2000, rng=None):
    if rng is None: rng = np.random.default_rng(SEED)
    aucs = []; n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2: continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    t_start = time.time()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"LOOCV_EPOCHS={LOOCV_EPOCHS}, SEED={SEED}")

    # ── 1. Load all 204 patients ────────────────────────────────────────────
    print("\n[1/4] Loading patient data...")
    df_tr_pcag = pd.read_csv(BASE_DIR / "pcag_train_aligned_v2.csv")
    df_te_pcag = pd.read_csv(BASE_DIR / "pcag_test_aligned_v2.csv")
    df_tr_meta = pd.read_csv(BASE_DIR / "kd_train_aligned_v2.csv")
    df_te_meta = pd.read_csv(BASE_DIR / "kd_test_aligned_v2.csv")

    df_tr = df_tr_pcag.merge(df_tr_meta[["subject_id", "source"]], on="subject_id", how="inner")
    df_te = df_te_pcag.merge(df_te_meta[["subject_id", "source"]], on="subject_id", how="inner")
    df_tr["split"] = "train"; df_te["split"] = "test"
    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    df_all["site"] = df_all["source"].apply(normalize_source)
    df_all["smri_origin"] = df_all["split"]  # to load correct feature CSV

    N = len(df_all)
    print(f"  Total patients: {N}  (train={len(df_tr)}, test={len(df_te)})")
    print(f"  Labels: {dict(zip(*np.unique(df_all['label'], return_counts=True)))}")

    # ── 2. Load sMRI raw features (768-dim BrainIAC) ───────────────────────
    smri_tr_df = pd.read_csv(BASE_DIR / "brainiac_features_train_v2.csv")
    smri_te_df = pd.read_csv(BASE_DIR / "brainiac_features_combined_test_v2.csv")

    smri_all_raw = np.zeros((N, 768), dtype=np.float32)
    for i, row in df_all.iterrows():
        if row["split"] == "train":
            smri_all_raw[i] = smri_tr_df.iloc[row["smri_feat_row"]][SMRI_COLS].values.astype(np.float32)
        else:
            smri_all_raw[i] = smri_te_df.iloc[row["smri_feat_row"]][SMRI_COLS].values.astype(np.float32)

    # ── 3. Pre-compute fMRI node features from globally harmonized matrices ─
    print("\n[2/4] Pre-computing fMRI node features (from globally harmonized matrices)...")
    t0 = time.time()
    all_feats = []; all_adjs = []
    for i, row in df_all.iterrows():
        sid = row["subject_id"]
        mat_path = FMRI_DIR / f"{sid}_combat.npy"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing harmonized fMRI: {mat_path}")
        mat  = np.load(mat_path)
        feat = extract_node_features(mat)
        adj  = build_adj(mat)
        all_feats.append(feat)
        all_adjs.append(adj)
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{N}] {sid}  ({time.time()-t0:.0f}s)")
    print(f"  Done in {time.time()-t0:.1f}s")

    sids    = df_all["subject_id"].values
    labels  = df_all["label"].values
    sites   = df_all["site"].values

    # ── 4. LOOCV main loop ─────────────────────────────────────────────────
    print("\n[3/4] Running LOOCV...")

    # Storage: per task, store (sid, true_label, loocv_prob)
    loocv_results = {task: {"sids": [], "true_labels": [], "probs": []} for task in TASKS}

    for i in range(N):
        sid_i   = sids[i]
        label_i = labels[i]
        site_i  = sites[i]
        t_fold  = time.time()

        relevant_tasks = LABEL_TO_TASKS.get(label_i, [])

        # Train mask (all except patient i)
        train_mask = np.ones(N, dtype=bool); train_mask[i] = False

        for task in relevant_tasks:
            cfg = TASKS[task]
            # Task-specific training patients (exclude patient i, keep task-relevant labels)
            task_mask = train_mask & np.isin(labels, cfg["classes"])
            task_idx  = np.where(task_mask)[0]

            if len(task_idx) < 5:
                print(f"  [SKIP] fold={i} task={task}: only {len(task_idx)} training patients")
                continue

            # Binary labels for task
            bin_labels_tr = (labels[task_idx] == cfg["pos"]).astype(np.int64)

            # sMRI ComBat: fit on task training patients, apply to patient i
            smri_train_harm, smri_test_harm = fit_and_apply_smri_combat(
                smri_all_raw[task_idx],
                sites[task_idx],
                bin_labels_tr,
                smri_all_raw[i],
                site_i,
            )

            # Build training data lists
            feats_tr = [all_feats[j] for j in task_idx]
            adjs_tr  = [all_adjs[j]  for j in task_idx]

            # Train model
            model = train_loocv_model(feats_tr, adjs_tr, smri_train_harm, bin_labels_tr)

            # Predict held-out patient
            prob = predict_prob(model, all_feats[i], all_adjs[i], smri_test_harm)

            # Store
            loocv_results[task]["sids"].append(sid_i)
            loocv_results[task]["true_labels"].append(int(label_i == cfg["pos"]))
            loocv_results[task]["probs"].append(prob)

            del model  # free GPU memory

        elapsed = time.time() - t_fold
        if (i+1) % 20 == 0 or i < 5:
            n_done = i + 1
            eta = (time.time() - t_start) / n_done * (N - n_done) / 60
            print(f"  Fold {i+1:3d}/{N}: sid={sid_i}, label={label_i}, tasks={relevant_tasks}, "
                  f"fold_time={elapsed:.0f}s, ETA={eta:.0f}min")

    # ── 5. Compute LOOCV metrics & save ────────────────────────────────────
    print("\n[4/4] Computing metrics...")
    rng = np.random.default_rng(SEED)
    summary = {}
    npz_data = {}

    print("\n" + "="*75)
    print(f"{'Task':<11} {'N':>5} {'AUC':>7} {'95% CI':>20} {'Sens':>7} {'Spec':>7}")
    print("="*75)

    for task in TASKS:
        r = loocv_results[task]
        if len(r["probs"]) == 0:
            print(f"{task:<11}: no predictions")
            continue
        probs  = np.array(r["probs"])
        labels_bin = np.array(r["true_labels"])
        sids_t = np.array(r["sids"])

        auc = roc_auc_score(labels_bin, probs)
        mu, lo, hi = bootstrap_auc(probs, labels_bin, rng=rng)

        # Threshold at 0.5 for sens/spec
        pred = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels_bin, pred, labels=[0,1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        summary[task] = {
            "n": int(len(probs)),
            "auc": float(auc),
            "bootstrap_mean": float(mu),
            "bootstrap_ci": [float(lo), float(hi)],
            "sensitivity": float(sens),
            "specificity": float(spec),
        }
        npz_data[f"{task}_probs"]  = probs
        npz_data[f"{task}_labels"] = labels_bin
        npz_data[f"{task}_sids"]   = sids_t

        ci_str = f"[{lo:.3f}, {hi:.3f}]"
        print(f"{task:<11} {len(probs):>5} {auc:>7.4f} {ci_str:>20} {sens:>7.3f} {spec:>7.3f}")

    # Save
    npz_path = RES_DIR / "loocv_results.npz"
    np.savez(npz_path, **npz_data)
    print(f"\n[SAVED] {npz_path}")

    json_path = RES_DIR / "loocv_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {json_path}")

    total_min = (time.time() - t_start) / 60
    print(f"\nTotal runtime: {total_min:.1f} minutes")


if __name__ == "__main__":
    main()
