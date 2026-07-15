"""
eval_loso_fmri.py
=================
W4 revision: Leave-One-Site-Out (LOSO) cross-site generalization for NC vs. MCI,
testing whether label-free fMRI ComBat improves transfer between sites.

Protocol (fMRI-only GAT/PCAG, to isolate the harmonization effect):
  - Pool all NC+MCI subjects (train+test split merged), split by SITE.
  - Two directions: train ADNI -> test TPMIC, and train TPMIC -> test ADNI.
  - Two conditions:
      raw         : original FC matrices (no fMRI ComBat)
      harmonized  : label-free ComBat matrices from fmri_combat_v2_nolabel/
                    (fit unsupervised on the full multi-site cohort)
  - Train with 5-fold CV on the train site; average the 5 folds' probabilities
    on the held-out test site. Report test-site AUC.

A 2-site cohort makes LOSO inherently limited (single-site train cannot itself
harmonize), so this is a complementary cross-site check, not the headline result.

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_loso_fmri.py
"""
import sys, re, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from training.train_pcag_combat_fusion import (
    PCAGModel, PCAGDataset, extract_node_features, build_adj)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARM_DIR = BASE / "fmri_combat_v2_nolabel"
SEED = 42


def detect_site(s):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(s)) else 'TPMIC'


def load_pool():
    tr = pd.read_csv(BASE / "pcag_train_aligned_v2.csv")
    te = pd.read_csv(BASE / "pcag_test_aligned_v2.csv")
    df = pd.concat([tr, te], ignore_index=True)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)   # NC vs MCI
    df["bin"] = (df["label"] == 1).astype(int)
    df["site"] = df["subject_id"].apply(detect_site)
    return df


def train_eval(train_df, test_df, harmonized):
    """5-fold CV on train_df; mean of folds' probs on test_df. fMRI-only."""
    def path_of(r):
        if harmonized:
            return str(HARM_DIR / f"{r['subject_id']}_combat.npy")
        return r["matrix_path"]
    zero_smri_tr = np.zeros((len(train_df), 768), dtype=np.float32)
    zero_smri_te = np.zeros((len(test_df), 768), dtype=np.float32)
    te_paths = [path_of(r) for _, r in test_df.iterrows()]
    te_loader = DataLoader(PCAGDataset(te_paths, zero_smri_te, test_df["bin"].values),
                           batch_size=16, shuffle=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    test_fold_probs = []
    for tr_idx, _ in skf.split(train_df, train_df["bin"]):
        sub = train_df.iloc[tr_idx]
        paths = [path_of(r) for _, r in sub.iterrows()]
        labels = sub["bin"].values
        counts = np.bincount(labels, minlength=2)
        w = 1.0 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler(w[labels], len(labels))
        loader = DataLoader(PCAGDataset(paths, zero_smri_tr[tr_idx], labels),
                            batch_size=16, sampler=sampler, num_workers=2)
        model = PCAGModel(fusion_dim=128, modality='fmri_only').to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-3)
        crit = nn.CrossEntropyLoss()
        model.train()
        for _ in range(60):
            for x, adj, smri, y in loader:
                x, adj, smri, y = x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                out, _ = model(x, adj, smri)
                loss = crit(out, y)
                loss.backward(); opt.step()
        model.eval(); probs = []
        with torch.no_grad():
            for x, adj, smri, y in te_loader:
                out, _ = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE))
                probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
        test_fold_probs.append(probs)
    mean_probs = np.mean(test_fold_probs, axis=0)
    return roc_auc_score(test_df["bin"].values, mean_probs)


def within_site_cv(site_df, harmonized):
    """5-fold CV within one site (train 4 folds, test held-out fold); OOF AUC."""
    def path_of(r):
        return str(HARM_DIR / f"{r['subject_id']}_combat.npy") if harmonized else r["matrix_path"]
    oof = np.zeros(len(site_df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in skf.split(site_df, site_df["bin"]):
        tr = site_df.iloc[tr_idx]; va = site_df.iloc[va_idx]
        paths = [path_of(r) for _, r in tr.iterrows()]; labels = tr["bin"].values
        counts = np.bincount(labels, minlength=2); w = 1.0 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler(w[labels], len(labels))
        loader = DataLoader(PCAGDataset(paths, np.zeros((len(tr), 768), np.float32), labels),
                            batch_size=16, sampler=sampler, num_workers=2)
        va_loader = DataLoader(PCAGDataset([path_of(r) for _, r in va.iterrows()],
                                           np.zeros((len(va), 768), np.float32), va["bin"].values),
                               batch_size=16, shuffle=False)
        model = PCAGModel(fusion_dim=128, modality='fmri_only').to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-3)
        crit = nn.CrossEntropyLoss(); model.train()
        for _ in range(60):
            for x, adj, smri, y in loader:
                x, adj, smri, y = x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(); out, _ = model(x, adj, smri); crit(out, y).backward(); opt.step()
        model.eval(); probs = []
        with torch.no_grad():
            for x, adj, smri, y in va_loader:
                out, _ = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE))
                probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
        oof[va_idx] = probs
    return roc_auc_score(site_df["bin"].values, oof)


def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    df = load_pool()
    print("NC vs MCI pool by site:")
    print(df.groupby(["site", "label"]).size().to_string())

    results = {}
    # Within-site (same-site CV): ADNI->ADNI, TPMIC->TPMIC
    for site in ["ADNI", "TPMIC"]:
        sdf = df[df["site"] == site].reset_index(drop=True)
        key = f"{site}->{site}"
        results[key] = {}
        print(f"\n=== Within-site {key}  (n={len(sdf)}, 5-fold CV) ===")
        for cond, harm in [("raw", False), ("harmonized", True)]:
            auc = within_site_cv(sdf, harm)
            results[key][cond] = float(auc)
            print(f"  {cond:11s}: within-site AUC = {auc:.4f}")
    # Cross-site (LOSO): ADNI->TPMIC, TPMIC->ADNI
    for train_site, test_site in [("ADNI", "TPMIC"), ("TPMIC", "ADNI")]:
        tr = df[df["site"] == train_site].reset_index(drop=True)
        te = df[df["site"] == test_site].reset_index(drop=True)
        key = f"{train_site}->{test_site}"
        results[key] = {}
        print(f"\n=== Cross-site {key}  (train n={len(tr)}, test n={len(te)}) ===")
        for cond, harm in [("raw", False), ("harmonized", True)]:
            auc = train_eval(tr, te, harm)
            results[key][cond] = float(auc)
            print(f"  {cond:11s}: test-site AUC = {auc:.4f}")
        results[key]["delta"] = results[key]["harmonized"] - results[key]["raw"]

    out = BASE / "results" / "loso_fmri_nc_vs_mci.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
