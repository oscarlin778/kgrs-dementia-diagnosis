"""
eval_site_data_regime.py
========================
Slide story (#B): show that single-site data is insufficient, and that merging
sites + label-free ComBat (our approach) improves NC vs. MCI discrimination.

All conditions: fMRI-only GAT/PCAG, NC vs. MCI, 3-seed ensemble (seeds 42,123,456)
for stability, evaluated by within-site OOF AUC where applicable.

Conditions (raw vs. label-free harmonized fMRI):
  single-site : ADNI-only CV (eval ADNI), TPMIC-only CV (eval TPMIC)
  cross-site  : ADNI->TPMIC, TPMIC->ADNI
  merged      : full-pool CV, eval on ADNI subjects and on TPMIC subjects

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_site_data_regime.py
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
from training.train_pcag_combat_fusion import PCAGModel, PCAGDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARM = BASE / "fmri_combat_v2_nolabel"
SEEDS = [42, 123, 456]


def detect_site(s):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(s)) else 'TPMIC'


def load_pool():
    tr = pd.read_csv(BASE / "pcag_train_aligned_v2.csv")
    te = pd.read_csv(BASE / "pcag_test_aligned_v2.csv")
    df = pd.concat([tr, te], ignore_index=True)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    df["bin"] = (df["label"] == 1).astype(int)
    df["site"] = df["subject_id"].apply(detect_site)
    return df


def path_of(r, harm):
    return str(HARM / f"{r['subject_id']}_combat.npy") if harm else r["matrix_path"]


def _train_one(train_df, harm, seed):
    paths = [path_of(r, harm) for _, r in train_df.iterrows()]
    labels = train_df["bin"].values
    counts = np.bincount(labels, minlength=2); w = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler(w[labels], len(labels))
    loader = DataLoader(PCAGDataset(paths, np.zeros((len(train_df), 768), np.float32), labels),
                        batch_size=16, sampler=sampler, num_workers=2)
    torch.manual_seed(seed); np.random.seed(seed)
    model = PCAGModel(fusion_dim=128, modality='fmri_only').to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-3)
    crit = nn.CrossEntropyLoss(); model.train()
    for _ in range(60):
        for x, adj, smri, y in loader:
            x, adj, smri, y = x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); out, _ = model(x, adj, smri); crit(out, y).backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def _predict(model, df, harm):
    loader = DataLoader(PCAGDataset([path_of(r, harm) for _, r in df.iterrows()],
                                    np.zeros((len(df), 768), np.float32), df["bin"].values),
                        batch_size=16, shuffle=False)
    probs = []
    for x, adj, smri, y in loader:
        out, _ = model(x.to(DEVICE), adj.to(DEVICE), smri.to(DEVICE))
        probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
    return np.array(probs)


def cv_oof(pool_df, harm):
    """5-fold CV over pool_df; return per-subject OOF probs averaged over seeds."""
    oof_seeds = []
    for seed in SEEDS:
        oof = np.zeros(len(pool_df))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(pool_df, pool_df["bin"]):
            model = _train_one(pool_df.iloc[tr_idx], harm, seed)
            oof[va_idx] = _predict(model, pool_df.iloc[va_idx], harm)
        oof_seeds.append(oof)
    return np.mean(oof_seeds, axis=0)


def cross(train_df, test_df, harm):
    """Train on train_df (one site), predict test_df (other site); 3-seed ensemble."""
    preds = []
    for seed in SEEDS:
        model = _train_one(train_df, harm, seed)
        preds.append(_predict(model, test_df, harm))
    return np.mean(preds, axis=0)


def main():
    df = load_pool()
    adni = df[df["site"] == "ADNI"].reset_index(drop=True)
    tpmic = df[df["site"] == "TPMIC"].reset_index(drop=True)
    res = {}

    for harm in [False, True]:
        cond = "harmonized" if harm else "raw"
        print(f"\n########## {cond.upper()} ##########")

        # single-site CV
        a_oof = cv_oof(adni, harm)
        t_oof = cv_oof(tpmic, harm)
        res.setdefault("single_ADNI", {})[cond] = float(roc_auc_score(adni["bin"], a_oof))
        res.setdefault("single_TPMIC", {})[cond] = float(roc_auc_score(tpmic["bin"], t_oof))
        print(f"  single ADNI->ADNI   : {res['single_ADNI'][cond]:.4f}")
        print(f"  single TPMIC->TPMIC : {res['single_TPMIC'][cond]:.4f}")

        # cross-site
        res.setdefault("cross_ADNI_to_TPMIC", {})[cond] = float(roc_auc_score(tpmic["bin"], cross(adni, tpmic, harm)))
        res.setdefault("cross_TPMIC_to_ADNI", {})[cond] = float(roc_auc_score(adni["bin"], cross(tpmic, adni, harm)))
        print(f"  cross ADNI->TPMIC   : {res['cross_ADNI_to_TPMIC'][cond]:.4f}")
        print(f"  cross TPMIC->ADNI   : {res['cross_TPMIC_to_ADNI'][cond]:.4f}")

        # merged CV, eval per site
        m_oof = cv_oof(df, harm)
        m_site = np.array([detect_site(s) for s in df["subject_id"]])
        res.setdefault("merged_ADNI", {})[cond] = float(roc_auc_score(df["bin"][m_site == "ADNI"], m_oof[m_site == "ADNI"]))
        res.setdefault("merged_TPMIC", {})[cond] = float(roc_auc_score(df["bin"][m_site == "TPMIC"], m_oof[m_site == "TPMIC"]))
        res.setdefault("merged_overall", {})[cond] = float(roc_auc_score(df["bin"], m_oof))
        print(f"  merged -> ADNI      : {res['merged_ADNI'][cond]:.4f}")
        print(f"  merged -> TPMIC     : {res['merged_TPMIC'][cond]:.4f}")
        print(f"  merged overall      : {res['merged_overall'][cond]:.4f}")

    out = BASE / "results" / "site_data_regime.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
