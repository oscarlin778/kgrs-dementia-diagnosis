"""
eval_site_prediction_probe.py
=============================
W2 revision experiment: direct empirical comparison of label-aware vs.
label-free ComBat on fMRI, via a SITE-PREDICTION PROBE.

Rationale (reviewer W2 / Q6):
  The paper claims that under severe site-class imbalance, label-aware ComBat
  inadvertently PRESERVES site signatures (because the diagnosis covariate is
  highly correlated with site, so "protecting" label variance protects site
  variance). label-free ComBat removes pure site effects.

  We test this directly: how well can a classifier predict the SITE
  (ADNI vs TPMIC) from the 6670-dim FC features under three conditions?
    1. Raw (no ComBat)        -> site should be highly predictable
    2. Label-free ComBat      -> site predictability should collapse to chance
    3. Label-aware ComBat      -> residual site predictability should remain
                                 HIGHER than label-free (signature preserved)

  Lower site-prediction AUC/accuracy = better de-confounding.

Probe: L2 logistic regression, 5-fold stratified CV (site as target).
ComBat fit on the SAME data being probed (this is a diagnostic of residual
site information, not a downstream model — no train/test split needed).

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_site_prediction_probe.py
"""
import json, re, time
import numpy as np
import pandas as pd
from pathlib import Path
from neuroCombat import neuroCombat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

BASE_DIR  = Path(__file__).parent.parent
TRAIN_CSV = BASE_DIR / "pcag_train_aligned_v2.csv"
RES_DIR   = BASE_DIR / "results"
N_ROIS    = 116


def detect_site(sid):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(sid)) else 'TPMIC'


def load_matrix(path):
    mat = np.load(path)
    if mat.ndim == 3:
        mat = mat[0]
    mat = (mat + mat.T) / 2
    return mat.astype(np.float32)


def matrix_to_tri(mat):
    iu = np.triu_indices(N_ROIS, k=1)
    return mat[iu]


def combat_harmonize(data, sites, labels, with_label):
    """Fit ComBat on the given data. Returns harmonized (n, 6670)."""
    covars = pd.DataFrame({"batch": sites})
    cat_cols = []
    if with_label:
        covars["label"] = labels
        cat_cols = ["label"]
    out = neuroCombat(
        dat=data.T,
        covars=covars,
        batch_col="batch",
        categorical_cols=cat_cols,
    )
    return out["data"].T


def site_probe(X, site_y, seed=42):
    """5-fold CV logistic regression predicting site. Returns (acc, auc)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for tr_idx, te_idx in skf.split(X, site_y):
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr = scaler.transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
        clf.fit(Xtr, site_y[tr_idx])
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]
        accs.append(accuracy_score(site_y[te_idx], pred))
        aucs.append(roc_auc_score(site_y[te_idx], prob))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(aucs)), float(np.std(aucs))


def main():
    df = pd.read_csv(TRAIN_CSV)
    df["site"] = df["subject_id"].apply(detect_site)

    print(f"Loaded {len(df)} train subjects")
    print(df.groupby(["site", "label"]).size().to_string())

    # Load FC features
    print("\nLoading FC matrices...")
    mats = [load_matrix(p) for p in df["matrix_path"]]
    X_raw = np.array([matrix_to_tri(m) for m in mats])  # (n, 6670)
    sites = df["site"].values
    labels = df["label"].values
    site_y = (sites == "TPMIC").astype(int)  # 1=TPMIC, 0=ADNI

    print(f"FC feature matrix: {X_raw.shape}")
    print(f"Site balance: ADNI={np.sum(site_y==0)}, TPMIC={np.sum(site_y==1)}")
    chance = max(np.mean(site_y), 1 - np.mean(site_y))
    print(f"Majority-class baseline accuracy = {chance:.3f}\n")

    conditions = {}

    # 1. Raw
    print("[1/3] Raw (no ComBat)...")
    conditions["raw"] = X_raw

    # 2. Label-free ComBat
    print("[2/3] Label-free ComBat...")
    conditions["label_free"] = combat_harmonize(X_raw, sites, labels, with_label=False)

    # 3. Label-aware ComBat (3-class diagnosis as covariate)
    print("[3/3] Label-aware ComBat...")
    conditions["label_aware"] = combat_harmonize(X_raw, sites, labels, with_label=True)

    # Probe each condition
    print(f"\n{'='*64}")
    print(f"{'Condition':<16}{'Site Acc':>16}{'Site AUC':>16}")
    print(f"{'-'*64}")
    results = {}
    for name, X in conditions.items():
        acc, acc_sd, auc, auc_sd = site_probe(X, site_y)
        results[name] = {"site_acc": acc, "site_acc_sd": acc_sd,
                         "site_auc": auc, "site_auc_sd": auc_sd}
        print(f"{name:<16}{acc:>8.3f}±{acc_sd:<5.3f}{auc:>8.3f}±{auc_sd:<5.3f}")
    print(f"{'='*64}")
    print(f"Majority-class baseline acc = {chance:.3f}; chance AUC = 0.500")

    out = {
        "description": "Site-prediction probe (ADNI vs TPMIC) from 6670-dim FC features",
        "n_subjects": int(len(df)),
        "n_adni": int(np.sum(site_y == 0)),
        "n_tpmic": int(np.sum(site_y == 1)),
        "majority_baseline_acc": float(chance),
        "probe": "L2 logistic regression, 5-fold stratified CV",
        "results": results,
    }
    out_path = RES_DIR / "site_prediction_probe.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
