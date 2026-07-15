"""
gen_labelaware_fmri_combat.py
=============================
W2 revision: generate LABEL-AWARE fMRI ComBat for NC_vs_MCI, so we can
retrain PCAG and compare within-site ADNI AUC against the label-free variant.

Fit on TRAIN only (binary NC/MCI label as categorical covariate), apply to
held-out TEST via neuroCombatFromTraining (leak-free). Output dir:
  fmri_combat_NC_vs_MCI_labelaware/

Usage:
  conda activate AD
  cd downstream/
  python data_prep/gen_labelaware_fmri_combat.py
"""
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from neuroCombat import neuroCombat, neuroCombatFromTraining

BASE_DIR  = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
TRAIN_CSV = BASE_DIR / "pcag_train_aligned_v2.csv"
TEST_CSV  = BASE_DIR / "pcag_test_aligned_v2.csv"
N_ROIS    = 116

TASK_CFG = {
    "NC_vs_AD":  {"classes": [0, 2], "pos": 2},
    "NC_vs_MCI": {"classes": [0, 1], "pos": 1},
    "MCI_vs_AD": {"classes": [1, 2], "pos": 2},
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="NC_vs_MCI", choices=list(TASK_CFG))
args = parser.parse_args()
TASK = args.task
CLASSES, POS = TASK_CFG[TASK]["classes"], TASK_CFG[TASK]["pos"]
OUT_DIR = BASE_DIR / f"fmri_combat_{TASK}_labelaware"
OUT_DIR.mkdir(exist_ok=True)


def detect_site(sid):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(sid)) else 'TPMIC'


def load_matrix(path):
    mat = np.load(path)
    if mat.ndim == 3:
        mat = mat[0]
    mat = (mat + mat.T) / 2
    return mat.astype(np.float32)


def matrix_to_tri(mat):
    return mat[np.triu_indices(N_ROIS, k=1)]


def tri_to_matrix(tri, diag_vals):
    iu = np.triu_indices(N_ROIS, k=1)
    mat = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
    mat[iu] = tri
    mat = mat + mat.T
    np.fill_diagonal(mat, diag_vals)
    return mat


def main():
    df_tr = pd.read_csv(TRAIN_CSV)
    df_te = pd.read_csv(TEST_CSV)
    tr = df_tr[df_tr["label"].isin(CLASSES)].copy()
    te = df_te[df_te["label"].isin(CLASSES)].copy()
    tr["bin_label"] = (tr["label"] == POS).astype(int)
    te["bin_label"] = (te["label"] == POS).astype(int)
    tr["site"] = tr["subject_id"].apply(detect_site)
    te["site"] = te["subject_id"].apply(detect_site)

    print(f"Train: {len(tr)} (ADNI={(tr.site=='ADNI').sum()}, TPMIC={(tr.site=='TPMIC').sum()})")
    print(f"Test:  {len(te)} (ADNI={(te.site=='ADNI').sum()}, TPMIC={(te.site=='TPMIC').sum()})")

    tr_mats = [load_matrix(p) for p in tr["matrix_path"]]
    te_mats = [load_matrix(p) for p in te["matrix_path"]]
    tr_data = np.array([matrix_to_tri(m) for m in tr_mats])
    te_data = np.array([matrix_to_tri(m) for m in te_mats])
    tr_diags = np.array([np.diag(m) for m in tr_mats])
    te_diags = np.array([np.diag(m) for m in te_mats])

    # Fit LABEL-AWARE ComBat on train only
    covars = pd.DataFrame({"batch": tr["site"].values, "label": tr["bin_label"].values})
    fit = neuroCombat(dat=tr_data.T, covars=covars,
                      batch_col="batch", categorical_cols=["label"])
    train_h = fit["data"].T
    test_h = neuroCombatFromTraining(
        dat=te_data.T, batch=te["site"].values, estimates=fit["estimates"]
    )["data"].T

    for i, sid in enumerate(tr["subject_id"]):
        np.save(OUT_DIR / f"{sid}_combat.npy",
                tri_to_matrix(train_h[i], tr_diags[i]).astype(np.float32))
    for i, sid in enumerate(te["subject_id"]):
        np.save(OUT_DIR / f"{sid}_combat.npy",
                tri_to_matrix(test_h[i], te_diags[i]).astype(np.float32))

    print(f"[SAVED] {len(tr)+len(te)} matrices -> {OUT_DIR}")


if __name__ == "__main__":
    main()
