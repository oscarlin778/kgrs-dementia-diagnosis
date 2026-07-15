"""
gen_fmri_combat_estimates.py
============================
Persist the LABEL-FREE fMRI ComBat estimates so the inference pipeline can
harmonize a new patient's raw FC matrix on the fly (the same transform that
produced fmri_combat_v2_nolabel/). Previously only the harmonized .npy outputs
were saved, not the estimator, so deployment could not harmonize new inputs.

Fits label-free ComBat on the full training cohort (site batch only, no
diagnosis covariate) and saves the estimates as JSON, mirroring the sMRI
combat_params_{task}.json format consumed by inference_pipeline_v2.apply_combat.

Output: checkpoints/fmri_combat_estimates_nolabel.json

Usage:
  conda activate AD
  cd downstream/
  python data_prep/gen_fmri_combat_estimates.py
"""
import json, re
import numpy as np
import pandas as pd
from pathlib import Path
from neuroCombat import neuroCombat

BASE_DIR  = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
TRAIN_CSV = BASE_DIR / "pcag_train_aligned_v2.csv"
OUT_PATH  = BASE_DIR / "checkpoints" / "fmri_combat_estimates_nolabel.json"
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
    return mat[np.triu_indices(N_ROIS, k=1)]


def main():
    df = pd.read_csv(TRAIN_CSV)
    df["site"] = df["subject_id"].apply(detect_site)
    all_sites = sorted(df["site"].unique())
    site_map = {s: i for i, s in enumerate(all_sites)}

    mats = [load_matrix(p) for p in df["matrix_path"]]
    tri = np.array([matrix_to_tri(m) for m in mats])  # (n, 6670)

    # Label-free: site batch only, NO diagnosis covariate
    covars = pd.DataFrame({"batch": df["site"].map(site_map).values})
    out = neuroCombat(dat=tri.T, covars=covars, batch_col="batch", categorical_cols=[])
    est = out["estimates"]

    params = {
        "site_map":   site_map,
        "stand_mean": np.asarray(est["stand.mean"]).tolist(),    # (6670, n) or (6670,)
        "var_pooled": np.asarray(est["var.pooled"]).flatten().tolist(),  # (6670,)
        "gamma_star": np.asarray(est["gamma.star"]).tolist(),    # (n_sites, 6670)
        "delta_star": np.asarray(est["delta.star"]).tolist(),    # (n_sites, 6670)
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(params, f)
    print(f"site_map={site_map}")
    print(f"stand_mean shape={np.asarray(est['stand.mean']).shape}, "
          f"var_pooled={len(params['var_pooled'])}, "
          f"gamma_star={np.asarray(est['gamma.star']).shape}")
    print(f"[SAVED] {OUT_PATH}")


if __name__ == "__main__":
    main()
