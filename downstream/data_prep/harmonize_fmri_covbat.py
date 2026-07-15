"""
harmonize_fmri_covbat.py
========================
Apply CovBat harmonization to fMRI FC features.

CovBat (Chen et al. 2022, Human Brain Mapping):
  Step 1: Standard label-free ComBat → harmonize mean/variance
  Step 2: PCA on ComBat-harmonized training data → get K leading PCs
  Step 3: Apply ComBat again to the K PC scores → harmonize covariance
  Step 4: Reconstruct = ComBat-harmonized residual + CovBat PC scores

Output: fmri_covbat_nolabel/ (same format as fmri_combat_v2_nolabel/)
"""
import re, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from neuroCombat import neuroCombat

DOWNSTREAM = Path(__file__).parent.parent
TRAIN_CSV  = DOWNSTREAM / "pcag_train_aligned_v2.csv"
TEST_CSV   = DOWNSTREAM / "pcag_test_aligned_v2.csv"
OUT_DIR    = DOWNSTREAM / "fmri_covbat_nolabel"
N_ROIS     = 116
VAR_THRESH = 0.80  # keep PCs that explain ≥80% cumulative variance

def detect_site(sid: str) -> str:
    return "ADNI" if re.search(r"\d{3}_S_\d{4}", str(sid)) else "TPMIC"

def load_matrix(path: str) -> np.ndarray:
    mat = np.load(path)
    if mat.ndim == 3:
        mat = mat[0]
    mat = (mat + mat.T) / 2
    return mat.astype(np.float32)

def matrix_to_tri(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(N_ROIS, k=1)
    return mat[iu]

def tri_to_matrix(tri: np.ndarray, diag_vals: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(N_ROIS, k=1)
    mat = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
    mat[iu] = tri
    mat = mat + mat.T
    np.fill_diagonal(mat, diag_vals)
    return mat

def _apply_combat(dat: np.ndarray, site_labels: np.ndarray,
                  site_map: dict, estimates: dict) -> np.ndarray:
    """Apply pre-fitted ComBat estimates to new data (workaround for neuroCombatFromTraining NumPy bug)."""
    site_indices = np.array([site_map[s] for s in site_labels])
    s_mean    = estimates["stand.mean"]
    g_mean    = s_mean.mean(axis=1) if (s_mean.ndim == 2 and s_mean.shape[1] > 1) else s_mean.flatten()
    v_pooled  = estimates["var.pooled"].flatten()
    gamma     = estimates["gamma.star"][site_indices]   # (n, d)
    delta     = estimates["delta.star"][site_indices]   # (n, d)
    dat_std   = (dat - g_mean) / (np.sqrt(v_pooled) + 1e-8)
    dat_adj   = (dat_std - gamma) / (np.sqrt(delta) + 1e-8)
    return dat_adj * np.sqrt(v_pooled) + g_mean


def covbat_harmonize(
    train_data: np.ndarray,   # (n_train, 6670)
    train_sites: np.ndarray,  # (n_train,)  str labels
    test_data: np.ndarray,    # (n_test, 6670)
    test_sites: np.ndarray,   # (n_test,)   str labels
    var_thresh: float = VAR_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CovBat harmonization.
    Returns (train_covbat, test_covbat) both shape (n, 6670).
    """
    print(f"  CovBat Step 1: Label-free ComBat (n_train={len(train_data)}, n_test={len(test_data)}) ...", end=" ")
    t0 = time.time()

    all_sites  = sorted(set(train_sites) | set(test_sites))
    site_map   = {s: i for i, s in enumerate(all_sites)}
    covars_tr  = pd.DataFrame({"batch": train_sites})

    fit_result = neuroCombat(
        dat=train_data.T,
        covars=covars_tr,
        batch_col="batch",
        categorical_cols=[],
    )
    train_c  = fit_result["data"].T         # (n_train, 6670) ComBat harmonized
    estimates = fit_result["estimates"]

    test_c = _apply_combat(test_data, test_sites, site_map, estimates)
    print(f"done ({time.time()-t0:.1f}s)")

    # Step 2: PCA on ComBat-harmonized training data
    print(f"  CovBat Step 2: PCA (var_thresh={var_thresh*100:.0f}%) ...", end=" ")
    t0 = time.time()
    pca = PCA(n_components=None, random_state=42)
    pca.fit(train_c)

    # Choose K: number of PCs explaining ≥ var_thresh cumulative variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    K = int(np.searchsorted(cum_var, var_thresh)) + 1
    K = max(K, 5)   # at least 5 PCs
    print(f"K={K} PCs (explains {cum_var[K-1]*100:.1f}% variance, done {time.time()-t0:.1f}s)")

    # Project to PC space
    train_pcs = pca.transform(train_c)[:, :K]  # (n_train, K)
    test_pcs  = pca.transform(test_c)[:, :K]   # (n_test, K)

    # Step 3: Apply ComBat to PC scores
    print(f"  CovBat Step 3: ComBat on {K} PC scores ...", end=" ")
    t0 = time.time()
    covars_tr_pc = pd.DataFrame({"batch": train_sites})
    pc_fit = neuroCombat(
        dat=train_pcs.T,
        covars=covars_tr_pc,
        batch_col="batch",
        categorical_cols=[],
    )
    train_pcs_h = pc_fit["data"].T          # (n_train, K)

    test_pcs_h = _apply_combat(test_pcs, test_sites, site_map, pc_fit["estimates"])
    print(f"done ({time.time()-t0:.1f}s)")

    # Step 4: Reconstruct = ComBat base + CovBat PC adjustment
    # Reconstruction: add back (harmonized_PC_scores - original_PC_scores) × V^T
    V = pca.components_[:K]                 # (K, 6670)

    train_delta_pc = train_pcs_h - train_pcs   # (n_train, K) adjustment in PC space
    test_delta_pc  = test_pcs_h  - test_pcs    # (n_test,  K)

    train_covbat = train_c + train_delta_pc @ V   # (n_train, 6670)
    test_covbat  = test_c  + test_delta_pc  @ V   # (n_test,  6670)

    return train_covbat.astype(np.float32), test_covbat.astype(np.float32)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    df_train["site"] = df_train["subject_id"].apply(detect_site)
    df_test["site"]  = df_test["subject_id"].apply(detect_site)

    # Load raw FC matrices
    print("\nLoading fMRI matrices...")
    tr_mats  = [load_matrix(p) for p in df_train["matrix_path"]]
    te_mats  = [load_matrix(p) for p in df_test["matrix_path"]]
    tr_data  = np.array([matrix_to_tri(m) for m in tr_mats])   # (n_train, 6670)
    te_data  = np.array([matrix_to_tri(m) for m in te_mats])   # (n_test,  6670)
    tr_diags = np.array([np.diag(m) for m in tr_mats])
    te_diags = np.array([np.diag(m) for m in te_mats])
    print(f"  tr_data shape: {tr_data.shape}, te_data shape: {te_data.shape}")

    # Site labels
    tr_sites = df_train["site"].values
    te_sites = df_test["site"].values
    print(f"  Train sites: ADNI={(tr_sites=='ADNI').sum()}, TPMIC={(tr_sites=='TPMIC').sum()}")
    print(f"  Test sites:  ADNI={(te_sites=='ADNI').sum()}, TPMIC={(te_sites=='TPMIC').sum()}")

    # CovBat
    print("\nRunning CovBat harmonization ...")
    train_h, test_h = covbat_harmonize(tr_data, tr_sites, te_data, te_sites)

    # Sanity check
    adni_pre  = tr_data[tr_sites == "ADNI"].mean()
    tpmic_pre = tr_data[tr_sites == "TPMIC"].mean()
    adni_post  = train_h[tr_sites == "ADNI"].mean()
    tpmic_post = train_h[tr_sites == "TPMIC"].mean()
    print(f"\n  Pre-CovBat:  ADNI mean={adni_pre:.4f}, TPMIC mean={tpmic_pre:.4f}, diff={abs(adni_pre-tpmic_pre):.4f}")
    print(f"  Post-CovBat: ADNI mean={adni_post:.4f}, TPMIC mean={tpmic_post:.4f}, diff={abs(adni_post-tpmic_post):.4f}")

    # Save per-subject .npy files (same format as fmri_combat_v2_nolabel/)
    print("\nSaving harmonized matrices...")
    for i, sid in enumerate(df_train["subject_id"]):
        mat = tri_to_matrix(train_h[i], tr_diags[i])
        np.save(OUT_DIR / f"{sid}_combat.npy", mat)
    for i, sid in enumerate(df_test["subject_id"]):
        mat = tri_to_matrix(test_h[i], te_diags[i])
        np.save(OUT_DIR / f"{sid}_combat.npy", mat)

    print(f"\nDone! Saved {len(df_train)+len(df_test)} files to {OUT_DIR}")
    print(f"  Train: {len(df_train)} files")
    print(f"  Test:  {len(df_test)} files")


if __name__ == "__main__":
    main()
