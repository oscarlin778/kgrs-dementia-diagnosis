"""
harmonize_fmri_combat_variants.py
==================================
產生兩種 fMRI ComBat 變體做比較：

選項 1: Task-specific ComBat
  - 對每個 task (NC_vs_AD / NC_vs_MCI / MCI_vs_AD) 用該 task 的 2-class 資料分別 fit
  - 用 binary label 為 categorical covariate
  - 輸出三個目錄：fmri_combat_NC_vs_AD/, fmri_combat_NC_vs_MCI/, fmri_combat_MCI_vs_AD/

選項 3: No-label ComBat
  - 全資料 (204 patients) 一起 fit
  - 不帶 label covariate (純粹 site removal)
  - 輸出：fmri_combat_v2_nolabel/
"""
import json, re, time
import numpy as np
import pandas as pd
from pathlib import Path
from neuroCombat import neuroCombat, neuroCombatFromTraining

BASE_DIR  = Path(__file__).parent
TRAIN_CSV = BASE_DIR / "pcag_train_aligned_v2.csv"
TEST_CSV  = BASE_DIR / "pcag_test_aligned_v2.csv"

N_ROIS = 116


def detect_site(sid):
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(sid)) else 'TPMIC'


def load_matrix(path):
    mat = np.load(path)
    if mat.ndim == 3: mat = mat[0]
    mat = (mat + mat.T) / 2
    return mat.astype(np.float32)


def matrix_to_tri(mat):
    iu = np.triu_indices(N_ROIS, k=1)
    return mat[iu]


def tri_to_matrix(tri, diag_vals):
    iu = np.triu_indices(N_ROIS, k=1)
    mat = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
    mat[iu] = tri
    mat = mat + mat.T
    np.fill_diagonal(mat, diag_vals)
    return mat


def run_combat(train_data, train_sites, train_labels,
               test_data, test_sites, test_labels,
               with_label=True, tag=""):
    """Fit on train, apply on test. Returns (train_h, test_h)."""
    print(f"  [{tag}] fitting ComBat (with_label={with_label})...")

    covars_train = pd.DataFrame({"batch": train_sites})
    cat_cols = []
    if with_label:
        covars_train["label"] = train_labels
        cat_cols = ["label"]

    t0 = time.time()
    fit_result = neuroCombat(
        dat=train_data.T,
        covars=covars_train,
        batch_col="batch",
        categorical_cols=cat_cols,
    )
    train_h = fit_result["data"].T  # (n_train, 6670)

    apply_result = neuroCombatFromTraining(
        dat=test_data.T,
        batch=test_sites,
        estimates=fit_result["estimates"],
    )
    test_h = apply_result["data"].T  # (n_test, 6670)
    print(f"    done in {time.time()-t0:.1f}s")
    return train_h, test_h


def variant_task_specific():
    """選項 1: 三個 task 分開做 ComBat。"""
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    TASK_CFG = {
        "NC_vs_AD":  {"classes": [0, 2], "pos": 2},
        "NC_vs_MCI": {"classes": [0, 1], "pos": 1},
        "MCI_vs_AD": {"classes": [1, 2], "pos": 2},
    }

    for task, cfg in TASK_CFG.items():
        print(f"\n{'='*60}")
        print(f"Task-specific ComBat: {task}")
        print('='*60)

        out_dir = BASE_DIR / f"fmri_combat_{task}"
        out_dir.mkdir(exist_ok=True)

        tr = df_train[df_train["label"].isin(cfg["classes"])].copy()
        te = df_test [df_test ["label"].isin(cfg["classes"])].copy()
        tr["bin_label"] = (tr["label"] == cfg["pos"]).astype(int)
        te["bin_label"] = (te["label"] == cfg["pos"]).astype(int)
        tr["site"] = tr["subject_id"].apply(detect_site)
        te["site"] = te["subject_id"].apply(detect_site)

        print(f"  Train: {len(tr)} (ADNI={(tr.site=='ADNI').sum()}, TPMIC={(tr.site=='TPMIC').sum()})")
        print(f"  Test:  {len(te)} (ADNI={(te.site=='ADNI').sum()}, TPMIC={(te.site=='TPMIC').sum()})")

        tr_mats = [load_matrix(p) for p in tr["matrix_path"]]
        te_mats = [load_matrix(p) for p in te["matrix_path"]]
        tr_data  = np.array([matrix_to_tri(m) for m in tr_mats])
        te_data  = np.array([matrix_to_tri(m) for m in te_mats])
        tr_diags = np.array([np.diag(m) for m in tr_mats])
        te_diags = np.array([np.diag(m) for m in te_mats])

        train_h, test_h = run_combat(
            tr_data, tr["site"].values, tr["bin_label"].values,
            te_data, te["site"].values, te["bin_label"].values,
            with_label=True, tag=task,
        )

        # 重組並存檔
        for i, sid in enumerate(tr["subject_id"]):
            mat = tri_to_matrix(train_h[i], tr_diags[i]).astype(np.float32)
            np.save(out_dir / f"{sid}_combat.npy", mat)
        for i, sid in enumerate(te["subject_id"]):
            mat = tri_to_matrix(test_h[i], te_diags[i]).astype(np.float32)
            np.save(out_dir / f"{sid}_combat.npy", mat)

        # sanity check
        adni_pre = tr_data[tr["site"]=='ADNI'].mean()
        tpmic_pre = tr_data[tr["site"]=='TPMIC'].mean()
        adni_post = train_h[tr["site"].values=='ADNI'].mean()
        tpmic_post = train_h[tr["site"].values=='TPMIC'].mean()
        print(f"  Pre:  ADNI={adni_pre:.4f}, TPMIC={tpmic_pre:.4f}  diff={abs(adni_pre-tpmic_pre):.4f}")
        print(f"  Post: ADNI={adni_post:.4f}, TPMIC={tpmic_post:.4f}  diff={abs(adni_post-tpmic_post):.4f}")
        print(f"  Saved to {out_dir}")


def variant_no_label():
    """選項 3: 全資料 + 不帶 label 的純 site removal。"""
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    print(f"\n{'='*60}")
    print(f"No-label ComBat (純 site removal)")
    print('='*60)

    out_dir = BASE_DIR / "fmri_combat_v2_nolabel"
    out_dir.mkdir(exist_ok=True)

    tr_mats = [load_matrix(p) for p in df_train["matrix_path"]]
    te_mats = [load_matrix(p) for p in df_test ["matrix_path"]]
    tr_data  = np.array([matrix_to_tri(m) for m in tr_mats])
    te_data  = np.array([matrix_to_tri(m) for m in te_mats])
    tr_diags = np.array([np.diag(m) for m in tr_mats])
    te_diags = np.array([np.diag(m) for m in te_mats])

    tr_sites = df_train["subject_id"].apply(detect_site).values
    te_sites = df_test ["subject_id"].apply(detect_site).values

    train_h, test_h = run_combat(
        tr_data, tr_sites, df_train["label"].values,
        te_data, te_sites, df_test["label"].values,
        with_label=False, tag="no-label",
    )

    for i, sid in enumerate(df_train["subject_id"]):
        mat = tri_to_matrix(train_h[i], tr_diags[i]).astype(np.float32)
        np.save(out_dir / f"{sid}_combat.npy", mat)
    for i, sid in enumerate(df_test["subject_id"]):
        mat = tri_to_matrix(test_h[i], te_diags[i]).astype(np.float32)
        np.save(out_dir / f"{sid}_combat.npy", mat)

    adni_pre = tr_data[tr_sites=='ADNI'].mean()
    tpmic_pre = tr_data[tr_sites=='TPMIC'].mean()
    adni_post = train_h[tr_sites=='ADNI'].mean()
    tpmic_post = train_h[tr_sites=='TPMIC'].mean()
    print(f"  Pre:  ADNI={adni_pre:.4f}, TPMIC={tpmic_pre:.4f}  diff={abs(adni_pre-tpmic_pre):.4f}")
    print(f"  Post: ADNI={adni_post:.4f}, TPMIC={tpmic_post:.4f}  diff={abs(adni_post-tpmic_post):.4f}")
    print(f"  Saved to {out_dir}")


if __name__ == "__main__":
    variant_task_specific()
    variant_no_label()
