"""
harmonize_fmri_combat.py
========================
對 fMRI connectivity matrix 做 ComBat 諧波化（site harmonization）。

問題：訓練 PCAG-ComBat 時，sMRI 有 ComBat，但 fMRI 沒有。
這導致：
  - 模型可從 fMRI connectivity 看出 site (ADNI vs TPMIC)
  - 訓練集 site-class 失衡 (ADNI 多 NC, TPMIC 多 disease) → 模型走捷徑
  - Within-site AUC (尤其 NC_vs_MCI) 大幅下降

流程：
  1. 載入所有 train + test FC matrix → arctanh (Fisher z transform)
  2. 取上三角 6670 維（不含對角）
  3. ComBat fit on train only，apply 到 train+test
  4. tanh 還原到 [-1, 1]
  5. 重組對稱矩陣（對角為 0），存到 fmri_combat_v2/{sid}_combat.npy

執行：
  conda activate AD
  python harmonize_fmri_combat.py
"""
import json, os, re, time
import numpy as np
import pandas as pd
from pathlib import Path
from neuroCombat import neuroCombat

BASE_DIR  = Path(__file__).parent
RES_DIR   = BASE_DIR / "results"
OUT_DIR   = BASE_DIR / "fmri_combat_v2"
TRAIN_CSV = BASE_DIR / "pcag_train_aligned_v2.csv"
TEST_CSV  = BASE_DIR / "pcag_test_aligned_v2.csv"

N_ROIS = 116
N_TRI = N_ROIS * (N_ROIS - 1) // 2   # 6670


def detect_site(sid: str) -> str:
    return 'ADNI' if re.search(r'\d{3}_S_\d{4}', str(sid)) else 'TPMIC'


def load_matrix(path: str) -> np.ndarray:
    """Load a 116x116 FC matrix as-is (檢驗：max>1 表示矩陣已是 z-transformed 或非 r)."""
    mat = np.load(path)
    if mat.ndim == 3:
        mat = mat[0]
    if mat.shape != (N_ROIS, N_ROIS):
        raise ValueError(f"Bad shape {mat.shape}: {path}")
    # ensure symmetric, save original diagonal value separately
    mat = (mat + mat.T) / 2
    return mat.astype(np.float32)


def matrix_to_tri(mat: np.ndarray) -> np.ndarray:
    """Upper triangle off-diagonal → flat vector of 6670."""
    iu = np.triu_indices(N_ROIS, k=1)
    return mat[iu]


def tri_to_matrix(tri: np.ndarray, diag_vals: np.ndarray) -> np.ndarray:
    """Reconstruct symmetric matrix from upper-tri vector, with original diagonal."""
    iu = np.triu_indices(N_ROIS, k=1)
    mat = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
    mat[iu] = tri
    mat = mat + mat.T
    np.fill_diagonal(mat, diag_vals)
    return mat


def main():
    OUT_DIR.mkdir(exist_ok=True)

    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    print(f"Train: {len(df_train)}  Test: {len(df_test)}")

    # 載入所有矩陣 → upper triangle（同時記錄對角線）
    print("\nLoading matrices...")
    all_data = []         # rows = subjects, cols = 6670 features
    all_diags = []        # 保存每位 patient 的對角值
    all_sites = []
    all_labels = []
    all_sids = []
    splits = []

    t0 = time.time()
    for tag, df in [("train", df_train), ("test", df_test)]:
        for _, row in df.iterrows():
            mat = load_matrix(row["matrix_path"])
            all_data.append(matrix_to_tri(mat))
            all_diags.append(np.diag(mat).copy())
            all_sites.append(detect_site(row["subject_id"]))
            all_labels.append(int(row["label"]))
            all_sids.append(row["subject_id"])
            splits.append(tag)
    all_diags = np.array(all_diags)
    all_data = np.array(all_data)
    all_sites = np.array(all_sites)
    all_labels = np.array(all_labels)
    all_sids = np.array(all_sids)
    splits = np.array(splits)

    print(f"  loaded {len(all_data)} matrices in {time.time()-t0:.1f}s")
    print(f"  data shape: {all_data.shape}")
    print(f"  sites: ADNI={(all_sites=='ADNI').sum()}, TPMIC={(all_sites=='TPMIC').sum()}")

    # ComBat：fit on train only, apply on all
    # neuroCombat 需要 (features, samples) shape
    print("\nFitting ComBat on TRAIN data only...")
    train_mask = splits == 'train'

    # 為避免 leakage：用 train 估參，再 apply 到 train+test
    # neuroCombat 一次處理所有 samples，所以要用 trick：
    #   把 test sample 標 site，但訓練時 fit 只看 train 的 batch effects
    #   實際上 neuroCombat 沒有 fit/apply API；正確做法是分開 fit + apply。
    # 用 neuroCombatFromTraining 來做兩階段
    from neuroCombat import neuroCombatFromTraining

    # 階段 1：fit on train
    train_data = all_data[train_mask].T    # (6670, n_train)
    train_covars = pd.DataFrame({
        "batch": all_sites[train_mask],
        "label": all_labels[train_mask],
    })
    t0 = time.time()
    fit_result = neuroCombat(
        dat=train_data,
        covars=train_covars,
        batch_col="batch",
        categorical_cols=["label"],
    )
    print(f"  ComBat fit done in {time.time()-t0:.1f}s")

    # 取出 estimate
    estimates = fit_result["estimates"]
    train_harmonized = fit_result["data"]   # (6670, n_train)

    # 階段 2：apply 到 test
    test_data = all_data[~train_mask].T    # (6670, n_test)
    test_covars = pd.DataFrame({
        "batch": all_sites[~train_mask],
        "label": all_labels[~train_mask],
    })
    t0 = time.time()
    apply_result = neuroCombatFromTraining(
        dat=test_data,
        batch=test_covars["batch"].values,
        estimates=estimates,
    )
    test_harmonized = apply_result["data"]
    print(f"  ComBat apply (test) done in {time.time()-t0:.1f}s")

    # 拼接回原順序
    out_data = np.zeros_like(all_data)
    train_idx = np.where(train_mask)[0]
    test_idx  = np.where(~train_mask)[0]
    for k, idx in enumerate(train_idx):
        out_data[idx] = train_harmonized[:, k]
    for k, idx in enumerate(test_idx):
        out_data[idx] = test_harmonized[:, k]

    # 重組矩陣（保留原對角值）
    print("\nReconstructing matrices...")
    saved = 0
    for i, sid in enumerate(all_sids):
        mat = tri_to_matrix(out_data[i], all_diags[i]).astype(np.float32)
        out_path = OUT_DIR / f"{sid}_combat.npy"
        np.save(out_path, mat)
        saved += 1

    print(f"  Saved {saved} matrices to {OUT_DIR}")

    # 存 ComBat 參數（給未來 inference 用）— deep-convert ndarray → list
    def _to_jsonable(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_jsonable(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o
    params_path = RES_DIR / "fmri_combat_v2_params.json"
    params_path.parent.mkdir(exist_ok=True)
    with open(params_path, "w") as f:
        json.dump({"estimates": _to_jsonable(estimates),
                   "n_features": N_TRI,
                   "n_train": int(train_mask.sum()),
                   "n_test":  int((~train_mask).sum())}, f, indent=2)
    print(f"  ComBat params saved to {params_path}")

    # 健全性檢查：site difference 應減少
    print("\n=== Sanity check: per-site mean of harmonized fMRI features ===")
    for site in ["ADNI", "TPMIC"]:
        m = all_data[(all_sites == site) & train_mask].mean()
        m_h = out_data[(all_sites == site) & train_mask].mean()
        print(f"  {site} (train): raw mean = {m:.4f}, harmonized = {m_h:.4f}")

if __name__ == "__main__":
    main()
