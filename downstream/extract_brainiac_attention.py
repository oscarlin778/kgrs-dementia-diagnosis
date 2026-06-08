"""
extract_brainiac_attention.py
=============================
對測試集每位病患抓取 BrainIAC ViT 的 attention：
  1. ViT 12 層 self-attention，每層多 head
  2. 用 attention rollout (Abnar & Zuidema 2020) 累積各層
  3. 將 216 patches (6×6×6) 的 attention 投影到 AAL116 ROI

Patch 對應到空間位置：
  - input 96×96×96, patch_size=16, 共 6×6×6=216 patches
  - patch i 對應空間立方 [zi*16:(zi+1)*16, yi*16:(yi+1)*16, xi*16:(xi+1)*16]
  - AAL atlas 已 ResizeWithPadOrCrop 到 (96,96,96)
  - 對每個 ROI mask 計算與每個 patch 的 overlap voxel count
  - ROI attention = Σ_patch (attention_patch × overlap_count) / total_voxel_count_of_ROI

輸出：
  results/brainiac_roi_v2.npz
    keys: subject_ids, roi_attention (n, 116), top_rois (n, 10), top_scores (n, 10)
"""
import os, time, pickle
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from pathlib import Path
from monai.transforms import ResizeWithPadOrCrop

from inference_pipeline_v2 import AAL116_NAMES, NETWORK_MAP, net_list, DEVICE

BASE_DIR = Path(__file__).parent
RES_DIR  = BASE_DIR / "results"
CSV_PATH = BASE_DIR / "pcag_test_aligned_v2.csv"
AAL_PATH = Path("/home/wei-chi/Alzheimers_Project/external_data/datasets/nilearn_data/aal_SPM12/aal/atlas/AAL.nii")

PATCH = 16
GRID  = 96 // PATCH  # 6
N_PATCHES = GRID ** 3  # 216


def load_and_resample_aal():
    atlas_img = nib.load(str(AAL_PATH))
    a = torch.from_numpy(atlas_img.get_fdata().astype(np.int64)).unsqueeze(0).float()
    resizer = ResizeWithPadOrCrop(spatial_size=(96, 96, 96), mode='constant')
    a_resized = resizer(a)
    return a_resized.numpy()[0].round().astype(np.int64)


def build_patch_roi_overlap(aal_96, label_to_idx):
    """
    回傳 (216, 116) overlap matrix。
    overlap[p, r] = patch p 含有 ROI r 的 voxel 數量
    """
    overlap = np.zeros((N_PATCHES, 116), dtype=np.float32)
    for pi in range(N_PATCHES):
        # patch index → (z, y, x) in 6×6×6 grid
        z = pi // (GRID * GRID)
        y = (pi % (GRID * GRID)) // GRID
        x = pi % GRID
        # voxel slice
        patch_vol = aal_96[z*PATCH:(z+1)*PATCH,
                           y*PATCH:(y+1)*PATCH,
                           x*PATCH:(x+1)*PATCH]
        # count voxels per label
        unique, counts = np.unique(patch_vol, return_counts=True)
        for lbl, cnt in zip(unique, counts):
            if lbl in label_to_idx:
                overlap[pi, label_to_idx[int(lbl)]] += cnt
    return overlap


def load_brainiac_model():
    """載入 BrainIAC + 開啟 save_attn"""
    from brainiac_extractor import load_vitmci_extractor, CKPT_PATH
    model = load_vitmci_extractor(CKPT_PATH, DEVICE, verbose=False)
    return model


def compute_attention_rollout(attention_layers):
    """
    attention_layers: list of (B=1, num_heads, N+1, N+1) tensors
      N+1 = 216 patches + 1 cls token (or no cls if classification=False)
    Returns: (N, N) rollout matrix
    """
    rolled = None
    for attn in attention_layers:
        # average over heads → (B, N, N)
        a = attn.mean(dim=1)[0].cpu().numpy()
        # 加上 identity 表示 skip
        a = a + np.eye(a.shape[0])
        # row normalize
        a = a / (a.sum(axis=1, keepdims=True) + 1e-8)
        if rolled is None:
            rolled = a
        else:
            rolled = a @ rolled
    return rolled


def main():
    print("Loading BrainIAC ViT (save_attn=True)...")
    model = load_brainiac_model()

    print("Loading AAL atlas + computing patch-ROI overlap...")
    aal_96 = load_and_resample_aal()
    unique_labels = sorted(set(aal_96.flatten()) - {0})
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels[:116])}
    overlap = build_patch_roi_overlap(aal_96, label_to_idx)
    print(f"  Overlap matrix shape: {overlap.shape}")
    print(f"  Patches with any ROI: {(overlap.sum(axis=1) > 0).sum()}/216")

    # 載入 T1 lookup (已在 inference_pipeline_v2 module level init)
    from inference_pipeline_v2 import find_t1_path
    from brainiac_extractor import get_preprocessing_transforms
    tfm = get_preprocessing_transforms(already_preprocessed=True)

    df = pd.read_csv(CSV_PATH)
    print(f"Test set: {len(df)} patients")

    n = len(df)
    out = {
        "subject_ids":     df["subject_id"].values,
        "labels":          df["label"].values,
        "roi_attention":   np.zeros((n, 116), dtype=np.float32),
        "top_rois":        np.zeros((n, 10), dtype=np.int32),
        "top_scores":      np.zeros((n, 10), dtype=np.float32),
        "missing":         np.zeros(n, dtype=bool),
    }

    t0 = time.time()
    for i, row in df.iterrows():
        sid = row["subject_id"]
        if i % 10 == 0:
            print(f"[{i+1}/{n}] {sid}  ({time.time()-t0:.0f}s)")
        t1_path = find_t1_path(sid)
        if t1_path is None:
            print(f"  [WARN] no T1 for {sid}")
            out["missing"][i] = True
            continue
        try:
            vol = tfm(t1_path).unsqueeze(0).to(DEVICE)  # (1, 1, 96, 96, 96)
            with torch.no_grad():
                _ = model(vol)  # forward
            # MONAI ViT save_attn=True 把 attention 存在每個 transformer block 的 .att_mat
            attentions = []
            for block in model.vit.blocks:
                if hasattr(block, "attn") and hasattr(block.attn, "att_mat"):
                    attentions.append(block.attn.att_mat)
                elif hasattr(block, "att_mat"):
                    attentions.append(block.att_mat)
            if not attentions:
                # fallback: try save_attn=True on different attr
                print(f"  [WARN] {sid}: no attention found in blocks")
                out["missing"][i] = True
                continue

            rollout = compute_attention_rollout(attentions)  # (N, N)
            # cls token (if exists) 在 dim 0；MONAI ViT classification=False 無 cls token
            # rollout shape should be (216, 216) if no cls
            if rollout.shape[0] != 216:
                # 假設 cls token 在 0，patches 在 1:
                rollout = rollout[1:, 1:]
            # patch importance：對所有 source 取平均（column sum 也可）
            patch_imp = rollout.sum(axis=0)  # (216,)

            # 投影到 ROI: roi_imp[r] = Σ_patch patch_imp[p] * overlap[p, r] / Σ_p overlap[:, r]
            roi_imp = np.zeros(116, dtype=np.float32)
            for r in range(116):
                total_vox = overlap[:, r].sum()
                if total_vox > 0:
                    roi_imp[r] = (patch_imp * overlap[:, r]).sum() / total_vox

            # normalize
            roi_imp = roi_imp / (roi_imp.sum() + 1e-8)
            top_idx = np.argsort(roi_imp)[-10:][::-1]
            out["roi_attention"][i] = roi_imp
            out["top_rois"][i]      = top_idx
            out["top_scores"][i]    = roi_imp[top_idx]
        except Exception as e:
            print(f"  [WARN] {sid} failed: {e}")
            out["missing"][i] = True

    out_path = RES_DIR / "brainiac_roi_v2.npz"
    np.savez(out_path, **out)
    print(f"\n[SAVED] {out_path}")
    print(f"  missing: {out['missing'].sum()}/{n}")

    # Sanity check
    print("\n=== Sample patient ===")
    valid_idx = np.where(~out["missing"])[0]
    if len(valid_idx) > 0:
        sample_i = valid_idx[0]
        sid = out["subject_ids"][sample_i]
        print(f"Patient {sid} (label={out['labels'][sample_i]})")
        for r, s in zip(out["top_rois"][sample_i][:5], out["top_scores"][sample_i][:5]):
            net = next((n_ for n_, idxs in NETWORK_MAP.items() if r in idxs), "Other")
            print(f"  {AAL116_NAMES[r]:30s}  ({net})  attn={s:.4f}")


if __name__ == "__main__":
    main()
