"""
brainiac_extractor.py
=====================
使用 vit_mci.ckpt (BrainIAC ViT-B，在 OASIS MCI 任務上 fine-tune) 作為凍結的特徵提取器。

輸出: (Batch, 768) 的潛在特徵向量，透過 Global Average Pooling 於所有 patch tokens 得到。

使用方式:
    python brainiac_extractor.py         # 執行自我測試，應印出 Output shape: (1, 768)
"""

import os
import torch
import torch.nn as nn

CKPT_PATH = (
    "/home/wei-chi/Alzheimers_Project/external_data/scripts/models/checkpoints/"
    "finetune_tpmic_full/vit_mci.ckpt"
)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  資料預處理 Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def get_preprocessing_transforms(already_preprocessed: bool = True):
    """
    回傳 MONAI Compose 預處理流程。

    論文規定的完整流程 (raw T1 sMRI):
      Step 1  N4 Bias Field Correction     (消除低頻強度不均勻)
      Step 2  Resample → 1×1×1 mm³         (線性插值)
      Step 3  Rigid registration → MNI     (★ 外部執行，見下方 bash 說明)
      Step 4  Skull-stripping              (★ 外部執行，見下方 bash 說明)
      Step 5  CropOrPad → (96, 96, 96)
      Step 6  Z-score 強度標準化
      Step 7  輸出 float32 Tensor (1, 96, 96, 96)

    本專案資料已完成 MNI 對齊與 Skull-strip，
    設 already_preprocessed=True 可跳過 Step 1-4，僅執行 Step 5-7。

    ── 外部前處理 bash 範例 ──────────────────────────────────────────────────
    # Step 3: ANTs 剛性配準至 MNI
    # antsRegistrationSyNQuick.sh -d 3 -t r \\
    #     -f MNI152_T1_1mm_brain.nii.gz \\
    #     -m input_brain.nii.gz \\
    #     -o output_mni_
    #
    # Step 4: HD-BET 去頭骨
    # hd-bet -i input.nii.gz -o output_brain.nii.gz
    ──────────────────────────────────────────────────────────────────────────
    """
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst, Orientation,
        Spacing, CropForeground, NormalizeIntensity,
        ResizeWithPadOrCrop, EnsureType,
    )

    TARGET = (96, 96, 96)

    if already_preprocessed:
        # 本專案使用此路徑：資料已 MNI 對齊、去頭骨
        return Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ResizeWithPadOrCrop(spatial_size=TARGET, mode="constant"),
            NormalizeIntensity(nonzero=True, channel_wise=True),   # Z-score
            EnsureType(data_type="tensor", dtype=torch.float32),
        ])
    else:
        # 完整原始影像前處理流程
        return Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),

            # Step 1: N4 Bias Field Correction (需 SimpleITK)
            # 若環境有 SimpleITK，可取消下方註解:
            # N4BiasFieldCorrection(),

            # Step 2: Resample to 1×1×1 mm³
            Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),

            # Step 3 & 4: Registration + Skull-strip → 外部執行後再 Load
            # 請先在 bash 執行 ANTs + HD-BET，再以已處理的影像路徑傳入

            # Step 5: Crop foreground + Pad/Crop to (96,96,96)
            CropForeground(),
            ResizeWithPadOrCrop(spatial_size=TARGET, mode="constant"),

            # Step 6: Z-score normalisation
            NormalizeIntensity(nonzero=True, channel_wise=True),

            # Step 7: 確保 float32 tensor
            EnsureType(data_type="tensor", dtype=torch.float32),
        ])


# ──────────────────────────────────────────────────────────────────────────────
# 2.  ViT Backbone Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class ViTFeatureExtractor(nn.Module):
    """
    以 MONAI ViT-B 為骨幹的凍結特徵提取器。
    forward() 輸出 (Batch, 768) 全域平均池化特徵。

    架構細節:
      - in_channels=1, img_size=(96,96,96), patch_size=(16,16,16)
      - hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12
      - classification=False → 不加分類頭，回傳 (B, 216, 768) patch 序列
      - qkv_bias=False → 與 vit_mci.ckpt 訓練時設定一致 (無 QKV bias)
      - Global Average Pooling (dim=1) → (B, 768)
    """

    def __init__(self):
        super().__init__()
        from monai.networks.nets import ViT
        self.vit = ViT(
            in_channels=1,
            img_size=(96, 96, 96),
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            classification=False,   # 不建構分類頭；回傳完整 patch 序列
            qkv_bias=False,         # checkpoint 無 QKV bias，須設 False 對齊
            save_attn=True,   # R-Fix-4: 保留 attention 給 interpretability 用
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 96, 96, 96) float32 tensor
        Returns:
            (B, 768) 特徵向量
        """
        seq_out, _ = self.vit(x)        # seq_out: (B, 216, 768)
        return seq_out.mean(dim=1)       # Global Avg Pool → (B, 768)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Checkpoint 載入
# ──────────────────────────────────────────────────────────────────────────────
def load_vitmci_extractor(
    ckpt_path: str,
    device: torch.device,
    verbose: bool = True,
) -> ViTFeatureExtractor:
    """
    將 vit_mci.ckpt 載入 ViTFeatureExtractor (凍結 eval 模式)。

    處理策略:
      1. 解包 PyTorch Lightning 的 state_dict
      2. 過濾分類頭 key (含 'classifier', 'fc', 'head', 'cls_head')
      3. 辨識並移除最常見的 backbone prefix
         優先順序: 'backbone.backbone.' > 'model.backbone.backbone.' > ...
      4. strict=False 載入，印出 Missing / Unexpected key 摘要

    vit_mci.ckpt 的 key 結構 (共 278 個):
      backbone.backbone.{patch_embedding, blocks.0-11, norm}  → 137 個 (主要 backbone)
      model.backbone.backbone.{blocks.0-11, norm}             → 137 個 (重複集，跳過)
      model.classifier.fc.{weight, bias}                      →   2 個 (過濾掉)
      classifier.fc.{weight, bias}                            →   2 個 (過濾掉)

    MONAI ViT 共 221 個 key；137 個與 checkpoint 完全吻合。
    剩餘 84 個 key 為 MONAI 1.5.2 新增的 cross_attn 層，
    但 with_cross_attention=False，forward pass 完全不使用，不影響特徵輸出。
    """
    FILTER_KEYWORDS = {"classifier", "fc", "head", "cls_head"}
    STRIP_PREFIXES = [
        "model.backbone.backbone.",
        "backbone.backbone.",
        "model.backbone.",
        "backbone.",
    ]

    if verbose:
        print(f"[INFO] 載入 checkpoint: {ckpt_path}")

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = raw_ckpt.get("state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt

    if verbose:
        print(f"[INFO] Checkpoint 原始 key 數量: {len(raw_sd)}")

    # ── 過濾分類頭 ────────────────────────────────────────────────────────────
    filtered_sd = {
        k: v for k, v in raw_sd.items()
        if not any(kw in k.lower() for kw in FILTER_KEYWORDS)
    }
    n_removed = len(raw_sd) - len(filtered_sd)
    if verbose:
        print(f"[INFO] 過濾分類頭 key: 移除 {n_removed} 個")

    # ── 找出最佳 prefix 並移除 ────────────────────────────────────────────────
    best_prefix, best_count = "", 0
    for prefix in STRIP_PREFIXES:
        cnt = sum(1 for k in filtered_sd if k.startswith(prefix))
        if cnt > best_count:
            best_count, best_prefix = cnt, prefix

    if best_prefix:
        if verbose:
            print(f"[INFO] 移除 prefix: '{best_prefix}' ({best_count} 個 key 吻合)")
        cleaned_sd = {
            k[len(best_prefix):]: v
            for k, v in filtered_sd.items()
            if k.startswith(best_prefix)
        }
    else:
        if verbose:
            print("[WARN] 未找到共同 prefix，保留原始 key")
        cleaned_sd = filtered_sd

    if verbose:
        print(f"[INFO] 清理後 key 數量: {len(cleaned_sd)}")

    # ── 建立模型並載入權重 ────────────────────────────────────────────────────
    model = ViTFeatureExtractor()
    result = model.vit.load_state_dict(cleaned_sd, strict=False)

    if verbose:
        n_miss = len(result.missing_keys)
        n_unex = len(result.unexpected_keys)
        miss_preview = result.missing_keys[:3]
        unex_preview = result.unexpected_keys[:3]
        print(f"[INFO] Missing keys  ({n_miss}): "
              f"{miss_preview}{'...' if n_miss > 3 else ''}")
        print(f"[INFO] Unexpected keys ({n_unex}): "
              f"{unex_preview}{'...' if n_unex > 3 else ''}")
        if n_miss == 84 and n_unex == 0:
            print("[INFO] 對齊狀態: 正常 ✓")
            print("       (84 個 missing = MONAI 1.5.2 cross_attn 層，")
            print("        with_cross_attention=False，forward 完全不使用)")
        elif n_unex > 0:
            print("[WARN] 有 Unexpected keys，請手動確認 key 對齊情況")

    # ── 凍結 + eval ──────────────────────────────────────────────────────────
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    if verbose:
        print(f"[INFO] 模型已凍結並移至 {device}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4.  特徵提取函數
# ──────────────────────────────────────────────────────────────────────────────
def extract_features(
    model: ViTFeatureExtractor,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    從 MRI tensor 提取 768-d 特徵。

    Args:
        model:        已凍結的 ViTFeatureExtractor
        image_tensor: (B, 1, 96, 96, 96) float32 tensor
        device:       目標裝置

    Returns:
        features: (B, 768) float32 tensor，已移至 CPU
    """
    assert image_tensor.ndim == 5 and image_tensor.shape[1] == 1, (
        f"輸入維度應為 (B, 1, 96, 96, 96)，實際為 {tuple(image_tensor.shape)}"
    )
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = model(image_tensor)   # (B, 768)
    return features.cpu()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  BrainIAC Attention Rollout (R-Fix-4)
# ──────────────────────────────────────────────────────────────────────────────

def get_attention_rollout(
    model: "ViTFeatureExtractor",
    image_tensor: torch.Tensor,
    device: torch.device,
    t1_path: str = None,
    out_path: str = None,
    aal_atlas_path: str = None,
) -> tuple:
    """
    Run BrainIAC ViT forward pass and compute Attention Rollout
    (Abnar & Zuidema, ACL 2020) over all 12 transformer layers.

    Args:
        model:         Loaded ViTFeatureExtractor (save_attn=True)
        image_tensor:  (1, 1, 96, 96, 96) float32
        device:        torch.device
        t1_path:       Path to patient T1 NIfTI (for affine registration)
        out_path:      If given, saves saliency as NIfTI here
        aal_atlas_path: If given, computes per-ROI mean saliency

    Returns:
        saliency_96:  np.ndarray (96,96,96), normalized [0,1]
        roi_scores:   list of {"name": str, "saliency": float}, top-10 by AAL116
    """
    import numpy as np
    import torch.nn.functional as F

    image_tensor = image_tensor.to(device)

    # Forward pass — attention saved into block.attn.att_mat
    with torch.no_grad():
        _ = model.vit(image_tensor)  # triggers save_attn hooks

    # Collect attention from all 12 layers: (B, heads, 216, 216)
    attn_maps = []
    for block in model.vit.blocks:
        att = getattr(block.attn, "att_mat", None)
        if att is None:
            continue
        att = att.detach().float().cpu()        # (B, 12, 216, 216)
        att = att.mean(dim=1)                   # avg heads → (B, 216, 216)
        attn_maps.append(att)

    if not attn_maps:
        # Fallback: uniform map if attention not accessible
        saliency_96 = np.ones((96, 96, 96), dtype=np.float32) * 0.5
        return saliency_96, []

    # Attention Rollout: R = A_L × A_{L-1} × ... × A_1
    # Each A_i = 0.5 * attn_i + 0.5 * I  (residual weighting)
    n_patches = attn_maps[0].shape[-1]  # 216
    rollout = torch.eye(n_patches).unsqueeze(0)  # (1, 216, 216)
    for att in attn_maps:
        att_r = 0.5 * att + 0.5 * torch.eye(n_patches).unsqueeze(0)
        row_sums = att_r.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        att_r = att_r / row_sums
        rollout = torch.bmm(att_r, rollout)

    # Per-patch importance: average incoming attention across all queries
    importance = rollout[0].mean(dim=0).numpy()    # (216,)

    # Reshape 216 → (6, 6, 6), upsample → (96, 96, 96)
    sal_3d = torch.tensor(importance.reshape(6, 6, 6)).unsqueeze(0).unsqueeze(0).float()
    sal_96 = F.interpolate(sal_3d, size=(96, 96, 96), mode="trilinear",
                           align_corners=False).squeeze().numpy()

    # Normalize to [0, 1]
    vmin, vmax = sal_96.min(), sal_96.max()
    if vmax > vmin:
        sal_96 = (sal_96 - vmin) / (vmax - vmin)
    sal_96 = sal_96.astype(np.float32)

    # ── Save as NIfTI ────────────────────────────────────────────────────────
    roi_scores = []
    if out_path is not None:
        import nibabel as nib
        from scipy.ndimage import zoom as ndimage_zoom

        # Get affine from patient T1
        if t1_path and os.path.exists(t1_path):
            ref_nii = nib.load(t1_path)
            affine   = ref_nii.affine
            ref_shape = tuple(ref_nii.shape[:3])
        else:
            affine    = np.diag([1.0, 1.0, 1.0, 1.0])
            ref_shape = (96, 96, 96)

        # Resample saliency to match T1 shape if needed
        if ref_shape != (96, 96, 96):
            scale  = [ref_shape[i] / 96.0 for i in range(3)]
            sal_out = ndimage_zoom(sal_96, scale, order=1).astype(np.float32)
        else:
            sal_out = sal_96

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nib.save(nib.Nifti1Image(sal_out, affine), out_path)

        # ── Per-ROI scores via AAL116 ────────────────────────────────────────
        if aal_atlas_path and os.path.exists(aal_atlas_path):
            try:
                atlas_nii  = nib.load(aal_atlas_path)
                atlas_data = np.asarray(atlas_nii.get_fdata(), dtype=np.int32)

                # Resample atlas to saliency output shape if needed
                if atlas_data.shape != sal_out.shape:
                    scale_a = [sal_out.shape[i] / atlas_data.shape[i] for i in range(3)]
                    atlas_data = ndimage_zoom(atlas_data, scale_a, order=0).astype(np.int32)

                labels = np.unique(atlas_data)
                labels = labels[labels > 0]  # skip background (0)

                # Load AAL label names if available
                label_txt = aal_atlas_path.replace(".nii.gz", ".txt").replace(".nii", ".txt")
                label_map = {}
                if os.path.exists(label_txt):
                    for line in open(label_txt):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                label_map[int(parts[0])] = parts[1]
                            except ValueError:
                                pass

                for lbl in labels:
                    mask = atlas_data == lbl
                    if mask.sum() == 0:
                        continue
                    mean_sal = float(sal_out[mask].mean())
                    name = label_map.get(int(lbl), f"ROI_{lbl}")
                    roi_scores.append({"name": name, "saliency": mean_sal})

                roi_scores.sort(key=lambda x: x["saliency"], reverse=True)
                roi_scores = roi_scores[:10]
            except Exception as e:
                print(f"[WARN] AAL ROI scoring failed: {e}")

    return sal_96, roi_scores


# ──────────────────────────────────────────────────────────────────────────────
# 6.  自我測試區塊
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 62)
    print("  BrainIAC vit_mci 特徵提取器 — 自我測試")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")
    print()

    # ── 載入模型 ─────────────────────────────────────────────────────────────
    model = load_vitmci_extractor(CKPT_PATH, device, verbose=True)
    print()

    # ── 確認凍結狀態 ─────────────────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"可訓練參數: {trainable:,} / {total:,}  (應為 0 / {total:,})")
    assert trainable == 0, "模型未完全凍結！"
    assert not model.training, "模型未處於 eval 模式！"
    print("凍結狀態: ✓  eval 模式: ✓")
    print()

    # ── Dummy tensor 測試 ────────────────────────────────────────────────────
    dummy = torch.randn(1, 1, 96, 96, 96, dtype=torch.float32)
    print(f"測試輸入 shape: {tuple(dummy.shape)}")
    feats = extract_features(model, dummy, device)
    print(f"Output shape:   {tuple(feats.shape)}")

    assert feats.shape == (1, 768), f"期望 (1, 768)，實際 {feats.shape}"
    assert not torch.isnan(feats).any(), "輸出包含 NaN！"
    assert not torch.isinf(feats).any(), "輸出包含 Inf！"
    print()
    print("  [PASS] 自我測試通過 — 輸出維度 (1, 768) 正確")
    print("=" * 62)
    print()
