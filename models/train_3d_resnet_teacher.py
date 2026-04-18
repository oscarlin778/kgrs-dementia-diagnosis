import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    NormalizeIntensityd, CropForegroundd,
    Resized, RandRotate90d, RandFlipd,
    RandAffined, RandGaussianNoised, RandGaussianSmoothd,
    RandScaleIntensityd, RandShiftIntensityd,
    RandBiasFieldd, RandCoarseDropoutd,
    SpatialCropd,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet50

import re
import warnings
warnings.filterwarnings('ignore')

def get_subject_id(path_str):
    """從任意路徑萃取乾淨的 subject ID（與 fMRI 側對齊）"""
    basename = os.path.basename(str(path_str))
    clean = re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy|_T1_MNI\.nii\.gz|_T1\.nii\.gz|\.nii\.gz)$', '', basename)
    clean = re.sub(r'^(sub-|sub_|old_dswau)', '', clean)
    return clean.strip()

# ===============================================================
# 設定區
# ===============================================================
DATA_DIR_TPMIC = "/home/wei-chi/Model/sMRI_data_MultiModal_Aligned_MNI"
DATA_DIR_ADNI  = "/home/wei-chi/Data/ADNI_sMRI_Aligned_MNI"

# ── Checkpoint 儲存目錄（cross-model KD 用）──
MODEL_SAVE_DIR = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Med3D 預訓練權重路徑 (None = 從頭訓練)
PRETRAIN_PATH = "/home/wei-chi/pretrain/resnet_50_23dataset.pth"

# ── Progressive Resizing ──
SPATIAL_SIZE_S1 = (64, 64, 64)   # Stage 1：小解析度，學整體結構
SPATIAL_SIZE_S2 = (96, 96, 96)   # Stage 2：大解析度，學細節

EPOCHS_S1       = 60             # Stage 1 epochs
EPOCHS_S2       = 90             # Stage 2 epochs (接續 Stage 1 weights)
BATCH_SIZE      = 4
ACCUM_STEPS     = 4              # 等效 batch=16
LR_S1           = 3e-4
LR_S2           = 1e-4           # Stage 2 用更小 LR fine-tune
WEIGHT_DECAY    = 1e-3
LABEL_SMOOTHING = 0.1
DROPOUT         = 0.3
N_FOLDS         = 5
SEED            = 42
PATIENCE        = 30

# ── ROI 裁切設定 (MNI 96³ 空間下的海馬迴中心座標) ──
# 海馬迴在 MNI 空間約位於左右半腦中央偏後下方
# 若你的影像是 96x96x96，可依實際情況調整
ROI_CENTER = (48, 38, 30)        # (x, y, z) 雙側海馬迴中心
ROI_SIZE_S1 = (52, 52, 40)       # Stage 1 ROI patch size
ROI_SIZE_S2 = (64, 64, 52)       # Stage 2 ROI patch size (更大視野)

# ===============================================================
# 1. 資料抓取
# ===============================================================
def get_3d_data_dicts(task_pair):
    """
    抓取指定類別的 T1 影像。
    支援 sMCI/pMCI 分層：
      - 若 task_pair 含 'sMCI' 或 'pMCI'，直接搜尋對應子資料夾
      - 否則走原本的 NC/MCI/AD 邏輯

    注意：MCI vs AD task 排除 ADNI MCI。
    原因：ADNI-3 MCI 幾乎全為 sMCI（輕微穩定型），結構上接近 NC 而非 AD，
    混入後會使 MCI class 往 NC 方向偏移，導致 MCI vs AD 邊界模糊、AD recall 下降。
    ADNI NC 與 ADNI AD 不受影響。
    """
    class_a, class_b = task_pair
    is_mci_vs_ad = set(task_pair) == {'MCI', 'AD'}
    data_dicts = []

    for label, class_name in enumerate([class_a, class_b]):
        # 來源 1: TPMIC（全部使用）
        folder_tpmic = os.path.join(DATA_DIR_TPMIC, class_name)
        if os.path.exists(folder_tpmic):
            files = list(set(glob.glob(os.path.join(folder_tpmic, "*[Tt]1*.nii.gz"))))
            for fp in files:
                data_dicts.append({"image": fp, "label": label})

        # 來源 2: ADNI
        # MCI vs AD task 時，跳過 ADNI MCI（避免 sMCI 稀釋邊界）
        if is_mci_vs_ad and class_name == 'MCI':
            continue
        folder_adni = os.path.join(DATA_DIR_ADNI, class_name)
        if os.path.exists(folder_adni):
            files = list(set(glob.glob(
                os.path.join(folder_adni, "**", "T1", "*.nii.gz"), recursive=True)))
            for fp in files:
                data_dicts.append({"image": fp, "label": label})

    return data_dicts


# ===============================================================
# 2. Transform 工廠（支援 ROI 模式 + Progressive Resizing）
# ===============================================================
def get_transforms(spatial_size, use_roi=False, roi_center=None, roi_size=None):
    """
    use_roi=True  → 在 Resize 前先裁切到海馬迴 ROI
    use_roi=False → 全腦訓練（NC/AD 任務）
    """

    # ── 共用的強度 & 幾何增強 ──
    augmentations = [
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandAffined(
            keys=["image"], prob=0.8,
            rotate_range=(0.15, 0.15, 0.15),
            scale_range=(0.1, 0.1, 0.1),
            translate_range=(5, 5, 5),
            mode="bilinear", padding_mode="border"
        ),
        RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)
        ),
        RandBiasFieldd(keys=["image"], prob=0.3, coeff_range=(0.0, 0.1)),
        RandCoarseDropoutd(
            keys=["image"], prob=0.3,
            holes=6, spatial_size=(8, 8, 8), fill_value=0.0
        ),
    ]

    # ── ROI crop（若啟用）──
    roi_crop = []
    if use_roi and roi_center and roi_size:
        # 計算裁切起點 (center - size//2)
        start = tuple(max(0, roi_center[i] - roi_size[i] // 2) for i in range(3))
        end   = tuple(start[i] + roi_size[i] for i in range(3))
        roi_crop = [
            SpatialCropd(keys=["image"], roi_start=list(start), roi_end=list(end))
        ]

    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),  # 先統一到 96³
        *roi_crop,                                            # 再 ROI crop
        Resized(keys=["image"], spatial_size=spatial_size),  # 最後縮放到目標大小
        *augmentations,
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        *roi_crop,
        Resized(keys=["image"], spatial_size=spatial_size),
    ])

    return train_transforms, val_transforms


# ===============================================================
# 3. SE Block
# ===============================================================
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


# ===============================================================
# 4. 模型建構
# ===============================================================
def build_model(device, pretrain_path=None):
    model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)

    model.layer4 = nn.Sequential(
        model.layer4,
        SEBlock3D(2048, reduction=16)
    )

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=DROPOUT / 2),
        nn.Linear(512, 2)
    )

    if pretrain_path and os.path.exists(pretrain_path):
        print(f"    📥 載入預訓練權重: {pretrain_path}")
        checkpoint  = torch.load(pretrain_path, map_location='cpu')
        state_dict  = checkpoint.get('state_dict', checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"       missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print("    ℹ️  從頭訓練（未使用預訓練權重）")

    return model.to(device)


# ===============================================================
# 5. 單 Stage 訓練
# ===============================================================
def train_one_stage(
    model, train_files, val_files, device,
    spatial_size, epochs, lr,
    use_roi=False, roi_center=None, roi_size=None,
    stage_name="Stage"
):
    train_trans, val_trans = get_transforms(
        spatial_size, use_roi=use_roi,
        roi_center=roi_center, roi_size=roi_size
    )

    train_ds = Dataset(data=train_files, transform=train_trans)
    val_ds   = Dataset(data=val_files,   transform=val_trans)

    class_counts   = np.bincount([d["label"] for d in train_files])
    sample_weights = [1.0 / class_counts[d["label"]] for d in train_files]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1, num_workers=2)

    weights   = torch.tensor(1. / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc    = 0.0
    patience_counter = 0
    best_state      = None

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for step, batch_data in enumerate(train_loader):
            inputs  = batch_data["image"].to(device)
            targets = batch_data["label"].to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets) / ACCUM_STEPS
            loss.backward()
            epoch_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # ── Val ──
        model.eval()
        fold_true, fold_pred = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                inp = batch_data["image"].to(device)
                tgt = batch_data["label"].to(device)
                out = model(inp)
                fold_pred.append(out.argmax(dim=1).item())
                fold_true.append(tgt.item())

        fold_acc = accuracy_score(fold_true, fold_pred)
        if fold_acc > best_val_acc:
            best_val_acc     = fold_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            cur_lr   = scheduler.get_last_lr()[0]
            print(f"      [{stage_name}] Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | Val: {fold_acc*100:.1f}% | "
                  f"Best: {best_val_acc*100:.1f}% | LR: {cur_lr:.2e}")

        if patience_counter >= PATIENCE:
            print(f"      [{stage_name}] Early stop @ epoch {epoch+1} "
                  f"(best {best_val_acc*100:.1f}%)")
            break

    # 恢復最佳權重
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    return model, best_val_acc


# ===============================================================
# 6. 完整任務（含 Progressive Resizing）
# ===============================================================
def run_3d_task(task_pair, device):
    task_name = f"{task_pair[0]} vs {task_pair[1]}"

    # MCI 相關任務啟用 ROI 模式
    mci_tasks   = {"MCI", "sMCI", "pMCI"}
    use_roi     = bool(mci_tasks & set(task_pair))
    roi_tag     = " [ROI+ProgResize]" if use_roi else " [全腦+ProgResize]"

    print(f"\n{'='*65}")
    print(f"  任務: {task_name}{roi_tag}")
    print(f"{'='*65}")

    data_dicts = get_3d_data_dicts(task_pair)
    if len(data_dicts) < 10:
        print("  ❌ 找不到足夠的 T1 影像！")
        return 0, None, None

    labels_arr = np.array([d["label"] for d in data_dicts])
    unique, counts = np.unique(labels_arr, return_counts=True)
    for cls, cnt in zip(task_pair, counts):
        print(f"  {cls}: {cnt} 筆")
    print(f"  總計: {len(data_dicts)} 筆 | ROI模式: {use_roi}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_true, all_pred, all_prob = [], [], []
    all_prob_full   = []   # (N, 2) — full softmax distribution for KD
    all_image_paths = []   # T1 路徑，用於跨 modality 對齊

    # 追蹤全局最佳 fold 模型
    global_best_acc   = 0.0
    global_best_state = None
    global_best_info  = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_dicts, labels_arr)):
        print(f"\n  ▶ Fold {fold+1}/{N_FOLDS}")

        train_files = [data_dicts[i] for i in train_idx]
        val_files   = [data_dicts[i] for i in val_idx]

        # 建模
        model = build_model(device, PRETRAIN_PATH)

        # ── Stage 1: 小解析度，學整體腦部結構 ──
        print(f"    [Stage 1] {SPATIAL_SIZE_S1}  {EPOCHS_S1} epochs  LR={LR_S1:.0e}")
        model, s1_best = train_one_stage(
            model, train_files, val_files, device,
            spatial_size = SPATIAL_SIZE_S1,
            epochs       = EPOCHS_S1,
            lr           = LR_S1,
            use_roi      = use_roi,
            roi_center   = ROI_CENTER,
            roi_size     = ROI_SIZE_S1,
            stage_name   = "S1"
        )
        print(f"    [Stage 1] 完成，best val acc = {s1_best*100:.1f}%")

        # ── Stage 2: 大解析度 fine-tune ──
        print(f"    [Stage 2] {SPATIAL_SIZE_S2}  {EPOCHS_S2} epochs  LR={LR_S2:.0e}")
        model, s2_best = train_one_stage(
            model, train_files, val_files, device,
            spatial_size = SPATIAL_SIZE_S2,
            epochs       = EPOCHS_S2,
            lr           = LR_S2,
            use_roi      = use_roi,
            roi_center   = ROI_CENTER,
            roi_size     = ROI_SIZE_S2,
            stage_name   = "S2"
        )
        print(f"    [Stage 2] 完成，best val acc = {s2_best*100:.1f}%")

        # ── 更新全局最佳模型 ──
        if s2_best > global_best_acc:
            global_best_acc   = s2_best
            global_best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
            global_best_info  = {'fold': fold + 1, 'val_acc': s2_best}

        # ── 最終推理（用 Stage 2 最佳權重，96³ 全大小）──
        _, final_val_trans = get_transforms(
            SPATIAL_SIZE_S2, use_roi=use_roi,
            roi_center=ROI_CENTER, roi_size=ROI_SIZE_S2
        )
        final_val_ds = Dataset(data=val_files, transform=final_val_trans)
        final_loader = DataLoader(final_val_ds, batch_size=1, num_workers=2)

        model.eval()
        with torch.no_grad():
            for local_i, batch_data in enumerate(final_loader):
                inp = batch_data["image"].to(device)
                tgt = batch_data["label"].to(device)
                out = model(inp)
                prob = torch.softmax(out, dim=1)
                all_pred.append(out.argmax(dim=1).item())
                all_true.append(tgt.item())
                all_prob.append(prob[0, 1].item())
                all_prob_full.append(prob[0].cpu().numpy())
                # 直接從 val_files 取路徑，不依賴 MONAI meta_dict
                all_image_paths.append(str(val_files[local_i]["image"]))

    acc = accuracy_score(all_true, all_pred)
    cm  = confusion_matrix(all_true, all_pred)
    try:
        auc = roc_auc_score(all_true, all_prob)
    except ValueError:
        auc = float('nan')

    # ---------------------------------------------------------------
    # 儲存最佳模型 + OOF soft probabilities（cross-model KD 用）
    # ---------------------------------------------------------------
    safe_name    = task_name.replace(' ', '_')
    all_prob_arr = np.stack(all_prob_full)   # (N, 2)

    # 1. 最佳單一模型權重
    ckpt_path = os.path.join(MODEL_SAVE_DIR,
                             f'smri_resnet_v3_{safe_name}_best.pth')
    torch.save({
        'model_state_dict': global_best_state,
        'model_config': {
            'dropout':       DROPOUT,
            'use_roi':       use_roi,
            'roi_center':    ROI_CENTER,
            'roi_size_s2':   ROI_SIZE_S2,
            'spatial_size':  SPATIAL_SIZE_S2,
            'pretrain_path': PRETRAIN_PATH,
        },
        'task':       task_name,
        'task_pair':  task_pair,
        'class_map':  {task_pair[0]: 0, task_pair[1]: 1},
        'best_val_acc': global_best_info.get('val_acc', 0.0),
        'best_from':    global_best_info,
    }, ckpt_path)
    print(f"\n  Saved best model → {ckpt_path}")
    print(f"  (fold={global_best_info.get('fold')}, "
          f"val_acc={global_best_info.get('val_acc', 0)*100:.1f}%)")

    # 2. OOF soft probabilities（不偏估計，可直接作為 KD soft labels）
    #    Shape: (N_samples, 2)   [class_a_prob, class_b_prob]
    #    image_paths 可用於和 fMRI side 的 matrix_paths 做受試者對齊
    oof_path = os.path.join(MODEL_SAVE_DIR,
                            f'smri_resnet_v3_{safe_name}_oof_probs.npy')
    np.save(oof_path, {
        'probs':       all_prob_arr,
        'labels':      np.array(all_true),
        'image_paths': all_image_paths,
        'class_map':   {task_pair[0]: 0, task_pair[1]: 1},
        'acc':         acc,
        'auc':         auc,
        'task':        task_name,
    }, allow_pickle=True)
    print(f"  Saved OOF soft probs → {oof_path}")
    print(f"  (shape: {all_prob_arr.shape}, {len(all_true)} samples)")

    # 3. 用 OOF 預測產出 teacher_logits（subject_id → prob，供 fMRI GNN ensemble）
    #    這是真正的 OOF 版本：每個樣本都是被 held-out 時做的預測
    teacher_logits_dict = {}
    skipped = 0
    for path, prob_vec in zip(all_image_paths, all_prob_arr):
        sid = get_subject_id(path)
        if sid:
            teacher_logits_dict[sid] = prob_vec
        else:
            skipped += 1
    tl_path = os.path.join(MODEL_SAVE_DIR, f"teacher_logits_{safe_name}.npy")
    np.save(tl_path, teacher_logits_dict, allow_pickle=True)
    print(f"  Saved OOF teacher_logits → {tl_path}")
    print(f"  ({len(teacher_logits_dict)} subjects aligned, {skipped} skipped)")

    return acc, auc, cm


# ===============================================================
# 7. 主程式
# ===============================================================
def main():
    print("🚀 sMRI 3D ResNet v3")
    print("   改進項目: ROI 聚焦 + Progressive Resizing + SE Block + 強化增強")
    print("   MCI 任務自動啟用海馬迴 ROI 模式\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 訓練硬體: {device}")
    if torch.cuda.is_available():
        print(f"   GPU : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 任務清單 ──
    # 若你已完成 ADNI 的 sMCI/pMCI 標籤分類，可改用下方的 progressive 任務：
    # tasks = [('NC', 'AD'), ('NC', 'MCI'), ('sMCI', 'pMCI')]
    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('3D ResNet v3 (sMRI) — ROI + Progressive Resizing', fontsize=16, fontweight='bold')

    for idx, task in enumerate(tasks):
        acc, auc, cm = run_3d_task(task, device)
        if cm is not None:
            key = f"{task[0]} vs {task[1]}"
            results[key] = (acc, auc, cm)

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Oranges', ax=axes[idx],
                xticklabels=[task[0], task[1]],
                yticklabels=[task[0], task[1]],
                annot_kws={"size": 16}
            )
            axes[idx].set_title(
                f"{task[0]} vs {task[1]}\nAcc: {acc*100:.1f}%  AUC: {auc:.3f}",
                fontsize=14
            )
            axes[idx].set_ylabel('True Label', fontsize=12)
            axes[idx].set_xlabel('Predicted Label', fontsize=12)

    print("\n" + "="*65)
    print("🏆 3D ResNet v3 效能總榜單")
    print("="*65)
    print(f"{'任務':<16} {'準確率':>8} {'AUC':>8}")
    print("-"*36)
    for task_name, (acc, auc, _) in results.items():
        flag = "✅" if acc >= 0.80 else "⚠️ "
        print(f"{flag} {task_name:<14} {acc*100:>7.1f}%  {auc:>7.3f}")

    plt.tight_layout()
    out_path = 'smri_3d_resnet_v3_confusion_matrices.png'
    plt.savefig(out_path, dpi=300)
    print(f"\n✅ 混淆矩陣已儲存: '{out_path}'")


if __name__ == "__main__":
    main()