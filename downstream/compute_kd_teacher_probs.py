"""
compute_kd_teacher_probs.py
============================
Phase 1 of Cross-Modal KD:
  1. Join fMRI (combined_train.csv) × sMRI (brainiac_combined_train.csv)
     on subject_id == orig_sid  → kd_train_aligned.csv
  2. Same for test sets          → kd_test_aligned.csv
  3. For each OVO task, run:
       a) BrainIAC 5-fold ensemble (LightFinetuneModel) on sMRI
       b) ResNet50+SEBlock3D on sMRI
       c) Fuse probs (average), save as npy teacher soft labels

Output files:
  downstream/kd_train_aligned.csv
  downstream/kd_test_aligned.csv
  downstream/teacher_probs/{task_safe}_train_teacher_probs.npy   # shape (N,2)
  downstream/teacher_probs/{task_safe}_train_subject_ids.npy     # shape (N,)
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream')

# ── Paths ───────────────────────────────────────────────────────────
FMRI_TRAIN_CSV  = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits/combined_train.csv"
FMRI_TEST_CSV   = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits/combined_test.csv"
SMRI_TRAIN_CSV  = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/brainiac_combined_train.csv"
SMRI_TEST_CSV   = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/brainiac_combined_test.csv"

LIGHTFT_CKPT_DIR = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/checkpoints/lightft"
FULLFT_CKPT_DIR = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/checkpoints/fullft"
RESNET_CKPT_DIR  = "/home/wei-chi/Alzheimers_Project/external_data/scripts/checkpoints/resnet_checkpoints"

OUT_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
PROB_DIR = OUT_DIR / "teacher_probs"
PROB_DIR.mkdir(exist_ok=True)

VIT_CKPT = "/home/wei-chi/Alzheimers_Project/external_data/scripts/models/checkpoints/finetune_tpmic_full/vit_mci.ckpt"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASKS = {
    "NC_vs_AD":  {"classes": [0, 2], "pos": 2},
    "NC_vs_MCI": {"classes": [0, 1], "pos": 1},
    "MCI_vs_AD": {"classes": [1, 2], "pos": 2},
}


# ── Step 1: 對齊 CSV ────────────────────────────────────────────────
fmri_tr = pd.read_csv(FMRI_TRAIN_CSV)   # subject_id, matrix_path, label, source
fmri_te = pd.read_csv(FMRI_TEST_CSV)
smri_tr = pd.read_csv(SMRI_TRAIN_CSV)   # pat_id, label, source, orig_sid
smri_te = pd.read_csv(SMRI_TEST_CSV)

# Rename for join
smri_tr = smri_tr.rename(columns={"orig_sid": "subject_id", "pat_id": "smri_path"})
smri_te = smri_te.rename(columns={"orig_sid": "subject_id", "pat_id": "smri_path"})

# Inner join on subject_id (keep subjects with BOTH modalities)
kd_tr = fmri_tr.merge(smri_tr[["subject_id", "smri_path"]], on="subject_id", how="inner")
kd_te = fmri_te.merge(smri_te[["subject_id", "smri_path"]], on="subject_id", how="inner")

kd_tr.to_csv(OUT_DIR / "kd_train_aligned.csv", index=False)
kd_te.to_csv(OUT_DIR / "kd_test_aligned.csv",  index=False)
print(f"kd_train_aligned: {len(kd_tr)} subjects")
print(f"kd_test_aligned:  {len(kd_te)} subjects")

# ── Transforms ──────────────────────────────────────────────────────
from monai.networks.nets import ViT as _ViT
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ResizeWithPadOrCrop, NormalizeIntensity, EnsureType,
    CropForeground, Resize,
)

SMRI_TRANSFORM = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ResizeWithPadOrCrop(spatial_size=(96, 96, 96), mode="constant"),
    NormalizeIntensity(nonzero=True, channel_wise=True),
    EnsureType(data_type="tensor", dtype=torch.float32),
])

RESNET_TRANSFORM = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    NormalizeIntensity(nonzero=True, channel_wise=True),
    CropForeground(return_coords=False),
    Resize(spatial_size=(96, 96, 96)),
    EnsureType(data_type="tensor", dtype=torch.float32),
])

# ── BrainIAC model def (must match finetune_brainiac_light.py) ──────
def load_vit_weights(vit_module):
    FILTER = {"classifier", "fc", "head", "cls_head"}
    STRIPS = ["model.backbone.backbone.", "backbone.backbone.",
              "model.backbone.", "backbone."]
    raw    = torch.load(VIT_CKPT, map_location="cpu", weights_only=False)
    raw_sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    filt   = {k: v for k, v in raw_sd.items()
              if not any(kw in k.lower() for kw in FILTER)}
    best_p, best_c = "", 0
    for p in STRIPS:
        c = sum(1 for k in filt if k.startswith(p))
        if c > best_c: best_c, best_p = c, p
    cleaned = {k[len(best_p):]: v for k, v in filt.items() if k.startswith(best_p)}
    vit_module.load_state_dict(cleaned, strict=False)

class LightFinetuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = _ViT(
            in_channels=1, img_size=(96,96,96), patch_size=(16,16,16),
            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
            classification=False, qkv_bias=False, save_attn=False,
        )
        load_vit_weights(self.vit)
        for p in self.vit.parameters(): p.requires_grad = False
        for p in self.vit.blocks[-2:].parameters(): p.requires_grad = True
        if hasattr(self.vit, "norm"):
            for p in self.vit.norm.parameters(): p.requires_grad = True
        self.classifier = nn.Sequential(
            nn.Linear(768, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1),
        )
    def forward(self, x):
        seq_out, _ = self.vit(x)
        return self.classifier(seq_out.mean(dim=1)).squeeze(-1)

# ── ResNet model def (must match train_3d_resnet_teacher.py) ────────
from monai.networks.nets import resnet50 as _resnet50

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

def build_resnet(task_safe):
    model = _resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)
    model.layer4 = nn.Sequential(model.layer4, SEBlock3D(2048, reduction=16))
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512), nn.ReLU(inplace=True),
        nn.Dropout(p=0.15), nn.Linear(512, 2)
    )
    ckpt_path = f"{RESNET_CKPT_DIR}/smri_resnet_v3_{task_safe}_best.pth"
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The checkpoint has a 'model_state_dict' key, and keys are prefixed with 'model.'
    if "model_state_dict" in sd:
        raw_sd = sd["model_state_dict"]
    else:
        raw_sd = sd
    
    # Remove 'model.' prefix from keys
    cleaned_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("model."):
            cleaned_sd[k[6:]] = v
        else:
            cleaned_sd[k] = v
            
    model.load_state_dict(cleaned_sd, strict=True)
    return model.eval().to(DEVICE)

# ── Build volume cache ───────────────────────────────────────────────
def build_volume_cache(smri_paths):
    cache = {}
    for p in tqdm(list(set(smri_paths)), desc="Caching sMRI (BrainIAC)"):
        nii = p + ".nii.gz"
        if os.path.exists(nii):
            cache[p] = SMRI_TRANSFORM(nii).unsqueeze(0)  # (1,1,96,96,96)
    print(f"  BrainIAC cached {len(cache)} volumes")
    return cache

def build_resnet_cache(paths):
    cache = {}
    for p in tqdm(list(set(paths)), desc="Caching sMRI (ResNet)"):
        nii = p + ".nii.gz"
        if os.path.exists(nii):
            try:
                cache[p] = RESNET_TRANSFORM(nii).unsqueeze(0)
            except Exception as e:
                print(f"  [WARN] ResNet cache error {p}: {e}")
    print(f"  ResNet cached {len(cache)} volumes")
    return cache

# ── Teacher inference ────────────────────────────────────────────────
@torch.no_grad()
def run_brainiac_ensemble(fold_ckpts, subjects_paths, cache):
    """Returns probs (N,) averaged over all folds"""
    all_fold_probs = []
    for ckpt in fold_ckpts:
        m = LightFinetuneModel().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        m.eval()
        fold_probs = []
        for p in subjects_paths:
            if p not in cache:
                fold_probs.append(0.5)
                continue
            vol = cache[p].to(DEVICE)
            logit = m(vol).item()
            fold_probs.append(torch.sigmoid(torch.tensor(logit)).item())
        all_fold_probs.append(fold_probs)
    return np.mean(all_fold_probs, axis=0)  # (N,)

@torch.no_grad()
def run_resnet(model, subjects_paths, resnet_cache):
    """Returns softmax probs (N,2)"""
    probs_list = []
    for p in subjects_paths:
        if p not in resnet_cache:
            probs_list.append([0.5, 0.5])
            continue
        vol = resnet_cache[p].to(DEVICE)
        logits = model(vol)  # (1,2)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probs_list.append(prob.tolist())
    return np.array(probs_list)  # (N,2)

# ── Main loop per task ───────────────────────────────────────────────
all_smri_paths = kd_tr["smri_path"].tolist() + kd_te["smri_path"].tolist()
vol_cache    = build_volume_cache(all_smri_paths)
resnet_cache = build_resnet_cache(all_smri_paths)

for task_safe, info in TASKS.items():
    print(f"\n{'='*55}\nTask: {task_safe}\n{'='*55}")
    classes, pos = info["classes"], info["pos"]

    # Filter to relevant classes
    def filter_df(df):
        sub = df[df["label"].isin(classes)].reset_index(drop=True)
        sub["bin_label"] = (sub["label"] == pos).astype(int)
        return sub

    tr_sub = filter_df(kd_tr)
    te_sub = filter_df(kd_te)

    # Load fold checkpoints for BrainIAC (Light FT)
    fold_ckpts = sorted([
        str(Path(LIGHTFT_CKPT_DIR) / f"lightft_{task_safe}_fold{i}.pt")
        for i in range(5)
    ])
    missing = [f for f in fold_ckpts if not os.path.exists(f)]
    if missing:
        print(f"  [WARN] Missing checkpoints: {missing}")
        continue

    # Load ResNet
    resnet_model = build_resnet(task_safe)

    # Run teacher on train
    print(f"  Running teacher on {len(tr_sub)} train subjects...")
    tr_paths = tr_sub["smri_path"].tolist()
    p_brainiac_tr     = run_brainiac_ensemble(fold_ckpts, tr_paths, vol_cache)
    p_brainiac_tr_2cls = np.stack([1 - p_brainiac_tr, p_brainiac_tr], axis=1)
    p_resnet_tr       = run_resnet(resnet_model, tr_paths, resnet_cache)
    p_teacher_tr      = (p_brainiac_tr_2cls + p_resnet_tr) / 2.0

    # Save
    np.save(PROB_DIR / f"{task_safe}_train_teacher_probs.npy", p_teacher_tr)
    np.save(PROB_DIR / f"{task_safe}_train_subject_ids.npy",
            tr_sub["subject_id"].values)
    np.save(PROB_DIR / f"{task_safe}_train_labels.npy",
            tr_sub["bin_label"].values)

    # Run teacher on test (for comparison)
    te_paths = te_sub["smri_path"].tolist()
    p_brainiac_te      = run_brainiac_ensemble(fold_ckpts, te_paths, vol_cache)
    p_brainiac_te_2cls = np.stack([1 - p_brainiac_te, p_brainiac_te], axis=1)
    p_resnet_te        = run_resnet(resnet_model, te_paths, resnet_cache)
    p_teacher_te       = (p_brainiac_te_2cls + p_resnet_te) / 2.0

    np.save(PROB_DIR / f"{task_safe}_test_teacher_probs.npy", p_teacher_te)
    np.save(PROB_DIR / f"{task_safe}_test_subject_ids.npy",
            te_sub["subject_id"].values)
    np.save(PROB_DIR / f"{task_safe}_test_labels.npy",
            te_sub["bin_label"].values)

    # Quick teacher AUC check
    from sklearn.metrics import roc_auc_score
    te_labels = te_sub["bin_label"].values
    if len(set(te_labels)) > 1:
        auc_brainiac = roc_auc_score(te_labels, p_brainiac_te)
        auc_resnet   = roc_auc_score(te_labels, p_resnet_te[:, 1])
        auc_ensemble = roc_auc_score(te_labels, p_teacher_te[:, 1])
        print(f"  Teacher test AUC  BrainIAC={auc_brainiac:.4f}  "
              f"ResNet={auc_resnet:.4f}  Ensemble={auc_ensemble:.4f}")

print("\n[DONE] Teacher soft labels saved to", PROB_DIR)
