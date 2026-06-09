"""
finetune_brainiac_full.py
Full fine-tuning: all ViT blocks unfrozen + strong augmentation.
V2: Added LLRD, EMA, Warmup, and TTA.
Train: brainiac_combined_train.csv (164)  Test: brainiac_combined_test.csv (40)
"""
import os, sys, json, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ResizeWithPadOrCrop, NormalizeIntensity, EnsureType,
    RandAffine, RandFlip, RandGaussianNoise, RandGaussianSmooth, RandScaleIntensity,
)

sys.path.insert(0, '/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream')

CKPT_PATH = "/home/wei-chi/Alzheimers_Project/external_data/scripts/models/checkpoints/finetune_tpmic_full/vit_mci.ckpt"
TRAIN_CSV = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/brainiac_combined_train.csv"
TEST_CSV  = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/brainiac_combined_test.csv"
OUT_JSON  = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results/vitmci_full_finetune_v4_combined_test.json"

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 4
MAX_EPOCHS   = 60
PATIENCE     = 12
BACKBONE_LR  = 5e-5
HEAD_LR      = 1e-4
WEIGHT_DECAY = 0.05
N_FOLDS      = 5
SEED         = 42
DROPOUT      = 0.4

WARMUP_EPOCHS = 5      # 前 5 epoch LR 線性 warmup
LLRD_DECAY    = 0.85   # 每往下一個 ViT block，LR 再乘以 0.85
LABEL_SMOOTH  = 0.1    # Label smoothing 強度

torch.set_float32_matmul_precision("medium")


# ── Weight loading ────────────────────────────────────────────────────────────
def load_vit_weights(vit_module):
    FILTER = {"classifier", "fc", "head", "cls_head"}
    STRIPS = ["model.backbone.backbone.", "backbone.backbone.",
              "model.backbone.", "backbone."]
    raw    = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    raw_sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    filt   = {k: v for k, v in raw_sd.items()
              if not any(kw in k.lower() for kw in FILTER)}
    best_p, best_c = "", 0
    for p in STRIPS:
        c = sum(1 for k in filt if k.startswith(p))
        if c > best_c: best_c, best_p = c, p
    cleaned = {k[len(best_p):]: v for k, v in filt.items() if k.startswith(best_p)}
    res = vit_module.load_state_dict(cleaned, strict=False)
    print(f"  [ViT] weights loaded: {len(cleaned)} keys, "
          f"missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")


# ── Model ─────────────────────────────────────────────────────────────────────
class FullFinetuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        from monai.networks.nets import ViT
        self.vit = ViT(
            in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
            classification=False, qkv_bias=False, save_attn=False,
        )
        load_vit_weights(self.vit)
        # Full Fine-tuning: unfreeze all parameters
        for p in self.vit.parameters():
            p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        seq_out, _ = self.vit(x)
        return self.classifier(seq_out.mean(dim=1)).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────
class AugDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        nii = self.paths[i] + ".nii.gz"
        return self.transform(nii), torch.tensor(self.labels[i], dtype=torch.float32)

class CachedDataset(Dataset):
    def __init__(self, paths, labels, cache):
        self.paths  = paths
        self.labels = labels
        self.cache  = cache
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.cache[self.paths[i]], torch.tensor(self.labels[i], dtype=torch.float32)


def build_cache(paths, transform):
    cache, skipped = {}, 0
    for p in tqdm(list(dict.fromkeys(paths)), desc="  Caching MRI"):
        nii = p + ".nii.gz"
        if not os.path.exists(nii):
            print(f"  [WARN] missing: {nii}"); skipped += 1; continue
        cache[p] = transform(nii)
    print(f"  Cached {len(cache)} volumes, skipped {skipped}")
    return cache


# ── Training ──────────────────────────────────────────────────────────────────
def get_llrd_groups(model):
    """Layer-wise LR Decay：越靠近輸入的層 LR 越小。"""
    lr_map = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            lr = HEAD_LR
        elif "blocks." in name:
            idx = int(name.split("blocks.")[1].split(".")[0])
            lr = BACKBONE_LR * (LLRD_DECAY ** (11 - idx))
        else:
            # patch_embedding / norm / cls_token / positional_embedding
            lr = BACKBONE_LR * (LLRD_DECAY ** 12)
        lr_map.setdefault(lr, []).append(param)
    return [{"params": params, "lr": lr} for lr, params in lr_map.items()]


def train_fold(tr_p, tr_l, val_p, val_l, train_transform, cache):
    model = FullFinetuneModel().to(DEVICE)

    # LLRD optimizer
    optimizer = torch.optim.AdamW(get_llrd_groups(model), weight_decay=WEIGHT_DECAY)

    # Warmup + Cosine Annealing scheduler
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, MAX_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    crit = nn.BCEWithLogitsLoss()

    tr_loader  = DataLoader(AugDataset(tr_p, tr_l, train_transform),
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(CachedDataset(val_p, val_l, cache),
                            batch_size=1, shuffle=False, num_workers=0)

    best_val_auc, patience_cnt = 0.0, 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(MAX_EPOCHS):
        # ── Training ─────────────────────────────────────────────────
        model.train()
        for imgs, lbls in tr_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            soft_lbls = lbls * (1 - LABEL_SMOOTH) + LABEL_SMOOTH * 0.5
            loss = crit(model(imgs), soft_lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # ── Validation：val AUC early stopping ───────────────────────
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                val_probs.append(torch.sigmoid(model(imgs.to(DEVICE))).cpu().item())
                val_true.append(int(lbls.item()))

        if len(set(val_true)) < 2:
            continue
        val_auc = roc_auc_score(val_true, val_probs)

        if val_auc > best_val_auc:
            best_val_auc, patience_cnt = val_auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model.eval()


@torch.no_grad()
def infer(model, paths, labels, cache):
    ds = CachedDataset(paths, labels, cache)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    probs = [torch.sigmoid(model(imgs.to(DEVICE))).cpu().item()
             for imgs, _ in loader]
    return np.array(probs)


@torch.no_grad()
def infer_tta(model, paths, labels, cache):
    """Test-Time Augmentation：對 6 種翻轉取平均。"""
    # 體積 shape = (1, 1, D, H, W)，翻轉維度從 dim=2 開始
    flip_configs = [[], [2], [3], [4], [2, 3], [2, 4]]
    all_probs = []
    ds = CachedDataset(paths, labels, cache)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for flip_dims in flip_configs:
        probs = []
        for imgs, _ in loader:
            vol = imgs.to(DEVICE)
            if flip_dims:
                vol = torch.flip(vol, dims=flip_dims)
            probs.append(torch.sigmoid(model(vol)).cpu().item())
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    torch.manual_seed(SEED); np.random.seed(SEED)

    train_transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ResizeWithPadOrCrop(spatial_size=(96, 96, 96), mode="constant"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        RandAffine(
            prob=0.8,
            rotate_range=(0.15, 0.15, 0.15),
            translate_range=(8, 8, 8),
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear", padding_mode="border"
        ),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandGaussianNoise(prob=0.3, mean=0.0, std=0.01),
        RandGaussianSmooth(prob=0.2,
            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        RandScaleIntensity(factors=0.15, prob=0.5),
        EnsureType(data_type="tensor", dtype=torch.float32),
    ])

    val_transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ResizeWithPadOrCrop(spatial_size=(96, 96, 96), mode="constant"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        EnsureType(data_type="tensor", dtype=torch.float32),
    ])

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}  Device: {DEVICE}")

    all_paths = list(set(train_df["pat_id"].tolist() + test_df["pat_id"].tolist()))
    print(f"\nCaching {len(all_paths)} volumes (val/test only)...")
    cache = build_cache(all_paths, val_transform)

    TASKS = {
        "NC vs AD":  {"classes": [0, 2], "pos": 2},
        "NC vs MCI": {"classes": [0, 1], "pos": 1},
        "MCI vs AD": {"classes": [1, 2], "pos": 2},
    }
    oof_results, test_results = {}, {}
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    for task_name, info in TASKS.items():
        print(f"\n{'='*60}\n  Task: {task_name}\n{'='*60}")
        cls, pos = info["classes"], info["pos"]

        def filter_df(df):
            sub = df[df["label"].isin(cls)].reset_index(drop=True)
            paths  = [p for p in sub["pat_id"].tolist() if p in cache]
            labels = [(1 if sub.loc[sub["pat_id"] == p, "label"].values[0] == pos else 0)
                      for p in paths]
            return paths, labels

        tr_p, tr_l = filter_df(train_df)
        te_p, te_l = filter_df(test_df)
        print(f"  Train: {len(tr_l)} | Test: {len(te_l)}")

        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        oof_probs   = np.zeros(len(tr_l))
        fold_models = []

        for fold, (tri, vali) in enumerate(skf.split(tr_p, tr_l)):
            print(f"  Fold {fold+1}/{N_FOLDS} ...", end="", flush=True)
            f_tr_p  = [tr_p[i] for i in tri];  f_tr_l  = [tr_l[i] for i in tri]
            f_val_p = [tr_p[i] for i in vali]; f_val_l = [tr_l[i] for i in vali]
            m = train_fold(f_tr_p, f_tr_l, f_val_p, f_val_l, train_transform, cache)
            fold_models.append(m)

            # --- 儲存 fold checkpoint ---
            from pathlib import Path as _P
            _ckpt_dir = _P("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/checkpoints/fullft_v4")
            _ckpt_dir.mkdir(parents=True, exist_ok=True)
            _task_safe = task_name.replace(" vs ", "_vs_")
            _ckpt_file = _ckpt_dir / f"fullft_{_task_safe}_fold{fold}.pt"
            torch.save(m.state_dict(), _ckpt_file)
            print(f"  [SAVED] {_ckpt_file}")
            # --- end ---

            vprobs = infer(m, f_val_p, f_val_l, cache)
            oof_probs[vali] = vprobs
            fa = roc_auc_score(f_val_l, vprobs) if len(set(f_val_l)) > 1 else 0.5
            print(f" AUC={fa:.3f}")

        oof_auc = roc_auc_score(tr_l, oof_probs)
        oof_acc = accuracy_score(tr_l, (oof_probs >= 0.5).astype(int))
        oof_results[task_name] = {"auc": round(oof_auc, 4), "acc": round(oof_acc, 4), "n": len(tr_l)}
        print(f"\n  OOF  AUC={oof_auc:.4f}  ACC={oof_acc:.4f}")

        te_stack = np.stack([infer_tta(m, te_p, te_l, cache) for m in fold_models])
        te_probs = te_stack.mean(axis=0)
        te_auc   = roc_auc_score(te_l, te_probs) if len(set(te_l)) > 1 else 0.5
        te_acc   = accuracy_score(te_l, (te_probs >= 0.5).astype(int))
        test_results[task_name] = {"auc": round(te_auc, 4), "acc": round(te_acc, 4), "n": len(te_l)}
        print(f"  Test AUC={te_auc:.4f}  ACC={te_acc:.4f}")

    out = {"oof": oof_results, "test": test_results,
           "train_n": len(train_df), "test_n": len(test_df)}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {OUT_JSON}")

    print("\n" + "="*65)
    print(f"{'Task':<12} {'OOF AUC':>9} {'OOF ACC':>9} {'Test AUC':>9} {'Test ACC':>9} {'n_te':>5}")
    print("-"*58)
    for t in TASKS:
        o, te = oof_results[t], test_results[t]
        print(f"{t:<12} {o['auc']:>9.4f} {o['acc']:>9.4f} {te['auc']:>9.4f} {te['acc']:>9.4f} {te['n']:>5}")


if __name__ == "__main__":
    run()
