import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd,
    NormalizeIntensityd, RandAffined, RandFlipd, RandGaussianNoised,
    RandGaussianSmoothd, RandScaleIntensityd
)
from monai.data import CacheDataset

import sys
sys.path.insert(0, '/home/wei-chi/BrainIAC/src')
from model import ViTBackboneNet

# ===============================================================
# 1. Configuration & Constants
# ===============================================================
CHECKPOINT_PATH = "/home/wei-chi/Alzheimers_Project/external_data/scripts/models/checkpoints/finetune_tpmic_full/BrainIAC.ckpt"
RESULTS_CSV = "finetune_smri_results_v2.csv"
TREND_PLOT = "smri_auc_trend_v2.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
MAX_EPOCHS = 100  # Increased for stability
BACKBONE_LR = 1e-5 # Extremely low as per paper hints
HEAD_LR = 1e-4     # Higher for the new head
WARMUP_EPOCHS = 10

torch.set_float32_matmul_precision('medium')

def get_transforms(train=True):
    keys = ["image"]
    transforms = [
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Resized(keys, spatial_size=(96, 96, 96)),
        ScaleIntensityd(keys),
        NormalizeIntensityd(keys)
    ]
    if train:
        transforms.extend([
            # Spatial Augmentations
            RandAffined(keys, prob=0.5, rotate_range=(0.2, 0.2, 0.2), translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1)),
            RandFlipd(keys, spatial_axis=[0, 1, 2], prob=0.5),
            # Intensity Augmentations (The "Nature Magic")
            RandGaussianNoised(keys, prob=0.2, mean=0.0, std=0.1),
            RandGaussianSmoothd(keys, prob=0.2, sigma_x=(0.5, 1.0)),
            RandScaleIntensityd(keys, factors=0.1, prob=0.5)
        ])
    return Compose(transforms)

# ===============================================================
# 2. Optimized Model Definition
# ===============================================================
class BrainIACFinetuner(pl.LightningModule):
    def __init__(self, checkpoint_path, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = ViTBackboneNet(checkpoint_path)
        
        # Linear Probing vs End-to-End
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone
            
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # Increased dropout
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1) # Added label smoothing
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        # Differential Learning Rates
        optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(), 'lr': BACKBONE_LR},
            {'params': self.classifier.parameters(), 'lr': HEAD_LR}
        ], weight_decay=1e-2)
        
        # Linear Warmup + Cosine Annealing
        def lr_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return float(epoch + 1) / float(WARMUP_EPOCHS)
            return 0.5 * (1.0 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (MAX_EPOCHS - WARMUP_EPOCHS)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]

# ===============================================================
# 3. Training Loop
# ===============================================================
def run_optimized_finetuning():
    train_pool_df = pd.read_csv('finetune_train_pool.csv')
    holdout_df = pd.read_csv('finetune_holdout_test.csv')
    
    # We focus on 1.0 (Full) vs 0.0 (Baseline) for this detailed run if time is tight, 
    # but I will keep the loop as requested.
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for frac in fractions:
        print(f"\n>>> [V2 Optimized] Data Fraction: {frac} <<<")
        
        if frac == 0.0:
            current_df = train_pool_df.copy()
            freeze = True
        else:
            current_df = train_pool_df.sample(frac=frac, random_state=42).reset_index(drop=True)
            freeze = False
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(current_df, current_df['label'])):
            print(f"--- Fold {fold+1}/5 ---")
            train_sub, val_sub = current_df.iloc[train_idx], current_df.iloc[val_idx]
            
            # Using CacheDataset for speed
            train_ds = CacheDataset(
                data=[{"image": p, "label": l} for p, l in zip(train_sub['path'], train_sub['label'])],
                transform=get_transforms(train=True), cache_rate=1.0, num_workers=8
            )
            val_ds = CacheDataset(
                data=[{"image": p, "label": l} for p, l in zip(val_sub['path'], val_sub['label'])],
                transform=get_transforms(train=False), cache_rate=1.0, num_workers=8
            )
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
            
            model = BrainIACFinetuner(CHECKPOINT_PATH, freeze_backbone=freeze)
            checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
            early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="min") # Higher patience
            
            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS, accelerator="gpu", devices=1,
                callbacks=[checkpoint_callback, early_stop], logger=False,
                enable_checkpointing=True
            )
            trainer.fit(model, train_loader, val_loader)
            
            best_model = BrainIACFinetuner.load_from_checkpoint(checkpoint_callback.best_model_path, weights_only=False)
            fold_models.append(best_model.to(DEVICE).eval())

        # Holdout Evaluation
        print("Evaluating on Holdout Test Set (V2 Ensemble)...")
        holdout_ds = CacheDataset(
            data=[{"image": p, "label": l} for p, l in zip(holdout_df['path'], holdout_df['label'])],
            transform=get_transforms(train=False), cache_rate=1.0, num_workers=8
        )
        holdout_loader = DataLoader(holdout_ds, batch_size=1, num_workers=4)
        
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in holdout_loader:
                img, lbl = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                ensemble_logits = torch.stack([m(img) for m in fold_models]).mean(dim=0)
                prob = F.softmax(ensemble_logits, dim=1)[:, 1].cpu().numpy()
                all_probs.append(prob[0]); all_labels.append(lbl.cpu().numpy()[0])
        
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
        print(f"Fraction {frac} V2 Result: AUC = {auc:.4f}, ACC = {acc:.4f}")
        results.append({"Data_Fraction": frac, "Holdout_AUC": auc, "Holdout_ACC": acc})
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

    # Visualization
    df_results = pd.read_csv(RESULTS_CSV)
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Data_Fraction'], df_results['Holdout_AUC'], marker='s', color='red', label='V2 (Optimized)')
    
    # Optional: Compare with V1 if exists
    if os.path.exists("/home/wei-chi/Alzheimers_Project/data/finetune_smri_results.csv"):
        v1 = pd.read_csv("/home/wei-chi/Alzheimers_Project/data/finetune_smri_results.csv")
        plt.plot(v1['Data_Fraction'], v1['Holdout_AUC'], marker='o', color='gray', linestyle='--', label='V1 (Baseline)')
    
    plt.title('BrainIAC Fine-tuning: AUC Trend V2 vs V1', fontsize=14, fontweight='bold')
    plt.xlabel('Data Fraction'); plt.ylabel('Holdout AUC'); plt.grid(True); plt.legend()
    plt.savefig(TREND_PLOT)

if __name__ == "__main__":
    run_optimized_finetuning()
