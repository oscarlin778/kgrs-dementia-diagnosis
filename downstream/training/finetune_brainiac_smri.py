import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd,
    NormalizeIntensityd, RandAffined, RandFlipd
)
from monai.data import CacheDataset

import sys
sys.path.insert(0, '/home/wei-chi/BrainIAC/src')
from model import ViTBackboneNet

# ===============================================================
# 1. Configuration & Constants
# ===============================================================
CHECKPOINT_PATH = "/home/wei-chi/Alzheimers_Project/external_data/scripts/models/checkpoints/finetune_tpmic_full/BrainIAC.ckpt"
RESULTS_CSV = "/home/wei-chi/Alzheimers_Project/data/finetune_smri_results.csv"
TREND_PLOT = "smri_auc_trend.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
MAX_EPOCHS = 50
HIDDEN_DIM = 768  # ViT-B

torch.set_float32_matmul_precision('medium')

# ===============================================================
# 2. Dataset & Transforms
# ===============================================================
class SMRIFinetuneDataset(Dataset):
    def __init__(self, paths, labels, transforms=None):
        self.data = [{"image": p, "label": l} for p, l in zip(paths, labels)]
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transforms:
            return self.transforms(self.data[idx])
        return self.data[idx]

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
            RandAffined(keys, prob=0.5, rotate_range=(0.1, 0.1, 0.1), translate_range=(5, 5, 5)),
            RandFlipd(keys, spatial_axis=[0, 1, 2], prob=0.5)
        ])
    return Compose(transforms)

# ===============================================================
# 3. Model Definition
# ===============================================================
class BrainIACFinetuner(pl.LightningModule):
    def __init__(self, checkpoint_path, freeze_backbone=True, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = ViTBackboneNet(checkpoint_path)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone FROZEN (Linear Probing mode)")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Backbone UNFROZEN (Fine-tuning mode)")
            
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        self.lr = lr

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
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
        return {"val_loss": loss, "logits": logits, "y": y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# ===============================================================
# 4. Training Loop
# ===============================================================
def run_fractional_finetuning():
    train_pool_df = pd.read_csv('finetune_train_pool.csv')
    holdout_df = pd.read_csv('finetune_holdout_test.csv')
    
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []

    for frac in fractions:
        print(f"\n{'='*60}\nProcessing Data Fraction: {frac}\n{'='*60}")
        
        # Sampling logic
        if frac == 0.0:
            # For Linear Probing, use 100% data but freeze backbone
            current_df = train_pool_df.copy()
            freeze = True
        else:
            # For End-to-End, sample fraction of data and unfreeze
            current_df = train_pool_df.sample(frac=frac, random_state=42).reset_index(drop=True)
            freeze = False
        
        print(f"Sampled {len(current_df)} subjects for training.")
        
        # 5-Fold Cross Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(current_df, current_df['label'])):
            print(f"--- Fold {fold+1}/5 ---")
            
            train_sub = current_df.iloc[train_idx]
            val_sub = current_df.iloc[val_idx]
            
            train_ds = CacheDataset(
                data=[{"image": p, "label": l} for p, l in zip(train_sub['path'], train_sub['label'])],
                transform=get_transforms(train=True),
                cache_rate=1.0, num_workers=8
            )
            val_ds = CacheDataset(
                data=[{"image": p, "label": l} for p, l in zip(val_sub['path'], val_sub['label'])],
                transform=get_transforms(train=False),
                cache_rate=1.0, num_workers=8
            )
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
            val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
            
            model = BrainIACFinetuner(CHECKPOINT_PATH, freeze_backbone=freeze)
            
            checkpoint_callback = ModelCheckpoint(
                monitor="val_acc", mode="max", save_top_k=1, filename=f"best-fold{fold}"
            )
            early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
            
            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                accelerator="gpu", devices=1,
                callbacks=[checkpoint_callback, early_stop_callback],
                enable_progress_bar=True,
                logger=False
            )
            
            trainer.fit(model, train_loader, val_loader)
            
            # Load best model for this fold (weights_only=False for numpy compatibility)
            best_model = BrainIACFinetuner.load_from_checkpoint(checkpoint_callback.best_model_path, weights_only=False)
            fold_models.append(best_model.to(DEVICE).eval())

        # Final Evaluation on Holdout Set (Ensemble of 5 folds)
        print("Evaluating on Holdout Test Set...")
        holdout_ds = CacheDataset(
            data=[{"image": p, "label": l} for p, l in zip(holdout_df['path'], holdout_df['label'])],
            transform=get_transforms(train=False),
            cache_rate=1.0, num_workers=8
        )
        holdout_loader = DataLoader(holdout_ds, batch_size=1, num_workers=4)
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in holdout_loader:
                img = batch["image"].to(DEVICE)
                lbl = batch["label"].to(DEVICE)
                
                # Ensemble: Average logits across folds
                ensemble_logits = torch.zeros((1, 2)).to(DEVICE)
                for m in fold_models:
                    ensemble_logits += m(img)
                ensemble_logits /= len(fold_models)
                
                prob = F.softmax(ensemble_logits, dim=1)[:, 1].cpu().numpy()
                all_probs.append(prob[0])
                all_labels.append(lbl.cpu().numpy()[0])
        
        auc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])
        
        print(f"Data Fraction {frac} Results: Holdout AUC = {auc:.4f}, ACC = {acc:.4f}")
        results.append({
            "Data_Fraction": frac,
            "Holdout_AUC": auc,
            "Holdout_ACC": acc
        })
        
        # Incremental save
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)

    # ===============================================================
    # 5. Visualization
    # ===============================================================
    df_results = pd.read_csv(RESULTS_CSV)
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Data_Fraction'], df_results['Holdout_AUC'], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Data Fraction', fontsize=12)
    plt.ylabel('Holdout Test AUC', fontsize=12)
    plt.title('BrainIAC Fine-tuning: AUC Trend across Data Fractions', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.5, 1.0)
    plt.savefig(TREND_PLOT)
    print(f"Trend plot saved to {TREND_PLOT}")

if __name__ == "__main__":
    run_fractional_finetuning()
