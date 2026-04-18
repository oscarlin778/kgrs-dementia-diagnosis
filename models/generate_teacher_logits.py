import os
import glob
import re
import torch
import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, CropForegroundd, Resized, SpatialCropd
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet50
import torch.nn as nn

# ==========================================
# 1. 檔案路徑與設定
# ==========================================
DATA_DIR_TPMIC = "/home/wei-chi/Model/sMRI_data_MultiModal_Aligned_MNI"
DATA_DIR_ADNI  = "/home/wei-chi/Data/ADNI_sMRI_Aligned_MNI"
MODEL_SAVE_DIR = "/home/wei-chi/Data/script/resnet_checkpoints"

# ==========================================
# 2. 萬用 ID 萃取器 (精確拔除各種前後綴)
# ==========================================
def get_clean_id(path):
    basename = os.path.basename(str(path))
    clean = re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy|_T1_MNI\.nii\.gz|_T1\.nii\.gz|\.nii\.gz)$', '', basename)
    clean = re.sub(r'^(sub-|sub_|old_dswau)', '', clean)
    return clean

# ==========================================
# 3. 確保與訓練時相同的 ROI 裁切邏輯
# ==========================================
def get_transforms(use_roi=False):
    roi_crop = []
    if use_roi:
        roi_center = (48, 38, 30)
        roi_size = (64, 64, 52)
        start = tuple(max(0, roi_center[i] - roi_size[i] // 2) for i in range(3))
        end   = tuple(start[i] + roi_size[i] for i in range(3))
        roi_crop = [SpatialCropd(keys=["image"], roi_start=list(start), roi_end=list(end))]

    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        *roi_crop,
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
    ])

# ==========================================
# 4. 模型架構
# ==========================================
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

def build_model(device):
    model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)
    model.layer4 = nn.Sequential(model.layer4, SEBlock3D(2048, reduction=16))
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.15),
        nn.Linear(512, 2)
    )
    return model.to(device)

# ==========================================
# 5. 執行推理與打包
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 啟動老師標籤重建引擎 ({device})")
    
    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    
    for task_pair in tasks:
        class_a, class_b = task_pair
        task_name = f"{class_a}_vs_{class_b}"
        print(f"\n[{task_name}] 正在處理...")
        
        # MCI 任務必須開啟 ROI 以匹配訓練時的模型
        use_roi = bool({"MCI", "sMCI", "pMCI"} & set(task_pair))
        val_transforms = get_transforms(use_roi=use_roi)
        
        # 收集影像路徑
        data_dicts = []
        for label, class_name in enumerate([class_a, class_b]):
            for d_dir in [DATA_DIR_TPMIC, DATA_DIR_ADNI]:
                folder = os.path.join(d_dir, class_name)
                if os.path.exists(folder):
                    for fp in glob.glob(os.path.join(folder, "**", "*[Tt]1*.nii.gz"), recursive=True):
                        data_dicts.append({"image": fp, "label": label, "orig_path": fp})
        
        if not data_dicts:
            continue
            
        model_path = os.path.join(MODEL_SAVE_DIR, f'smri_resnet_v3_{task_name}_best.pth')
        if not os.path.exists(model_path):
            print(f"  ❌ 找不到模型: {model_path}")
            continue
            
        model = build_model(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        ds = Dataset(data=data_dicts, transform=val_transforms)
        loader = DataLoader(ds, batch_size=1, num_workers=2)
        
        teacher_dict = {}
        with torch.no_grad():
            for i, batch_data in enumerate(loader):
                inp = batch_data["image"].to(device)
                prob = torch.softmax(model(inp), dim=1).cpu().numpy()[0]
                
                # 手動繞過 MONAI，直接從原始清單拿路徑並萃取 ID
                subj_id = get_clean_id(data_dicts[i]["orig_path"])
                teacher_dict[subj_id] = prob
                
        out_npy = os.path.join(MODEL_SAVE_DIR, f"teacher_logits_{task_name}.npy")
        np.save(out_npy, teacher_dict, allow_pickle=True)
        print(f"  ✅ 成功產出 {len(teacher_dict)} 筆清晰 ID 對齊標籤 -> {out_npy}")

if __name__ == "__main__":
    main()