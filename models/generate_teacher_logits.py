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
# 產出的檔案要放的地方 (GNN 訓練時會去這裡讀 teacher logits)
OUTPUT_DIR     = "/home/wei-chi/Data/script/checkpoints/resnet_checkpoints"

# ==========================================
# 2. 萬用 ID 萃取器 (拔除前後綴)
# ==========================================
def get_clean_id(path):
    basename = os.path.basename(str(path))
    # 對 ADNI ID 的特別處理: 例如 003_S_6833
    adni_match = re.search(r'(\d{3}_S_\d{4})', basename)
    if adni_match:
        return adni_match.group(1)
    
    # 處理 TPMIC ID: 例如 0076
    clean = re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy|_T1_MNI\.nii\.gz|_T1\.nii\.gz|\.nii\.gz)$', '', basename)
    clean = re.sub(r'^(sub-|sub_|old_dswau)', '', clean)
    return clean.strip()

# ==========================================
# 3. 預處理
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
# 5. 執行推理並儲存
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 啟動 Teacher Logits 生成器 (支援 TPMIC + ADNI)")
    
    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for task_pair in tasks:
        class_a, class_b = task_pair
        task_name = f"{class_a}_vs_{class_b}"
        print(f"\n[{task_name}] 正在掃描影像...")
        
        use_roi = bool({"MCI", "sMCI", "pMCI"} & set(task_pair))
        val_transforms = get_transforms(use_roi=use_roi)
        
        # 收集路徑
        data_dicts = []
        for label, class_name in enumerate([class_a, class_b]):
            # 搜尋 TPMIC 與 ADNI
            for d_dir in [DATA_DIR_TPMIC, DATA_DIR_ADNI]:
                folder = os.path.join(d_dir, class_name)
                if os.path.exists(folder):
                    # 搜尋符合 T1 命名的所有影像
                    files = glob.glob(os.path.join(folder, "**", "*.nii.gz"), recursive=True)
                    for fp in files:
                        # 簡單過濾一下，確保是結構像而非 FC 或其他中間產物
                        if "T1" in os.path.basename(fp) or "sub-" in os.path.basename(fp):
                            data_dicts.append({"image": fp, "label": label, "orig_path": fp})
        
        if not data_dicts:
            print(f"  ⚠️ 找不到任何影像，跳過任務。")
            continue
            
        print(f"  🔍 發現 {len(data_dicts)} 筆候選影像，準備推理...")
            
        model_path = os.path.join(MODEL_SAVE_DIR, f'smri_resnet_v3_{task_name}_best.pth')
        if not os.path.exists(model_path):
            print(f"  ❌ 找不到對應的 ResNet 模型權重: {model_path}")
            continue
            
        model = build_model(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        ds = Dataset(data=data_dicts, transform=val_transforms)
        loader = DataLoader(ds, batch_size=1, num_workers=4)
        
        teacher_dict = {}
        with torch.no_grad():
            for i, batch_data in enumerate(loader):
                inp = batch_data["image"].to(device)
                # 獲取機率
                logits = model(inp)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                subj_id = get_clean_id(data_dicts[i]["orig_path"])
                teacher_dict[subj_id] = prob
                
        # 存成小寫檔名以對應 GNN 訓練腳本
        out_npy = os.path.join(OUTPUT_DIR, f"teacher_logits_{task_name.lower()}.npy")
        np.save(out_npy, teacher_dict, allow_pickle=True)
        print(f"  ✅ 成功！產出 {len(teacher_dict)} 筆標籤 -> {out_npy}")

if __name__ == "__main__":
    main()
