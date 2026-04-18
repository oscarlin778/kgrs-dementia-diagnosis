import os
import glob
import nibabel as nib
import random

def check_nifti_dimensions(data_dirs, num_samples=3):
    for data_dir in data_dirs:
        print(f"\n🔍 正在掃描目錄: {data_dir}")
        
        # 尋找所有 .nii 或 .nii.gz 檔案 (包含子目錄)
        nii_files = glob.glob(os.path.join(data_dir, "**", "*.nii"), recursive=True)
        nii_gz_files = glob.glob(os.path.join(data_dir, "**", "*.nii.gz"), recursive=True)
        all_files = nii_files + nii_gz_files
        
        if not all_files:
            print("❌ 找不到任何 NIfTI 檔案，將跳過此目錄。")
            continue
            
        print(f"📂 共找到 {len(all_files)} 個 NIfTI 檔案。隨機抽取 {min(num_samples, len(all_files))} 個進行維度開箱...\n")
        print("=" * 65)
        
        # 隨機打亂以確保抽樣多樣性
        random.shuffle(all_files)
        
        for file_path in all_files[:num_samples]:
            try:
                img = nib.load(file_path)
                shape = img.shape
                dim = len(shape)
                
                filename = os.path.basename(file_path)
                folder_name = os.path.basename(os.path.dirname(file_path))
                
                if dim == 3:
                    modality = "🧠 3D 結構性影像 (Structure / sMRI / T1)"
                elif dim == 4:
                    time_points = shape[3]
                    modality = f"⏱️ 4D 功能性影像 (Functional / rs-fMRI, 包含 {time_points} 個時間點)"
                else:
                    modality = f"❓ 未知格式 ({dim} 維度)"
                    
                print(f"📁 所在資料夾: {folder_name}")
                print(f"📄 檔案名稱: {filename}")
                print(f"📏 矩陣維度: {shape}")
                print(f"🏷️ 影像判定: {modality}")
                print("-" * 65)
                
            except Exception as e:
                print(f"⚠️ 無法讀取檔案 {file_path}: {e}")

if __name__ == "__main__":
    # 同時掃描 Data 底下的 fMRI 以及上一層的 Model 目錄
    TARGET_DIRS = [
        "/home/wei-chi/Data/fMRI", 
        "/home/wei-chi/Model"
    ]
    check_nifti_dimensions(TARGET_DIRS)