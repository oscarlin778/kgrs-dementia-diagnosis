import os
import glob
import ants
import time

# ================= 1. 路徑設定 =================
INPUT_DIR = "/home/wei-chi/Data/Matched_Dual_Modal_Dataset"
OUTPUT_DIR = "/home/wei-chi/Data/ADNI_sMRI_Aligned_MNI"

def align_adni_to_mni():
    print("🚀 啟動 ADNI to MNI 空間對齊引擎 (ANTs) - 遞迴增強版...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🧠 正在載入 MNI152 標準模板...")
    mni_template = ants.image_read(ants.get_ants_data('mni'))
    
    groups = ["AD", "MCI", "NC"]
    success_count = 0
    skip_count = 0
    
    for group in groups:
        group_in = os.path.join(INPUT_DIR, group)
        group_out = os.path.join(OUTPUT_DIR, group)
        os.makedirs(group_out, exist_ok=True)
        
        if not os.path.exists(group_in): continue
            
        subjects = os.listdir(group_in)
        for subj in subjects:
            t1_dir = os.path.join(group_in, subj, "T1")
            
            # 🔥【關鍵修改】使用 recursive=True，讓 glob 能夠鑽進所有的子資料夾找 nii.gz
            t1_files = glob.glob(os.path.join(t1_dir, "**", "*.nii"), recursive=True) + \
                       glob.glob(os.path.join(t1_dir, "**", "*.nii.gz"), recursive=True)
            
            if not t1_files:
                # print(f"  ⚠️ 找不到 T1 影像: {subj}") # 註解掉避免太吵
                continue
            
            input_path = t1_files[0]
            output_name = f"{subj}_T1_MNI.nii.gz"
            output_path = os.path.join(group_out, output_name)
            
            if os.path.exists(output_path):
                # print(f"  ⏩ 已存在，跳過: {output_name}") # 註解掉，眼不見為淨
                skip_count += 1
                continue
                
            try:
                start_time = time.time()
                print(f"  🔄 正在對齊: {subj} [{group}] ...", end="", flush=True)
                
                moving_img = ants.image_read(input_path)
                
                reg = ants.registration(
                    fixed=mni_template, 
                    moving=moving_img, 
                    type_of_transform='Affine'
                )
                
                ants.image_write(reg['warpedmovout'], output_path)
                
                elapsed = time.time() - start_time
                print(f" ✅ 完成! ({elapsed:.1f}s)")
                success_count += 1
                
            except Exception as e:
                print(f" ❌ 失敗: {subj} - {e}")

    print("=" * 50)
    print(f"🏁 任務結束！")
    print(f"✅ 成功將 {success_count} 筆新 ADNI 影像對齊至 MNI 空間！")
    print(f"⏩ 跳過了 {skip_count} 筆已存在的舊資料。")
    print(f"📁 新資料存放於: {OUTPUT_DIR}")

if __name__ == "__main__":
    align_adni_to_mni()