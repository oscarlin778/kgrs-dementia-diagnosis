import os
import subprocess

# 設定配對好的資料集路徑
BASE_DIR = "/home/wei-chi/Data/Matched_Dual_Modal_Dataset"

def convert_dicom_to_nifti(base_dir):
    # 遍歷 AD, MCI, NC 資料夾
    for group in ["AD", "MCI", "NC"]:
        group_path = os.path.join(base_dir, group)
        if not os.path.exists(group_path): continue
        
        # 遍歷每個受試者
        for subj in os.listdir(group_path):
            subj_path = os.path.join(group_path, subj)
            
            for modal in ["T1", "fMRI"]:
                input_dir = os.path.join(subj_path, modal)
                if not os.path.exists(input_dir): continue
                
                # 執行 dcm2niix
                # -z y: 壓縮成 .nii.gz
                # -f: 命名格式 (受試者_模態)
                # -o: 輸出位置
                print(f"🔄 正在轉換: {subj} [{modal}]...")
                cmd = [
                    "dcm2niix",
                    "-z", "y",
                    "-f", f"{subj}_{modal}",
                    "-o", input_dir,
                    input_dir
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL)

    print("✅ 全部轉換完成！現在每個 T1/fMRI 資料夾下應該都有 .nii.gz 檔了。")

if __name__ == "__main__":
    convert_dicom_to_nifti(BASE_DIR)