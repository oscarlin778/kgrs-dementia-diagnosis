import numpy as np
import os

# 換成你存放老師機率檔的實際路徑 (TEACHER_PROBS_DIR)
teacher_dir = "/home/wei-chi/Alzheimers_Project/external_data/scripts/checkpoints/resnet_checkpoints"

# 我們剛剛那三個苦主的 ID
subjects = ["sub-TPMIC03002", "sub-TPMIC03007", "sub-TPMIC03010"]
tasks = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]

print("="*50)
print(" 🔍 偷看 3D ResNet 老師的答案卷 (Soft Labels)")
print("="*50)

for subj in subjects:
    print(f"\n病患: {subj}")
    for task in tasks:
        # 假設你的老師機率檔是存成 npy，檔名格式請依照你實際的狀況修改
        # 如果是存在 CSV 裡，你可能要用 pandas 讀取
        prob_file = os.path.join(teacher_dir, f"{task}_{subj}_prob.npy") 
        
        if os.path.exists(prob_file):
            prob = np.load(prob_file)
            print(f"  [{task}] 老師給的機率: [Index 0: {prob[0]:.3f}, Index 1: {prob[1]:.3f}]")
        else:
            print(f"  [{task}] ⚠️ 找不到機率檔 (請確認路徑或檔名格式)")