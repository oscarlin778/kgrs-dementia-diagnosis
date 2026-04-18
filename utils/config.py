import os

# ================= 安全性設定 =================
# Google Gemini API Key
API_KEY = os.getenv("GOOGLE_API_KEY")


# ================= 系統路徑設定 (自動適應使用者) =================
# 自動抓取目前使用者的 Home 目錄 (例如 /home/wei-chi)
HOME_DIR = os.path.expanduser("~")

# 基礎資料夾路徑
BASE_DIR = os.path.join(HOME_DIR, "Data")

# 1. 訓練好的模型權重檔 (.pth)
MODEL_PATH = os.path.join(BASE_DIR, "script", "best_gnn_model.pth")

# 2. 資料集索引檔 (.csv)
DATASET_CSV_PATH = os.path.join(BASE_DIR, "fMRI", "processed_data", "dataset_index.csv")

# 3. AAL 圖譜資料夾 (解決 Permission Denied 的關鍵)
AAL_DIR = os.path.join(BASE_DIR, "fMRI", "nilearn_data")

# 4. processed_data 資料夾 (用於搜尋 .npy)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "fMRI", "processed_data")

# 5. 視覺化報告圖片的輸出資料夾
OUTPUT_DIR = os.path.join(BASE_DIR, "script", "report_assets")