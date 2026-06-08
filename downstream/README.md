# KGRS 失智症輔助診斷系統 — 啟動說明

## 系統架構概覽

```
瀏覽器 (localhost:5173)
    │
    ├─ Vite Dev Server      ← 你要手動啟動
    │
    └─ FastAPI Backend (localhost:8080)    ← 你要手動啟動
            │
            ├─ Neo4j Graph DB (bolt:7687)  ← 已常駐（Docker）
            ├─ Ollama Embedding (11434)    ← 已常駐（系統服務）
            └─ PCAG-ComBat + KD GNN 模型  ← 由 backend 載入
```

**預設資料涵蓋：** 142 位 TPMIC 病患 + 74 位 ADNI 病患 = 共 216 位  
**推論硬體：** NVIDIA RTX 5090（CUDA 自動偵測）

---

## 前置確認（Neo4j & Ollama 是否在線）

開始之前先確認兩個常駐服務是否正常：

```bash
# Neo4j（應出現 LISTEN 7687）
ss -tlnp | grep 7687

# Ollama（應出現 LISTEN 11434）
ss -tlnp | grep 11434
```

若 Neo4j 未啟動，知識圖譜查詢與報告中的病患背景資料將無法使用（分析仍可執行）。

---

## 啟動步驟

### 方法 A：Supervisord（推薦，自動重啟）

```bash
conda activate AD
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
mkdir -p logs
supervisord -c supervisord.conf
```

確認兩個服務都啟動：
```bash
supervisorctl -c supervisord.conf status
# 應顯示：kgrs-backend  RUNNING  kgrs-frontend  RUNNING
```

查看即時 log：
```bash
tail -f logs/backend.log      # 後端
tail -f logs/frontend.log     # 前端
```

停止所有服務：
```bash
supervisorctl -c supervisord.conf stop all
supervisorctl -c supervisord.conf shutdown
```

> supervisord 在背景執行，關掉終端機服務仍繼續。若 backend crash 會自動重啟（最多 5 次）。

---

### 方法 B：手動分開啟動

#### 第一步：啟動後端（Terminal 1）

```bash
conda activate AD
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
uvicorn api_server:app --host 0.0.0.0 --port 8080
```

**啟動成功的標誌（約 10～20 秒）：**
```
[init] T1 lookup 建立完成，共 136 筆 sMRI 影像。
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8080
```

> **這個 Terminal 不能關，關掉後端就停了。**

#### 第二步：啟動前端（Terminal 2）

另開一個新的終端機：

```bash
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/kgrs-frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

> **這個 Terminal 也不能關。**

### 第三步：VSCode Port Forwarding（SSH 必做）

因為是 SSH 進去遠端，瀏覽器無法直接存取遠端的 port，**必須透過 VSCode 轉發**。

1. VSCode 底部點選 **PORTS** 分頁
2. 確認清單裡**同時存在**以下兩個 port（沒有就按 **Add Port** 加）：

   | Port | 用途 |
   |------|------|
   | `5173` | 前端 Vite |
   | `8080` | 後端 FastAPI |

3. 兩個都加好之後，本機瀏覽器開啟 `http://localhost:5173`

> ⚠️ **最常見的錯誤：只 forward 5173，沒有 forward 8080。**  
> 前端頁面會打開，但一直轉圈或沒有病患資料，就是這個原因。

---

## 停止系統

- 後端：在 Terminal 1 按 `Ctrl+C`
- 前端：在 Terminal 2 按 `Ctrl+C`

若要強制終止背景殘留的進程：

```bash
# 停後端
kill $(ps aux | grep "uvicorn api_server" | grep -v grep | awk '{print $2}') 2>/dev/null

# 停前端
kill $(ps aux | grep "vite --host 0.0.0.0 --port 5173" | grep -v grep | awk '{print $2}') 2>/dev/null
```

---

## 關鍵路徑與設定

### 環境變數（`.env`）
```
位置：/home/wei-chi/Alzheimers_Project/external_data/scripts/.env
內容：NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
說明：後端啟動時自動載入，不需要手動 source
```

### 資料目錄（`config.py`）
| 變數 | 路徑 | 說明 |
|---|---|---|
| `MATRIX_DIR` | `external_models/processed_116_matrices` | TPMIC fMRI 矩陣 |
| `ADNI_MATRIX_DIR` | `external_data/features/ADNI_processed_116_matrices` | ADNI fMRI 矩陣 |
| `TPMIC_SMRI_ROOT` | `external_models/sMRI_data_MultiModal_Aligned_MNI` | TPMIC T1 MRI |
| `SMRI_ROOT` | `external_data/datasets/ADNI_sMRI_Aligned_MNI` | ADNI T1 MRI |
| `SALIENCY_DIR` | `external_data/scripts/results/saliency` | 預計算 Grad-CAM |

### 模型 Checkpoint
```
PCAG-ComBat (NC/AD, NC/MCI)：
  external_models/resnet_checkpoints/pcag_combat/

KD GNN (MCI/AD)：
  external_models/resnet_checkpoints/gnn_checkpoints/
```

---

## 推論模型說明

| 任務 | 模型 | 輸入 |
|---|---|---|
| NC vs AD | PCAG-ComBat（5-fold ensemble） | fMRI + sMRI |
| NC vs MCI | PCAG-ComBat（5-fold ensemble） | fMRI + sMRI |
| MCI vs AD | KD GNN（3-seed ensemble） | fMRI only |

- 雙模態（fMRI + sMRI）：自動走 PCAG-ComBat 三個任務全跑
- 單 fMRI：僅跑 MCI vs AD（KD GNN）
- 單 sMRI：無推論（系統回傳空結果）

---

## 常見問題排查

### 前端開得起來，但沒有病患資料 / 一直轉圈
**最可能原因：VSCode PORTS 面板沒有 forward 8080。**
1. 確認 PORTS 面板同時有 `5173` 和 `8080`
2. 用 curl 確認後端是否正常：
   ```bash
   curl http://localhost:8080/api/v1/patients
   ```
   應回傳 JSON，若回傳錯誤或無回應則後端有問題。
3. F12 → Network 分頁，找 `patients` 請求，看狀態碼是什麼

### 後端啟動失敗 / Port 8080 被佔用
```bash
# 查看誰在用 8080
ss -tlnp | grep 8080

# 強制終止舊的 uvicorn
kill $(ps aux | grep "uvicorn api_server" | grep -v grep | awk '{print $2}')
```

### 前端 Port 5173 被佔用
```bash
kill $(ps aux | grep "vite --host 0.0.0.0 --port 5173" | grep -v grep | awk '{print $2}')
```

### 報告生成失敗（LLM 錯誤）
- 確認 Ollama 服務在線：`curl http://localhost:11434/api/tags`
- 確認使用的模型已下載：`ollama list`

### 病患無人口統計資料
- TPMIC 病患（sub_0001~sub_0142）：Neo4j 中無年齡等資料，屬正常現象
- ADNI 病患（sub-XXX_S_XXXX）：29 位有完整人口統計，其餘僅有診斷標籤

### Grad-CAM 不顯示
- 確認 `SALIENCY_DIR` 下有 `.nii.gz` 檔（預計算 630 個）
- 分析完成後在 Brain Viewer 右上角開啟 toggle 開關
