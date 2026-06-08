# BrainIAC 系統服務管理指南

## 服務架構

| 服務 | 說明 | Port |
|------|------|------|
| `brainiac-backend` | FastAPI API Server (api_server.py) | 8082 |
| `brainiac-frontend` | Vite 前端開發伺服器 | 5173 |
| Neo4j | 圖資料庫（Docker，由系統管理） | 7687 |
| Ollama | LLM 推論（gemma3:12b + nomic-embed-text） | 11434 |

---

## 一次性設定（只需做一次）

### 開機自動啟動

```bash
sudo loginctl enable-linger wei-chi
```

執行後重開機，backend 和 frontend 會自動啟動，不需要手動開終端機。

---

## 日常操作

### 查看狀態

```bash
systemctl --user status brainiac-backend brainiac-frontend
```

### 啟動

```bash
systemctl --user start brainiac-backend
systemctl --user start brainiac-frontend
```

### 停止

```bash
systemctl --user stop brainiac-backend
systemctl --user stop brainiac-frontend
```

### 重啟（改完程式碼後用這個）

```bash
# 只改了後端程式碼（api_server.py、inference_pipeline_v2.py、graph_rag_retriever.py 等）
systemctl --user restart brainiac-backend

# 只改了前端程式碼（App.jsx、元件等）
# → 前端有 hot reload，通常不需要重啟，直接存檔瀏覽器就更新

# 改了 vite.config.js（proxy 設定）
systemctl --user restart brainiac-frontend

# 兩個都重啟
systemctl --user restart brainiac-backend brainiac-frontend
```

---

## 查看 Log

### 即時追蹤（推薦用於 debug）

```bash
# 後端 log
journalctl --user -u brainiac-backend -f

# 前端 log
journalctl --user -u brainiac-frontend -f
```

### 查看 log 檔案

```bash
tail -f ~/logs/brainiac-backend.log
tail -f ~/logs/brainiac-frontend.log
```

---

## 重要路徑

| 項目 | 路徑 |
|------|------|
| 後端主程式 | `scripts/downstream/api_server.py` |
| 推論管線 | `scripts/downstream/inference_pipeline_v2.py` |
| RAG 檢索 | `scripts/downstream/graph_rag_retriever.py` |
| 前端 | `scripts/downstream/kgrs-frontend/src/App.jsx` |
| Vite 設定 | `scripts/downstream/kgrs-frontend/vite.config.js` |
| Systemd 服務檔 | `~/.config/systemd/user/brainiac-backend.service` |
| Systemd 服務檔 | `~/.config/systemd/user/brainiac-frontend.service` |
| Backend Log | `~/logs/brainiac-backend.log` |
| Frontend Log | `~/logs/brainiac-frontend.log` |

---

## 修改程式碼的標準流程

```bash
# 1. 修改程式碼（以後端為例）
vim api_server.py   # 或用 VS Code 改

# 2. 重啟對應服務
systemctl --user restart brainiac-backend

# 3. 確認服務正常啟動
systemctl --user status brainiac-backend

# 4. 追蹤 log 確認沒有錯誤
journalctl --user -u brainiac-backend -f
```

---

## 常見問題

### Port 被占用導致啟動失敗

```bash
# 查看誰占用 8082
ss -tlnp | grep 8082

# 強制殺掉後重啟服務
kill <PID>
systemctl --user restart brainiac-backend
```

### 服務反覆 crash（查原因）

```bash
journalctl --user -u brainiac-backend -n 50 --no-pager
```

### 手動更新 Systemd 服務設定後

```bash
# 修改 ~/.config/systemd/user/brainiac-backend.service 後必須執行
systemctl --user daemon-reload
systemctl --user restart brainiac-backend
```
