import os
import glob
import re
import json
import torch
import numpy as np
import requests
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# ── 引入你的核心邏輯 ──
from inference_pipeline import (
    preprocess_matrix, preprocess_t1, 
    infer_fmri_task, infer_smri_task, 
    query_knowledge_graph, find_t1_path,
    TASKS, AAL116_NAMES
)

# 直接手動定義路徑，最安全！
MATRIX_DIR = "/home/wei-chi/Model/processed_116_matrices"
ADNI_MATRIX_DIR = "/home/wei-chi/Data/ADNI_processed_116_matrices"
DATA_ROOT = "/home/wei-chi/Data"  # 對應 static_data mount 根目錄

app = FastAPI(title="KGRS Dynamic API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 靜態檔案服務：讓前端 NiiVue 能直接 fetch T1 NIfTI
app.mount("/static_data", StaticFiles(directory=DATA_ROOT), name="static_data")

device = torch.device("cpu")

# ── 9 大功能網路定義 (AAL116, 0-based index) ──
NETWORK_LABELS = ['DMN', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'VIS', 'SUB', 'CER']

_AAL_NETWORK_MAP = {
    'DMN': [22,23,24,25,26,27,32,33,34,35,36,37,38,39,64,65,66,67,84,85,88,89],
    'SMN': [0,1,16,17,18,19,56,57,58,59,68,69],
    'DAN': [8,9,58,59,60,61,64,65],
    'VAN': [10,11,12,13,14,15,28,29,62,63,80,81],
    'LIM': [20,21,36,37,38,39,40,41,82,83,86,87],
    'FPN': [4,5,6,7,48,49,50,51,60,61],
    'VIS': [42,43,44,45,46,47,48,49,50,51,52,53,54,55],
    'SUB': [70,71,72,73,74,75,76,77],
    'CER': list(range(90, 116)),
}

def compute_network_matrix(matrix_path: str) -> list:
    """從原始 116×116 FC 矩陣計算 9×9 network-level 平均連結強度"""
    raw = np.load(matrix_path).astype(np.float32)
    if raw.ndim == 3:          # (1, 116, 116) → (116, 116)
        raw = raw[0]
    np.fill_diagonal(raw, 0)

    n = len(NETWORK_LABELS)
    net_mat = np.zeros((n, n), dtype=np.float32)

    for i, ni in enumerate(NETWORK_LABELS):
        for j, nj in enumerate(NETWORK_LABELS):
            idx_i = [x for x in _AAL_NETWORK_MAP[ni] if x < 116]
            idx_j = [x for x in _AAL_NETWORK_MAP[nj] if x < 116]
            sub = raw[np.ix_(idx_i, idx_j)]
            if i == j:
                # within-network: upper triangle only
                triu = sub[np.triu_indices(len(idx_i), k=1)]
                net_mat[i, j] = float(np.mean(triu)) if triu.size else 0.0
            else:
                net_mat[i, j] = float(np.mean(sub))

    return net_mat.tolist()

def clean_subject_id(filename):
    """將檔名轉換為乾淨的 Subject ID"""
    name = os.path.basename(filename)
    name = re.sub(r'(_matrix_116\.npy|_matrix_clean_116\.npy|_task-rest_bold_matrix_clean_116\.npy)$', '', name)
    return name

# ===============================================================
# 1. 自動掃描 API：動態抓取資料夾內的病患
# ===============================================================
@app.get("/api/v1/patients")
async def get_patients():
    """自動掃描多個矩陣目錄，並對齊 sMRI 路徑與靜態 URL"""
    patient_list = []
    seen = set()

    # 掃描兩個矩陣目錄
    scan_dirs = [d for d in [MATRIX_DIR, ADNI_MATRIX_DIR] if os.path.isdir(d)]
    matrix_files = []
    for d in scan_dirs:
        matrix_files.extend(glob.glob(os.path.join(d, "*.npy")))

    for m_path in matrix_files:
        sid = clean_subject_id(m_path)
        if sid in seen:
            continue
        seen.add(sid)

        t1_path = find_t1_path(sid)

        # 將本地 t1_path 轉換成前端可用的 HTTP URL
        t1_url = ""
        if t1_path and t1_path.startswith(DATA_ROOT):
            relative = t1_path[len(DATA_ROOT):].lstrip("/")
            t1_url = f"http://localhost:8080/static_data/{relative}"

        patient_list.append({
            "id": sid,
            "label": f"{sid} ({'Dual-Modal' if t1_path else 'fMRI only'})",
            "matrix_path": m_path,
            "t1_path": t1_path or "",
            "t1_url": t1_url,
        })

    return {"patients": sorted(patient_list, key=lambda x: x['id'])}

# ===============================================================
# 2. 影像分析 API：(加入 fmri_weight 參數與歸因數據)
# ===============================================================
@app.post("/api/v1/analyze")
async def analyze_patient(
    subject_id: str = Form(...),
    matrix_path: str = Form(...),
    t1_path: Optional[str] = Form(None),
    fmri_weight: float = Form(0.5)
):
    smri_weight = 1.0 - fmri_weight
    
    # 執行推論 (這部分會用到我們在 inference_pipeline 改好的 Saliency 邏輯)
    x, adj = preprocess_matrix(matrix_path)
    t1_tensor = preprocess_t1(t1_path) if t1_path and os.path.exists(t1_path) else None

    task_results = {}
    radar_data = {}
    all_top_rois = []

    for task_pair in TASKS:
        task_name = f"{task_pair[0]} vs {task_pair[1]}"
        
        # 🌟 這裡 infer_fmri_task 現在回傳的是 Saliency (歸因)，解決 ROI 一樣的問題
        prob_f, top_idx, saliency = infer_fmri_task(x, adj, task_pair, device, subject_id)
        prob_s = infer_smri_task(t1_tensor, task_pair, device) if t1_tensor is not None else None
        
        # 融合機率
        prob_fused = (prob_f * fmri_weight + prob_s * smri_weight) if prob_s is not None else prob_f
        
        task_results[task_name] = {
            "prob_fused": prob_fused,
            "prob_fmri": prob_f,
            "prob_smri": prob_s,
            "top_rois": [AAL116_NAMES[i] for i in top_idx[:5]]
        }
        radar_data[task_name] = saliency.tolist()
        all_top_rois.extend([AAL116_NAMES[i] for i in top_idx[:5]])

    # 查詢 Neo4j
    kg_context = query_knowledge_graph(list(set(all_top_rois)), ("NC", "AD"))

    # 9×9 network-level FC matrix (用於前端熱力圖)
    network_matrix = compute_network_matrix(matrix_path)

    return {
        "task_results": task_results,
        "radar_data": radar_data,
        "kg_context": kg_context,
        "network_matrix": network_matrix,
        "network_labels": NETWORK_LABELS,
    }

# ===============================================================
# 3. 報告生成 API：(加入速度優化參數)
# ===============================================================
class ReportRequest(BaseModel):
    subject_id: str
    task_results: dict
    kg_context: dict
    mode: str = "fast" # 新增：'fast' 或 'detailed'

@app.post("/api/v1/report/stream")
async def generate_report_stream(req: ReportRequest):
    # 根據模式調整指令以優化速度
    length_hint = "請提供極簡短的條列式重點 (200字內)" if req.mode == "fast" else "請提供詳盡的臨床解釋與診斷建議"
    
    prompt = f"你是一位神經影像專家。分析病患 {req.subject_id} 的結果：{json.dumps(req.task_results)}。{length_hint}。使用繁體中文。"
    
    def ollama_stream():
        # 優化：降低 temperature 並限制長度以加速生成
        payload = {
            "model": "gemma4:26b",
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": 400 if req.mode == "fast" else 1000, "temperature": 0.2}
        }
        with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    yield json.loads(line).get("response", "")

    return StreamingResponse(ollama_stream(), media_type="text/event-stream")