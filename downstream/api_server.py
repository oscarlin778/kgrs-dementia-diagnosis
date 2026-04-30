import os
import glob
import re
import json
import torch
import numpy as np
import requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# ── 引入你的核心邏輯 ──
from inference_pipeline import (
    ModalityInput,
    run_multimodal_inference,
    query_knowledge_graph, find_t1_path,
    TASKS, AAL116_NAMES
)
from graph_rag_retriever import get_patient_graph_context, retrieve_medical_literature, retrieve_multimodal

# 直接手動定義路徑，最安全！
MATRIX_DIR = "/home/wei-chi/Model/processed_116_matrices"
ADNI_MATRIX_DIR = "/home/wei-chi/Data/ADNI_processed_116_matrices"
DATA_ROOT = "/home/wei-chi/Data"  # 對應 static_data mount 根目錄
MODEL_ROOT = "/home/wei-chi/Model" # 對應 static_model mount 根目錄

app = FastAPI(title="KGRS Dynamic API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 靜態檔案服務：讓前端 NiiVue 能直接 fetch T1 NIfTI
app.mount("/static_data", StaticFiles(directory=DATA_ROOT), name="static_data")
app.mount("/static_model", StaticFiles(directory=MODEL_ROOT), name="static_model")

def path_to_url(path: str) -> str:
    """將本地路徑轉換為 HTTP URL"""
    if not path:
        return ""
    if path.startswith(DATA_ROOT):
        relative = path[len(DATA_ROOT):].lstrip("/")
        return f"http://localhost:8080/static_data/{relative}"
    if path.startswith(MODEL_ROOT):
        relative = path[len(MODEL_ROOT):].lstrip("/")
        return f"http://localhost:8080/static_model/{relative}"
    return ""

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

    size = raw.shape[0]
    for i, ni in enumerate(NETWORK_LABELS):
        for j, nj in enumerate(NETWORK_LABELS):
            idx_i = [x for x in _AAL_NETWORK_MAP[ni] if x < size]
            idx_j = [x for x in _AAL_NETWORK_MAP[nj] if x < size]
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

def is_matrix_valid(path: str) -> bool:
    """檢查矩陣是否有效（非全零且可讀取）"""
    try:
        m = np.load(path)
        return np.count_nonzero(m) > 0
    except:
        return False

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

        # 驗證 fMRI 與 sMRI 的可用性
        fmri_ok = is_matrix_valid(m_path)
        t1_path = find_t1_path(sid)

        # 決定標籤顯示
        if fmri_ok and t1_path:
            label_tag = "Dual-Modal"
        elif fmri_ok:
            label_tag = "fMRI only"
        elif t1_path:
            label_tag = "sMRI only"
        else:
            label_tag = "Invalid Data"

        # 將本地 t1_path 轉換成前端可用的 HTTP URL
        t1_url = path_to_url(t1_path)

        patient_list.append({
            "id": sid,
            "label": f"{sid} ({label_tag})",
            "matrix_path": m_path if fmri_ok else "",
            "t1_path": t1_path or "",
            "t1_url": t1_url,
            "fmri_valid": fmri_ok
        })

    return {"patients": sorted(patient_list, key=lambda x: x['id'])}


# ===============================================================
# 2. 影像分析 API：(加入 fmri_weight 參數與歸因數據)
# ===============================================================
@app.post("/api/v1/analyze")
async def analyze_patient(
    subject_id: str = Form(...),
    matrix_path: Optional[str] = Form(None),
    t1_path: Optional[str] = Form(None),
    fmri_weight: float = Form(0.5)
):
    # 空字串統一轉為 None，讓推論管線正確判斷模態是否可用
    matrix_path = matrix_path or None
    t1_path = t1_path or None

    # Prepare input
    modality_input = ModalityInput(
        matrix_path=matrix_path,
        t1_path=t1_path,
        subject_id=subject_id
    )
    
    # Run multi-modal inference
    inference_results = run_multimodal_inference(modality_input, device)
    
    task_results = {}
    radar_data = {}
    all_top_rois = []
    smri_saliency_url = ""  # 取第一個有效的 saliency 路徑

    for task_name, res in inference_results.items():
        # Map back to the key format expected by frontend if needed (e.g., "NC vs AD")
        display_name = task_name.replace("_vs_", " vs ")

        task_results[display_name] = {
            "prob_fused": res["prob_positive"],
            "prediction": res["prediction"],
            "modality_used": res["modality_used"],
            "fmri_findings": res["fmri_findings"],
            "smri_findings": res["smri_findings"],
            "fmri_pred": res["fmri_pred"],
            "smri_pred": res["smri_pred"],
        }

        # Saliency for radar chart (from fMRI)
        if res["fmri_findings"]:
            saliency_116 = res["fmri_findings"].get("saliency_116")
            if saliency_116 and len(saliency_116) == 116:
                radar_data[display_name] = saliency_116
            else:
                radar_data[display_name] = [0.0] * 116
                for roi in res["fmri_findings"]["top_regions"]:
                    if roi["name"] in AAL116_NAMES:
                        idx = AAL116_NAMES.index(roi["name"])
                        radar_data[display_name][idx] = roi["saliency"]

            all_top_rois.extend([r["name"] for r in res["fmri_findings"]["top_regions"][:5]])

        # 從 sMRI findings 取出 Grad-CAM NIfTI 路徑並轉換成 HTTP URL
        if not smri_saliency_url and res.get("smri_findings"):
            sal_path = res["smri_findings"].get("saliency_path")
            if sal_path and os.path.exists(sal_path):
                smri_saliency_url = path_to_url(sal_path)

    # 查詢 Neo4j
    kg_context = query_knowledge_graph(list(set(all_top_rois)), ("NC", "AD"))

    # 9×9 network-level FC matrix（sMRI-only 病患無矩陣時回傳空陣列）
    network_matrix = compute_network_matrix(matrix_path) if matrix_path and os.path.exists(matrix_path) else []

    return {
        "task_results": task_results,
        "radar_data": radar_data,
        "kg_context": kg_context,
        "network_matrix": network_matrix,
        "network_labels": NETWORK_LABELS,
        "smri_saliency_url": smri_saliency_url,
    }

# ===============================================================
# 3. 報告生成 API：(加入速度優化參數)
# ===============================================================
class ReportRequest(BaseModel):
    subject_id: str
    task_results: dict
    kg_context: dict
    mode: str = "fast"  # 'fast' 或 'detailed'
    patient_context: Optional[str] = None  # 若前端已取得則直接傳入，否則由後端查詢


@app.post("/api/v1/report/stream")
async def generate_report_stream(req: ReportRequest):
    # 若前端未傳入病患脈絡，從 Neo4j 圖譜動態擷取
    patient_ctx = req.patient_context or get_patient_graph_context(req.subject_id)

    # 向量檢索多模態文獻
    # Get findings from the first available task
    first_task_res = list(req.task_results.values())[0]
    fmri_findings = first_task_res.get("fmri_findings")
    smri_findings = first_task_res.get("smri_findings")
    
    literature_ctx = retrieve_multimodal(fmri_findings, smri_findings, patient_ctx)

    # 模態一致性檢測 (Modality Concordance)
    concordance_lines = []
    for task_str, res in req.task_results.items():
        f_pred = res.get("fmri_pred")
        s_pred = res.get("smri_pred")
        if f_pred is not None and s_pred is not None:
            if f_pred == s_pred:
                concordance_lines.append(f"  • {task_str}：結構與功能預測一致。")
            else:
                concordance_lines.append(f"  • {task_str}：⚠️ 注意：結構與功能預測存在分歧（fMRI={f_pred}, sMRI={s_pred}）。")

    concordance_block = ""
    if concordance_lines:
        concordance_block = "\n\n### 模態一致性分析：\n" + "\n".join(concordance_lines)

    # 判斷實際可用的模態，動態生成報告指示
    has_fmri = any(v.get("fmri_findings") for v in req.task_results.values())
    has_smri = any(v.get("smri_findings") for v in req.task_results.values())

    if has_fmri and has_smri:
        modality_note = "本次為雙模態分析。"
        modality_instruction = (
            "- 【影像分析洞察】中，請分別描述【結構性發現 (sMRI)】與【功能性發現 (fMRI)】。\n"
            "- 明確點出兩模態結果是否一致：若不一致，請引用實際數據並說明臨床意義。\n"
        )
    elif has_smri:
        modality_note = "本次為 sMRI 單模態分析（無有效 fMRI 數據）。"
        modality_instruction = (
            "- 【影像分析洞察】中，請僅描述【結構性發現 (sMRI)】，不要捏造 fMRI 相關內容。\n"
            "- 請在報告中說明本次分析僅基於結構影像，缺乏功能性佐證。\n"
        )
    else:
        modality_note = "本次為 fMRI 單模態分析（無 T1 結構影像）。"
        modality_instruction = (
            "- 【影像分析洞察】中，請僅描述【功能性發現 (fMRI)】，不要捏造 sMRI 相關內容。\n"
            "- 請在報告中說明本次分析僅基於功能影像，缺乏結構性佐證。\n"
        )

    prompt = (
        f"你是一位專精失智症神經影像診斷的臨床 AI 助理。\n"
        f"⚠️ 注意：{modality_note}\n\n"
        f"## 病患背景脈絡（來自知識圖譜）\n{patient_ctx}\n\n"
        f"## 參考醫學文獻（來自 Neo4j Vector RAG 檢索）\n{literature_ctx}\n\n"
        f"## 影像分析結果\n{json.dumps(req.task_results, ensure_ascii=False)}\n"
        f"{concordance_block}\n\n"
        f"請依以下指示生成報告，段落標題需以【】標示：\n"
        f"- 優先考量病患認知儲備（Cognitive Reserve）對影像結果的影響。\n"
        f"{modality_instruction}"
        f"- 【臨床診斷建議】中，⚠️ 強制要求：必須引用上方【參考醫學文獻】中的論點來支持診斷建議，"
        f"並在引用句尾標註來源標籤（例如：「[fMRI 相關文獻 1]」或「[sMRI 相關文獻 1]」）。\n"
        f"- 使用繁體中文，病患代碼：{req.subject_id}。"
    )

    def ollama_stream():
        payload = {
            "model": "gemma3:12b",
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 2000 if req.mode == "fast" else 3000,
                "temperature": 0.2,
            },
        }
        with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    yield json.loads(line).get("response", "")

    return StreamingResponse(ollama_stream(), media_type="text/event-stream")


# ===============================================================
# 4. 病患圖譜脈絡測試端點 (Task 3 驗證用)
# ===============================================================
@app.get("/api/v1/test/patient-context/{subject_id}")
async def test_patient_context(subject_id: str):
    """
    驗證用端點：查詢指定病患的 Neo4j 圖譜脈絡。
    範例：GET /api/v1/test/patient-context/012_S_6760
    """
    context = get_patient_graph_context(subject_id)
    return {
        "subject_id": subject_id,
        "patient_context": context,
        "has_data": not context.startswith("[病患脈絡]"),
    }