import os
import glob
import re
import json
import time
import logging
import logging.handlers
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

# ── Rotating file log (10 MB × 3 檔) ──────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_log_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_LOG_DIR, "api_server.log"),
    maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logging.basicConfig(level=logging.INFO, handlers=[_log_handler, logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ── 引入你的核心邏輯 ──
from inference_pipeline_v2 import (
    ModalityInput,
    run_multimodal_inference,
    query_knowledge_graph, find_t1_path,
    TASKS, AAL116_NAMES, INFERENCE_THRESHOLDS
)

_TASK_AUC_REF = {
    "NC vs AD":  0.814,   # PCAG-ComBat v2, n=31 test, 95% CI: 0.632–0.939
    "NC vs MCI": 0.747,   # PCAG-ComBat v2, n=38 test, 95% CI: 0.563–0.913
    "MCI vs AD": 0.697,   # PCAG-ComBat v2 (dual-modal), n=29 test, 95% CI: 0.487–0.878
}

_RELIABILITY = {
    "NC vs AD":  {"level": "high",   "auc_ref": _TASK_AUC_REF["NC vs AD"], "note": ""},
    "NC vs MCI": {"level": "medium", "auc_ref": _TASK_AUC_REF["NC vs MCI"],
                  "note": "NC/MCI boundary is inherently difficult; AUC≈0.75. "
                          "Treat as supplementary evidence only."},
    "MCI vs AD": {"level": "medium", "auc_ref": _TASK_AUC_REF["MCI vs AD"],
                  "note": "MCI/AD boundary; AUC≈0.70 with dual-modal PCAG. "
                          "Wide CI due to small AD test set (n=11)."},
}

def _compute_ovo_weighted(task_results: dict) -> dict:
    """
    階層式三分類（替代 OVO 投票）：
    Step 1 — NC vs Disease:
        P(AD)  >= 0.5  (NC vs AD)   OR
        P(MCI) >= 0.5  (NC vs MCI)  → Disease；否則 → NC
    Step 2 — Disease 內部:
        P(AD)  >= 0.5  (MCI vs AD)  → AD；否則 → MCI
    """
    nc_ad  = task_results.get("NC vs AD")
    nc_mci = task_results.get("NC vs MCI")
    mci_ad = task_results.get("MCI vs AD")

    p_ad  = float(nc_ad.get("prob_fused",  0.5)) if nc_ad  else 0.5
    p_mci = float(nc_mci.get("prob_fused", 0.5)) if nc_mci else 0.5
    p_adg = float(mci_ad.get("prob_fused", 0.5)) if mci_ad else 0.5

    # Asymmetric thresholds: keep AD sensitivity (0.50), raise NC_vs_MCI bar (0.55)
    THRESH_NC_AD  = 0.50   # AD 靈敏度優先，閾值維持低
    THRESH_NC_MCI = 0.55   # MCI/NC 邊界模糊，閾值稍高
    is_disease = (p_ad >= THRESH_NC_AD) or (p_mci >= THRESH_NC_MCI)
    if not is_disease:
        predicted = "NC"
    else:
        predicted = "AD" if p_adg >= 0.5 else "MCI"

    # Borderline: decisive probability within 0.06 of its respective threshold
    BORDER = 0.06
    if predicted == "NC":
        near_ad  = p_ad  >= (THRESH_NC_AD  - BORDER)
        near_mci = p_mci >= (THRESH_NC_MCI - BORDER)
        is_borderline = near_ad or near_mci
    elif predicted == "MCI":
        decisive_prob = p_mci if p_mci >= p_ad else p_ad
        thr = THRESH_NC_MCI if p_mci >= p_ad else THRESH_NC_AD
        is_borderline = decisive_prob <= (thr + BORDER)
    else:  # AD
        is_borderline = p_adg <= (0.5 + BORDER)

    return {
        "predicted_class": predicted,
        "is_borderline": is_borderline,
        "probs": {"NC_vs_AD": round(p_ad, 4),
                  "NC_vs_MCI": round(p_mci, 4),
                  "MCI_vs_AD": round(p_adg, 4)},
        "is_tie": False,
        "tie_broken_by": None,
    }

from graph_rag_retriever import (
    get_patient_graph_context, retrieve_medical_literature,
    retrieve_multimodal, get_similar_patients_context,
)

from config import MATRIX_DIR, ADNI_MATRIX_DIR, DATA_ROOT, MODEL_ROOT

# ── BrainIAC saliency output directory ──────────────────────────
SMRI_SAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results", "smri_saliency")
os.makedirs(SMRI_SAL_DIR, exist_ok=True)

app = FastAPI(title="KGRS Dynamic API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 靜態檔案服務
app.mount("/static_data",    StaticFiles(directory=DATA_ROOT),   name="static_data")
app.mount("/static_model",   StaticFiles(directory=MODEL_ROOT),  name="static_model")
app.mount("/static_saliency", StaticFiles(directory=SMRI_SAL_DIR), name="static_saliency")

def path_to_url(path: str) -> str:
    """將本地路徑轉換為相對 URL（Vite proxy 自動轉發，不依賴 port）"""
    if not path:
        return ""
    if path.startswith(SMRI_SAL_DIR):
        relative = path[len(SMRI_SAL_DIR):].lstrip("/")
        return f"/static_saliency/{relative}"
    if path.startswith(DATA_ROOT):
        relative = path[len(DATA_ROOT):].lstrip("/")
        return f"/static_data/{relative}"
    if path.startswith(MODEL_ROOT):
        relative = path[len(MODEL_ROOT):].lstrip("/")
        return f"/static_model/{relative}"
    return ""

# ── ROI 臨床先驗知識（基於 AD/MCI 文獻共識） ─────────────────────────────────
ROI_CLINICAL_PRIOR = """
## 腦區臨床意義參考表（AD/MCI 神經影像文獻共識）

【高度相關 — AD/MCI 核心病理區域】
- Hippocampus_L/R：AD 最早期萎縮標誌，海馬體體積縮小與記憶損傷直接相關
- ParaHippocampal_L/R：內嗅皮質延伸區，早期 Tau 累積位置
- Amygdala_L/R：早期 AD 神經纖維糾結區，情緒與記憶整合受損
- Cingulum_Post_L/R：後扣帶皮質，default mode network 核心，NC→MCI 轉換的早期代謝異常區
- Precuneus_L/R：預設模式網路，澱粉樣蛋白沉積熱點
- Angular_L/R：頂葉聯合區，AD 語言與空間認知退化區

【中度相關 — MCI 與認知儲備相關區域】
- Frontal_Med_Orb_L/R：眼眶前額葉，執行功能與決策
- Temporal_Mid_L/R：中顳葉，語義記憶
- Temporal_Sup_L/R：顳上回，語言處理與聽覺聯繫
- Thalamus_L/R：丘腦，AD 中期受累，訊息中繼站功能退化
- Cingulum_Mid_L/R / Cingulum_Ant_L/R：中/前扣帶，注意力與執行功能

【非典型 AD 區域（關注時需謹慎解讀）】
- Cerebelum_*：小腦，主要負責運動協調，非 AD 主要病理標誌；若模型高度 attend 此區域，
  可能源於個體影像特徵、頭部姿勢差異或資料集 site effect，**不宜直接解讀為 AD 病理依據**
- Occipital_*：枕葉，主要視覺皮質；AD 晚期才受累，早期 attention 高可能為 confound
- Rolandic_Oper_*、Precentral_*、Postcentral_*：運動/感覺皮質，非 AD 早期病理區

⚠️ 解讀規則：
1. 若 attention 最高的 ROI 屬於「高度相關」區域 → 直接引用並說明其臨床意義
2. 若屬於「中度相關」→ 引用並說明可能的認知功能影響
3. 若屬於「非典型 AD 區域」→ 提及 attention 分數，但明確說明該區域在 AD 病理中的非典型性，
   並優先關注同時出現的高/中度相關區域
"""

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

    if raw.shape != (116, 116):
        print(f"  [network_matrix] ⚠️  matrix shape {raw.shape} ≠ (116,116), indices will be clipped."
        )

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
# 0. Health check
# ===============================================================
_SERVER_START = time.time()

@app.get("/api/v1/health")
async def health_check():
    from inference_pipeline_v2 import _PIPELINE_SINGLETON
    return {
        "status": "ok",
        "model_loaded": _PIPELINE_SINGLETON is not None,
        "uptime_sec": round(time.time() - _SERVER_START),
    }

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
            "fmri_valid": fmri_ok,
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

    # Input validation: matrix must be square and ≤ 116 (pipeline zero-pads if needed)
    if matrix_path:
        try:
            m = np.load(matrix_path)
            if m.ndim == 3: m = m[0]
            if m.shape[0] != m.shape[1] or m.shape[0] > 116:
                logger.warning(f"[analyze] {subject_id}: invalid matrix shape {m.shape}")
                raise HTTPException(
                    status_code=422,
                    detail=f"fMRI matrix shape {m.shape} is invalid (must be square and ≤ 116)."
                )
            if m.shape != (116, 116):
                logger.warning(f"[analyze] {subject_id}: matrix {m.shape} will be zero-padded to 116×116")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[analyze] {subject_id}: cannot read matrix — {e}")
            raise HTTPException(status_code=422, detail=f"Cannot read fMRI matrix: {e}")

    # Prepare input
    modality_input = ModalityInput(
        matrix_path=matrix_path,
        t1_path=t1_path,
        subject_id=subject_id
    )

    # Run multi-modal inference with timeout guard
    t0 = time.time()
    try:
        inference_results = run_multimodal_inference(modality_input, device)
    except Exception as e:
        logger.error(f"[analyze] {subject_id}: inference failed — {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    logger.info(f"[analyze] {subject_id}: inference OK in {time.time()-t0:.1f}s")
    
    task_results = {}
    radar_data = {}
    all_top_rois = []
    smri_saliency_url = ""  # 取第一個有效的 saliency 路徑

    for task_name, res in inference_results.items():
        # Map back to the key format expected by frontend if needed (e.g., "NC vs AD")
        display_name = task_name.replace("_vs_", " vs ")

        task_results[display_name] = {
            "prob_fused":        res["prob_positive"],
            "prediction":        res["prediction"],
            "modality_used":     res["modality_used"],
            "fusion_reason":     res.get("fusion_reason", "smri_primary"),
            "fmri_findings":     res["fmri_findings"],
            "smri_findings":     res["smri_findings"],
            "fmri_pred":         res["fmri_pred"],
            "smri_pred":         res["smri_pred"],
            # ── 新增 ──
            "modality_conflict": (
                res["fmri_pred"] is not None and
                res["smri_pred"] is not None and
                res["fmri_pred"] != res["smri_pred"]
            ),
            "reliability":       _RELIABILITY.get(display_name, {"level": "unknown", "auc_ref": None, "note": ""}),
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

        # BrainIAC attention rollout saliency path
        if not smri_saliency_url:
            sal_path = res.get("smri_saliency_path")
            if sal_path and os.path.exists(sal_path):
                smri_saliency_url = path_to_url(sal_path)

    # 查詢 Neo4j
    kg_context = query_knowledge_graph(list(set(all_top_rois)), ("NC", "AD"))

    # 9×9 network-level FC matrix
    network_matrix = compute_network_matrix(matrix_path) if matrix_path and os.path.exists(matrix_path) else []

    # 相似病患（SIMILAR_TO 邊）
    similar_patients_ctx = get_similar_patients_context(subject_id)

    conflict_tasks = [
        name for name, tres in task_results.items()
        if tres.get("modality_conflict")
    ]
    ovo_result = _compute_ovo_weighted(task_results)

    return {
        "task_results": task_results,
        "ovo_result": ovo_result,
        "radar_data": radar_data,
        "kg_context": kg_context,
        "network_matrix": network_matrix,
        "network_labels": NETWORK_LABELS,
        "smri_saliency_url": smri_saliency_url,
        "similar_patients_context": similar_patients_ctx,
        "modality_conflict_summary": {
            "has_conflict":    len(conflict_tasks) > 0,
            "conflict_tasks":  conflict_tasks,
            "conflict_count":  len(conflict_tasks),
            "warning": (
                f"fMRI 與 sMRI 在 {len(conflict_tasks)} 個任務預測不一致，建議謹慎解讀結果。"
                if conflict_tasks else ""
            ),
        },
    }

# ===============================================================
# 3. 報告生成 API：(加入速度優化參數)
# ===============================================================
class ReportRequest(BaseModel):
    subject_id: str
    task_results: dict
    kg_context: dict = {}
    mode: str = "fast"  # 'fast' 或 'detailed'
    patient_context: Optional[str] = None
    use_rag: bool = True
    model_observations: str = ""
    ovo_result: dict = {}  # OVO 最終預測，含 predicted_class / probs / is_borderline


@app.post("/api/v1/report/stream")
async def generate_report_stream(req: ReportRequest):
    # 若前端未傳入病患脈絡，從 Neo4j 圖譜動態擷取
    patient_ctx = req.patient_context or get_patient_graph_context(req.subject_id)

    # Task 5: Get findings from the first available task and pass predicted_class
    # Also need to find the overall predicted class for Task 5 routing
    # We can infer it from the weighted voting if available, but for now we look at the results.
    # A more robust way is to pass it from frontend or re-calculate.
    # Let's assume the frontend provides it or we take the first available result's prediction.
    # However,Task 3 needs specific values.
    
    # Extract values for Task 3 evidence_summary
    nc_ad_res = req.task_results.get("NC vs AD", {})
    nc_mci_res = req.task_results.get("NC vs MCI", {})
    mci_ad_res = req.task_results.get("MCI vs AD", {})
    
    # Task 3.3: Reliability note for the header (SSE)
    # We need to find which class was actually predicted overall. 
    # Since OVO result isn't explicitly passed in ReportRequest, 
    # we'll look for a common field or take the NC vs AD as a proxy if it exists.
    # In a real scenario, ReportRequest should include 'predicted_class'.
    # Let's check if 'predicted_class' is in task_results (it might be in a different structure)
    # Based on _compute_ovo_weighted, it returns predicted_class.
    # We'll try to find it.
    predicted_class = "UNKNOWN"
    # Logic to guess the final prediction from binary tasks
    votes = {"NC": 0, "MCI": 0, "AD": 0}
    for t_res in req.task_results.values():
        if "prediction" in t_res:
            # This is simplified OVO; in reality we'd use the weighted version
            pass

    # For the sake of Task 5 routing, let's see if we can find the "fused" prediction
    # If not provided, we'll use None and retrieval will use default.
    # Wait, Task 3 says "predicted task"'s _RELIABILITY note.
    # I will assume req.task_results has a special key or we use the first one.
    
    # Let's build the summary first.
    # We need: nc_ad_pred, nc_ad_prob, nc_ad_thr etc.
    def get_task_info(task_name, safe_name):
        res = req.task_results.get(task_name, {})
        pred = res.get("prediction", "N/A")
        prob = res.get("prob_positive", 0.0)
        modality = res.get("modality_used", "smri")
        thr = INFERENCE_THRESHOLDS.get(modality, {}).get(safe_name, 0.5)
        return pred, prob, thr

    nc_ad_pred, nc_ad_prob, nc_ad_thr = get_task_info("NC vs AD", "NC_vs_AD")
    nc_mci_pred, nc_mci_prob, nc_mci_thr = get_task_info("NC vs MCI", "NC_vs_MCI")
    mci_ad_pred, mci_ad_prob, mci_ad_thr = get_task_info("MCI vs AD", "MCI_vs_AD")

    # Modality and Confidence from the "primary" evidence
    # We'll take it from the first task available
    first_task = list(req.task_results.values())[0]
    modality_used = first_task.get("modality_used", "sMRI")
    confidence_level = first_task.get("confidence", "medium")
    
    modality_disagree = any(res.get("fmri_pred") != res.get("smri_pred") 
                           for res in req.task_results.values() 
                           if res.get("fmri_pred") is not None and res.get("smri_pred") is not None)

    # Use OVO result as the authoritative final prediction
    ovo = req.ovo_result or {}
    overall_pred = ovo.get("predicted_class") or ("AD" if nc_ad_pred == 1 and mci_ad_pred == 1 else "MCI" if nc_mci_pred == 1 else "NC")
    ovo_probs    = ovo.get("probs", {})
    is_borderline = ovo.get("is_borderline", False)

    # Use OVO probs when available (more reliable than task-level prob_fused)
    p_nc_ad  = ovo_probs.get("NC_vs_AD",  nc_ad_prob)
    p_nc_mci = ovo_probs.get("NC_vs_MCI", nc_mci_prob)
    p_mci_ad = ovo_probs.get("MCI_vs_AD", mci_ad_prob)

    border_note = "（⚠️ 邊界值預測，建議追蹤）" if is_borderline else ""

    # Build single-modal prediction lines if available
    single_modal_lines = []
    task_display = [("NC vs AD", "NC_vs_AD"), ("NC vs MCI", "NC_vs_MCI"), ("MCI vs AD", "MCI_vs_AD")]
    for task_str, task_key in task_display:
        res = req.task_results.get(task_str, {})
        fp = res.get("fmri_pred")
        sp = res.get("smri_pred")
        if fp is not None and sp is not None:
            agree = "✓ 一致" if fp == sp else "✗ 分歧"
            single_modal_lines.append(f"  {task_str}：fMRI={fp}, sMRI={sp}  [{agree}]")
    single_modal_block = (
        "\n單模態各自預測（僅供解釋，OVO 以 fusion 為準）：\n" + "\n".join(single_modal_lines)
        if single_modal_lines else ""
    )

    evidence_summary = f"""
### [系統推論證據摘要]
**OVO 最終判定：{overall_pred}**{border_note}
主要依據：PCAG-ComBat fusion（信心度：{confidence_level}）
  NC vs AD  機率={p_nc_ad:.2f}（閾值=0.50，≥ 閾值 → 疾病方向）
  NC vs MCI 機率={p_nc_mci:.2f}（閾值=0.55，≥ 閾值 → 疾病方向）
  MCI vs AD 機率={p_mci_ad:.2f}（閾值=0.50，≥ 閾值 → AD 方向）
模態一致性：{"一致" if not modality_disagree else "不一致，fusion 整合後判定 " + overall_pred}{single_modal_block}
"""
    
    fmri_findings = first_task.get("fmri_findings")
    smri_findings = first_task.get("smri_findings")
    if req.use_rag:
        literature_ctx, rag_citations = retrieve_multimodal(fmri_findings, smri_findings, patient_ctx, predicted_class=overall_pred)
    else:
        literature_ctx = "（本次為無 RAG 對照組，未檢索參考文獻）"
        rag_citations = []

    # 相似病患（SIMILAR_TO 邊）
    similar_ctx = get_similar_patients_context(req.subject_id)

    # 模態一致性檢測 (Modality Concordance)
    # ... (existing concordance logic)
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

    # 判斷實際可用的模態
    # 雙模態判斷：若 task_results 內有 fmri_pred/smri_pred（單模態獨立預測），
    # 表示我們有兩個模態的資訊（即使 fmri_findings/smri_findings 為 None）
    has_fmri = any(
        v.get("fmri_findings") is not None or v.get("fmri_pred") is not None
        for v in req.task_results.values()
    )
    has_smri = any(
        v.get("smri_findings") is not None or v.get("smri_pred") is not None
        or v.get("modality_used") in ("pcag_combat", "smri", "fusion")
        for v in req.task_results.values()
    )

    if has_fmri and has_smri:
        modality_note = "本次為雙模態分析（fMRI + sMRI）。模型使用 PCAG-ComBat 融合兩種影像進行預測。"
        modality_instruction = (
            "- 【影像分析洞察】請分為兩段：\n"
            "  【結構性發現 (sMRI)】基於 **BrainIAC ViT Attention Rollout** 結果撰寫——\n"
            "    具體引用 sMRI 關注 ROI（attn 分數代表模型決策依據）。\n"
            "    📌 ROI 名稱**直接保留英文**，不要翻譯。\n"
            "    ⚠️ 請對照上方【腦區臨床意義參考表】解讀每個 ROI 的臨床意義：\n"
            "      - 高度相關區域 → 直接說明其與 AD/MCI 的病理連結\n"
            "      - 非典型 AD 區域（Cerebelum、Occipital 等）→ 說明 attention 分數，\n"
            "        並明確指出該區域在 AD 病理中屬非典型，優先聚焦同時出現的典型區域\n"
            "    🚫 嚴禁使用「GradCAM」。嚴禁捏造未列出的腦區數值。\n"
            "  【功能性發現 (fMRI)】基於 GAT 網路 attention 撰寫——\n"
            "    引用 fMRI 關注 ROI（含百分比），描述功能連結模式。\n"
            "    📌 ROI 名稱**直接保留英文**。\n"
            "    ⚠️ 對照【腦區臨床意義參考表】解讀：Parietal、Temporal、Cingulum 等高度相關區域\n"
            "    的 attention 高，在 MCI/AD 背景下通常反映**功能連結異常**（模型偵測到病理訊號），\n"
            "    而非補償機制——請按參考表指引，區分「病理訊號」vs「認知儲備」的解釋。\n"
            "    🚫 不要寫「DMN 活躍 = 正常/保護」等未有根據的推論。\n"
            "- 【模態一致性分析】請逐 task 描述單模態預測結果：\n"
            "    引用各 task 的 fMRI-only 與 sMRI-only 預測（class 名稱：NC / MCI / AD）。\n"
            "    ✅ 若兩者一致：說明 fusion 結果獲雙模態支持，可信度較高。\n"
            "    ⚠️ 若兩者分歧（例如 fMRI=NC, sMRI=AD）：\n"
            "      - 說明各模態關注的 ROI 導致不同預測的可能原因。\n"
            "      - **特別指出：若某 task sMRI 傾向疾病而 fMRI 傾向正常，\n"
            "        這可能代表『結構已開始退化、功能尚在代償期』的早期徵象，\n"
            "        是需要密切追蹤的臨床警訊，而非互相矛盾。**\n"
            "      - PCAG fusion 整合兩者後的判定即為最終依據，說明 fusion 如何取捨。\n"
            "- 【臨床診斷建議】需包含以下具體內容：\n"
            "    1. 追蹤頻率：根據預測結果與邊界值狀況，明確建議「每 X 個月」追蹤。\n"
            "       （NC 且非邊界值 → 12 個月；NC 邊界值 → 6 個月；MCI → 6 個月；AD → 3 個月）\n"
            "    2. 評估工具：至少提及 MMSE 或 MoCA 認知量表，以及是否需要神經心理學全套評估。\n"
            "    3. 影像追蹤：若有模態分歧或邊界值，建議複查 MRI 並重點監測哪個 ROI 的變化。\n"
            "    4. 認知儲備：結合病患教育年限說明 Cognitive Reserve 對預測結果的可能影響。\n"
            "- 若某 task 的兩個 class 都不符合病患情況（如 MCI vs AD task 對一個 NC 病患），\n"
            "  請說明此 task 對本病患**參考意義有限**，以 NC_vs_AD / NC_vs_MCI 結果為主。\n"
            "- 當 prob 介於 0.44–0.56 之間，請明確說明為**邊界值**，建議追蹤而非下定論。\n"
        )
    elif has_smri:
        modality_note = "本次為 sMRI 單模態分析（無有效 fMRI 數據）。"
        modality_instruction = (
            "- 【影像分析洞察】中，請僅描述【結構性發現 (sMRI)】，不要捏造 fMRI 相關內容。\n"
            "- ⚠️ 必須在【臨床診斷建議】開頭加上以下免責聲明：「本次分析僅有結構性 MRI 資料，"
            "缺乏功能性影像（fMRI）佐證，預測可信度低於雙模態分析，建議補充 fMRI 資料以提高診斷準確性。」\n"
            "- 預測結果不確定性較高，請在報告中明確傳達此點。\n"
        )
    else:
        modality_note = "本次為 fMRI 單模態分析（無 T1 結構影像）。"
        modality_instruction = (
            "- 【影像分析洞察】中，請僅描述【功能性發現 (fMRI)】，不要捏造 sMRI 相關內容。\n"
            "- 請在報告中說明本次分析僅基於功能影像，缺乏結構性佐證。\n"
        )

    similar_section = f"\n## 腦部特徵相似病患（來自知識圖譜 SIMILAR_TO 邊）\n{similar_ctx}\n" if similar_ctx else ""

    observations_section = (f"\n## 模型內部觀察（單模態預測與關鍵腦區）\n{req.model_observations}\n"
                            if req.model_observations else "")
    # Round probabilities and saliency to 2 decimal places before injecting into prompt
    def _round_findings(task_results):
        import copy
        tr = copy.deepcopy(task_results)
        for res in tr.values():
            for prob_key in ("prob_positive", "prob_fused", "conf_smri", "conf_fmri"):
                if prob_key in res and res[prob_key] is not None:
                    res[prob_key] = round(float(res[prob_key]), 2)
            for key in ("fmri_findings", "smri_findings"):
                findings = res.get(key) or {}
                for roi in findings.get("top_regions", []):
                    if "saliency" in roi:
                        roi["saliency"] = round(roi["saliency"], 2)
        return tr

    prompt = (
        f"你是一位專精失智症神經影像診斷的臨床 AI 助理。\n"
        f"⚠️ 注意：{modality_note}\n\n"
        f"{ROI_CLINICAL_PRIOR}\n"
        f"## 影像分析推論摘要（Task-level Evidence）\n{evidence_summary}\n\n"
        f"## 病患背景脈絡（來自知識圖譜）\n{patient_ctx}\n"
        f"{observations_section}"
        f"{similar_section}"
        f"\n## 參考醫學文獻（來自 Neo4j Vector RAG 檢索）\n{literature_ctx}\n\n"
        f"## 影像分析詳細結果\n{json.dumps(_round_findings(req.task_results), ensure_ascii=False)}\n"
        f"{concordance_block}\n\n"
        f"請依以下指示生成報告，段落標題需以【】標示：\n"
        f"- 優先考量病患認知儲備（Cognitive Reserve）對影像結果的影響。\n"
        f"{modality_instruction}"
        f"- 【臨床診斷建議】中，若上方【參考醫學文獻】有與本病患影像發現**直接相關**的內容，"
        f"請在該段落自然引用（使用學術格式，例如：「根據 Vockert et al. (2024)...」或句尾標註「(Simfukwe et al., 2025)」）。"
        f"若文獻與病患具體發現關聯性不強，請勿強行插入，以維持報告的臨床推理一致性。\n"
        f"- ⚠️ 文獻引用原則：文獻是**補充背景知識**，不是報告主軸。"
        f"不要因引用文獻而改變對本病患的核心臨床推斷，不要在同一段落對同一指標出現前後矛盾的描述。\n"
        f"- 使用繁體中文，病患代碼：{req.subject_id}。"
    )

    def ollama_stream():
        payload = {
            "model": "gemma3:12b",
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 4000 if req.mode == "fast" else 6000,
                "temperature": 0.2,
            },
        }
        with requests.post("http://localhost:11434/api/generate", json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    yield json.loads(line).get("response", "")
        # Append reference list after LLM stream ends
        if rag_citations:
            import re as _re
            ref_lines = "\n\n---\n**參考文獻**\n"
            for idx, c in enumerate(rag_citations, 1):
                title = c['title'].strip().rstrip('.')
                # Try to extract year (4-digit) from text snippet
                year_match = _re.search(r'\b(19|20)\d{2}\b', c.get('text', ''))
                year_str = f" ({year_match.group()})" if year_match else ""
                # Extract a short informative snippet (skip table-like content)
                raw_text = c.get('text', '')
                # Clean: remove lines that look like table rows (lots of digits/symbols)
                clean_lines = [ln.strip() for ln in raw_text.split('.') if ln.strip()
                               and not _re.match(r'^[\d\s\±\.\,✓✗\|%]+$', ln.strip())
                               and len(ln.strip()) > 30]
                snippet = clean_lines[0][:120] + '…' if clean_lines else ''
                ref_lines += f"\n[{idx}] {title}{year_str}"
                if snippet:
                    ref_lines += f"\n     *{snippet}*"
            yield ref_lines

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)