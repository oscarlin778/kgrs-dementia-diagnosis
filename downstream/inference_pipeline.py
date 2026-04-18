"""
Dual-Modal Inference Pipeline: fMRI + sMRI → GraphRAG Clinical Report
流程：FC matrix + T1 NIfTI → GNN / 3D ResNet 雙流推論 → GAT attention → Neo4j → Gemma 報告

Usage (自動雙模態，T1 會依 subject_id 自動尋找):
    python inference_pipeline.py \
        --matrix /path/sub_matrix_116.npy \
        --subject_id sub-011_S_6303_AD

Usage (手動指定 T1):
    python inference_pipeline.py \
        --matrix /path/sub_matrix_116.npy \
        --t1_image /path/sub_T1.nii.gz \
        --subject_id sub-011_S_6303_AD

Usage (fMRI only，找不到 T1 時自動 fallback):
    python inference_pipeline.py \
        --matrix /path/sub_matrix_116.npy \
        --subject_id sub_0001_NC
"""

import os, sys, argparse, warnings
import numpy as np
import torch
import torch.nn.functional as F
import requests
from monai.networks.nets import resnet50 as monai_resnet50

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))

from train_hierarchical_gnn_kd import (
    FNPGNNv8_KD, extract_node_features,
    K_RATIO, TEACHER_PROBS_DIR, NETWORK_MAP
)

# ── sMRI ResNet 架構（與 train_3d_resnet_teacher.py 的 build_model 完全一致）──
class _SEBlock3D(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channels // reduction, channels, bias=False),
            torch.nn.Sigmoid(),
        )
    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1, 1)

def _build_smri_model():
    model = monai_resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)
    model.layer4 = torch.nn.Sequential(model.layer4, _SEBlock3D(2048, reduction=16))
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.15),
        torch.nn.Linear(512, 2),
    )
    return model

# ── 路徑設定 ─────────────────────────────────────────────────────────
CKPT_DIR      = os.path.join(TEACHER_PROBS_DIR, "gnn_checkpoints")
SMRI_CKPT_DIR = TEACHER_PROBS_DIR   # smri_resnet_v3_{task}_best.pth 放在這裡
SMRI_ROOT     = "/home/wei-chi/Data/ADNI_sMRI_Aligned_MNI"  # T1 根目錄
TASKS         = [("NC", "AD"), ("NC", "MCI"), ("MCI", "AD")]
SEEDS         = [42, 123, 456]

# ── T1 自動查找：啟動時掃描所有 NIfTI，建立 ADNI_ID → 路徑的 lookup ──
import re as _re, glob as _glob

def _build_t1_lookup(smri_root: str) -> dict:
    lookup = {}
    for path in _glob.glob(os.path.join(smri_root, "**", "*_T1_MNI.nii.gz"), recursive=True):
        fname = os.path.basename(path)                  # e.g. 011_S_6303_T1_MNI.nii.gz
        m = _re.match(r"(\d+_S_\d+)_T1_MNI\.nii\.gz", fname)
        if m:
            lookup[m.group(1)] = path                   # key = "011_S_6303"
    return lookup

T1_LOOKUP: dict = _build_t1_lookup(SMRI_ROOT) if os.path.isdir(SMRI_ROOT) else {}
if T1_LOOKUP:
    print(f"[init] T1 lookup 建立完成，共 {len(T1_LOOKUP)} 筆 sMRI 影像。")


def find_t1_path(subject_id: str) -> str | None:
    """
    從 subject_id 中抽取 ADNI ID（格式 NNN_S_XXXX），
    到 T1_LOOKUP 中尋找對應的 T1 NIfTI 路徑。
    找不到時回傳 None（不報錯，退回單模態）。
    """
    m = _re.search(r"(\d{3}_S_\d{4})", subject_id)
    if not m:
        return None
    adni_id = m.group(1)
    return T1_LOOKUP.get(adni_id, None)

# ── Neo4j ────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://192.168.51.183:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ── 雙模態融合權重（可依 validation 結果調整）────────────────────────
FMRI_WEIGHT = 0.5
SMRI_WEIGHT = 0.5

# ── Teacher ensemble 權重（訓練時選出的最佳值）───────────────────────
ENSEMBLE_WEIGHTS = {
    "NC_vs_AD":  0.5,
    "NC_vs_MCI": 0.6,
    "MCI_vs_AD": 0.5,
}

AAL116_NAMES = [
    'Precentral_L','Precentral_R','Frontal_Sup_L','Frontal_Sup_R',
    'Frontal_Sup_Orb_L','Frontal_Sup_Orb_R','Frontal_Mid_L','Frontal_Mid_R',
    'Frontal_Mid_Orb_L','Frontal_Mid_Orb_R','Frontal_Inf_Oper_L','Frontal_Inf_Oper_R',
    'Frontal_Inf_Tri_L','Frontal_Inf_Tri_R','Frontal_Inf_Orb_L','Frontal_Inf_Orb_R',
    'Rolandic_Oper_L','Rolandic_Oper_R','Supp_Motor_Area_L','Supp_Motor_Area_R',
    'Olfactory_L','Olfactory_R','Frontal_Sup_Med_L','Frontal_Sup_Med_R',
    'Frontal_Med_Orb_L','Frontal_Med_Orb_R','Rectus_L','Rectus_R',
    'Insula_L','Insula_R','Cingulum_Ant_L','Cingulum_Ant_R',
    'Cingulum_Mid_L','Cingulum_Mid_R','Cingulum_Post_L','Cingulum_Post_R',
    'Hippocampus_L','Hippocampus_R','ParaHippocampal_L','ParaHippocampal_R',
    'Amygdala_L','Amygdala_R','Calcarine_L','Calcarine_R',
    'Cuneus_L','Cuneus_R','Lingual_L','Lingual_R',
    'Occipital_Sup_L','Occipital_Sup_R','Occipital_Mid_L','Occipital_Mid_R',
    'Occipital_Inf_L','Occipital_Inf_R','Fusiform_L','Fusiform_R',
    'Postcentral_L','Postcentral_R','Parietal_Sup_L','Parietal_Sup_R',
    'Parietal_Inf_L','Parietal_Inf_R','SupraMarginal_L','SupraMarginal_R',
    'Angular_L','Angular_R','Precuneus_L','Precuneus_R',
    'Paracentral_Lob_L','Paracentral_Lob_R','Caudate_L','Caudate_R',
    'Putamen_L','Putamen_R','Pallidum_L','Pallidum_R',
    'Thalamus_L','Thalamus_R','Heschl_L','Heschl_R',
    'Temporal_Sup_L','Temporal_Sup_R','Temporal_Pole_Sup_L','Temporal_Pole_Sup_R',
    'Temporal_Mid_L','Temporal_Mid_R','Temporal_Pole_Mid_L','Temporal_Pole_Mid_R',
    'Temporal_Inf_L','Temporal_Inf_R','Cerebelum_Crus1_L','Cerebelum_Crus1_R',
    'Cerebelum_Crus2_L','Cerebelum_Crus2_R','Cerebelum_3_L','Cerebelum_3_R',
    'Cerebelum_4_5_L','Cerebelum_4_5_R','Cerebelum_6_L','Cerebelum_6_R',
    'Cerebelum_7b_L','Cerebelum_7b_R','Cerebelum_8_L','Cerebelum_8_R',
    'Cerebelum_9_L','Cerebelum_9_R','Cerebelum_10_L','Cerebelum_10_R',
    'Vermis_1_2','Vermis_3','Vermis_4_5','Vermis_6',
    'Vermis_7','Vermis_8','Vermis_9','Vermis_10',
]


# ═══════════════════════════════════════════════════════════════════
# 1a. fMRI 前處理：FC matrix → (x_feat, adj_norm)
# ═══════════════════════════════════════════════════════════════════
def preprocess_matrix(matrix_path: str):
    adj_raw = np.load(matrix_path)
    assert adj_raw.shape == (116, 116), f"Expected (116,116), got {adj_raw.shape}"

    adj_z = np.arctanh(np.clip(adj_raw, -0.999, 0.999))
    x_feat = extract_node_features(adj_z)

    adj_abs = np.abs(adj_z)
    np.fill_diagonal(adj_abs, 0)
    k = int(116 * K_RATIO)
    adj_mask = np.zeros_like(adj_z)
    for i in range(116):
        top_idx = np.argsort(adj_abs[i])[-k:]
        adj_mask[i, top_idx] = adj_z[i, top_idx]
    adj_mask = np.maximum(adj_mask, adj_mask.T)
    np.fill_diagonal(adj_mask, 1.0)
    rowsum = np.abs(adj_mask).sum(1)
    rowsum[rowsum == 0] = 1e-10
    d_mat = np.diag(np.power(rowsum, -0.5))
    adj_norm = (d_mat @ adj_mask @ d_mat).astype(np.float32)

    return (
        torch.FloatTensor(x_feat).unsqueeze(0),
        torch.FloatTensor(adj_norm).unsqueeze(0),
    )


# ═══════════════════════════════════════════════════════════════════
# 1b. sMRI 前處理：T1 NIfTI / .npy → (1, 1, D, H, W) tensor
# ═══════════════════════════════════════════════════════════════════
def preprocess_t1(t1_path: str) -> torch.Tensor:
    """完全複製 train_3d_resnet_teacher.py 的 val_transforms → (1,1,96,96,96)"""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        NormalizeIntensityd, CropForegroundd, Resized,
    )
    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
    ])
    data = val_transforms({"image": t1_path})
    return data["image"].unsqueeze(0)   # (1, 1, 96, 96, 96)


# ═══════════════════════════════════════════════════════════════════
# 2a. fMRI 推論（GNN + teacher ensemble）
# ═══════════════════════════════════════════════════════════════════
def infer_fmri_task(x, adj, task_pair, device, subject_id):
    class_a, class_b = task_pair
    safe = f"{class_a}_vs_{class_b}"
    x, adj = x.to(device), adj.to(device)

    all_probs, all_saliency = [], []
    for seed in SEEDS:
        ckpt = os.path.join(CKPT_DIR, f"gnn_{safe}_seed{seed}.pt")
        if not os.path.exists(ckpt):
            continue
        model = FNPGNNv8_KD().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        # 🌟 核心修改 1：開啟梯度追蹤
        x.requires_grad_(True)
        
        logits, _, _ = model(x, adj, return_attn=True)
        prob = F.softmax(logits, dim=1).detach().cpu().numpy()[0]   # (2,)

        # 🌟 核心修改 2：計算對目標類別 (class_b, 也就是 Index 1) 的梯度
        target_score = logits[0, 1] # 疾病的機率分數
        model.zero_grad()
        target_score.backward() # 反向傳播計算梯度

        # 🌟 核心修改 3：取輸入特徵梯度的絕對值總和，這就是該腦區對預測的「貢獻度」
        # shape: (116, feature_dim) -> sum -> (116,)
        saliency = x.grad[0].abs().sum(dim=-1).detach().cpu().numpy() 
        
        all_probs.append(prob)
        all_saliency.append(saliency)
        
        # 重置梯度
        x.grad = None

    if not all_probs:
        return None, None, None

    avg_prob = np.mean(all_probs, axis=0)
    avg_saliency = np.mean(all_saliency, axis=0) # 現在我們用 Saliency 取代 Attention

    top_k = 10
    top_idx = np.argsort(avg_saliency)[-top_k:][::-1]
    
    # 這裡回傳的 avg_saliency 將會讓前端雷達圖出現截然不同的形狀！
    return float(avg_prob[1]), top_idx.tolist(), avg_saliency


# ═══════════════════════════════════════════════════════════════════
# 2b. sMRI 推論（3D ResNet）
# ═══════════════════════════════════════════════════════════════════
def infer_smri_task(t1_tensor: torch.Tensor, task_pair: tuple, device) -> float:
    """
    MONAI resnet50 推論，回傳 P(class_b)。
    checkpoint 的 class_map = {class_a:0, class_b:1}，與 GNN 一致，不需 flip。
    """
    class_a, class_b = task_pair
    safe = f"{class_a}_vs_{class_b}"

    ckpt_path = os.path.join(SMRI_CKPT_DIR, f"smri_resnet_v3_{safe}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"      [sMRI] ⚠️  找不到 checkpoint: {ckpt_path}")
        return None

    model = _build_smri_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(t1_tensor.to(device))
        prob = F.softmax(logits, dim=1).cpu().numpy()[0]  # [P(class_a), P(class_b)]

    prob_class_b = float(prob[1])
    print(f"      [sMRI] P({class_b}) = {prob_class_b*100:.1f}%")
    return prob_class_b


# ═══════════════════════════════════════════════════════════════════
# 3. Neo4j 查詢
# ═══════════════════════════════════════════════════════════════════
def query_knowledge_graph(top_roi_names: list, task_pair: tuple) -> dict:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return {"error": "neo4j 套件未安裝"}
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        return {"error": str(e)}

    class_a, class_b = task_pair
    task_str = f"{class_a}_vs_{class_b}"
    results = {"roi_details": [], "network_summary": []}

    with driver.session() as session:
        for roi_name in top_roi_names:
            rec = session.run("""
                MATCH (r:ROI {name: $name})
                OPTIONAL MATCH (r)-[:BELONGS_TO]->(n:BrainNetwork)
                OPTIONAL MATCH (n)-[a:ATTENTION_DIFF {task: $task}]->(d:DiseaseStage)
                RETURN r.function AS func, r.ad_relevance AS relevance, r.notes AS notes,
                       n.abbr AS network, n.fullname AS net_fullname,
                       a.diff_value AS attn_diff, a.interpretation AS interpretation
            """, name=roi_name, task=task_str).single()
            if rec:
                results["roi_details"].append({
                    "roi": roi_name, "function": rec["func"],
                    "ad_relevance": rec["relevance"], "notes": rec["notes"],
                    "network": rec["network"], "net_fullname": rec["net_fullname"],
                    "attn_diff": rec["attn_diff"], "interpretation": rec["interpretation"],
                })

        net_recs = session.run("""
            MATCH (n:BrainNetwork)-[a:ATTENTION_DIFF {task: $task}]->(d:DiseaseStage)
            RETURN n.abbr AS network, n.fullname AS fullname,
                   a.diff_value AS diff, a.interpretation AS interp
            ORDER BY abs(a.diff_value) DESC
        """, task=task_str)
        for r in net_recs:
            results["network_summary"].append({
                "network": r["network"], "fullname": r["fullname"],
                "diff": r["diff"], "interp": r["interp"],
            })

    driver.close()
    return results


# ═══════════════════════════════════════════════════════════════════
# 4. 雙模態報告生成（本地端 Ollama / Gemma 4）
# ═══════════════════════════════════════════════════════════════════
def init_ollama():
    model_name = "gemma4:26b"
    print(f"  本地端私有模型: {model_name}")
    return model_name


def generate_report(subject_id: str, task_results: dict, kg_context: dict, model_name: str) -> str:
    # ── fMRI 分類摘要 ────────────────────────────────────────────────
    fmri_lines, smri_lines, fused_lines = [], [], []
    for task_str, res in task_results.items():
        if res["prob_fused"] is None:
            continue
        p_fmri  = res["prob_fmri"]
        p_smri  = res["prob_smri"]
        p_fused = res["prob_fused"]
        cb      = res["class_b"]

        def risk(p):
            if p is None: return "N/A"
            return f"{p*100:.1f}% ({'HIGH' if p > 0.6 else 'MODERATE' if p > 0.4 else 'LOW'})"

        fmri_lines.append(f"- {task_str}: P({cb}) = {risk(p_fmri)}")
        smri_lines.append(f"- {task_str}: P({cb}) = {risk(p_smri)}")
        fused_lines.append(f"- {task_str}: P({cb}) = {risk(p_fused)}")

    # ── Top ROI 摘要 ─────────────────────────────────────────────────
    roi_lines = []
    for detail in kg_context.get("roi_details", [])[:8]:
        sign = "↑disease" if (detail.get("attn_diff") or 0) > 0 else "↑control"
        roi_lines.append(
            f"  • {detail['roi']} [{detail.get('network','?')}] "
            f"({sign}, relevance={detail.get('ad_relevance','?')})\n"
            f"    → {detail.get('function','')}"
        )

    # ── 網路層級摘要 ─────────────────────────────────────────────────
    net_lines = []
    task_keys = list(task_results.keys())
    for net in kg_context.get("network_summary", []):
        bar = "█" * int(abs(net["diff"] or 0) * 10)
        ref_task = task_keys[0] if task_keys else "NC vs AD"
        parts = ref_task.split(" vs ")
        direction = f"↑{parts[1]}" if (net["diff"] or 0) > 0 else f"↑{parts[0]}"
        net_lines.append(f"  {net['network']:6s} {direction}: {bar} ({net['diff']:+.2f})")

    has_smri = any(r["prob_smri"] is not None for r in task_results.values())
    modality_note = "雙模態融合 (fMRI-GNN + sMRI-ResNet)" if has_smri else "單模態 (fMRI-GNN only，sMRI 未提供)"

    prompt = f"""You are a clinical neuroimaging AI assistant specializing in dementia diagnosis.
Analyze the multimodal brain imaging results for subject: {subject_id}
Analysis mode: {modality_note}

## [模態一] 功能性影像 fMRI — GNN 分類結果
{chr(10).join(fmri_lines) if fmri_lines else "（未取得）"}

## [模態二] 結構性影像 sMRI — 3D ResNet 分類結果
{chr(10).join(smri_lines) if smri_lines else "（未提供 T1 影像，此欄位為 N/A）"}

## [融合決策] 雙模態加權平均 (fMRI×{FMRI_WEIGHT} + sMRI×{SMRI_WEIGHT})
{chr(10).join(fused_lines) if fused_lines else "（同 fMRI 結果）"}

## Top Salient Brain Regions (GAT Attention, fMRI)
{chr(10).join(roi_lines)}

## Network-level Attention Difference
{chr(10).join(net_lines)}

## Task
Based on the above multimodal results, generate a concise clinical neuroimaging report:

1. **整體評估 (Overall Assessment)**：綜合雙模態融合機率，判斷最可能的疾病階段。
2. **關鍵發現 (Key Findings)**：
   - 結構性發現 (sMRI)：描述 ResNet 推斷的大腦結構性改變，重點提及海馬迴、皮質萎縮等。
   - 功能性發現 (fMRI)：描述 GNN/GAT 偵測到的功能性連結異常腦區與網路。
3. **雙模態整合解釋 (Integrative Interpretation)**：
   - 討論結構萎縮 (sMRI) 與功能性網路代償/衰退 (fMRI) 之間的關聯。
   - 特別指出：若 fMRI 偵測到小腦或 VAN 的代償性活化，但 sMRI 同時顯示海馬萎縮，
     這往往代表患者正處於 AD 初期的功能重塑階段，而非真正的 MCI。
4. **信心水準與限制 (Confidence & Limitations)**：提及模型準確率並給出保守聲明。
5. **後續建議 (Recommended Follow-up)**：建議下一步診斷步驟。

請務必以「繁體中文 (Traditional Chinese)」撰寫正式臨床報告，字數控制在 600 字以內。
"""

    print(f"  正在將雙模態特徵送入本地端 {model_name} 進行推理...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"[本地端模型生成失敗: {e}]\n\n請確認 Ollama 服務是否在背景運行。"


# ═══════════════════════════════════════════════════════════════════
# 5. 主推論流程
# ═══════════════════════════════════════════════════════════════════
def run_inference(matrix_path: str, subject_id: str, t1_path: str = None):
    # 強制 CPU，避免與背景 Ollama 搶 VRAM
    device = torch.device("cpu")

    # 若未手動指定 T1，嘗試從 subject_id 自動查找
    if t1_path is None:
        t1_path = find_t1_path(subject_id)
        if t1_path:
            print(f"  [auto] T1 自動配對成功：{t1_path}")

    print(f"\n{'='*62}")
    print(f"  Subject : {subject_id}")
    print(f"  fMRI    : {matrix_path}")
    print(f"  sMRI    : {t1_path if t1_path else '(未找到 T1，單模態模式)'}")
    print(f"  Device  : {device}")
    print(f"{'='*62}")

    # ── 前處理 ───────────────────────────────────────────────────────
    print("\n[1/4] 前處理影像...")
    x, adj = preprocess_matrix(matrix_path)

    t1_tensor = None
    if t1_path:
        try:
            t1_tensor = preprocess_t1(t1_path)
            print(f"  sMRI tensor shape: {t1_tensor.shape}")
        except Exception as e:
            print(f"  ⚠️  sMRI 前處理失敗：{e}  → 退回單模態模式")
            t1_tensor = None

    # ── 雙流推論 ────────────────────────────────────────────────────
    print("\n[2/4] GNN + ResNet 推論（3 tasks）...")
    task_results = {}
    all_top_rois = set()

    for task_pair in TASKS:
        class_a, class_b = task_pair
        task_str = f"{class_a} vs {class_b}"
        safe = f"{class_a}_vs_{class_b}"
        print(f"  [{task_str}]")

        # fMRI 推論
        prob_fmri, top_idx, attn = infer_fmri_task(x, adj, task_pair, device, subject_id)

        # sMRI 推論（若有 T1）
        prob_smri = None
        if t1_tensor is not None:
            prob_smri = infer_smri_task(t1_tensor, task_pair, device)

        # 決策融合
        if prob_fmri is not None and prob_smri is not None:
            prob_fused = FMRI_WEIGHT * prob_fmri + SMRI_WEIGHT * prob_smri
            modal_tag = "fused"
        elif prob_fmri is not None:
            prob_fused = prob_fmri
            modal_tag = "fMRI-only"
        else:
            prob_fused = None
            modal_tag = "failed"

        if prob_fused is not None and top_idx is not None:
            top_names = [AAL116_NAMES[i] for i in top_idx]
            all_top_rois.update(top_names[:5])
            smri_str = f" | sMRI P({class_b})={prob_smri*100:.1f}%" if prob_smri is not None else ""
            print(f"      fMRI P({class_b})={prob_fmri*100:.1f}%{smri_str} → Fused={prob_fused*100:.1f}% [{modal_tag}]")
            print(f"      Top ROI: {top_names[0]}, {top_names[1]}, {top_names[2]}")
        else:
            top_names = []

        task_results[task_str] = {
            "class_a": class_a, "class_b": class_b,
            "prob_fmri": prob_fmri, "prob_smri": prob_smri, "prob_fused": prob_fused,
            "top_rois": top_names, "attn": attn,
        }

    # ── Neo4j 查詢 ──────────────────────────────────────────────────
    print("\n[3/4] 查詢 Neo4j 知識圖譜...")
    top_roi_names = list(all_top_rois)[:10]
    if not top_roi_names:
        print("  ⚠️  無 checkpoint，請先訓練模型")
        return

    kg_context = query_knowledge_graph(top_roi_names, ("NC", "AD"))
    if "error" in kg_context:
        print(f"  ⚠️  Neo4j 查詢失敗：{kg_context['error']}")
        kg_context = {"roi_details": [], "network_summary": []}
    else:
        print(f"  查詢到 {len(kg_context['roi_details'])} 個 ROI，{len(kg_context['network_summary'])} 個網路摘要")

    # ── 報告生成 ────────────────────────────────────────────────────
    print("\n[4/4] 生成雙模態臨床報告（Ollama / Gemma 4）...")
    ollama_model = init_ollama()
    report = generate_report(subject_id, task_results, kg_context, ollama_model)

    print("\n" + "=" * 62)
    print("  CLINICAL REPORT")
    print("=" * 62)
    print(report)

    out_dir = "/home/wei-chi/Data/script/results/reports"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject_id}_report.txt")
    with open(out_path, "w") as f:
        f.write(f"Subject: {subject_id}\n")
        f.write(f"fMRI matrix: {matrix_path}\n")
        if t1_path:
            f.write(f"sMRI T1: {t1_path}\n")
        f.write("\n")
        f.write(report)
    print(f"\n  報告已儲存 → {out_path}")
    return report


# ═══════════════════════════════════════════════════════════════════
# 6. CLI 入口
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Modal fMRI+sMRI GNN Inference Pipeline")
    parser.add_argument("--matrix",     required=True,  help="fMRI FC matrix (.npy, 116×116)")
    parser.add_argument("--subject_id", default="unknown", help="Subject identifier")
    parser.add_argument("--t1_image",   default=None,   help="sMRI T1 影像路徑 (.nii.gz 或 .npy)，選填")
    args = parser.parse_args()

    run_inference(args.matrix, args.subject_id, args.t1_image)
