"""
Dual-Modal Inference Pipeline: fMRI + sMRI → GraphRAG Clinical Report
流程：FC matrix + T1 NIfTI → GNN / 3D ResNet 雙流推論 → GAT attention → Neo4j → Gemma 報告

Performance:
    fMRI only:  NC/AD AUC=0.784 ACC=79.3% | NC/MCI AUC=0.658 ACC=63.0% | MCI/AD AUC=0.687 ACC=72.0%
    sMRI only:  NC/AD AUC=0.825 ACC=85.0% | NC/MCI AUC=0.703 ACC=71.0% | MCI/AD AUC=0.848 ACC=83.7%
    Both:       Classification from sMRI + functional evidence from fMRI for report

Usage (自動雙模態，T1 會依 subject_id 自動尋找):
    python inference_pipeline.py \
        --matrix /path/sub_matrix_116.npy \
        --subject_id sub-011_S_6303_AD
"""

import os, sys, argparse, warnings
from dataclasses import dataclass
from typing import Optional, Union, List, Dict
import numpy as np
import torch
import torch.nn.functional as F
import requests
import nibabel as nib
from monai.networks.nets import resnet50 as monai_resnet50

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))

from train_hierarchical_gnn_kd import (
    extract_node_features,
    K_RATIO, TEACHER_PROBS_DIR,
)
# We will use FNPGNNv8_E13 from the new training script
from train_hierarchical_gnn_e13_gsl import FNPGNNv8_E13, GraphLearner

@dataclass
class ModalityInput:
    matrix_path: Optional[str]        # fMRI FC matrix .npy
    t1_path:     Optional[str]        # sMRI T1 .nii.gz (auto-found or explicit)
    subject_id:  str

@dataclass
class WorkerEvidence:
    modality:       str            # "fmri" or "smri"
    task:           str            # "NC_vs_AD" etc.
    prob_positive:  float          # P(positive class)
    prediction:     int            # 0 or 1 after threshold
    confidence:     str            # "high" / "medium" / "low"
    findings:       dict           # modality-specific evidence for RAG

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

SALIENCY_DIR = "/home/wei-chi/Data/script/results/saliency"


def compute_gradcam(model: torch.nn.Module, t1_tensor: torch.Tensor,
                    device: torch.device, target_class: int = 1) -> np.ndarray:
    """
    對 sMRI ResNet 計算 Grad-CAM 熱力圖。
    target_class=1 對應正類別（AD/MCI）。
    回傳 shape (96,96,96) 的 numpy array，值域 [0,1]。
    """
    activations: dict = {}
    gradients: dict = {}

    def _fwd_hook(module, inp, out):
        activations["feat"] = out

    def _bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0]

    fwd_handle = model.layer4.register_forward_hook(_fwd_hook)
    bwd_handle = model.layer4.register_full_backward_hook(_bwd_hook)

    try:
        t = t1_tensor.to(device)
        model.zero_grad()
        logits = model(t)
        logits[0, target_class].backward()

        act  = activations["feat"].detach()                     # (1,C,d,h,w)
        grad = gradients["feat"].detach()                       # (1,C,d,h,w)
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)        # (1,C,1,1,1)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))  # (1,1,d,h,w)

        cam = F.interpolate(cam, size=(96, 96, 96),
                            mode="trilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def save_saliency_nifti(cam: np.ndarray, subject_id: str, task: str) -> str:
    """將 Grad-CAM 陣列儲存為 NIfTI，回傳檔案路徑。"""
    os.makedirs(SALIENCY_DIR, exist_ok=True)
    # 2 mm 等向性仿射矩陣（與 96^3 尺寸對應）
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(cam.astype(np.float32), affine)
    fname = f"{subject_id}_{task}_gradcam.nii.gz"
    out_path = os.path.join(SALIENCY_DIR, fname)
    nib.save(img, out_path)
    return out_path


# ── 路徑設定 ─────────────────────────────────────────────────────────
CKPT_DIR      = os.path.join(TEACHER_PROBS_DIR, "gnn_checkpoints")
SMRI_CKPT_DIR = TEACHER_PROBS_DIR   # smri_resnet_v3_{task}_best.pth 放在這裡
SMRI_ROOT     = "/home/wei-chi/Data/ADNI_sMRI_Aligned_MNI"  # ADNI T1 根目錄
TPMIC_SMRI_ROOT = "/home/wei-chi/Model/sMRI_data_MultiModal_Aligned_MNI" # TPMIC T1 根目錄
TASKS         = [("NC", "AD"), ("NC", "MCI"), ("MCI", "AD")]
SEEDS         = [42, 123, 456]

INFERENCE_THRESHOLDS = {
    "fmri": {"NC_vs_AD": 0.522, "NC_vs_MCI": 0.398, "MCI_vs_AD": 0.232},
    "smri": {"NC_vs_AD": 0.500, "NC_vs_MCI": 0.500, "MCI_vs_AD": 0.500},
}

# ── T1 自動查找：啟動時掃描所有 NIfTI，建立 ID → 路徑的 lookup ──
import re as _re, glob as _glob

def _build_t1_lookup(adni_root: str, tpmic_root: str) -> dict:
    lookup = {}
    # ADNI scan
    if os.path.isdir(adni_root):
        for path in _glob.glob(os.path.join(adni_root, "**", "*_T1_MNI.nii.gz"), recursive=True):
            fname = os.path.basename(path)
            m = _re.match(r"(\d+_S_\d+)_T1_MNI\.nii\.gz", fname)
            if m:
                lookup[m.group(1)] = path
    # TPMIC scan (sub_XXXX_T1.nii.gz)
    if os.path.isdir(tpmic_root):
        for path in _glob.glob(os.path.join(tpmic_root, "**", "sub_*_T1.nii.gz"), recursive=True):
            fname = os.path.basename(path)
            m = _re.match(r"(sub_\d+)_T1\.nii\.gz", fname)
            if m:
                lookup[m.group(1)] = path
    return lookup

T1_LOOKUP: dict = _build_t1_lookup(SMRI_ROOT, TPMIC_SMRI_ROOT)
if T1_LOOKUP:
    print(f"[init] T1 lookup 建立完成，共 {len(T1_LOOKUP)} 筆 sMRI 影像。")


def find_t1_path(subject_id: str) -> str | None:
    """
    從 subject_id 中抽取 ID，到 T1_LOOKUP 中尋找對應的 T1 NIfTI 路徑。
    支持 ADNI (NNN_S_XXXX) 與 TPMIC (sub_XXXX) 格式。
    """
    # Try ADNI format
    m_adni = _re.search(r"(\d{3}_S_\d{4})", subject_id)
    if m_adni:
        return T1_LOOKUP.get(m_adni.group(1), None)
    
    # Try TPMIC format
    m_tpmic = _re.search(r"(sub_\d+)", subject_id)
    if m_tpmic:
        return T1_LOOKUP.get(m_tpmic.group(1), None)
    
    return None

# ── Patient-Centric GraphRAG ─────────────────────────────────────────
from graph_rag_retriever import get_patient_graph_context, retrieve_medical_literature, retrieve_multimodal

# ── Neo4j ────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
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
    if adj_raw.ndim == 3:
        adj_raw = adj_raw[0]

    n = adj_raw.shape[0]
    if n != 116:
        if n < 116:
            # 零填補至 116×116，保留已有的 ROI 連結資訊
            padded = np.zeros((116, 116), dtype=adj_raw.dtype)
            padded[:n, :n] = adj_raw
            print(f"  ⚠️  矩陣維度 {n}×{n}，已零填補至 116×116")
            adj_raw = padded
        else:
            raise ValueError(f"矩陣維度 {n}×{n} 超過 116，無法處理")

    if np.count_nonzero(adj_raw) == 0:
        raise ValueError("fMRI 矩陣為全零，影像資料可能已損壞或未正確處理。")

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


# ── 模態特有常數 ──────────────────────────────────────────────────
# AAL116 NETWORK_MAP from train_hierarchical_gnn_e13_gsl.py
E13_NETWORK_MAP = {
    'DMN': [34, 35, 66, 67, 64, 65, 22, 23, 24, 25],
    'SMN': [0, 1, 56, 57, 68, 69],
    'VIS': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    'VAN': [28, 29, 30, 31, 32, 33],
    'FPN': [6, 7, 58, 59, 60, 61],
    'LIM': [36, 37, 38, 39, 40, 41],
    'DAN': [10, 11, 14, 15],
    'SUB': [70, 71, 72, 73, 74, 75, 76, 77],
    'CER': list(range(90, 116)),
}

# ═══════════════════════════════════════════════════════════════════
# 2a. fMRI Worker
# ═══════════════════════════════════════════════════════════════════
def infer_fmri_task(x: torch.Tensor, adj: torch.Tensor, task_pair: tuple, device: torch.device) -> Optional[WorkerEvidence]:
    class_a, class_b = task_pair
    safe = f"{class_a}_vs_{class_b}"
    x, adj = x.to(device), adj.to(device)

    all_probs, all_saliency = [], []
    task_idx = 0 if safe == "NC_vs_AD" else 1 if safe == "NC_vs_MCI" else 2
    
    for seed in SEEDS:
        # E13 is multi-task, we use the unified checkpoint
        ckpt = os.path.join(CKPT_DIR, f"gnn_e13_seed{seed}.pt")
        if not os.path.exists(ckpt):
            continue
        model = FNPGNNv8_E13().to(device)
        try:
            model.load_state_dict(torch.load(ckpt, map_location=device))
        except Exception as e:
            print(f"  [fMRI] ⚠️  載入權重失敗 ({ckpt}): {e}")
            continue
        model.eval()

        x.requires_grad_(True)
        # E13 returns (logits_list, progression, flat)
        outputs = model(x, adj)
        logits_list = outputs[:3]
        logits = logits_list[task_idx]
        
        prob = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        
        target_score = logits[0, 1]
        model.zero_grad()
        target_score.backward()
        
        saliency = x.grad[0].abs().sum(dim=-1).detach().cpu().numpy()
        all_probs.append(prob)
        all_saliency.append(saliency)
        x.grad = None

    if not all_probs:
        print(f"  [fMRI] ⚠️  找不到 {safe} 的有效權重")
        return None

    avg_prob = np.mean(all_probs, axis=0)
    avg_saliency = np.mean(all_saliency, axis=0)
    prob_positive = float(avg_prob[1])

    top_k = 10
    top_idx = np.argsort(avg_saliency)[-top_k:][::-1]
    
    # Calculate network weights based on saliency
    net_weights = {}
    for net_name, indices in E13_NETWORK_MAP.items():
        net_weights[net_name] = float(np.mean(avg_saliency[indices]))
    
    top_network = max(net_weights, key=net_weights.get)
    top_region = AAL116_NAMES[top_idx[0]]
    
    findings = {
        "modality": "fMRI functional connectivity",
        "top_regions": [{"name": AAL116_NAMES[i], "saliency": float(avg_saliency[i])} for i in top_idx],
        "network_weights": net_weights,
        "saliency_116": avg_saliency.tolist(),
        "summary": f"Functional connectivity analysis identified abnormal patterns in {top_network}. The {top_region} showed highest saliency."
    }
    
    threshold = INFERENCE_THRESHOLDS["fmri"].get(safe, 0.5)
    prediction = 1 if prob_positive >= threshold else 0
    confidence = "high" if abs(prob_positive - threshold) > 0.2 else "medium" if abs(prob_positive - threshold) > 0.1 else "low"

    return WorkerEvidence(
        modality="fmri",
        task=safe,
        prob_positive=prob_positive,
        prediction=prediction,
        confidence=confidence,
        findings=findings
    )


# ═══════════════════════════════════════════════════════════════════
# 2b. sMRI Worker
# ═══════════════════════════════════════════════════════════════════
def infer_smri_task(t1_tensor: torch.Tensor, task_pair: tuple, device: torch.device,
                    subject_id: str = "unknown") -> Optional[WorkerEvidence]:
    class_a, class_b = task_pair
    safe = f"{class_a}_vs_{class_b}"

    ckpt_path = os.path.join(SMRI_CKPT_DIR, f"smri_resnet_v3_{safe}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"      [sMRI] ⚠️  找不到 checkpoint: {ckpt_path}")
        return None

    model = _build_smri_model().to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as e:
        print(f"      [sMRI] ⚠️  載入權重失敗: {e}")
        return None
    model.eval()

    # 前向推論（需啟用梯度以計算 Grad-CAM）
    t1 = t1_tensor.to(device)
    with torch.enable_grad():
        logits = model(t1)
        prob = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    prob_positive = float(prob[1])

    # 計算 Grad-CAM（對正類別，即 AD/MCI）
    saliency_path = None
    try:
        cam = compute_gradcam(model, t1_tensor, device, target_class=1)
        saliency_path = save_saliency_nifti(cam, subject_id, safe)
    except Exception as e:
        print(f"      [sMRI] ⚠️  Grad-CAM 計算失敗：{e}")

    threshold = INFERENCE_THRESHOLDS["smri"].get(safe, 0.5)
    findings = {
        "modality": "sMRI structural MRI",
        "saliency_path": saliency_path,
        "summary": f"Structural MRI analysis for {safe} task indicates a pattern more consistent with {class_b if prob_positive > threshold else class_a} (probability: {prob_positive:.3f})."
    }

    prediction = 1 if prob_positive >= threshold else 0
    confidence = "high" if abs(prob_positive - threshold) > 0.2 else "medium" if abs(prob_positive - threshold) > 0.1 else "low"

    return WorkerEvidence(
        modality="smri",
        task=safe,
        prob_positive=prob_positive,
        prediction=prediction,
        confidence=confidence,
        findings=findings
    )


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
    model_name = "gemma3:12b"
    print(f"  本地端私有模型: {model_name}")
    return model_name


def generate_report(subject_id: str, task_results: dict, kg_context: dict, model_name: str,
                    patient_context: str = "") -> str:
    
    # Extract fMRI and sMRI findings from task_results (using the first available task's findings)
    first_task = list(task_results.values())[0]
    fmri_findings = first_task.get("fmri_findings")
    smri_findings = first_task.get("smri_findings")

    # Use multi-modal retrieval
    literature_ctx = retrieve_multimodal(fmri_findings, smri_findings, patient_context)

    print("\n" + "="*40 + " [Debug: Multi-modal RAG Context] " + "="*40)
    print(literature_ctx)
    print("="*110 + "\n")
    
    # ── 模態分析摘要 ────────────────────────────────────────────────
    fmri_summary = ""
    smri_summary = ""
    concordance_lines = []

    for task_name, res in task_results.items():
        task_label = task_name.replace("_vs_", " vs ")
        f_findings = res["fmri_findings"]
        s_findings = res["smri_findings"]
        
        if f_findings:
            fmri_summary += f"- {task_label}: {f_findings.get('summary', '')}\n"
        
        if s_findings:
            smri_summary += f"- {task_label}: {s_findings.get('summary', '')}\n"

        # Check concordance
        f_pred = res.get("fmri_pred")
        s_pred = res.get("smri_pred")
        if f_pred is not None and s_pred is not None:
            if f_pred == s_pred:
                concordance_lines.append(f"  • {task_label}：結構與功能預測一致。")
            else:
                concordance_lines.append(f"  • {task_label}：⚠️ 分歧（fMRI={f_pred}, sMRI={s_pred}）。")

    concordance_report = "\n".join(concordance_lines) if concordance_lines else "（單模態分析，無一致性比對數據）"

    prompt = f"""你是一位專精於失智症神經影像診斷的臨床 AI 助理。
請根據以下多模態影像分析結果與病患背景脈絡，撰寫一份正式的繁體中文臨床神經影像報告。

病患代碼：{subject_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【背景脈絡】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{patient_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【參考醫學文獻】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{literature_ctx}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【影像分析數據】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[功能性影像 fMRI 發現]
{fmri_summary if fmri_summary else "（未取得有效 fMRI 數據）"}

[結構性影像 sMRI 發現]
{smri_summary if smri_summary else "（未提供 T1 影像）"}

[模態一致性分析]
{concordance_report}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【報告撰寫指示】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
請依以下結構撰寫報告，字數控制在 800 字以內，每個段落標題必須以【】標示：

**【背景脈絡】**
整合病患年齡、性別、教育年限對認知儲備的影響，說明其如何影響影像解讀。

**【影像分析洞察】**
- **結構性發現 (sMRI)**：詳細描述結構影像顯示的腦區變化（如萎縮情形）。
- **功能性發現 (fMRI)**：詳細描述功能連接異常模式（如網絡連接強度變化）。
- **模態一致性分析**：比較結構與功能發現，討論其臨床意義。
  - 若一致，註明：「結構與功能成像結果一致。」
  - 若不一致，說明可能的原因（如功能代償或結構領先功能）。

**【臨床診斷建議】**
- 綜合多模態證據給出診斷判斷。
- ⚠️ 【強制要求】：你必須運用上方【參考醫學文獻】中的內容來支持你的診斷建議，並在引用該句的句尾明確標註出處（例如：「[fMRI 相關文獻 1]」）。

請務必以「繁體中文（Traditional Chinese）」撰寫，格式清晰。
"""
    print(f"  正在呼叫本地端 {model_name} 生成多模態報告...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"[報告生成失敗: {e}]"


# ═══════════════════════════════════════════════════════════════════
# 5. 主推論流程與 Router
# ═══════════════════════════════════════════════════════════════════
def run_multimodal_inference(modality_input: ModalityInput, device: torch.device) -> dict:
    """
    Router: detect available modalities, decide classification strategy.
    Synthesizer: merge evidence for RAG.
    """
    # ── 前處理 ───────────────────────────────────────────────────────
    x, adj = None, None
    if modality_input.matrix_path:
        try:
            x, adj = preprocess_matrix(modality_input.matrix_path)
        except Exception as e:
            print(f"  [fMRI] ⚠️  前處理失敗：{e}")

    t1_tensor = None
    if modality_input.t1_path:
        try:
            t1_tensor = preprocess_t1(modality_input.t1_path)
        except Exception as e:
            print(f"  [sMRI] ⚠️  前處理失敗：{e}")

    results = {}
    for task_pair in TASKS:
        class_a, class_b = task_pair
        task_name = f"{class_a}_vs_{class_b}"
        
        # Always run fMRI if available (for RAG richness)
        fmri_ev = infer_fmri_task(x, adj, task_pair, device) if x is not None else None
        
        # Run sMRI if available
        smri_ev = infer_smri_task(t1_tensor, task_pair, device, modality_input.subject_id) if t1_tensor is not None else None

        if fmri_ev is None and smri_ev is None:
            continue

        # Classification Strategy: sMRI is primary when available (most accurate)
        primary_ev = smri_ev if smri_ev else fmri_ev
        
        # Build combined evidence for RAG
        evidence_list = [ev.findings for ev in [fmri_ev, smri_ev] if ev]
        
        results[task_name] = {
            "prediction":    primary_ev.prediction,
            "prob_positive": primary_ev.prob_positive,
            "modality_used": primary_ev.modality,
            "evidence":      evidence_list,
            "fmri_findings": fmri_ev.findings if fmri_ev else None,
            "smri_findings": smri_ev.findings if smri_ev else None,
            "fmri_pred":     fmri_ev.prediction if fmri_ev else None,
            "smri_pred":     smri_ev.prediction if smri_ev else None,
            "confidence":    primary_ev.confidence,
            "class_a":       class_a,
            "class_b":       class_b
        }
    return results

MATRIX_ROOTS = [
    "/home/wei-chi/Model/processed_116_matrices",
    "/home/wei-chi/Data/ADNI_processed_116_matrices"
]

def find_matrix_path(subject_id: str) -> str | None:
    """
    從 subject_id 自動尋找對應的 .npy 矩陣路徑。
    """
    for root in MATRIX_ROOTS:
        if not os.path.isdir(root):
            continue
        # 搜尋包含 subject_id 的 .npy 檔案
        pattern = os.path.join(root, f"*{subject_id}*.npy")
        matches = _glob.glob(pattern)
        if matches:
            return matches[0]
    return None

def run_inference(matrix_path: str = None, subject_id: str = "unknown", t1_path: str = None):
    # 強制 CPU，避免與背景 Ollama 搶 VRAM
    device = torch.device("cpu")

    # 若未提供 matrix_path，嘗試從 subject_id 自動查找
    if not matrix_path or not os.path.exists(matrix_path):
        auto_matrix = find_matrix_path(subject_id)
        if auto_matrix:
            print(f"  [auto] fMRI 矩陣自動配對成功：{auto_matrix}")
            matrix_path = auto_matrix
        else:
            print(f"  ❌ 錯誤：找不到 ID 為 {subject_id} 的 fMRI 矩陣。")
            return

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

    input_data = ModalityInput(matrix_path=matrix_path, t1_path=t1_path, subject_id=subject_id)
    
    print("\n[1/3] 執行多模態推論流程...")
    task_results = run_multimodal_inference(input_data, device)
    
    if not task_results:
        print("  ⚠️  推論失敗，未取得任何結果。")
        return

    # Extract all top ROIs for KG query (from fMRI findings)
    all_top_rois = set()
    for res in task_results.values():
        if res["fmri_findings"]:
            for roi_info in res["fmri_findings"]["top_regions"][:5]:
                all_top_rois.add(roi_info["name"])

    # ── Neo4j 查詢 ──────────────────────────────────────────────────
    print("\n[2/3] 查詢 Neo4j 知識圖譜...")
    top_roi_names = list(all_top_rois)
    if not top_roi_names:
        kg_context = {"roi_details": [], "network_summary": []}
    else:
        kg_context = query_knowledge_graph(top_roi_names, ("NC", "AD"))
        if "error" in kg_context:
            print(f"  ⚠️  Neo4j 查詢失敗：{kg_context['error']}")
            kg_context = {"roi_details": [], "network_summary": []}
        else:
            print(f"  查詢到 {len(kg_context['roi_details'])} 個 ROI，{len(kg_context['network_summary'])} 個網路摘要")

    # ── 報告生成 ────────────────────────────────────────────────────
    print("\n[3/3] 生成多模態臨床報告（Ollama / Gemma 4）...")
    patient_ctx = get_patient_graph_context(subject_id)
    ollama_model = init_ollama()
    
    report = generate_report(subject_id, task_results, kg_context, ollama_model, patient_ctx)

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
    parser.add_argument("--matrix",     required=False, help="fMRI FC matrix (.npy, 116×116)")
    parser.add_argument("--subject_id", default="unknown", help="Subject identifier")
    parser.add_argument("--t1_image",   default=None,   help="sMRI T1 影像路徑 (.nii.gz 或 .npy)，選填")
    args = parser.parse_args()

    run_inference(args.matrix, args.subject_id, args.t1_image)
