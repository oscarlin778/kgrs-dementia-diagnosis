"""
eval_report_quality.py
======================
對測試集病患批次產生「有 RAG」和「無 RAG」兩版診斷報告，
用兩位本地 LLM (Gemma3-12B + Qwen3.6-35B) 當 G-Eval judges 各打分四個維度，
量化 KG + RAG 對報告品質的貢獻，並做配對統計檢定。

評分維度（各 0–3）：
  1. Factual Accuracy   — 報告內容是否與分類結果一致
  2. Clinical Relevance — 醫療建議是否具臨床意義
  3. Completeness       — 是否涵蓋 fMRI、sMRI、診斷建議等所有必要段落
  4. Coherence          — 報告是否結構清晰、邏輯流暢

統計：
  - 每個 dimension 跑 Wilcoxon signed-rank（paired）→ p-value
  - Δ 用 2000-bootstrap 求 95% CI
  - 兩個 judge 同時跑，可比較 inter-judge agreement

執行：
  conda activate AD
  python eval_report_quality.py              # 跑全部 test patients
  python eval_report_quality.py --n 4        # 只跑每類前 4 位（debug 用）
"""
import argparse, json, os, re, sys, time, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from scipy.stats import wilcoxon

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

BASE_DIR    = Path(__file__).parent.parent
RES_DIR     = BASE_DIR / "results"
API_BASE    = "http://localhost:8081"
OLLAMA_URL  = "http://localhost:11434/api/generate"

JUDGES = [
    {"model": "gemma3:12b",  "label": "Gemma3",   "num_predict": 256},
    # Cross-family second judge (qwen3.6:35b no longer installed); Llama3.1-8B is available.
    {"model": "llama3.1:8b", "label": "Llama3.1", "num_predict": 256},
]

LABEL_MAP  = {0: "NC", 1: "MCI", 2: "AD"}
DIMENSIONS = ["factual_accuracy", "clinical_relevance", "completeness", "coherence"]
DIM_LABELS = ["Factual\nAccuracy", "Clinical\nRelevance", "Completeness", "Coherence"]
TASK_KEYS  = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]

N_BOOT = 2000

# AAL116 ROI names + network mapping (from inference_pipeline_v2.py, inlined to avoid heavy import)
NETWORK_MAP = {
    'DMN':   [34,35,66,67,64,65,22,23,24,25],
    'SMN':   [0,1,56,57,68,69],
    'VN':    [42,43,44,45,46,47,48,49,50,51,52,53],
    'SN':    [28,29,30,31,32,33],
    'FPN':   [6,7,58,59,60,61],
    'LN':    [36,37,38,39,40,41],
    'VAN':   [10,11,14,15],
    'BGN':   [70,71,72,73,74,75,76,77],
    'CereN': list(range(90, 116)),
}
NETWORK_FULL = {
    'DMN': 'Default Mode Network', 'SMN': 'Sensorimotor Network',
    'VN':  'Visual Network',       'SN':  'Salience Network',
    'FPN': 'Frontoparietal Network','LN': 'Limbic Network',
    'VAN': 'Ventral Attention Network','BGN':'Basal Ganglia Network',
    'CereN':'Cerebellar Network',
}
ROI_TO_NET = {node: net for net, nodes in NETWORK_MAP.items() for node in nodes}

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


def compute_fmri_findings(matrix_path: str, top_k: int = 8) -> dict | None:
    """FC connectivity-based ROI saliency + network grouping."""
    try:
        mat = np.load(matrix_path)
        if mat.ndim == 3:
            mat = mat[0]
        if mat.shape != (116, 116):
            return None
        mat = np.arctanh(np.clip(mat, -0.999, 0.999))
        np.fill_diagonal(mat, 0)
        sal = np.abs(mat).mean(axis=1)
        sal_mean = sal.mean()
        sal_norm = np.clip((sal - sal_mean) / (sal.max() - sal_mean + 1e-8), 0, 1)
        top_idx = np.argsort(sal_norm)[-top_k:][::-1]
        top_regions = [
            {
                "name":     AAL116_NAMES[i] if i < len(AAL116_NAMES) else f"ROI_{i}",
                "saliency": round(float(sal_norm[i]), 3),
                "network":  ROI_TO_NET.get(i, "Other"),
            }
            for i in top_idx
        ]
        # Network-level aggregated saliency
        net_scores = {}
        for net, idxs in NETWORK_MAP.items():
            vals = [sal_norm[i] for i in idxs if i < len(sal_norm)]
            if vals:
                net_scores[net] = float(np.mean(vals))
        top_networks = sorted(net_scores.items(), key=lambda x: -x[1])[:4]
        return {
            "top_regions":  top_regions,
            "top_networks": [{"abbr": n, "name": NETWORK_FULL[n],
                              "score": round(s, 3)} for n, s in top_networks],
        }
    except Exception as e:
        print(f"  [WARN] fmri findings failed for {matrix_path}: {e}")
        return None


# ── Load predictions ─────────────────────────────────────────────────────────
def load_predictions(use_ensemble: bool = False) -> dict:
    """
    讀 results/full_predictions_v2_nolabel{_ensemble}.npz —— 每個病患都有 3 個 task 的機率。
    回傳 {subject_id: {task: {"prob", "pred", "fmri_prob/pred", "smri_prob/pred"}}}
    """
    fname = "full_predictions_v2_nolabel_ensemble.npz" if use_ensemble else "full_predictions_v2_nolabel.npz"
    full_path = RES_DIR / fname
    if not full_path.exists():
        print(f"[WARN] {full_path} not found — falling back to per-task probs (incomplete!)")
        return _load_predictions_legacy()

    d = np.load(full_path, allow_pickle=True)
    pred_map = {}
    has_modality = "fmri_probs" in d.files
    for i, sid in enumerate(d["subject_ids"]):
        pred_map[sid] = {}
        for j, task in enumerate(d["tasks"]):
            fused_p = float(d["fused_probs"][i, j])
            if has_modality:
                fmri_p = float(d["fmri_probs"][i, j])
                smri_p = float(d["smri_probs"][i, j])
            else:
                fmri_p = None
                smri_p = None
            pred_map[sid][task] = {
                "prob":       fused_p,
                "pred":       int(fused_p >= 0.5),
                "true_label": int(d["labels"][i]),
                "fmri_prob":  fmri_p,
                "fmri_pred":  int(fmri_p >= 0.5) if fmri_p is not None else None,
                "smri_prob":  smri_p,
                "smri_pred":  int(smri_p >= 0.5) if smri_p is not None else None,
            }

    # 補充單模態預測：從 ablation fmri_only / smri_only npz 載入
    # 這些檔案只含各 task 的 test set 成員（不是全部 49 人）
    TASK_LIST = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]
    for task in TASK_LIST:
        for modality in ("fmri", "smri"):
            npz_path = RES_DIR / f"pcag_combat_{task}_probs_v2_{modality}_only_fmricombat_nolabel.npz"
            if not npz_path.exists():
                continue
            ab = np.load(npz_path, allow_pickle=True)
            for sid_ab, prob_ab in zip(ab["test_subject_ids"], ab["test_probs"]):
                if sid_ab in pred_map and task in pred_map[sid_ab]:
                    p = float(prob_ab)
                    pred_map[sid_ab][task][f"{modality}_prob"] = p
                    pred_map[sid_ab][task][f"{modality}_pred"] = int(p >= 0.5)
    return pred_map


def _load_predictions_legacy() -> dict:
    """Fallback: 用 per-task npz files（不完整，only test set members per task）"""
    pred_map = {}
    def _load(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        return {sid: {"prob": float(p), "pred": int(p >= 0.5), "label": int(l)}
                for sid, p, l in zip(d["test_subject_ids"], d["test_probs"], d["test_labels"])}
    for task in TASK_KEYS:
        fused_path = RES_DIR / f"pcag_combat_{task}_probs_v2_fmricombat_nolabel.npz"
        fmri_path  = RES_DIR / f"pcag_combat_{task}_probs_v2_fmri_only_fmricombat_nolabel.npz"
        smri_path  = RES_DIR / f"pcag_combat_{task}_probs_v2_smri_only_fmricombat_nolabel.npz"
        if not fused_path.exists():
            continue
        fused = _load(fused_path); fmri = _load(fmri_path) if fmri_path.exists() else {}; smri = _load(smri_path) if smri_path.exists() else {}
        for sid, info in fused.items():
            pred_map.setdefault(sid, {})[task] = {
                "prob": info["prob"], "pred": info["pred"], "true_label": info["label"],
                "fmri_prob": fmri.get(sid, {}).get("prob"), "fmri_pred": fmri.get(sid, {}).get("pred"),
                "smri_prob": smri.get(sid, {}).get("prob"), "smri_pred": smri.get(sid, {}).get("pred"),
            }
    return pred_map


def load_matrix_paths() -> dict:
    df = pd.read_csv(BASE_DIR / "pcag_test_aligned_v2.csv")
    return dict(zip(df["subject_id"], df["matrix_path"]))


# ── Load model interpretability caches (R-1, R-2, R-3 outputs) ──────────────
def load_interpretability_caches() -> dict:
    """
    載入三份模型可解釋性快取：
      - PCAG attention (R-1)
      - GradCAM ROI (R-2)
      - GAT attention rollout (R-3)
    回傳 {(subject_id, task): {pcag, gradcam, gat}}
    """
    caches = {"pcag": None, "brainiac": None, "gat": None}
    files = {
        # F-Fix: 改用 no-label ComBat 重訓模型抽出的 attention
        # GradCAM 已廢棄（來自舊 ResNet teacher），改用 BrainIAC ViT attention
        "pcag":     RES_DIR / "pcag_attention_v2_nolabel.npz",
        "brainiac": RES_DIR / "brainiac_roi_v2.npz",
        "gat":      RES_DIR / "gat_attention_v2_nolabel.npz",
    }
    for k, fp in files.items():
        if fp.exists():
            caches[k] = dict(np.load(fp, allow_pickle=True))
        else:
            print(f"[WARN] missing cache: {fp}")

    # index by subject
    indexed = {"pcag": {}, "brainiac": {}, "gat": {}}
    for k, c in caches.items():
        if c is None:
            continue
        for i, sid in enumerate(c["subject_ids"]):
            indexed[k][str(sid)] = i

    return caches, indexed


NETWORK_LABELS = ['DMN', 'SMN', 'VN', 'SN', 'FPN', 'LN', 'VAN', 'BGN', 'CereN']
NETWORK_FULL_NAMES = {
    'DMN': 'Default Mode Network', 'SMN': 'Sensorimotor Network',
    'VN': 'Visual Network', 'SN': 'Salience Network',
    'FPN': 'Frontoparietal Network', 'LN': 'Limbic Network',
    'VAN': 'Ventral Attention Network', 'BGN': 'Basal Ganglia Network',
    'CereN': 'Cerebellar Network',
}


def get_roi_network(roi_idx: int) -> str:
    """ROI index → 網路 abbr"""
    return ROI_TO_NET.get(roi_idx, "Other")


def build_observation_block(sid: str, caches: dict, indexed: dict,
                            tasks_info: dict) -> str:
    """
    根據三份 cache 組合 model_observations 字串。
    tasks_info: {task: {fmri_pred, smri_pred, fused_pred, fused_prob}}

    ⚠️ 注意：不可在此處放入「真實標籤」— LLM 不應該知道真實答案，否則報告會 cheating。
    真實標籤只用於我們離線驗證 (records["true_label"])。
    """
    # task → (negative class, positive class) for pred=0/1 mapping
    TASK_CLASSES = {
        "NC_vs_AD":  ("NC", "AD"),
        "NC_vs_MCI": ("NC", "MCI"),
        "MCI_vs_AD": ("MCI", "AD"),
    }

    def _class_name(task, pred):
        if pred is None:
            return "N/A"
        neg, pos = TASK_CLASSES[task]
        return pos if pred == 1 else neg

    lines = []

    # 三個 task 各自的證據
    for task in ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]:
        task_display = task.replace("_vs_", " vs ")
        t_idx_in_cache = ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"].index(task)
        neg_cls, pos_cls = TASK_CLASSES[task]

        info = tasks_info.get(task)
        # 若該病患不屬於此 task 的兩類之一，no prob from PCAG
        if info is None or info.get("fused_pred") is None:
            lines.append(f"\n### {task_display}")
            lines.append(f"  ⓘ 此 task ({neg_cls} vs {pos_cls}) 模型未對本病患輸出機率，"
                         f"因為其分類類別不屬於此 binary 任務的範圍。")
            continue

        lines.append(f"\n### {task_display}（{neg_cls} vs {pos_cls}）")
        f_pred = info.get("fmri_pred"); s_pred = info.get("smri_pred")
        fused_prob = info.get("fused_prob"); fused_pred = info.get("fused_pred")
        f_prob = info.get('fmri_prob') or 0
        s_prob = info.get('smri_prob') or 0

        fused_name = _class_name(task, fused_pred)
        f_name = _class_name(task, f_pred)
        s_name = _class_name(task, s_pred)

        lines.append(f"  Fused (PCAG-ComBat) 預測={fused_name}, prob={fused_prob:.3f} → 判為 {fused_name}")
        if f_pred is None or s_pred is None:
            lines.append(f"  (單模態獨立預測結果不在此 ensemble 檔中，無法做模態分歧分析)")
        else:
            modality_status = (f"✓ 兩模態一致都判 {f_name}" if f_pred == s_pred
                               else f"⚠️ 兩模態分歧：fMRI→{f_name}, sMRI→{s_name}")
            lines.append(f"  fMRI-only 預測={f_name} (prob={f_prob:.3f}), "
                         f"sMRI-only 預測={s_name} (prob={s_prob:.3f})  → {modality_status}")

        # sMRI evidence (BrainIAC ViT attention rollout — replaces deprecated GradCAM)
        # 註：BrainIAC ViT 對所有 task 共用同一份 attention（whole-brain feature）
        if caches["brainiac"] is not None and sid in indexed["brainiac"]:
            i = indexed["brainiac"][sid]
            if not caches["brainiac"]["missing"][i]:
                top_r = caches["brainiac"]["top_rois"][i][:5]
                top_s = caches["brainiac"]["top_scores"][i][:5]
                roi_str = "、".join(
                    f"{AAL116_NAMES[r]}({get_roi_network(int(r))}, attn={s:.3f})"
                    for r, s in zip(top_r, top_s)
                )
                lines.append(f"  [sMRI 模型關注區 (BrainIAC ViT Attention Rollout)]：{roi_str}")

        # fMRI evidence (GAT rollout)
        if caches["gat"] is not None and sid in indexed["gat"]:
            i = indexed["gat"][sid]
            top_r = caches["gat"]["top_rois"][i, t_idx_in_cache][:5]
            top_s = caches["gat"]["top_roi_scores"][i, t_idx_in_cache][:5]
            net_attn = caches["gat"]["net_attention"][i, t_idx_in_cache]
            net_pairs = sorted(zip(NETWORK_LABELS, net_attn), key=lambda x: -x[1])[:3]

            roi_str = "、".join(
                f"{AAL116_NAMES[r]}({get_roi_network(int(r))})"
                for r in top_r
            )
            net_str = "、".join(f"{NETWORK_FULL_NAMES[n]} ({v*100:.1f}%)" for n, v in net_pairs)
            lines.append(f"  [fMRI 模型關注網路 (GNN Attention Rollout)]：{net_str}")
            lines.append(f"  [fMRI 關注 ROI]：{roi_str}")

        # PCAG attention summary (caveat: abstract space)
        if caches["pcag"] is not None and sid in indexed["pcag"]:
            i = indexed["pcag"][sid]
            entropy = float(caches["pcag"]["entropy"][i, t_idx_in_cache])
            top_dims = caches["pcag"]["top_active_dims"][i, t_idx_in_cache][:3]
            max_entropy = np.log(20)  # ln(20) ≈ 3.0
            focus_pct = (1 - entropy / max_entropy) * 100
            lines.append(f"  [PCAG 融合]：integration entropy={entropy:.2f} (focus={focus_pct:.0f}% — "
                         f"{'集中' if focus_pct > 30 else '分散'} integration); "
                         f"top fusion dims: {list(top_dims)}")

    if not lines:
        return ""

    header = "本病患的模型決策依據（per-task 模態獨立預測 + 影像關注區）：" + "".join(lines)

    # 模態分歧的整體說明
    disagree_tasks = []
    for task in ["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"]:
        info = tasks_info.get(task)
        if info and info.get("fmri_pred") is not None and info.get("smri_pred") is not None:
            if info["fmri_pred"] != info["smri_pred"]:
                neg_cls, pos_cls = TASK_CLASSES[task]
                disagree_tasks.append(f"{task.replace('_vs_', ' vs ')}（fMRI→{_class_name(task, info['fmri_pred'])}, sMRI→{_class_name(task, info['smri_pred'])}）")
    if disagree_tasks:
        header += "\n\n[整體模態分歧摘要] 以下 task(s) 出現分歧：" + "；".join(disagree_tasks)
        header += "\n  → 撰寫報告時，請對照上方各 task 的 BrainIAC 關注區 (sMRI) 與 GAT 關注網路 (fMRI)，"
        header += "說明兩模態各看到什麼具體的影像特徵而導致預測不同。"

    return header


def select_patients(pred_map: dict, n_per_class: int | None) -> list[str]:
    """所有病患都有 3 個 task 的機率（從 full_predictions_v2_nolabel.npz）"""
    df = pd.read_csv(BASE_DIR / "pcag_test_aligned_v2.csv")
    selected = []
    for label in [0, 1, 2]:
        pool = [sid for sid in df[df["label"] == label]["subject_id"]
                if sid in pred_map and len(pred_map[sid]) == 3]
        if n_per_class is None:
            selected.extend(pool)
        else:
            selected.extend(pool[:n_per_class])
    return selected


def build_inference_payload(subject_id: str, pred_map: dict,
                            matrix_paths: dict,
                            caches: dict | None = None,
                            indexed: dict | None = None) -> dict:
    tasks = pred_map.get(subject_id, {})

    task_results = {}
    tasks_info_for_obs = {}
    for task, info in tasks.items():
        display = task.replace("_vs_", " vs ")
        fmri_pred = info.get("fmri_pred")
        smri_pred = info.get("smri_pred")
        fmri_prob = info.get("fmri_prob")
        smri_prob = info.get("smri_prob")

        task_results[display] = {
            "prob_fused":    info["prob"],
            "prob_positive": info["prob"],   # api_server evidence_summary 用此欄位名
            "prediction":    info["pred"],
            "modality_used": "pcag_combat",
            "fusion_reason": "pcag_combat",
            "fmri_findings": None,
            "smri_findings": None,
            "fmri_pred":     fmri_pred,
            "smri_pred":     smri_pred,
            "fmri_prob":     fmri_prob,
            "smri_prob":     smri_prob,
            "modality_conflict": (fmri_pred is not None and smri_pred is not None
                                  and fmri_pred != smri_pred),
        }
        tasks_info_for_obs[task] = {
            "fmri_pred": fmri_pred, "smri_pred": smri_pred,
            "fmri_prob": fmri_prob, "smri_prob": smri_prob,
            "fused_pred": info["pred"], "fused_prob": info["prob"],
        }

    # OVO weighted vote
    p_nc_ad  = tasks.get("NC_vs_AD",  {}).get("prob", 0.5)
    p_nc_mci = tasks.get("NC_vs_MCI", {}).get("prob", 0.5)
    p_mci_ad = tasks.get("MCI_vs_AD", {}).get("prob", 0.5)
    score = {"NC": 0.0, "MCI": 0.0, "AD": 0.0}
    score["NC"]  += (1 - p_nc_ad) + (1 - p_nc_mci)
    score["MCI"] += p_nc_mci      + (1 - p_mci_ad)
    score["AD"]  += p_nc_ad       + p_mci_ad
    overall_pred = max(score, key=score.get)

    # 用新的 R-4 observation block：整合 R-2 GradCAM + R-3 GAT + R-1 PCAG
    observation_block = ""
    if caches is not None and indexed is not None:
        observation_block = build_observation_block(
            subject_id, caches, indexed, tasks_info_for_obs
        )

    # patient_context 設為簡短訊息 → bypass Neo4j 查詢（避免使用舊版預測值）
    return {
        "task_results":        task_results,
        "overall_prediction":  overall_pred,
        "patient_context":     "（本研究資料庫無此病患詳細人口統計資料；請使用下方「模型內部觀察」中的 AI 預測值為準）",
        "model_observations":  observation_block,
    }


# ── Generate report ──────────────────────────────────────────────────────────
def generate_report(subject_id: str, inference: dict, use_rag: bool) -> str:
    payload = {
        "subject_id":         subject_id,
        "task_results":       inference.get("task_results", {}),
        "overall_pred":       inference.get("overall_prediction", ""),
        "patient_context":    inference.get("patient_context", ""),
        "model_observations": inference.get("model_observations", ""),
        "use_rag":            use_rag,
    }
    try:
        full_text = ""
        with requests.post(f"{API_BASE}/api/v1/report/stream",
                           json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode())
                        full_text += chunk.get("response", "")
                    except Exception:
                        full_text += line.decode()
        return full_text.strip()
    except Exception as e:
        print(f"  [WARN] report generation failed (use_rag={use_rag}): {e}")
        return ""


# ── Judge ────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are a clinical AI report evaluator specializing in dementia diagnosis.

Rate the following diagnostic report on FOUR dimensions. Each score is 0–3:
  0 = Poor / missing
  1 = Partial / acceptable
  2 = Good / mostly complete
  3 = Excellent / fully satisfies criteria

Scoring criteria:
  factual_accuracy   : Does the report correctly reflect the classification result ({diagnosis})?
                       Are all stated findings consistent with the prediction scores?
  clinical_relevance : Are the clinical recommendations medically appropriate and specific to
                       the patient's predicted condition?
  completeness       : Does the report cover all required sections (imaging findings, diagnosis
                       suggestion, follow-up recommendation)?
  coherence          : Is the report logically structured, readable, and free of contradictions?

Respond ONLY with a JSON object like:
{{"factual_accuracy": <int>, "clinical_relevance": <int>, "completeness": <int>, "coherence": <int>, "reason": "<one sentence>"}}

--- REPORT START ---
{report}
--- REPORT END ---
"""

def judge_report(report: str, diagnosis: str, judge_model: str,
                 num_predict: int = 256) -> dict | None:
    if not report or len(report) < 50:
        return None
    prompt = JUDGE_PROMPT.format(diagnosis=diagnosis, report=report[:6000])
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": judge_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": num_predict}
        }, timeout=600)
        text = r.json().get("response", "")
        # 移除 <think>...</think> 區塊（Qwen3 reasoning trace）
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # 在剩餘文字中找 JSON object — 抓最後一個（reasoning 後的最終答案）
        matches = re.findall(r'\{[^{}]*"factual_accuracy"[^{}]*\}', text, re.DOTALL)
        if matches:
            return json.loads(matches[-1])
        # fallback: greedy match
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"  [WARN] judge {judge_model} failed: {e}")
    return None


# ── Statistical analysis ─────────────────────────────────────────────────────
def paired_stats(scores_with: list, scores_no: list) -> dict:
    """配對 Wilcoxon test + bootstrap 95% CI for Δ"""
    a = np.array(scores_with, dtype=float)
    b = np.array(scores_no,   dtype=float)
    n = len(a)
    if n < 2:
        return {"n": n, "mean_with": None, "mean_no": None,
                "delta": None, "ci_low": None, "ci_high": None, "p_value": None}

    diff = a - b
    mean_with = float(a.mean())
    mean_no   = float(b.mean())
    delta     = float(diff.mean())

    # bootstrap CI for mean delta
    rng = np.random.default_rng(42)
    boot = np.array([
        diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)
    ])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]).tolist()

    # Wilcoxon signed-rank（差為 0 時用 'zsplit' 處理）
    if np.allclose(diff, 0):
        p_value = 1.0
    else:
        try:
            stat, p_value = wilcoxon(a, b, zero_method="zsplit",
                                     alternative="two-sided")
            p_value = float(p_value)
        except Exception:
            p_value = None

    return {
        "n": n,
        "mean_with": round(mean_with, 3),
        "mean_no":   round(mean_no,   3),
        "delta":     round(delta,     3),
        "ci_low":    round(ci_low,    3),
        "ci_high":   round(ci_high,   3),
        "p_value":   round(p_value,   4) if p_value is not None else None,
    }


def aggregate_by_judge(records: list, judge_label: str) -> dict:
    """對指定 judge 收集每個病患在 with_rag / no_rag 下的分數，做配對分析"""
    per_pid = {}
    for r in records:
        if r["judge"] != judge_label or not r["scores"]:
            continue
        per_pid.setdefault(r["subject_id"], {})[r["condition"]] = r["scores"]

    result = {}
    for dim in DIMENSIONS:
        with_list, no_list = [], []
        for pid, conds in per_pid.items():
            if "with_rag" in conds and "no_rag" in conds:
                v1 = conds["with_rag"].get(dim)
                v2 = conds["no_rag"].get(dim)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    with_list.append(v1)
                    no_list.append(v2)
        result[dim] = paired_stats(with_list, no_list)
    return result


def inter_judge_agreement(records: list) -> dict:
    """每個 dimension 計算兩個 judge 的 Pearson 相關（在同一 condition+pid 上）"""
    by_key = {}
    for r in records:
        if not r["scores"]:
            continue
        key = (r["subject_id"], r["condition"])
        by_key.setdefault(key, {})[r["judge"]] = r["scores"]

    labels = [j["label"] for j in JUDGES]
    out = {}
    for dim in DIMENSIONS:
        pairs = []
        for key, jd in by_key.items():
            if labels[0] in jd and labels[1] in jd:
                v1 = jd[labels[0]].get(dim)
                v2 = jd[labels[1]].get(dim)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    pairs.append((v1, v2))
        if len(pairs) >= 2:
            arr = np.array(pairs, dtype=float)
            if arr[:, 0].std() > 0 and arr[:, 1].std() > 0:
                corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
            else:
                corr = None
            out[dim] = {"n_pairs": len(pairs), "pearson_r": round(corr, 3) if corr is not None else None}
        else:
            out[dim] = {"n_pairs": len(pairs), "pearson_r": None}
    return out


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ensemble", action="store_true", default=False,
                        help="Use 5-seed ensemble predictions (full_predictions_v2_nolabel_ensemble.npz)")
    parser.add_argument("--n", type=int, default=None,
                        help="patients per class (None = all test set)")
    parser.add_argument("--out", default=str(RES_DIR / "report_quality_results.json"))
    args = parser.parse_args()

    RES_DIR.mkdir(exist_ok=True)

    pred_map      = load_predictions(use_ensemble=args.use_ensemble)
    matrix_paths  = load_matrix_paths()
    caches, indexed = load_interpretability_caches()
    print(f"[caches loaded] pcag={len(indexed['pcag'])}, "
          f"brainiac={len(indexed['brainiac'])}, gat={len(indexed['gat'])}")
    patients      = select_patients(pred_map, args.n)
    judge_names = ", ".join(j["label"] for j in JUDGES)
    print(f"Evaluating {len(patients)} patients  (judges: {judge_names})\n")

    df_label = pd.read_csv(BASE_DIR / "pcag_test_aligned_v2.csv")\
                 .set_index("subject_id")["label"].to_dict()

    records = []
    t_start = time.time()

    for idx, pid in enumerate(patients, 1):
        label     = df_label.get(pid, -1)
        diagnosis = LABEL_MAP.get(label, "Unknown")
        elapsed   = time.time() - t_start
        print(f"[{idx}/{len(patients)}] {pid}  true={diagnosis}  ({elapsed:.0f}s)")

        inference = build_inference_payload(pid, pred_map, matrix_paths,
                                            caches=caches, indexed=indexed)
        overall   = inference["overall_prediction"]

        for use_rag in [True, False]:
            tag = "with_rag" if use_rag else "no_rag"
            print(f"  Generating ({tag})...", end=" ", flush=True)
            report = generate_report(pid, inference, use_rag)
            print(f"{len(report)} chars")

            for judge in JUDGES:
                print(f"  Judging ({tag} / {judge['label']})...", end=" ", flush=True)
                scores = judge_report(report, diagnosis,
                                      judge["model"], judge.get("num_predict", 256))
                if scores:
                    short = {"factual_accuracy": "FA", "clinical_relevance": "CR",
                             "completeness": "CP", "coherence": "CH"}
                    parts = [f"{short[d]}={scores.get(d)}" for d in DIMENSIONS]
                    print(" ".join(parts))
                else:
                    print("None")

                records.append({
                    "subject_id":   pid,
                    "true_label":   label,
                    "diagnosis":    diagnosis,
                    "overall_pred": overall,
                    "condition":    tag,
                    "judge":        judge["label"],
                    "report_len":   len(report),
                    "report":       report[:8000],
                    "scores":       scores,
                })
                time.sleep(0.5)

    # ── Aggregate per judge ──────────────────────────────────────────────────
    per_judge = {j["label"]: aggregate_by_judge(records, j["label"]) for j in JUDGES}
    agree     = inter_judge_agreement(records)

    summary = {
        "n_patients": len(patients),
        "judges":     [j["label"] for j in JUDGES],
        "per_judge":  per_judge,
        "inter_judge_agreement": agree,
    }

    output = {"summary": summary, "records": records}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {args.out}")

    # ── Print tables ─────────────────────────────────────────────────────────
    for judge_label, stats in per_judge.items():
        print(f"\n=== Judge: {judge_label} ===")
        print(f"{'Dimension':<22} {'With RAG':>10} {'No RAG':>10} {'Δ (95% CI)':>22} {'p':>8}")
        print("-" * 76)
        for d in DIMENSIONS:
            s = stats[d]
            if s["mean_with"] is None:
                print(f"{d:<22} {'N/A':>10} {'N/A':>10} {'N/A':>22} {'N/A':>8}")
                continue
            ci = f"[{s['ci_low']:+.2f}, {s['ci_high']:+.2f}]"
            print(f"{d:<22} {s['mean_with']:>10.3f} {s['mean_no']:>10.3f} "
                  f"{s['delta']:>+7.3f} {ci:>14}  {s['p_value']:>7.4f}")

    print(f"\n=== Inter-judge agreement (Pearson r) ===")
    for d in DIMENSIONS:
        a = agree[d]
        r = a["pearson_r"]
        r_s = f"{r:+.3f}" if r is not None else "N/A"
        print(f"  {d:<22}  r={r_s}  (n_pairs={a['n_pairs']})")

    make_figure(summary, args.out.replace(".json", ".png"))


# ── Figure ───────────────────────────────────────────────────────────────────
def make_figure(summary, out_path):
    """每個 judge 一張子圖：grouped bars + Δ panel with CI"""
    n_judges = len(summary["judges"])
    fig, axes = plt.subplots(n_judges, 2, figsize=(13, 5 * n_judges), facecolor="white")
    if n_judges == 1:
        axes = axes[np.newaxis, :]
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.07,
                        wspace=0.30, hspace=0.40)

    x = np.arange(len(DIMENSIONS))
    w = 0.32

    for row_i, judge_label in enumerate(summary["judges"]):
        stats = summary["per_judge"][judge_label]
        wr_v  = [stats[d]["mean_with"] or 0 for d in DIMENSIONS]
        nr_v  = [stats[d]["mean_no"]   or 0 for d in DIMENSIONS]
        delta = [stats[d]["delta"]     or 0 for d in DIMENSIONS]
        ci_l  = [stats[d]["ci_low"]    or 0 for d in DIMENSIONS]
        ci_h  = [stats[d]["ci_high"]   or 0 for d in DIMENSIONS]
        pvals = [stats[d]["p_value"]   for d in DIMENSIONS]

        ax1, ax2 = axes[row_i]

        # Panel A: grouped bars
        b1 = ax1.bar(x - w/2, wr_v, w, color="#2563eb", label="With KG+RAG",
                     edgecolor="#1d4ed8", linewidth=1.2, zorder=3)
        b2 = ax1.bar(x + w/2, nr_v, w, color="#94a3b8", label="Without RAG",
                     edgecolor="#64748b", linewidth=0.8, zorder=3, alpha=0.85)
        for bar, v in zip(b1, wr_v):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                     f"{v:.2f}", ha="center", fontsize=9, fontweight="bold", color="#1d4ed8")
        for bar, v in zip(b2, nr_v):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                     f"{v:.2f}", ha="center", fontsize=9, color="#475569")

        ax1.set_xticks(x); ax1.set_xticklabels(DIM_LABELS, fontsize=10)
        ax1.set_ylim(0, 3.6); ax1.set_ylabel("Score (0–3)", fontsize=11)
        ax1.set_title(f"({chr(65+row_i*2)}) {judge_label} — Mean Scores",
                      fontsize=12, fontweight="bold")
        ax1.axhline(1.5, color="#cbd5e1", linewidth=0.8, linestyle="--", alpha=0.7)
        ax1.legend(fontsize=10); ax1.grid(axis="y", alpha=0.3); ax1.set_facecolor("#f8fafc")

        # Panel B: Δ with 95% CI error bars
        colors = ["#16a34a" if d >= 0 else "#dc2626" for d in delta]
        err_lower = [delta[i] - ci_l[i] for i in range(len(delta))]
        err_upper = [ci_h[i] - delta[i] for i in range(len(delta))]
        bars2 = ax2.bar(x, delta, color=colors, edgecolor="#334155",
                        linewidth=0.9, zorder=3, alpha=0.88,
                        yerr=[err_lower, err_upper], capsize=5,
                        error_kw={"elinewidth": 1.2, "ecolor": "#334155"})

        for i, (bar, d, p) in enumerate(zip(bars2, delta, pvals)):
            y_top = ci_h[i] if d >= 0 else ci_l[i]
            va    = "bottom" if d >= 0 else "top"
            yp    = y_top + 0.04 if d >= 0 else y_top - 0.04
            sig   = " *" if (p is not None and p < 0.05) else ""
            ax2.text(bar.get_x() + bar.get_width()/2, yp,
                     f"{d:+.3f}{sig}", ha="center", va=va,
                     fontsize=9.5, fontweight="bold",
                     color="#166534" if d >= 0 else "#991b1b")
            # p-value below x-axis
            ax2.text(bar.get_x() + bar.get_width()/2, -0.55,
                     f"p={p:.3f}" if p is not None else "p=N/A",
                     ha="center", va="top", fontsize=8, color="#475569",
                     transform=ax2.get_xaxis_transform())

        ax2.axhline(0, color="#334155", linewidth=1.2, zorder=2)
        ax2.set_xticks(x); ax2.set_xticklabels(DIM_LABELS, fontsize=10)
        ax2.set_ylabel("Δ Score (With − Without)", fontsize=11)
        ax2.set_title(f"({chr(65+row_i*2+1)}) {judge_label} — Δ with 95% CI",
                      fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3); ax2.set_facecolor("#f8fafc")

    n = summary["n_patients"]
    fig.suptitle(f"LLM-Judge Report Quality Evaluation  (n={n} patients, 2 judges)",
                 fontsize=14, fontweight="bold", y=0.995)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
