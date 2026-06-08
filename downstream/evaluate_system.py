#!/usr/bin/env python3
"""
End-to-end system evaluation for the Dementia Multi-Modal Diagnostic System.

Covers:
  1. Task-level accuracy   – AUC / ACC / F1 / confusion-matrix for NC_AD, NC_MCI, MCI_AD
  2. Modality stratification – Dual-Modal / fMRI-only / sMRI-only performance breakdown
  3. OVO voting analysis   – 1:1:1 tie rate + modality disagreement rate
  4. Stability test        – 10 repeated calls for 5 random patients (determinism + latency)

Usage:
    python evaluate_system.py [--api-url URL] [--output results/system_eval.json]
    python evaluate_system.py --no-neo4j --stability-n 3 --stability-k 5
"""

import os
import re
import glob
import json
import time
import random
import argparse
import warnings
from collections import defaultdict
from typing import Optional

import requests
import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, confusion_matrix,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    print("[WARN] scikit-learn not installed — AUC / F1 skipped.")

warnings.filterwarnings("ignore")

from config import SMRI_ROOT as _SMRI_ROOT, MATCHED_ROOT as _MATCHED_ROOT, TPMIC_SMRI_ROOT as _TPMIC_SMRI_ROOT
_API_BASE     = "http://localhost:8080"

TASKS = ["NC vs AD", "NC vs MCI", "MCI vs AD"]
TASK_CLASSES = {
    "NC vs AD":  ("NC", "AD"),
    "NC vs MCI": ("NC", "MCI"),
    "MCI vs AD": ("MCI", "AD"),
}
LABELS = ("NC", "MCI", "AD")

# Positive class index per task (class_b = 1)
#   NC vs AD  → AD=1   NC vs MCI → MCI=1   MCI vs AD → AD=1

# ══════════════════════════════════════════════════════════════════════════════
# 1. Ground-truth builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_smri_gt() -> dict:
    """Scan ADNI_sMRI_Aligned_MNI/{NC,MCI,AD}/ → {NNN_S_XXXX: label}."""
    gt: dict = {}
    for label in LABELS:
        d = os.path.join(_SMRI_ROOT, label)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            m = re.match(r"(\d+_S_\d+)_T1_MNI\.nii\.gz$", fname)
            if m:
                gt[m.group(1)] = label
    return gt


def _build_matched_gt() -> dict:
    """Scan Matched_Dual_Modal_Dataset/{NC,MCI,AD}/ → {NNN_S_XXXX: label}."""
    gt: dict = {}
    for label in LABELS:
        d = os.path.join(_MATCHED_ROOT, label)
        if not os.path.isdir(d):
            continue
        for sid in os.listdir(d):
            if re.match(r"\d+_S_\d+$", sid):
                gt[sid] = label
    return gt


def _build_tpmic_gt() -> dict:
    """Scan TPMIC sMRI_data_MultiModal_Aligned_MNI/{NC,MCI,AD}/ → {sub_XXXX: label}."""
    gt: dict = {}
    for label in LABELS:
        d = os.path.join(_TPMIC_SMRI_ROOT, label)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            m = re.match(r"(sub_\d+)_T1\.nii\.gz$", fname)
            if m:
                gt[m.group(1)] = label
    return gt


def _query_neo4j_gt(subject_id: str, api_base: str) -> Optional[str]:
    """Fallback: parse clinical_dx from GET /api/v1/test/patient-context/{id}."""
    try:
        r = requests.get(
            f"{api_base}/api/v1/test/patient-context/{subject_id}",
            timeout=8,
        )
        if r.status_code != 200:
            return None
        ctx = r.json().get("patient_context", "")
        for pat in [
            r"臨床診斷[：:]\s*(NC|MCI|AD)",
            r"診斷[結果為是]?\s*[：:]?\s*(NC|MCI|AD)",
            r"\bclinical_dx\b.*?(NC|MCI|AD)",
        ]:
            m = re.search(pat, ctx, re.IGNORECASE)
            if m:
                dx = m.group(1).upper()
                if dx in LABELS:
                    return dx
    except Exception:
        pass
    return None


def resolve_gt(
    subject_id: str,
    smri_gt: dict,
    matched_gt: dict,
    api_base: str,
    use_neo4j: bool = True,
    tpmic_gt: dict = None,
) -> Optional[str]:
    """
    Priority order:
      1. ADNI ID (NNN_S_XXXX) → sMRI directory lookup
      2. ADNI ID → Matched_Dual_Modal_Dataset lookup
      3. TPMIC ID (sub_XXXX) → TPMIC sMRI directory lookup
      4. Label suffix in subject_id (e.g., "sub-011_S_6303_AD" or "011_S_6303_AD")
      5. Neo4j patient-context API (if use_neo4j)
    """
    m = re.search(r"(\d{3}_S_\d{4})", subject_id)
    if m:
        adni_id = m.group(1)
        if adni_id in smri_gt:
            return smri_gt[adni_id]
        if adni_id in matched_gt:
            return matched_gt[adni_id]

    # TPMIC format: sub_XXXX
    if tpmic_gt and re.match(r"sub_\d+$", subject_id):
        if subject_id in tpmic_gt:
            return tpmic_gt[subject_id]

    # suffix: sub-011_S_6303_AD  or  someprefix_MCI
    m2 = re.search(r"_(NC|MCI|AD)(?:[_\-]|$)", subject_id, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    if use_neo4j:
        return _query_neo4j_gt(subject_id, api_base)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. API helpers
# ══════════════════════════════════════════════════════════════════════════════

def fetch_patients(api_base: str) -> list:
    r = requests.get(f"{api_base}/api/v1/patients", timeout=30)
    r.raise_for_status()
    return r.json()["patients"]


def analyze_patient(
    api_base: str,
    patient: dict,
    fmri_weight: float = 0.5,
    timeout: int = 180,
) -> dict:
    """POST /api/v1/analyze with multipart form-data."""
    data = {
        "subject_id":  patient["id"],
        "matrix_path": patient.get("matrix_path") or "",
        "t1_path":     patient.get("t1_path")     or "",
        "fmri_weight": str(fmri_weight),
    }
    r = requests.post(f"{api_base}/api/v1/analyze", data=data, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_thresholds(task_raw: dict) -> dict:
    """
    task_raw: {task: {"y_true": [...], "y_prob": [...]}}
    對每個任務以 Youden's J (sensitivity + specificity - 1) 搜尋最佳閾值。
    """
    if not _SKLEARN:
        return {}
    from sklearn.metrics import roc_curve
    results = {}
    for task, d in task_raw.items():
        if len(d["y_true"]) < 4 or len(set(d["y_true"])) < 2:
            results[task] = {"note": "insufficient data"}
            continue
        fpr, tpr, thresholds = roc_curve(d["y_true"], d["y_prob"])
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        results[task] = {
            "optimal_threshold": float(thresholds[best_idx]),
            "youden_j":          float(j_scores[best_idx]),
            "sensitivity":       float(tpr[best_idx]),
            "specificity":       float(1 - fpr[best_idx]),
        }
    return results


def _safe_auc(y_true, y_prob) -> Optional[float]:
    if not _SKLEARN or len(set(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def compute_binary_metrics(y_true: list, y_pred: list, y_prob: list) -> dict:
    n = len(y_true)
    if n == 0:
        return {"n": 0}

    result: dict = {"n": n}

    if _SKLEARN:
        result["acc"] = float(accuracy_score(y_true, y_pred))
        result["f1"]  = float(f1_score(y_true, y_pred, zero_division=0))
        auc = _safe_auc(y_true, y_prob)
        if auc is not None:
            result["auc"] = auc
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        result["confusion_matrix"] = cm.tolist()   # [[TN,FP],[FN,TP]]
    else:
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        result["acc"] = correct / n

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. OVO voting
# ══════════════════════════════════════════════════════════════════════════════

def ovo_vote(task_results: dict) -> tuple:
    """
    OVO majority vote from the 3 binary predictions.
    Returns (predicted_class_str, votes_dict, is_111_tie).

    Vote assignment:
      NC vs AD  → pred=0 ⇒ +1 NC,  pred=1 ⇒ +1 AD
      NC vs MCI → pred=0 ⇒ +1 NC,  pred=1 ⇒ +1 MCI
      MCI vs AD → pred=0 ⇒ +1 MCI, pred=1 ⇒ +1 AD
    """
    votes = {"NC": 0, "MCI": 0, "AD": 0}

    mapping = {
        "NC vs AD":  {0: "NC",  1: "AD"},
        "NC vs MCI": {0: "NC",  1: "MCI"},
        "MCI vs AD": {0: "MCI", 1: "AD"},
    }

    available = 0
    for task, vote_map in mapping.items():
        tres = task_results.get(task)
        if tres is None:
            continue
        available += 1
        pred = int(tres.get("prediction", 0))
        votes[vote_map[pred]] += 1

    if available == 0:
        return "UNKNOWN", votes, False

    max_v = max(votes.values())
    winners = [c for c, v in votes.items() if v == max_v]

    is_111_tie = (votes["NC"] == 1 and votes["MCI"] == 1 and votes["AD"] == 1)
    predicted = winners[0] if len(winners) == 1 else "TIE"
    return predicted, votes, is_111_tie


def has_modality_disagreement(task_results: dict) -> bool:
    """True if any task has both fmri_pred and smri_pred available and they differ."""
    for tres in task_results.values():
        fp = tres.get("fmri_pred")
        sp = tres.get("smri_pred")
        if fp is not None and sp is not None and fp != sp:
            return True
    return False


def infer_modality_group(task_results: dict) -> str:
    """Determine Dual-Modal / fMRI only / sMRI only from response findings."""
    has_fmri = any(tres.get("fmri_findings") for tres in task_results.values())
    has_smri = any(tres.get("smri_findings") for tres in task_results.values())
    if has_fmri and has_smri:
        return "Dual-Modal"
    if has_smri:
        return "sMRI only"
    if has_fmri:
        return "fMRI only"
    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Stability test
# ══════════════════════════════════════════════════════════════════════════════

def run_stability_test(
    api_base: str,
    patients: list,
    n_patients: int = 5,
    n_repeats: int = 10,
) -> dict:
    """
    Randomly sample n_patients valid patients, call analyze n_repeats times each.
    Verify that prob_fused is identical across all repeats (deterministic inference).
    Record per-call latency (mean ± std).
    """
    valid = [p for p in patients if p.get("fmri_valid") or p.get("t1_path")]
    if not valid:
        return {"error": "No valid patients available for stability test."}

    sampled = random.sample(valid, min(n_patients, len(valid)))
    per_patient: dict = {}

    for p in sampled:
        sid = p["id"]
        probs_by_task: dict = defaultdict(list)
        latencies: list = []

        for _ in range(n_repeats):
            t0 = time.perf_counter()
            try:
                resp = analyze_patient(api_base, p)
                latencies.append(time.perf_counter() - t0)
                for task, tres in resp.get("task_results", {}).items():
                    probs_by_task[task].append(tres.get("prob_fused"))
            except Exception as e:
                latencies.append(time.perf_counter() - t0)
                probs_by_task["__error__"].append(str(e))

        # Determinism check: max(prob) - min(prob) across repeats per task
        deterministic = True
        max_dev = 0.0
        task_deviations: dict = {}
        for task, vals in probs_by_task.items():
            if task == "__error__":
                continue
            valid_vals = [v for v in vals if v is not None]
            if len(valid_vals) > 1:
                dev = float(max(valid_vals) - min(valid_vals))
                task_deviations[task] = dev
                max_dev = max(max_dev, dev)
                # float32 precision floor; two identical forward passes should differ by < 1e-6
                if dev > 1e-6:
                    deterministic = False

        per_patient[sid] = {
            "deterministic":      deterministic,
            "max_prob_deviation": max_dev,
            "task_deviations":    task_deviations,
            "n_repeats":          n_repeats,
            "latency_mean_s":     float(np.mean(latencies)),
            "latency_std_s":      float(np.std(latencies)),
            "latency_min_s":      float(np.min(latencies)),
            "latency_max_s":      float(np.max(latencies)),
        }

    all_det = all(v["deterministic"] for v in per_patient.values())
    all_lats = [v["latency_mean_s"] for v in per_patient.values()]
    return {
        "patients":             per_patient,
        "all_deterministic":    all_det,
        "global_latency_mean_s": float(np.mean(all_lats)) if all_lats else 0.0,
        "global_latency_std_s":  float(np.std(all_lats))  if all_lats else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    api_base: str = _API_BASE,
    stability_n: int = 5,
    stability_k: int = 10,
    use_neo4j: bool = True,
    verbose: bool = True,
    adni_only: bool = False,
    test_csv: str = None,
) -> dict:

    print(f"\n{'='*70}")
    print("  DEMENTIA MULTI-MODAL DIAGNOSTIC SYSTEM  —  SYSTEM EVALUATION")
    print(f"{'='*70}")

    # ── Build GT maps ─────────────────────────────────────────────────────────
    # ... (building GT logic)
    print("\n[1/5] Building ground-truth from filesystem …")
    smri_gt    = _build_smri_gt()
    matched_gt = _build_matched_gt()
    tpmic_gt   = {} if adni_only else _build_tpmic_gt()
    print(f"      sMRI dir GT   : {len(smri_gt)} patients")
    print(f"      Matched-DS GT : {len(matched_gt)} patients")
    print(f"      TPMIC GT      : {len(tpmic_gt)} patients" + (" (skipped: --adni-only)" if adni_only else ""))

    # ── Fetch patient list from API ───────────────────────────────────────────
    print("\n[2/5] Fetching patient list from API …")
    try:
        all_patients = fetch_patients(api_base)
    except Exception as e:
        raise RuntimeError(f"Cannot reach API at {api_base}: {e}")
    
    if adni_only:
        all_patients = [p for p in all_patients if not p["id"].startswith("sub_")]
        print(f"      Total patients : {len(all_patients)} (ADNI only)")
    else:
        print(f"      Total patients : {len(all_patients)}")

    if test_csv:
        if os.path.exists(test_csv):
            import pandas as pd
            df_test = pd.read_csv(test_csv)
            allowed_ids = set(df_test['subject_id'].astype(str).tolist())
            
            def normalize_id(sid):
                # ADNI: sub-011_S_6303 -> 011_S_6303
                # TPMIC: sub_0001 -> sub_0001
                if sid.startswith("sub-"):
                    return sid.replace("sub-", "")
                return sid

            all_patients = [p for p in all_patients if normalize_id(p["id"]) in allowed_ids]
            print(f"      Filtered by {test_csv}: {len(all_patients)} patients remain")
        else:
            print(f"      Warning: Test CSV not found: {test_csv}")

    # ── Resolve ground-truth per patient ─────────────────────────────────────
    gt_map: dict = {}
    neo4j_hits = 0
    for p in all_patients:
        if "Invalid Data" in p.get("label", ""):
            continue
        gt = resolve_gt(p["id"], smri_gt, matched_gt, api_base,
                        use_neo4j=use_neo4j, tpmic_gt=tpmic_gt)
        if gt:
            gt_map[p["id"]] = gt
            if gt not in smri_gt.values() and gt not in matched_gt.values():
                neo4j_hits += 1

    patients_with_gt = [p for p in all_patients if p["id"] in gt_map]
    label_dist = defaultdict(int)
    for v in gt_map.values():
        label_dist[v] += 1

    print(f"      Patients with GT : {len(patients_with_gt)} / {len(all_patients)}")
    print(f"      Label dist       : {dict(label_dist)}")
    if use_neo4j:
        print(f"      Neo4j fallbacks  : {neo4j_hits}")

    # ── Per-task / per-modality accumulators ──────────────────────────────────
    # task_data[task] = {y_true, y_pred, y_prob}
    task_data: dict = {t: {"y_true": [], "y_pred": [], "y_prob": []} for t in TASKS}

    # strat_data[modality_group][task] = {y_true, y_pred, y_prob}
    strat_data: dict = defaultdict(
        lambda: {t: {"y_true": [], "y_pred": [], "y_prob": []} for t in TASKS}
    )

    ovo_records: list = []
    failed: list = []

    # ── Inference loop ────────────────────────────────────────────────────────
    print(f"\n[3/5] Running inference for {len(patients_with_gt)} patients …")

    for i, patient in enumerate(patients_with_gt):
        sid       = patient["id"]
        gt_label  = gt_map[sid]
        modality  = patient.get("label", "").split("(")[-1].rstrip(")")

        if verbose:
            print(f"  [{i+1:3d}/{len(patients_with_gt)}] {sid:<38} GT={gt_label}  {modality}")

        try:
            resp = analyze_patient(api_base, patient)
        except Exception as e:
            failed.append({"id": sid, "error": str(e)})
            if verbose:
                print(f"         ✗ API error: {e}")
            continue

        task_results = resp.get("task_results", {})
        if not task_results:
            failed.append({"id": sid, "error": "empty task_results"})
            continue

        # OVO vote
        ovo_pred, ovo_votes, is_111_tie = ovo_vote(task_results)
        modality_disagree  = has_modality_disagreement(task_results)
        mod_group          = infer_modality_group(task_results)

        # 記錄 API 直接回傳的 OVO 結果（Task 2 新增欄位）
        api_ovo = resp.get("ovo_result", {})
        api_ovo_pred = api_ovo.get("predicted_class", "UNKNOWN")

        ovo_records.append({
            "id":               sid,
            "gt":               gt_label,
            "ovo_pred":         ovo_pred,           # 評估腳本自己算的（保留）
            "api_ovo_pred":     api_ovo_pred,       # API 回傳的（新增）
            "ovo_correct":      (ovo_pred == gt_label),
            "api_ovo_correct":  (api_ovo_pred == gt_label),  # 新增
            "ovo_votes":        ovo_votes,
            "api_weighted_votes": api_ovo.get("weighted_votes", {}),  # 新增
            "is_111_tie":       is_111_tie,
            "api_is_tie":       api_ovo.get("is_tie", False),   # 新增
            "tie_broken_by":    api_ovo.get("tie_broken_by"),   # 新增
            "modality_disagree": modality_disagree,
            "mod_group":        mod_group,
        })

        # Per-task binary accumulation
        for task, (cls_a, cls_b) in TASK_CLASSES.items():
            if gt_label not in (cls_a, cls_b):
                continue  # patient not in scope for this task

            tres = task_results.get(task)
            if not tres:
                continue

            y_true = 1 if gt_label == cls_b else 0
            y_pred = int(tres.get("prediction", 0))
            y_prob = float(tres.get("prob_fused", 0.5))

            task_data[task]["y_true"].append(y_true)
            task_data[task]["y_pred"].append(y_pred)
            task_data[task]["y_prob"].append(y_prob)

            strat_data[mod_group][task]["y_true"].append(y_true)
            strat_data[mod_group][task]["y_pred"].append(y_pred)
            strat_data[mod_group][task]["y_prob"].append(y_prob)

    n_ok = len(patients_with_gt) - len(failed)
    print(f"      Completed : {n_ok}   Failed : {len(failed)}")

    # ── Compute task-level metrics ────────────────────────────────────────────
    print("\n[4/5] Computing metrics …")
    task_metrics: dict = {}
    for task in TASKS:
        d = task_data[task]
        task_metrics[task] = compute_binary_metrics(
            d["y_true"], d["y_pred"], d["y_prob"]
        )

    strat_metrics: dict = {}
    for mod_grp in ["Dual-Modal", "fMRI only", "sMRI only"]:
        strat_metrics[mod_grp] = {}
        tdata = strat_data.get(mod_grp, {})
        for task in TASKS:
            d = tdata.get(task, {"y_true": [], "y_pred": [], "y_prob": []})
            strat_metrics[mod_grp][task] = compute_binary_metrics(
                d["y_true"], d["y_pred"], d["y_prob"]
            )

    # ── OVO analysis ──────────────────────────────────────────────────────────
    total          = len(ovo_records)
    n_tie          = sum(1 for r in ovo_records if r["is_111_tie"])
    n_dual         = sum(1 for r in ovo_records if r["mod_group"] == "Dual-Modal")
    n_disagree     = sum(1 for r in ovo_records if r["modality_disagree"])
    ovo_correct    = sum(1 for r in ovo_records if r["ovo_correct"])

    api_ovo_correct = sum(1 for r in ovo_records if r.get("api_ovo_correct", False))
    api_tie_count   = sum(1 for r in ovo_records if r.get("api_is_tie", False))

    ovo_analysis = {
        "total_patients":             total,
        "ovo_correct":                ovo_correct,
        "ovo_accuracy":               ovo_correct / total if total else 0.0,
        "n_111_tie":                  n_tie,
        "tie_rate":                   n_tie / total if total else 0.0,
        "n_dual_modal":               n_dual,
        "n_modality_disagree":        n_disagree,
        "disagree_rate_among_dual":   n_disagree / n_dual if n_dual else 0.0,
        "api_ovo_correct":            api_ovo_correct,
        "api_ovo_accuracy":           api_ovo_correct / total if total else 0.0,
        "api_tie_rate":               api_tie_count / total if total else 0.0,
    }

    # ── Stability test ────────────────────────────────────────────────────────
    print(f"\n[5/5] Stability test ({stability_n} patients × {stability_k} repeats) …")
    stability = run_stability_test(api_base, all_patients, stability_n, stability_k)

    # Threshold search
    threshold_analysis = find_optimal_thresholds({
        task: {"y_true": task_data[task]["y_true"], "y_prob": task_data[task]["y_prob"]}
        for task in TASKS
    })

    return {
        "summary": {
            "api_base":            api_base,
            "total_patients":      len(all_patients),
            "patients_with_gt":    len(patients_with_gt),
            "successful_inferences": n_ok,
            "failed_inferences":   len(failed),
            "label_distribution":  dict(label_dist),
        },
        "task_metrics":        task_metrics,
        "modality_stratified": strat_metrics,
        "ovo_analysis":        ovo_analysis,
        "ovo_records":         ovo_records,
        "stability":           stability,
        "failed":              failed,
        "threshold_analysis":  threshold_analysis,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. Formatted report
# ══════════════════════════════════════════════════════════════════════════════

_W = 72
_SEP = "─" * _W


def _fmt_row(task: str, m: dict) -> str:
    n   = m.get("n", 0)
    auc = f"{m['auc']:.4f}" if m.get("auc") is not None else "  N/A  "
    acc = f"{m['acc']:.4f}" if m.get("acc") is not None else "  N/A  "
    f1  = f"{m['f1']:.4f}"  if m.get("f1")  is not None else "  N/A  "
    return f"  {task:<14}  {n:>5}  {auc:>7}  {acc:>7}  {f1:>7}"


def _header_row() -> str:
    return (f"  {'Task':<14}  {'N':>5}  {'AUC':>7}  {'ACC':>7}  {'F1':>7}\n"
            f"  {'-'*14}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}")


def print_report(results: dict) -> None:
    s = results["summary"]
    print(f"\n{'='*_W}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*_W}")
    print(f"  API endpoint        : {s['api_base']}")
    print(f"  Total patients      : {s['total_patients']}")
    print(f"  Patients with GT    : {s['patients_with_gt']}")
    print(f"  Successful infer.   : {s['successful_inferences']}")
    print(f"  Failed inferences   : {s['failed_inferences']}")
    print(f"  Label distribution  : {s['label_distribution']}")

    # ── Task-level ────────────────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  TASK-LEVEL METRICS  (ref: fMRI AUC ≈ 0.784/0.658/0.687 | sMRI AUC ≈ 0.825/0.703/0.848)")
    print(_SEP)
    print(_header_row())
    for task in TASKS:
        print(_fmt_row(task, results["task_metrics"].get(task, {})))

    print("\n  Confusion matrices  [rows = True label, cols = Predicted]  [[TN FP] / [FN TP]]:")
    for task in TASKS:
        cm = results["task_metrics"].get(task, {}).get("confusion_matrix")
        if cm:
            cls_a, cls_b = TASK_CLASSES[task]
            print(f"    {task:<14} {cls_a}=0 / {cls_b}=1 → {cm}")

    # ── Modality-stratified ───────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  MODALITY-STRATIFIED ACCURACY")
    print(_SEP)
    for mod_grp in ["Dual-Modal", "fMRI only", "sMRI only"]:
        tdata = results["modality_stratified"].get(mod_grp, {})
        n_total = sum(tdata.get(t, {}).get("n", 0) for t in TASKS)
        print(f"\n  [{mod_grp}]  (unique task-patient entries: {n_total})")
        print(_header_row())
        for task in TASKS:
            print(_fmt_row(task, tdata.get(task, {})))

    # ── OVO voting ────────────────────────────────────────────────────────────
    ovo = results["ovo_analysis"]
    print(f"\n{_SEP}")
    print("  OVO VOTING ANALYSIS")
    print(_SEP)
    print(f"  3-class OVO accuracy      : {ovo['ovo_accuracy']:.4f}  "
          f"({ovo['ovo_correct']}/{ovo['total_patients']})")
    print(f"  Weighted OVO accuracy (API) : {ovo.get('api_ovo_accuracy', 0):.4f}  "
          f"({ovo.get('api_ovo_correct', 0)}/{ovo['total_patients']})")
    print(f"  API tie rate (after break)  : {ovo.get('api_tie_rate', 0):.4f}")
    print(f"  1:1:1 tie rate            : {ovo['tie_rate']:.4f}  "
          f"({ovo['n_111_tie']}/{ovo['total_patients']})")
    print(f"  Dual-modal patients       : {ovo['n_dual_modal']}")
    print(f"  Modality disagreement     : {ovo['disagree_rate_among_dual']:.4f}  "
          f"({ovo['n_modality_disagree']}/{ovo['n_dual_modal']} dual-modal)")

    # ── Threshold search ──────────────────────────────────────────────────────
    ta = results.get("threshold_analysis", {})
    if ta:
        print(f"\n{_SEP}")
        print("  OPTIMAL THRESHOLD SEARCH  (Youden's J = sensitivity + specificity - 1)")
        print(_SEP)
        print(f"  {'Task':<14}  {'Current':>9}  {'Optimal':>9}  {'J-score':>8}  {'Sens':>6}  {'Spec':>6}")
        print(f"  {'-'*14}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*6}  {'-'*6}")
        current_thresholds = {"NC vs AD": 0.500, "NC vs MCI": 0.500, "MCI vs AD": 0.500}
        for task in TASKS:
            m = ta.get(task, {})
            if "note" in m:
                print(f"  {task:<14}  {'N/A':>9}  {'N/A':>9}  {'N/A':>8}")
                continue
            curr = current_thresholds.get(task, 0.5)
            print(f"  {task:<14}  {curr:>9.3f}  {m['optimal_threshold']:>9.3f}  "
                  f"{m['youden_j']:>8.4f}  {m['sensitivity']:>6.3f}  {m['specificity']:>6.3f}")

    # ── Stability ─────────────────────────────────────────────────────────────
    stab = results["stability"]
    print(f"\n{_SEP}")
    print("  STABILITY TEST  (determinism + latency)")
    print(_SEP)
    if "error" in stab:
        print(f"  {stab['error']}")
    else:
        det_str = "YES ✓" if stab["all_deterministic"] else "NO  ✗"
        print(f"  All inferences deterministic : {det_str}")
        print(f"  Global latency (mean ± std)  : "
              f"{stab['global_latency_mean_s']:.3f}s ± {stab['global_latency_std_s']:.3f}s")
        print(f"\n  {'Subject ID':<42} {'det':>3}  {'lat mean':>9}  {'lat std':>8}  {'max dev':>10}")
        print(f"  {'-'*42}  {'-'*3}  {'-'*9}  {'-'*8}  {'-'*10}")
        for pid, pd in stab.get("patients", {}).items():
            det = "✓" if pd["deterministic"] else "✗"
            print(f"  {pid[:42]:<42}  {det:>3}  "
                  f"{pd['latency_mean_s']:>8.3f}s  "
                  f"{pd['latency_std_s']:>7.3f}s  "
                  f"{pd['max_prob_deviation']:>10.2e}")

    print(f"\n{'='*_W}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation of the Dementia Multi-Modal Diagnostic System.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url",     default=_API_BASE,
                        help="FastAPI base URL")
    parser.add_argument("--output",      default="results/system_eval.json",
                        help="Output JSON path (relative to this script's directory)")
    parser.add_argument("--stability-n", type=int, default=5,
                        help="Number of patients sampled for stability test")
    parser.add_argument("--stability-k", type=int, default=10,
                        help="Number of API calls per patient in stability test")
    parser.add_argument("--no-neo4j",    action="store_true",
                        help="Skip Neo4j fallback for ground-truth resolution")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for stability test patient sampling")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress per-patient inference progress lines")
    parser.add_argument("--adni-only",   action="store_true",
                        help="Evaluate ADNI patients only, exclude TPMIC")
    parser.add_argument("--test-csv",    default=None,
                        help="Path to CSV containing subject_id list for filtering")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        results = run_evaluation(
            api_base    = args.api_url,
            stability_n = args.stability_n,
            stability_k = args.stability_k,
            use_neo4j   = not args.no_neo4j,
            verbose     = not args.quiet,
            adni_only   = args.adni_only,
            test_csv    = args.test_csv,
        )
    except RuntimeError as e:
        print(f"\n[FATAL] {e}")
        return 1

    print_report(results)

    # Save JSON output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"Results saved → {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
