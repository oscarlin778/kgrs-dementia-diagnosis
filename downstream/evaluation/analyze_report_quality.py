"""
analyze_report_quality.py
=========================
兩種客觀方式評估 RAG 對報告品質的影響：

Method A: Classification-Consistency Analysis
  - 正確預測的病患報告品質是否高於錯誤預測？
  - 若是，說明 LLM judge 確實在評估「準確性」，而非純粹文字豐富度

Method B: Factual Checklist (可機器驗證的客觀指標)
  - 報告是否正確引用機率值（誤差 < 0.05）
  - 是否提及具名腦區（ROI）
  - 是否包含臨床建議
  - 是否指出不確定性（邊界值）
  - with_rag 是否有文獻引用（Paper name / 作者 et al.）
"""
import json
import re
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
# 用 v2（原始版）因為 v3 分析矛盾，用 v2 做基準
V2_PATH = BASE_DIR / "results" / "report_quality_v2_nolabel_results.json"
V3_PATH = BASE_DIR / "results" / "report_quality_v3_fixed_results.json"

LABEL_TO_CLASS = {0: "NC", 1: "MCI", 2: "AD"}
DIMS = ["factual_accuracy", "clinical_relevance", "completeness", "coherence"]

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def load_records(path):
    d = json.load(open(path))
    return d["records"]

def get_correct(rec):
    """病患的 overall 預測是否與 true_label 一致（診斷層級）"""
    # diagnosis = true label string, overall_pred = model prediction string
    return rec.get("overall_pred") == rec.get("diagnosis")

def checklist_score(report: str, condition: str, actual_probs: dict = None) -> dict:
    """
    客觀 checklist（0/1 每項）：
    1. roi_mentioned:  是否提到具名腦區（ROI）
    2. recommendation: 是否有臨床建議段落
    3. uncertainty:    是否提到不確定性/邊界值
    4. cite_paper:     (with_rag only) 是否有文獻引用（et al. 或括號年份）
    5. report_length:  字元數（資訊量代理指標）
    """
    roi_pattern = re.compile(
        r'(Frontal|Temporal|Parietal|Occipital|Hippocampus|Amygdala|'
        r'Cerebellum|Cerebelum|Thalamus|Putamen|Cingulate|Precuneus|'
        r'Insula|Caudate|Parahippocampal|DMN|default mode|SMA|'
        r'[A-Z][a-z]+_[LR]|[A-Z][a-z]+_Sup|[A-Z][a-z]+_Mid|'
        r'海馬|前額葉|顳葉|頂葉|枕葉|杏仁核|小腦|扣帶迴|楔前葉|'
        r'丘腦|尾狀核|殼核|島葉|旁海馬|額葉|DMN|預設模式)',
        re.IGNORECASE
    )
    cite_pattern = re.compile(r'et al\..*?\d{4}|\(\w+.*?,?\s*\d{4}\)')
    uncertainty_kw = ["邊界", "追蹤", "不確定", "謹慎", "建議複查", "borderline", "uncertain"]
    recommend_kw  = ["建議", "臨床", "follow", "追蹤", "recommend", "suggest", "定期", "診斷"]

    return {
        "roi_mentioned":  int(bool(roi_pattern.search(report))),
        "recommendation": int(any(kw in report.lower() for kw in recommend_kw)),
        "uncertainty":    int(any(kw in report for kw in uncertainty_kw)),
        "cite_paper":     int(bool(cite_pattern.search(report))) if condition == "with_rag" else None,
        "report_length":  len(report),
    }


# ─────────────────────────────────────────
# Method A: Classification-Consistency
# ─────────────────────────────────────────

def method_a(records, version_label="v2"):
    print(f"\n{'='*65}")
    print(f"METHOD A: Classification-Consistency Analysis ({version_label})")
    print(f"{'='*65}")
    print("假說：正確分類病患的報告品質 > 錯誤分類病患的報告品質\n")

    # 只看 with_rag（更 informative）
    for judge in ["Gemma3", "Llama3.1"]:
        correct_scores  = defaultdict(list)
        wrong_scores    = defaultdict(list)
        for rec in records:
            if rec["condition"] != "with_rag" or rec["judge"] != judge:
                continue
            if not rec.get("scores"):
                continue
            is_correct = get_correct(rec)
            for dim in DIMS:
                v = rec["scores"].get(dim)
                if isinstance(v, (int, float)):
                    (correct_scores if is_correct else wrong_scores)[dim].append(v)

        n_correct = len(correct_scores[DIMS[0]])
        n_wrong   = len(wrong_scores[DIMS[0]])
        print(f"  Judge: {judge}  (correct n={n_correct}, wrong n={n_wrong})")
        print(f"  {'Dimension':<20} {'Correct':>8} {'Wrong':>8} {'Δ':>7}  p")
        print(f"  {'-'*55}")
        for dim in DIMS:
            c = correct_scores[dim]
            w = wrong_scores[dim]
            if len(c) < 2 or len(w) < 2:
                continue
            mc, mw = np.mean(c), np.mean(w)
            _, p = stats.mannwhitneyu(c, w, alternative="greater")
            sig = "✓" if p < 0.10 else " "
            print(f"  {dim:<20} {mc:>8.3f} {mw:>8.3f} {mc-mw:>+7.3f}  p={p:.3f} {sig}")
        print()

    # 同時看 no_rag
    print("  [no_rag 版本作為對照]")
    for judge in ["Gemma3", "Llama3.1"]:
        correct_scores  = defaultdict(list)
        wrong_scores    = defaultdict(list)
        for rec in records:
            if rec["condition"] != "no_rag" or rec["judge"] != judge:
                continue
            if not rec.get("scores"):
                continue
            is_correct = get_correct(rec)
            for dim in DIMS:
                v = rec["scores"].get(dim)
                if isinstance(v, (int, float)):
                    (correct_scores if is_correct else wrong_scores)[dim].append(v)

        n_correct = len(correct_scores[DIMS[0]])
        n_wrong   = len(wrong_scores[DIMS[0]])
        print(f"  Judge: {judge}/no_rag  (correct n={n_correct}, wrong n={n_wrong})")
        for dim in DIMS:
            c = correct_scores[dim]
            w = wrong_scores[dim]
            if len(c) < 2 or len(w) < 2:
                continue
            mc, mw = np.mean(c), np.mean(w)
            _, p = stats.mannwhitneyu(c, w, alternative="greater")
            sig = "✓" if p < 0.10 else " "
            print(f"    {dim:<20} {mc:.3f} vs {mw:.3f}  Δ={mc-mw:+.3f}  p={p:.3f} {sig}")
        print()


# ─────────────────────────────────────────
# Method B: Factual Checklist
# ─────────────────────────────────────────

def method_b(records, version_label="v2"):
    print(f"\n{'='*65}")
    print(f"METHOD B: Factual Checklist Evaluation ({version_label})")
    print(f"{'='*65}")
    print("客觀可驗證指標（不依賴 LLM judge）\n")

    wr_checks = defaultdict(list)
    nr_checks = defaultdict(list)

    # De-dup per patient×condition（只保留 Gemma3 那筆，records 重複）
    seen = set()
    for rec in records:
        key = (rec["subject_id"], rec["condition"])
        if key in seen:
            continue
        seen.add(key)

        report = rec.get("report", "")
        cond   = rec["condition"]
        chk    = checklist_score(report, cond)

        target = wr_checks if cond == "with_rag" else nr_checks
        for k, v in chk.items():
            if v is not None:
                target[k].append(v)

    print(f"  {'指標':<20} {'With RAG':>10} {'No RAG':>10}  {'Δ':>7}  p")
    print(f"  {'-'*60}")

    for key in ["roi_mentioned", "recommendation", "uncertainty", "report_length"]:
        wr = wr_checks[key]
        nr = nr_checks[key]
        mwr, mnr = np.mean(wr), np.mean(nr)
        if key == "report_length":
            _, p = stats.wilcoxon(wr, nr)
            print(f"  {key:<20} {mwr:>10.0f} {mnr:>10.0f} {mwr-mnr:>+7.0f}  p={p:.3f}")
        else:
            # binary proportion test
            from scipy.stats import chi2_contingency
            n = min(len(wr), len(nr))
            a, b = int(np.sum(wr[:n])), int(np.sum(nr[:n]))
            ct = np.array([[a, n-a], [b, n-b]])
            if ct.min() > 0:
                _, p, _, _ = chi2_contingency(ct)
            else:
                p = 1.0
            print(f"  {key:<20} {mwr:>10.3f} {mnr:>10.3f} {mwr-mnr:>+7.3f}  p={p:.3f}")

    # cite_paper: only for with_rag
    cite_rate = np.mean(wr_checks["cite_paper"])
    print(f"\n  cite_paper (with_rag): {cite_rate:.1%} 的報告有引用文獻")
    print(f"  報告平均字元數: with_rag={np.mean(wr_checks['report_length']):.0f}, "
          f"no_rag={np.mean(nr_checks['report_length']):.0f}")


# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────

def summarize_llm_judge(records, version_label):
    print(f"\n{'='*65}")
    print(f"LLM Judge 彙整 ({version_label}，with_rag vs no_rag)")
    print(f"{'='*65}")

    by = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if not rec.get("scores"):
            continue
        cond, judge = rec["condition"], rec["judge"]
        for dim in DIMS:
            v = rec["scores"].get(dim)
            if isinstance(v, (int, float)):
                by[judge][f"{cond}/{dim}"].append(v)

    for judge in ["Gemma3", "Llama3.1"]:
        print(f"\n  Judge: {judge}")
        print(f"  {'Dimension':<20} {'With RAG':>9} {'No RAG':>9} {'Δ':>7}  p")
        print(f"  {'-'*55}")
        for dim in DIMS:
            wr = by[judge][f"with_rag/{dim}"]
            nr = by[judge][f"no_rag/{dim}"]
            if len(wr) < 2: continue
            mwr, mnr = np.mean(wr), np.mean(nr)
            _, p = stats.wilcoxon(wr, nr)
            sig = "*" if p < 0.05 else ("†" if p < 0.10 else " ")
            print(f"  {dim:<20} {mwr:>9.3f} {mnr:>9.3f} {mwr-mnr:>+7.3f}  p={p:.3f} {sig}")


V5_PATH = BASE_DIR / "results" / "report_quality_v5_full_fix.json"

if __name__ == "__main__":
    import sys
    use_v3 = "--v3" in sys.argv
    use_v5 = "--v5" in sys.argv

    if use_v5:
        path, version = V5_PATH, "v5 (ROI+modality fix)"
    elif use_v3:
        path, version = V3_PATH, "v3 (fixed prompt)"
    else:
        path, version = V2_PATH, "v2 (original)"
    print(f"\nLoading: {path.name}")
    records = load_records(path)
    print(f"Total records: {len(records)}")

    summarize_llm_judge(records, version)
    method_a(records, version)
    method_b(records, version)

    print("\n" + "="*65)
    print("完成")
