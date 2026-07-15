"""
eval_scale_judge.py
===================
從 report_quality_v5_full_fix.json 載入已生成的報告，
用三個規模不同的非中國模型當 judge，比較 scale 對鑑別力的影響：

  Small  : llama3.2:1b  (Meta, 1B)
  Medium : llama3.1:8b  (Meta, 8B)
  Large  : gemma3:12b   (Google, 12B)

目的：驗證「≥8B 模型才有足夠鑑別力評估醫療報告品質」

執行：
  conda activate AD
  cd downstream/
  python evaluation/eval_scale_judge.py
  python evaluation/eval_scale_judge.py --out results/report_quality_scale.json
"""
import argparse, json, re, time
import numpy as np
import requests
from pathlib import Path
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
RES_DIR  = BASE_DIR / "results"

OLLAMA_URL = "http://localhost:11434/api/generate"

JUDGES = [
    {"model": "llama3.2:1b",  "label": "Llama3.2-1B",  "num_predict": 512},
    {"model": "llama3.1:8b",  "label": "Llama3.1-8B",  "num_predict": 512},
    {"model": "gemma3:12b",   "label": "Gemma3-12B",   "num_predict": 512},
]

DIMENSIONS = ["factual_accuracy", "clinical_relevance", "completeness", "coherence"]
N_BOOT = 2000

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


def judge_report(report: str, diagnosis: str, model: str, num_predict: int) -> dict | None:
    if not report or len(report) < 50:
        return None
    prompt = JUDGE_PROMPT.format(diagnosis=diagnosis, report=report[:6000])
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": num_predict}
        }, timeout=300)
        text = r.json().get("response", "")
        # Strip any thinking blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        matches = re.findall(r'\{[^{}]*"factual_accuracy"[^{}]*\}', text, re.DOTALL)
        if matches:
            return json.loads(matches[-1])
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f"  [WARN] {model} failed: {e}")
    return None


def paired_stats(scores_with: list, scores_no: list) -> dict:
    a = np.array(scores_with, dtype=float)
    b = np.array(scores_no,   dtype=float)
    n = len(a)
    if n < 2:
        return {"n": n, "mean_with": None, "mean_no": None,
                "delta": None, "ci_low": None, "ci_high": None, "p_value": None}
    diff = a - b
    rng  = np.random.default_rng(42)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]).tolist()
    if np.allclose(diff, 0):
        p_value = 1.0
    else:
        try:
            _, p_value = wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
            p_value = float(p_value)
        except Exception:
            p_value = None
    return {
        "n":         n,
        "mean_with": round(float(a.mean()), 3),
        "mean_no":   round(float(b.mean()), 3),
        "delta":     round(float(diff.mean()), 3),
        "ci_low":    round(ci_low,  3),
        "ci_high":   round(ci_high, 3),
        "p_value":   round(p_value, 4) if p_value is not None else None,
    }


def aggregate(records: list, judge_label: str) -> dict:
    per_pid = {}
    for r in records:
        if r["judge"] != judge_label or not r["scores"]:
            continue
        per_pid.setdefault(r["subject_id"], {})[r["condition"]] = r["scores"]
    result = {}
    for dim in DIMENSIONS:
        w_list, n_list = [], []
        for pid, conds in per_pid.items():
            if "with_rag" in conds and "no_rag" in conds:
                v1 = conds["with_rag"].get(dim)
                v2 = conds["no_rag"].get(dim)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    w_list.append(v1); n_list.append(v2)
        result[dim] = paired_stats(w_list, n_list)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5",  default=str(RES_DIR / "report_quality_v5_full_fix.json"),
                        help="v5 JSON with saved reports")
    parser.add_argument("--out", default=str(RES_DIR / "report_quality_scale.json"))
    args = parser.parse_args()

    # ── Load v5 reports ──────────────────────────────────────────────────────
    with open(args.v5, encoding="utf-8") as f:
        v5 = json.load(f)

    # Build lookup: (subject_id, condition) → {report, diagnosis}
    report_lookup = {}
    for rec in v5["records"]:
        key = (rec["subject_id"], rec["condition"])
        if key not in report_lookup:
            report_lookup[key] = {
                "report":    rec.get("report", ""),
                "diagnosis": rec.get("diagnosis", "Unknown"),
            }

    pairs = sorted({(k[0], k[1]) for k in report_lookup})
    print(f"Loaded {len(pairs)} (patient, condition) pairs from v5")
    print(f"Judges: {[j['label'] for j in JUDGES]}\n")

    # ── Run judges ───────────────────────────────────────────────────────────
    new_records = []
    t0 = time.time()

    for idx, (pid, condition) in enumerate(pairs, 1):
        info = report_lookup[(pid, condition)]
        report    = info["report"]
        diagnosis = info["diagnosis"]
        elapsed   = time.time() - t0
        print(f"[{idx}/{len(pairs)}] {pid} / {condition}  ({elapsed:.0f}s)")

        for judge in JUDGES:
            print(f"  judging with {judge['label']}...", end=" ", flush=True)
            scores = judge_report(report, diagnosis, judge["model"], judge["num_predict"])
            if scores:
                short = {"factual_accuracy": "FA", "clinical_relevance": "CR",
                         "completeness": "CP", "coherence": "CH"}
                print(" ".join(f"{short[d]}={scores.get(d)}" for d in DIMENSIONS))
            else:
                print("FAILED")

            new_records.append({
                "subject_id": pid,
                "diagnosis":  diagnosis,
                "condition":  condition,
                "judge":      judge["label"],
                "scores":     scores,
            })
            time.sleep(0.3)

    # ── Aggregate ────────────────────────────────────────────────────────────
    per_judge = {j["label"]: aggregate(new_records, j["label"]) for j in JUDGES}

    summary = {
        "n_patients": len({r["subject_id"] for r in new_records}),
        "judges":     [j["label"] for j in JUDGES],
        "per_judge":  per_judge,
    }
    output = {"summary": summary, "records": new_records}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {args.out}")

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "="*90)
    header = f"{'Metric':<22}"
    for j in JUDGES:
        header += f"  {j['label']:>14} Δ      p"
    print(header)
    print("-"*90)
    for dim in DIMENSIONS:
        row = f"{dim:<22}"
        for j in JUDGES:
            s = per_judge[j["label"]][dim]
            if s["delta"] is None:
                row += f"  {'N/A':>14}    N/A"
            else:
                sig = "*" if (s["p_value"] is not None and s["p_value"] < 0.05) else " "
                row += f"  {s['delta']:>+8.3f}{sig}  {s['p_value'] if s['p_value'] is not None else 'N/A':>6}"
        print(row)

    # ── Generate LaTeX table ──────────────────────────────────────────────────
    print("\n" + "="*90)
    print("LaTeX table:")
    print(r"""\begin{table}[h]
\caption{LLM-Judge Report Quality Evaluation by Model Scale (RAG vs.\ No-RAG, $n{=}49$)}
\label{tab:reportquality}
\centering
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc|cccc|cccc}
\toprule
& \multicolumn{4}{c|}{Llama3.2-1B (Small)} & \multicolumn{4}{c|}{Llama3.1-8B (Medium)} & \multicolumn{4}{c}{Gemma3-12B (Large)} \\
Metric & $\mu_+$ & $\mu_-$ & $\Delta$ & $p$ & $\mu_+$ & $\mu_-$ & $\Delta$ & $p$ & $\mu_+$ & $\mu_-$ & $\Delta$ & $p$ \\
\midrule""")

    dim_labels = {
        "factual_accuracy":   "Factual Accuracy",
        "clinical_relevance": "Clinical Relevance",
        "completeness":       "Completeness",
        "coherence":          "Coherence",
    }
    for dim in DIMENSIONS:
        row_parts = [dim_labels[dim]]
        for jlabel in [j["label"] for j in JUDGES]:
            s = per_judge[jlabel][dim]
            if s["delta"] is None:
                row_parts.append(r"\multicolumn{4}{c|}{N/A}")
            else:
                sig = r"\textbf{" + f"{s['delta']:+.3f}" + "}" if (s["p_value"] is not None and s["p_value"] < 0.05) else f"{s['delta']:+.3f}"
                row_parts.append(
                    f"{s['mean_with']:.2f} & {s['mean_no']:.2f} & {sig} & {s['p_value']:.3f}"
                )
        # last column group has no trailing |
        line = row_parts[0]
        for i, part in enumerate(row_parts[1:], 1):
            line += " & " + part
        print(line + r" \\")

    print(r"""\bottomrule
\multicolumn{13}{l}{\small $\mu_+$: with RAG, $\mu_-$: without RAG, $\Delta = \mu_+ - \mu_-$; bold $\Delta$ indicates $p{<}0.05$.}
\end{tabular}}
\end{table}""")


if __name__ == "__main__":
    main()
