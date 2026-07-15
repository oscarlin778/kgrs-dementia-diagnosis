"""
eval_citation_v9.py
===================
Re-run citation rate & precision evaluation on all 62 test patients.

For each patient:
  1. Generate with_rag report → API appends **參考文獻** with retrieved papers
  2. Generate no_rag  report → no appended references

Metrics:
  - citation_rate_rag    : fraction of RAG reports that contain appended reference section
  - citation_rate_norag  : fraction of no-RAG reports with any spontaneous in-text citation
  - citation_precision   : for RAG reports with ref section, fraction of in-text [N] markers
                           that correspond to an entry in the appended reference list

Saves:  results/report_quality_citation_v9.json
        results/report_quality_ragas_v9.json  (RAGAS faithfulness, re-run)

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_citation_v9.py
  python evaluation/eval_citation_v9.py --skip-ragas
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).parent.parent
RES_DIR  = BASE_DIR / "results"
sys.path.insert(0, str(BASE_DIR))

API_BASE = "http://localhost:8080"
LABEL_MAP = {0: "NC", 1: "MCI", 2: "AD"}

# ── Reuse payload builders from eval_report_quality ──────────────────────────
from evaluation.eval_report_quality import (
    load_predictions, load_matrix_paths, load_interpretability_caches,
    select_patients, build_inference_payload, compute_fmri_findings,
)

# ── Report generation (streaming, captures full text) ────────────────────────
def generate_report(subject_id: str, inference: dict, use_rag: bool,
                    timeout: int = 240) -> str:
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
                           json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode())
                        full_text += chunk.get("response", "")
                    except Exception:
                        # Non-JSON lines come from the appended reference section.
                        # iter_lines() strips newlines, so we restore them here so
                        # multi-line reference sections are structurally reconstructed.
                        full_text += line.decode() + "\n"
        return full_text.strip()
    except Exception as e:
        print(f"  [WARN] report generation failed (use_rag={use_rag}): {e}")
        return ""


# ── Citation analysis ─────────────────────────────────────────────────────────
# Matches the appended reference section that api_server.py yields at the end.
# The API yields:  \n\n---\n**參考文獻**\n\n[1] Title ...
# iter_lines() strips \n, so non-JSON lines are reconstructed with \n added back.
# Pattern accepts optional whitespace/newlines between --- and **參考文獻**.
REF_SECTION_PATTERN = re.compile(
    r'---\s*\n?\s*\*\*參考文獻\*\*',
    re.MULTILINE,
)
# Numbered entry in the appended ref section: [1] Title ...
REF_ENTRY_PATTERN = re.compile(r'^\[(\d+)\]\s+.+', re.MULTILINE)
# In-text citation marker anywhere in body: [1] or [1,2] or [1–3]
INTEXT_CITE_PATTERN = re.compile(r'\[(\d+(?:[,–\-]\d+)*)\]')
# Any spontaneous citation (author/year) in no-rag body
SPONTANEOUS_CITE = re.compile(
    r'et al\..*?\d{4}'
    r'|\(\w+.*?,?\s*\d{4}\)'
    r'|\[\d+\][^\n—─]',
    re.IGNORECASE,
)


def analyse_citation(report: str, use_rag: bool) -> dict:
    """
    Returns:
        has_ref_section   (bool)  – RAG-only: API appended a **參考文獻** section
        n_appended_refs   (int)   – number of entries in appended ref list
        n_intext_cites    (int)   – number of distinct [N] markers in body
        n_valid_cites     (int)   – in-text [N] where N ≤ n_appended_refs
        citation_rate     (bool)  – the per-report contribution to citation-rate metric
        citation_precision(float) – precision for this report
    """
    m = REF_SECTION_PATTERN.search(report)
    if m:
        ref_section  = report[m.start():]
        body         = report[:m.start()]
        has_ref_sec  = True
    else:
        ref_section  = ""
        body         = report
        has_ref_sec  = False

    # Count appended ref entries
    n_appended = len(REF_ENTRY_PATTERN.findall(ref_section))

    # In-text [N] in body
    intext_nums = set()
    for m2 in INTEXT_CITE_PATTERN.finditer(body):
        raw = m2.group(1)
        for part in re.split(r'[,–\-]', raw):
            part = part.strip()
            if part.isdigit():
                intext_nums.add(int(part))

    n_intext = len(intext_nums)
    n_valid  = sum(1 for n in intext_nums if n <= n_appended) if n_appended else 0
    precision = (n_valid / n_intext) if n_intext > 0 else (1.0 if has_ref_sec else float('nan'))

    if use_rag:
        cite_rate = has_ref_sec  # RAG: cited retrieved lit ⟺ ref section present
    else:
        cite_rate = bool(SPONTANEOUS_CITE.search(body))  # no-RAG: any spontaneous ref

    return {
        "has_ref_section":    has_ref_sec,
        "n_appended_refs":    n_appended,
        "n_intext_cites":     n_intext,
        "n_valid_cites":      n_valid,
        "citation_rate":      cite_rate,
        "citation_precision": precision,
    }


# ── RAGAS (faithfulness only, re-retrieves contexts from Neo4j) ──────────────
def run_ragas(records: list[dict], out_path: Path) -> dict | None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except ImportError as e:
        print(f"  [SKIP RAGAS] missing package: {e}")
        return None

    _runcfg = RunConfig(max_workers=2, timeout=600)
    llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0.1))

    sys.path.insert(0, str(BASE_DIR))
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env", override=True)
        import graph_rag_retriever as grr
        if not grr._VECTOR_AVAILABLE:
            print("  [SKIP RAGAS] Neo4j vector not available")
            return None
        retrieve_fn = grr.retrieve_medical_literature
    except Exception as e:
        print(f"  [SKIP RAGAS] {e}")
        return None

    ROI_PATTERN = re.compile(r'([A-Z][a-z]+(?:_[A-Za-z0-9]+){1,4})\s*\(')
    def extract_roi_names(report: str) -> list[str]:
        return list(dict.fromkeys(ROI_PATTERN.findall(report)))[:5]

    def get_contexts(report: str, diagnosis: str) -> list[str]:
        roi_names = extract_roi_names(report)
        try:
            lit_text, citations = retrieve_fn(diagnosis, roi_names)
            return [c["text"] for c in citations if c.get("text")]
        except Exception as e:
            print(f"    [WARN] retrieval: {e}")
            return []

    QUESTION_TMPL = (
        "What do the neuroimaging findings indicate for patient {subject_id} "
        "with predicted {diagnosis} diagnosis? "
        "Provide specific brain-region evidence and clinical follow-up recommendations."
    )

    rows = {"question": [], "answer": [], "contexts": []}
    pids_faith = []
    for rec in records:
        if rec["condition"] != "with_rag" or not rec["report"]:
            continue
        pid    = rec["subject_id"]
        diag   = rec["diagnosis"]
        ctxs   = get_contexts(rec["report"], diag)
        if not ctxs:
            print(f"  [SKIP] {pid}: no contexts")
            continue
        rows["question"].append(QUESTION_TMPL.format(subject_id=pid, diagnosis=diag))
        rows["answer"].append(rec["report"][:6000])
        rows["contexts"].append(ctxs)
        pids_faith.append(pid)

    if not pids_faith:
        print("  [SKIP RAGAS] no patients with contexts")
        return None

    print(f"  Running RAGAS faithfulness for {len(pids_faith)} patients...")
    ds = Dataset.from_dict(rows)
    t0 = time.time()
    res = evaluate(ds, metrics=[Faithfulness(llm=llm)],
                   raise_exceptions=False, run_config=_runcfg)
    scores = list(res["faithfulness"])
    mean_faith = float(np.nanmean(scores))
    print(f"  Done ({time.time()-t0:.0f}s). Faithfulness={mean_faith:.3f}  n={len(scores)}")

    result = {
        "n_patients": len(pids_faith),
        "faithfulness": {"mean": mean_faith, "n_evaluated": len(scores), "scores": scores},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {out_path}")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--out", default=str(RES_DIR / "report_quality_citation_v9.json"))
    parser.add_argument("--ragas-out", default=str(RES_DIR / "report_quality_ragas_v9.json"))
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    pred_map     = load_predictions(use_ensemble=True)
    matrix_paths = load_matrix_paths()
    caches, indexed = load_interpretability_caches()
    patients     = select_patients(pred_map, n_per_class=None)  # all 62 test patients
    df_label     = pd.read_csv(BASE_DIR / "pcag_test_aligned_v2.csv")\
                     .set_index("subject_id")["label"].to_dict()

    print(f"Evaluating {len(patients)} patients for citation metrics\n")

    records = []
    rag_cite_flags  = []
    norag_cite_flags = []
    rag_precision_list = []

    t_start = time.time()
    for idx, pid in enumerate(patients, 1):
        label    = df_label.get(pid, -1)
        diagnosis = LABEL_MAP.get(label, "Unknown")
        elapsed  = time.time() - t_start
        print(f"[{idx}/{len(patients)}] {pid}  true={diagnosis}  ({elapsed:.0f}s)")

        inference = build_inference_payload(pid, pred_map, matrix_paths,
                                            caches=caches, indexed=indexed)
        overall   = inference["overall_prediction"]

        # Inject fmri_findings into task_results so the API can retrieve papers.
        # build_inference_payload sets fmri_findings=None to bypass old Neo4j lookup;
        # we re-compute it here from the FC matrix so retrieve_multimodal gets ROI names.
        mpath = matrix_paths.get(pid)
        fmri_findings = compute_fmri_findings(mpath) if mpath else None
        if fmri_findings:
            for task_info in inference["task_results"].values():
                task_info["fmri_findings"] = fmri_findings

        for use_rag in [True, False]:
            tag = "with_rag" if use_rag else "no_rag"
            print(f"  Generating ({tag})...", end=" ", flush=True)
            report = generate_report(pid, inference, use_rag)
            print(f"{len(report)} chars")

            stats = analyse_citation(report, use_rag)
            print(f"    ref_section={stats['has_ref_section']}  "
                  f"appended={stats['n_appended_refs']}  "
                  f"intext={stats['n_intext_cites']}  "
                  f"valid={stats['n_valid_cites']}  "
                  f"cite_rate={stats['citation_rate']}  "
                  f"precision={stats['citation_precision']:.2f}" if not isinstance(stats['citation_precision'], float) or not np.isnan(stats['citation_precision']) else
                  f"    ref_section={stats['has_ref_section']}  cite_rate={stats['citation_rate']}")

            records.append({
                "subject_id":   pid,
                "true_label":   label,
                "diagnosis":    diagnosis,
                "overall_pred": overall,
                "condition":    tag,
                "report_len":   len(report),
                "report":       report,
                **stats,
            })

            if use_rag:
                rag_cite_flags.append(int(stats["citation_rate"]))
                if not np.isnan(stats["citation_precision"]):
                    rag_precision_list.append(stats["citation_precision"])
            else:
                norag_cite_flags.append(int(stats["citation_rate"]))

    # ── Summary ──────────────────────────────────────────────────────────────
    rag_rate    = float(np.mean(rag_cite_flags))
    norag_rate  = float(np.mean(norag_cite_flags))
    rag_prec    = float(np.nanmean(rag_precision_list)) if rag_precision_list else float('nan')

    n_patients  = len(patients)
    print(f"\n{'='*55}")
    print(f"n={n_patients}")
    print(f"RAG   citation rate:      {rag_rate:.1%}  ({sum(rag_cite_flags)}/{len(rag_cite_flags)})")
    print(f"no-RAG citation rate:     {norag_rate:.1%}  ({sum(norag_cite_flags)}/{len(norag_cite_flags)})")
    print(f"RAG   citation precision: {rag_prec:.1%}")
    print(f"{'='*55}")

    summary = {
        "n_patients":              n_patients,
        "citation_rate_rag":       rag_rate,
        "citation_rate_norag":     norag_rate,
        "citation_precision_rag":  rag_prec,
        "n_rag_with_ref_section":  sum(r["has_ref_section"] for r in records if r["condition"] == "with_rag"),
        "n_appended_refs_mean":    float(np.mean([r["n_appended_refs"] for r in records if r["condition"] == "with_rag"])),
    }

    output = {"summary": summary, "records": records}
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {out_path}")

    # ── RAGAS ────────────────────────────────────────────────────────────────
    if not args.skip_ragas:
        print("\n--- RAGAS Faithfulness ---")
        run_ragas(records, Path(args.ragas_out))

    print("\nDone.")
    return summary


if __name__ == "__main__":
    main()
