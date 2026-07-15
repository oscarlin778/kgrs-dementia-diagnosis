"""
eval_context_precision.py
=========================
Measures retrieval quality of the KG-RAG pipeline via Context Precision.

For each patient (with_rag condition):
  1. Re-retrieve k=5 papers using the same MMR process as generation time
  2. For each retrieved paper chunk, ask LLM: is this relevant to the patient's diagnosis?
  3. Context Precision = fraction of retrieved chunks judged relevant
  4. Report mean ± std across all patients

Judge model: llama3.1:8b (cross-family from Gemma3 generator)

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_context_precision.py
"""
import json, os, re, sys, time
import numpy as np
import requests
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
RES_DIR    = BASE_DIR / "results"
OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE      = "llama3.1:8b"

sys.path.insert(0, str(BASE_DIR))

RELEVANCE_PROMPT = """\
You are a clinical neuroscience expert evaluating retrieval quality for a dementia diagnostic system.

The system performs neuroimaging analysis (fMRI/sMRI) to differentiate between Normal Cognition (NC), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD). A patient has been assessed with predicted diagnosis: {diagnosis}.

The following paper excerpt was retrieved to support this patient's clinical report:

--- PAPER EXCERPT ---
{chunk}
--- END EXCERPT ---

Question: Is this paper excerpt relevant to dementia neuroimaging research, including topics such as resting-state fMRI connectivity, sMRI structural analysis, cognitive reserve, MCI/AD biomarkers, brain network changes, or harmonisation of neuroimaging data?

Answer with ONLY "YES" or "NO".
"""


def judge_relevance(chunk: str, diagnosis: str) -> bool | None:
    prompt = RELEVANCE_PROMPT.format(
        diagnosis=diagnosis,
        chunk=chunk[:1500]
    )
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": JUDGE,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 8}
        }, timeout=60)
        text = r.json().get("response", "").strip().upper()
        if "YES" in text:
            return True
        elif "NO" in text:
            return False
    except Exception as e:
        print(f"    [WARN] judge failed: {e}")
    return None


def load_retriever():
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=True)
    import graph_rag_retriever as grr
    if not grr._VECTOR_AVAILABLE:
        raise RuntimeError("Neo4j vector store not available.")
    return grr.retrieve_medical_literature


ROI_PATTERN = re.compile(r'([A-Z][a-z]+(?:_[A-Za-z0-9]+){1,4})\s*\(')

def extract_roi_names(report: str) -> list[str]:
    return list(dict.fromkeys(ROI_PATTERN.findall(report)))[:5]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5",  default=str(RES_DIR / "report_quality_v5_full_fix.json"))
    parser.add_argument("--out", default=str(RES_DIR / "report_quality_context_precision.json"))
    args = parser.parse_args()

    print(f"Judge: {JUDGE}")
    retrieve_fn = load_retriever()
    print("Neo4j vector store connected.\n")

    with open(args.v5, encoding="utf-8") as f:
        v5 = json.load(f)

    # Build per-patient lookup (with_rag only)
    patients = {}
    for rec in v5["records"]:
        pid = rec["subject_id"]
        if rec["condition"] == "with_rag" and rec.get("report"):
            patients[pid] = {
                "diagnosis": rec.get("diagnosis", "Unknown"),
                "report":    rec["report"],
            }

    print(f"Evaluating context precision for {len(patients)} patients...\n")

    patient_precisions = []
    records = []

    for i, (pid, info) in enumerate(patients.items(), 1):
        diagnosis = info["diagnosis"]
        report    = info["report"]
        roi_names = extract_roi_names(report)

        print(f"[{i}/{len(patients)}] {pid} ({diagnosis})")
        print(f"  Retrieving context...", end=" ", flush=True)

        try:
            _, citations = retrieve_fn(diagnosis, roi_names)
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        if not citations:
            print("no citations retrieved, skipping")
            continue

        print(f"{len(citations)} chunks retrieved")

        chunk_results = []
        for j, cit in enumerate(citations):
            chunk = cit.get("text", "")
            title = cit.get("title", f"Ref {j+1}")
            if not chunk:
                continue
            relevant = judge_relevance(chunk, diagnosis)
            label = "YES" if relevant else ("NO" if relevant is False else "?")
            print(f"    [{j+1}] {title[:60]:60s} → {label}")
            chunk_results.append({
                "title":    title,
                "relevant": relevant,
            })
            time.sleep(0.1)

        valid = [c for c in chunk_results if c["relevant"] is not None]
        if valid:
            precision = sum(1 for c in valid if c["relevant"]) / len(valid)
            patient_precisions.append(precision)
            print(f"  Context Precision = {precision:.3f} ({sum(1 for c in valid if c['relevant'])}/{len(valid)})")
        else:
            precision = None
            print("  No valid judgements.")

        records.append({
            "subject_id": pid,
            "diagnosis":  diagnosis,
            "n_retrieved": len(citations),
            "chunks":     chunk_results,
            "precision":  precision,
        })

    # Summary
    mean_cp = float(np.mean(patient_precisions)) if patient_precisions else None
    std_cp  = float(np.std(patient_precisions))  if patient_precisions else None

    print(f"\n{'='*55}")
    print(f"Context Precision (n={len(patient_precisions)} patients)")
    if mean_cp is not None:
        print(f"  Mean  = {mean_cp:.3f}")
        print(f"  Std   = {std_cp:.3f}")
        print(f"  Min   = {min(patient_precisions):.3f}")
        print(f"  Max   = {max(patient_precisions):.3f}")

    output = {
        "judge":         JUDGE,
        "n_patients":    len(patient_precisions),
        "mean_precision": mean_cp,
        "std_precision":  std_cp,
        "patient_precisions": patient_precisions,
        "records":       records,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
