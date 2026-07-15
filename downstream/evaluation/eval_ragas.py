"""
eval_ragas.py
=============
RAGAS evaluation of RAG vs no-RAG report quality.

Metrics:
  1. Answer Relevance  – RAG vs no-RAG (no contexts needed)
  2. Faithfulness      – RAG only: % of report claims grounded in retrieved literature

Answer Relevance uses nomic-embed-text (local Ollama).
Faithfulness re-retrieves contexts from Neo4j vector index.

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_ragas.py
  python evaluation/eval_ragas.py --skip-faithfulness   # Answer Relevance only
"""
import argparse, json, os, re, sys, time
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# ── RAGAS + LangChain ────────────────────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# Local Ollama can't serve 16 concurrent requests -> jobs time out and scores
# become NaN. Throttle concurrency and lengthen the timeout for reliability.
_RUNCFG = RunConfig(max_workers=2, timeout=600)
from langchain_ollama import ChatOllama, OllamaEmbeddings

LLM_MODEL = "llama3.1:8b"
EMB_MODEL  = "nomic-embed-text"
RES_DIR    = BASE_DIR / "results"

QUESTION_TMPL = (
    "What do the neuroimaging findings indicate for patient {subject_id} "
    "with predicted {diagnosis} diagnosis? "
    "Provide specific brain-region evidence and clinical follow-up recommendations."
)


def make_question(subject_id: str, diagnosis: str) -> str:
    return QUESTION_TMPL.format(subject_id=subject_id, diagnosis=diagnosis)


# ── Context retrieval (for Faithfulness) ─────────────────────────────────────
ROI_PATTERN = re.compile(
    r'([A-Z][a-z]+(?:_[A-Za-z0-9]+){1,4})\s*\('
)

def extract_roi_names(report: str) -> list[str]:
    """Extract AAL-style ROI names from report text."""
    return list(dict.fromkeys(ROI_PATTERN.findall(report)))[:5]


_retriever_loaded = False
_retrieve_fn = None

def load_retriever():
    global _retriever_loaded, _retrieve_fn
    if _retriever_loaded:
        return _retrieve_fn is not None
    _retriever_loaded = True
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env", override=True)
        import graph_rag_retriever as grr
        if grr._VECTOR_AVAILABLE:
            _retrieve_fn = grr.retrieve_medical_literature
            print("Neo4j vector store connected.")
            return True
        else:
            print("Vector store unavailable — skipping Faithfulness.")
    except Exception as e:
        print(f"Could not load retriever: {e}")
    return False


def get_contexts(report: str, diagnosis: str) -> list[str]:
    """Re-retrieve literature contexts used for this patient's report."""
    if _retrieve_fn is None:
        return []
    roi_names = extract_roi_names(report)
    patient_ctx = diagnosis
    try:
        lit_text, citations = _retrieve_fn(patient_ctx, roi_names)
        return [c["text"] for c in citations if c.get("text")]
    except Exception as e:
        print(f"  [WARN] retrieval failed: {e}")
        return []


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5",  default=str(RES_DIR / "report_quality_v5_full_fix.json"))
    parser.add_argument("--out", default=str(RES_DIR / "report_quality_ragas.json"))
    parser.add_argument("--skip-faithfulness", action="store_true")
    args = parser.parse_args()

    # ── Build LLM / embedding wrappers ───────────────────────────────────────
    print(f"LLM: {LLM_MODEL}  |  Embeddings: {EMB_MODEL}")
    llm = LangchainLLMWrapper(ChatOllama(model=LLM_MODEL, temperature=0.1))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMB_MODEL))
    ar_metric = AnswerRelevancy(llm=llm, embeddings=emb)
    fi_metric  = Faithfulness(llm=llm)

    # ── Load reports ──────────────────────────────────────────────────────────
    with open(args.v5, encoding="utf-8") as f:
        v5 = json.load(f)

    patients = {}
    for rec in v5["records"]:
        pid = rec["subject_id"]
        if pid not in patients:
            patients[pid] = {"diagnosis": rec.get("diagnosis", "Unknown"),
                             "with_rag": None, "no_rag": None}
        if rec.get("report"):
            patients[pid][rec["condition"]] = rec["report"]

    valid = [(pid, info) for pid, info in patients.items()
             if info["with_rag"] and info["no_rag"]]
    print(f"Patients with both conditions: {len(valid)}")

    # ── Load retriever for Faithfulness ───────────────────────────────────────
    do_faith = not args.skip_faithfulness and load_retriever()

    # ── Answer Relevance: both conditions ────────────────────────────────────
    print("\n--- Answer Relevance (RAG vs no-RAG) ---")
    ar_records = {"with_rag": [], "no_rag": []}

    for cond in ("with_rag", "no_rag"):
        rows = {"question": [], "answer": [], "contexts": []}
        pids_order = []
        for pid, info in valid:
            report = info[cond]
            if not report:
                continue
            rows["question"].append(make_question(pid, info["diagnosis"]))
            rows["answer"].append(report[:6000])
            rows["contexts"].append([])      # AR doesn't use contexts
            pids_order.append(pid)

        print(f"  Evaluating {cond} ({len(pids_order)} reports)...", flush=True)
        ds = Dataset.from_dict(rows)
        t0 = time.time()
        res = evaluate(ds, metrics=[ar_metric], raise_exceptions=False, run_config=_RUNCFG)
        print(f"  Done ({time.time()-t0:.0f}s). Mean AR = {np.mean(res['answer_relevancy']):.3f}")
        ar_records[cond] = list(res["answer_relevancy"])

    # Paired stats
    rag_ar   = np.array(ar_records["with_rag"])
    norag_ar = np.array(ar_records["no_rag"])
    diff = rag_ar - norag_ar
    if len(diff) >= 2 and not np.allclose(diff, 0):
        _, p_ar = wilcoxon(rag_ar, norag_ar, zero_method="zsplit", alternative="two-sided")
    else:
        p_ar = 1.0
    delta_ar = float(diff.mean())
    print(f"\n  Answer Relevance: RAG={rag_ar.mean():.3f}  no-RAG={norag_ar.mean():.3f}"
          f"  Δ={delta_ar:+.3f}  p={p_ar:.4f}")

    # ── Faithfulness: RAG only ────────────────────────────────────────────────
    faith_scores = []
    if do_faith:
        print("\n--- Faithfulness (RAG only, re-retrieving contexts) ---")
        rows = {"question": [], "answer": [], "contexts": []}
        pids_faith = []
        for pid, info in valid:
            report = info["with_rag"]
            if not report:
                continue
            ctxs = get_contexts(report, info["diagnosis"])
            if not ctxs:
                print(f"  [SKIP] {pid}: no contexts retrieved")
                continue
            rows["question"].append(make_question(pid, info["diagnosis"]))
            rows["answer"].append(report[:6000])
            rows["contexts"].append(ctxs)
            pids_faith.append(pid)

        if pids_faith:
            print(f"  Evaluating faithfulness for {len(pids_faith)} patients...", flush=True)
            ds = Dataset.from_dict(rows)
            t0 = time.time()
            res = evaluate(ds, metrics=[fi_metric], raise_exceptions=False, run_config=_RUNCFG)
            faith_scores = list(res["faithfulness"])
            print(f"  Done ({time.time()-t0:.0f}s). Mean Faithfulness = {np.mean(faith_scores):.3f}")
    else:
        print("\nFaithfulness skipped.")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "llm": LLM_MODEL,
        "embeddings": EMB_MODEL,
        "n_patients": len(valid),
        "answer_relevance": {
            "rag_mean":   float(rag_ar.mean()),
            "norag_mean": float(norag_ar.mean()),
            "delta":      delta_ar,
            "p_wilcoxon": float(p_ar),
            "rag_scores":   ar_records["with_rag"],
            "norag_scores": ar_records["no_rag"],
        },
        "faithfulness": {
            "n_evaluated": len(faith_scores),
            "mean": float(np.mean(faith_scores)) if faith_scores else None,
            "scores": faith_scores,
        }
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {args.out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"Answer Relevance: Δ={delta_ar:+.3f}  p={p_ar:.4f}")
    if faith_scores:
        print(f"Faithfulness (RAG): mean={np.mean(faith_scores):.3f}  "
              f"n={len(faith_scores)}")


if __name__ == "__main__":
    main()
