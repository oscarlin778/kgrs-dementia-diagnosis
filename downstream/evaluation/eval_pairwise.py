"""
eval_pairwise.py
================
Pairwise comparison of RAG vs no-RAG reports using Gemma3-12B.
Addresses ceiling effect of absolute 1-3 scoring.

For each patient: judge sees both reports (A/B randomized), picks better one.
Win rate tested with one-sided binomial test (H0: RAG wins <= 50%).

Usage:
  conda activate AD
  cd downstream/
  python evaluation/eval_pairwise.py
"""
import argparse, json, re, time, random
import numpy as np
import requests
from pathlib import Path
from scipy.stats import binomtest

BASE_DIR   = Path(__file__).parent.parent
RES_DIR    = BASE_DIR / "results"
OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE      = "gemma3:12b"

PROMPT = """\
You are a senior neurologist reviewing AI-generated diagnostic reports for dementia staging.

Two reports were generated for the SAME patient (ID: {subject_id}, predicted diagnosis: {diagnosis}).
They differ only in whether external medical literature was used during generation.

=== REPORT A ===
{report_a}
=== END REPORT A ===

=== REPORT B ===
{report_b}
=== END REPORT B ===

Compare the two reports on clinical evidence quality. Consider ONLY:
1. Specificity of neural findings (are specific brain regions cited with quantitative values?)
2. Evidence grounding (are recommendations backed by clinical research?)
3. Clinical utility (how actionable are the follow-up recommendations for THIS patient?)

Do NOT favour a report simply because it is longer.

Respond ONLY with a JSON object:
{{"choice": "A" or "B" or "Tie", "reason": "<one sentence>"}}
"""


def judge_pair(subject_id, diagnosis, rep_rag, rep_norag, model=JUDGE):
    rag_is_a = random.random() < 0.5
    ra = rep_rag[:4000]  if rag_is_a else rep_norag[:4000]
    rb = rep_norag[:4000] if rag_is_a else rep_rag[:4000]

    prompt = PROMPT.format(subject_id=subject_id, diagnosis=diagnosis,
                           report_a=ra, report_b=rb)
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256}
        }, timeout=180)
        text = re.sub(r'<think>.*?</think>', '', r.json().get("response", ""), flags=re.DOTALL)
        matches = re.findall(r'\{[^{}]*"choice"[^{}]*\}', text, re.DOTALL)
        if matches:
            obj = json.loads(matches[-1])
        else:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            obj = json.loads(m.group()) if m else None
        if obj:
            choice = obj.get("choice", "").strip().upper()
            reason = obj.get("reason", "")
            if choice == "TIE":
                return "tie", reason, rag_is_a
            elif choice in ("A", "B"):
                rag_wins = (choice == "A") == rag_is_a
                return "rag" if rag_wins else "norag", reason, rag_is_a
    except Exception as e:
        print(f"  [WARN] {e}")
    return "failed", "", rag_is_a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v5",    default=str(RES_DIR / "report_quality_v5_full_fix.json"))
    parser.add_argument("--out",   default=str(RES_DIR / "report_quality_pairwise.json"))
    parser.add_argument("--judge", default=JUDGE, help="Ollama model tag for judge")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    with open(args.v5, encoding="utf-8") as f:
        v5 = json.load(f)

    # Build per-patient dict
    patients = {}
    for rec in v5["records"]:
        pid = rec["subject_id"]
        if pid not in patients:
            patients[pid] = {"diagnosis": rec.get("diagnosis", "Unknown"),
                             "with_rag": None, "no_rag": None}
        if rec.get("report"):
            patients[pid][rec["condition"]] = rec["report"]

    pairs = [(pid, info) for pid, info in patients.items()
             if info["with_rag"] and info["no_rag"]]
    judge_model = args.judge
    print(f"Pairwise judge: {judge_model}  |  n={len(pairs)} patients\n")

    records = []
    counts = {"rag": 0, "norag": 0, "tie": 0, "failed": 0}

    for i, (pid, info) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {pid} ({info['diagnosis']})... ", end="", flush=True)
        outcome, reason, rag_is_a = judge_pair(
            pid, info["diagnosis"], info["with_rag"], info["no_rag"],
            model=judge_model)
        counts[outcome] = counts.get(outcome, 0) + 1
        tag = {"rag": "RAG wins", "norag": "no-RAG wins",
               "tie": "Tie", "failed": "FAILED"}[outcome]
        print(tag)
        records.append({"subject_id": pid, "diagnosis": info["diagnosis"],
                         "outcome": outcome, "rag_was_A": rag_is_a, "reason": reason})
        time.sleep(0.2)

    # Stats
    n_total   = counts["rag"] + counts["norag"] + counts["tie"]
    n_decided = counts["rag"] + counts["norag"]
    p_binom   = None
    if n_decided > 0:
        bt = binomtest(counts["rag"], n_decided, 0.5, alternative="greater")
        p_binom = float(bt.pvalue)

    # Including ties as 0.5 win
    win_rate_incl = (counts["rag"] + 0.5 * counts["tie"]) / n_total if n_total else None

    print(f"\n{'='*55}")
    print(f"n={n_total}  RAG wins: {counts['rag']}  Ties: {counts['tie']}  no-RAG wins: {counts['norag']}  Failed: {counts['failed']}")
    if n_decided > 0:
        print(f"RAG win rate (excl. ties): {counts['rag']/n_decided:.1%}")
    if win_rate_incl is not None:
        print(f"RAG win rate (ties=0.5):   {win_rate_incl:.1%}")
    if p_binom is not None:
        print(f"Binomial p (RAG > 50%):    {p_binom:.4f}")

    out = {"judge": judge_model, "n_patients": n_total,
           "rag_wins": counts["rag"], "ties": counts["tie"],
           "norag_wins": counts["norag"], "failures": counts["failed"],
           "rag_win_rate_excl_ties": counts["rag"]/n_decided if n_decided else None,
           "rag_win_rate_incl_ties": win_rate_incl,
           "p_binomial_greater": p_binom,
           "records": records}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
