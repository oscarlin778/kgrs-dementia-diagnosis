"""
eval_rag_llm_judge.py
=====================
用多個 LLM（本機 Ollama + Gemini API）對 RAG 取回的 chunk 打相關性分數（0–3），
比較 Baseline (k=3) vs MMR (k=5) 兩種 retrieval 方法的 Mean Relevance Score。

評分標準：
  0 = 完全不相關
  1 = 部分相關（同主題但沒直接回答問題）
  2 = 相關（觸及問題核心概念）
  3 = 高度相關（直接提供支持問題的具體證據）

執行：
  conda activate AD
  python eval_rag_llm_judge.py
  python eval_rag_llm_judge.py --input results/rag_eval_result.json \
                               --output results/rag_llm_judge_result.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# ── 設定 ───────────────────────────────────────────────────────────────────
OLLAMA_BASE   = "http://localhost:11434"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 本機 Ollama 模型
OLLAMA_JUDGE_MODELS = [
    {"name": "gemma3:12b",  "label": "Gemma3-12B",  "backend": "ollama"},
    {"name": "qwen3.6:35b", "label": "Qwen3.6-35B", "backend": "ollama"},
]

# Gemini API 模型（有 key 才加入）
GEMINI_JUDGE_MODELS = [
    {"name": "gemini-2.5-flash", "label": "Gemini-2.5-Flash", "backend": "gemini"},
] if GEMINI_API_KEY else []

JUDGE_MODELS = OLLAMA_JUDGE_MODELS + GEMINI_JUDGE_MODELS

DEFAULT_INPUT  = Path(__file__).parent / "results/rag_eval_result.json"
DEFAULT_OUTPUT = Path(__file__).parent / "results/rag_llm_judge_result.json"

SCORE_PROMPT_TEMPLATE = """\
You are an expert evaluator for a medical literature retrieval system focused on Alzheimer's disease and dementia.

Your task: Rate how relevant the retrieved text chunk is to the given question.

Question:
{query}

Retrieved chunk:
{chunk}

Scoring scale:
0 = Completely irrelevant (off-topic)
1 = Partially relevant (same broad topic, but does not address the question)
2 = Relevant (addresses the core concept of the question)
3 = Highly relevant (directly provides evidence or information that answers the question)

Respond with ONLY a single integer: 0, 1, 2, or 3. No explanation."""


# ── Ollama 呼叫 ────────────────────────────────────────────────────────────
def call_ollama(model: str, prompt: str, retries: int = 3) -> str:
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 16},
    }
    if "qwen3" in model.lower():
        payload["think"] = False

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"    ⚠️  {model} 第 {attempt}/{retries} 次失敗: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return ""


# ── Gemini API 呼叫 ────────────────────────────────────────────────────────
def call_gemini(model: str, prompt: str, retries: int = 3) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = response.text
            return text.strip() if text else ""
        except Exception as e:
            print(f"    ⚠️  Gemini {model} 第 {attempt}/{retries} 次失敗: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return ""


# ── 統一呼叫入口 ───────────────────────────────────────────────────────────
def call_judge(model_cfg: dict, prompt: str) -> str:
    if model_cfg["backend"] == "gemini":
        return call_gemini(model_cfg["name"], prompt)
    return call_ollama(model_cfg["name"], prompt)


# ── 解析分數 ───────────────────────────────────────────────────────────────
def parse_score(text: str) -> int | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    matches = re.findall(r"\b([0-3])\b", text)
    if matches:
        return int(matches[0])
    return None


# ── 單題評分 ───────────────────────────────────────────────────────────────
def score_results(model_cfg: dict, query: str, retrieval_results: list[dict]) -> dict:
    scores = []
    for r in retrieval_results:
        chunk_text = r.get("snippet", "")[:500]
        prompt = SCORE_PROMPT_TEMPLATE.format(query=query, chunk=chunk_text)
        raw = call_judge(model_cfg, prompt)
        score = parse_score(raw)
        scores.append(score)

    valid = [s for s in scores if s is not None]
    mean  = sum(valid) / len(valid) if valid else 0.0
    return {"scores": scores, "mean": round(mean, 3), "valid_count": len(valid)}


# ── 主評估流程 ─────────────────────────────────────────────────────────────
def run_judge_eval(eval_data: dict) -> dict:
    per_query = eval_data["per_query"]
    total_q   = len(per_query)
    all_results = []

    for model_cfg in JUDGE_MODELS:
        label = model_cfg["label"]
        print(f"\n{'═'*70}")
        print(f"Judge Model: {label}  (backend={model_cfg['backend']})")
        print(f"{'─'*70}")
        print(f"{'QID':<6} {'Category':<22} {'Baseline Mean':<16} {'MMR Mean'}")
        print(f"{'─'*70}")

        model_rows = []
        b_means, m_means = [], []

        for i, pq in enumerate(per_query, 1):
            qid      = pq["id"]
            category = pq["category"]
            query    = pq["query"]

            b_scored = score_results(model_cfg, query, pq["baseline"]["results"])
            m_scored = score_results(model_cfg, query, pq["mmr"]["results"])

            b_means.append(b_scored["mean"])
            m_means.append(m_scored["mean"])
            print(f"{qid:<6} {category:<22} {b_scored['mean']:<16.3f} {m_scored['mean']:.3f}")

            model_rows.append({
                "id":       qid,
                "category": category,
                "query":    query,
                "baseline_judge": b_scored,
                "mmr_judge":      m_scored,
            })

            if i % 5 == 0:
                print(f"  ... {i}/{total_q} 題完成")

        b_overall = round(sum(b_means) / len(b_means), 3)
        m_overall = round(sum(m_means) / len(m_means), 3)
        print(f"\n{'─'*70}")
        print(f"{'Overall Mean Relevance':<30} {b_overall:<16} {m_overall}")

        from collections import defaultdict
        cat_b: dict = defaultdict(list)
        cat_m: dict = defaultdict(list)
        for row in model_rows:
            cat_b[row["category"]].append(row["baseline_judge"]["mean"])
            cat_m[row["category"]].append(row["mmr_judge"]["mean"])

        print(f"\n  按類別 Mean Relevance（{label}）：")
        for cat in sorted(cat_b.keys()):
            b_cat = round(sum(cat_b[cat]) / len(cat_b[cat]), 3)
            m_cat = round(sum(cat_m[cat]) / len(cat_m[cat]), 3)
            print(f"    {cat:<22} Baseline={b_cat}  MMR={m_cat}")

        all_results.append({
            "model":    label,
            "model_id": model_cfg["name"],
            "backend":  model_cfg["backend"],
            "overall": {
                "baseline_mean_relevance": b_overall,
                "mmr_mean_relevance":      m_overall,
            },
            "by_category": {
                cat: {
                    "baseline": round(sum(cat_b[cat]) / len(cat_b[cat]), 3),
                    "mmr":      round(sum(cat_m[cat]) / len(cat_m[cat]), 3),
                }
                for cat in sorted(cat_b.keys())
            },
            "per_query": model_rows,
        })

    # ── Inter-model agreement ──────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("Inter-Model Agreement Summary")
    print(f"{'─'*70}")
    header = f"{'Method':<12}"
    for r in all_results:
        header += f"  {r['model']:<18}"
    print(header)
    print(f"{'─'*70}")
    for method_key, label in [("baseline_mean_relevance", "BASELINE"), ("mmr_mean_relevance", "MMR")]:
        row_str = f"{label:<12}"
        for r in all_results:
            row_str += f"  {r['overall'][method_key]:<18}"
        print(row_str)

    # ── 三模型平均（若有三個模型）─────────────────────────────────────────
    if len(all_results) >= 2:
        print(f"\n{'─'*70}")
        print("3-Judge Average:")
        avg_b = round(sum(r["overall"]["baseline_mean_relevance"] for r in all_results) / len(all_results), 3)
        avg_m = round(sum(r["overall"]["mmr_mean_relevance"]      for r in all_results) / len(all_results), 3)
        print(f"  BASELINE  avg={avg_b}")
        print(f"  MMR       avg={avg_m}")

    return {"judge_results": all_results}


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"找不到輸入檔: {input_path}")
        print("請先執行 eval_rag_retrieval.py --save-json results/rag_eval_result.json")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    total_chunks = sum(
        len(pq["baseline"]["results"]) + len(pq["mmr"]["results"])
        for pq in eval_data["per_query"]
    )
    total_calls = total_chunks * len(JUDGE_MODELS)
    print(f"載入 {len(eval_data['per_query'])} 題，共需評分 {total_calls} 次 LLM 呼叫")
    print(f"Judge 模型：{[m['label'] for m in JUDGE_MODELS]}")
    if not GEMINI_API_KEY:
        print("⚠️  未偵測到 GEMINI_API_KEY，跳過 Gemini judge")
    print("（Ollama 模型約 15–25 分鐘；Gemini 很快）\n")

    result = run_judge_eval(eval_data)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果已儲存至 {out}")


if __name__ == "__main__":
    main()
