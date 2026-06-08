"""
eval_rag_retrieval.py
=====================
評估 Neo4j Vector RAG 系統的 retrieval 效能。

指標：
  - Diversity@k    : k 個結果來自幾篇不同的 paper（越高越好，最高 k）
  - Title Coverage : 30 題共用到幾篇不同 paper（最高 19）
  - Papers Hit     : 每題命中的 paper 清單

比較：
  - Method A (Baseline): similarity_search, k=3（修改前）
  - Method B (MMR-5)   : max_marginal_relevance_search, k=5 + paper dedup（修改後）

執行：
  conda activate AD
  python eval_rag_retrieval.py
  python eval_rag_retrieval.py --save-json results/rag_eval_result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

# ── 連線設定 ────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "nm6131027")

QUERIES_FILE = Path(__file__).parent / "docs" / "rag_test_queries.txt"

# ── 讀取 30 題 ───────────────────────────────────────────────────────────────
def load_queries(path: Path) -> list[dict]:
    """解析 rag_test_queries.txt，回傳 [{id, category, text}, ...]"""
    queries = []
    category = ""
    cat_map = {
        "A": "Hippocampus/sMRI",
        "B": "DMN/fMRI",
        "C": "Deep Learning",
        "D": "Cognitive Reserve",
        "E": "Biomarkers",
        "F": "Clinical AI",
    }
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("A.") or line.startswith("B.") or line.startswith("C.") \
               or line.startswith("D.") or line.startswith("E.") or line.startswith("F."):
                category = cat_map.get(line[0], line[0])
            elif line.startswith("Q") and ":" in line:
                qid, text = line.split(":", 1)
                queries.append({"id": qid.strip(), "category": category, "text": text.strip()})
    return queries


# ── 初始化 Vector Store ──────────────────────────────────────────────────────
def init_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="chunk_embedding_index",
        text_node_property="text",
    )
    return vs


# ── 兩種 retrieval 方法 ──────────────────────────────────────────────────────
def retrieve_baseline(vs, query_text: str, k: int = 3) -> list[dict]:
    """Method A: 原始 similarity_search k=3"""
    docs = vs.similarity_search(query_text, k=k)
    return [
        {
            "title": d.metadata.get("title", "Unknown"),
            "source": d.metadata.get("source", d.metadata.get("title", "Unknown")),
            "snippet": d.page_content[:120].replace("\n", " "),
        }
        for d in docs
    ]


def retrieve_mmr(vs, query_text: str, k: int = 5, fetch_k: int = 20) -> list[dict]:
    """Method B: MMR k=5 + paper-level dedup"""
    try:
        candidates = vs.max_marginal_relevance_search(query_text, k=k, fetch_k=fetch_k)
    except Exception as e:
        print(f"    ⚠️  MMR failed ({e}), fallback to similarity_search k={fetch_k}")
        candidates = vs.similarity_search(query_text, k=fetch_k)

    seen: set = set()
    deduped = []
    for d in candidates:
        src = d.metadata.get("source") or d.metadata.get("title") or str(len(seen))
        if src not in seen:
            seen.add(src)
            deduped.append(d)
        if len(deduped) >= k:
            break

    return [
        {
            "title": d.metadata.get("title", "Unknown"),
            "source": d.metadata.get("source", d.metadata.get("title", "Unknown")),
            "snippet": d.page_content[:120].replace("\n", " "),
        }
        for d in deduped
    ]


# ── 計算指標 ─────────────────────────────────────────────────────────────────
def compute_diversity(results: list[dict]) -> int:
    """Diversity@k：結果來自幾篇不同的 paper"""
    return len(set(r["source"] for r in results))


def run_evaluation(vs, queries: list[dict]) -> dict:
    all_baseline, all_mmr = [], []
    per_query = []

    print(f"\n{'─'*70}")
    print(f"{'QID':<6} {'Category':<20} {'Baseline Div@3':<16} {'MMR Div@5':<12}")
    print(f"{'─'*70}")

    for q in queries:
        b_results = retrieve_baseline(vs, q["text"], k=3)
        m_results = retrieve_mmr(vs, q["text"], k=5, fetch_k=20)

        b_div = compute_diversity(b_results)
        m_div = compute_diversity(m_results)

        b_titles = [r["title"] for r in b_results]
        m_titles = [r["title"] for r in m_results]

        all_baseline.extend(r["source"] for r in b_results)
        all_mmr.extend(r["source"] for r in m_results)

        print(f"{q['id']:<6} {q['category']:<20} {b_div:<16} {m_div:<12}")

        per_query.append({
            "id": q["id"],
            "category": q["category"],
            "query": q["text"],
            "baseline": {"diversity": b_div, "titles": b_titles, "results": b_results},
            "mmr":      {"diversity": m_div, "titles": m_titles, "results": m_results},
        })

    # 彙總
    b_coverage = len(set(all_baseline))
    m_coverage = len(set(all_mmr))
    b_avg_div  = sum(pq["baseline"]["diversity"] for pq in per_query) / len(per_query)
    m_avg_div  = sum(pq["mmr"]["diversity"]      for pq in per_query) / len(per_query)

    print(f"\n{'═'*70}")
    print(f"{'指標':<30} {'Baseline (k=3)':<18} {'MMR (k=5)'}")
    print(f"{'─'*70}")
    print(f"{'Avg Diversity@k':<30} {b_avg_div:<18.2f} {m_avg_div:.2f}")
    print(f"{'Title Coverage (unique papers)':<30} {b_coverage:<18} {m_coverage}")
    print(f"{'Total queries':<30} {len(queries)}")
    print(f"{'Total papers in DB':<30} 25")
    print(f"{'═'*70}\n")

    return {
        "summary": {
            "total_queries": len(queries),
            "baseline": {"avg_diversity": round(b_avg_div, 3), "title_coverage": b_coverage},
            "mmr":      {"avg_diversity": round(m_avg_div, 3), "title_coverage": m_coverage},
        },
        "per_query": per_query,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-json", metavar="PATH", help="將完整結果存成 JSON 檔")
    args = parser.parse_args()

    if not QUERIES_FILE.exists():
        print(f"找不到 {QUERIES_FILE}")
        sys.exit(1)

    queries = load_queries(QUERIES_FILE)
    print(f"載入 {len(queries)} 題測試 query")

    print("初始化 Neo4j Vector Store...")
    vs = init_vector_store()
    print("連線成功，開始評估...\n")

    result = run_evaluation(vs, queries)

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"結果已儲存至 {out}")


if __name__ == "__main__":
    main()
