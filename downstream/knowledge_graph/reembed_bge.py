"""
reembed_bge.py
==============
Re-embed all :Chunk literature nodes with bge-m3 (1024-d) into a SEPARATE
property `embedding_bge` + a parallel vector index `chunk_bge_index`.
The original nomic embedding/index is left intact (safe rollback).

Usage:
  conda activate AD
  cd downstream/
  python knowledge_graph/reembed_bge.py
"""
import sys, time, requests
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
import graph_rag_retriever as grr

OLLAMA = "http://localhost:11434/api/embeddings"
MODEL = "bge-m3"
DIM = 1024


def embed(text):
    r = requests.post(OLLAMA, json={"model": MODEL, "prompt": text[:8000]}, timeout=60)
    return r.json()["embedding"]


def main():
    drv = grr.graph_driver
    with drv.session() as s:
        rows = s.run("MATCH (n:Chunk) WHERE n.text IS NOT NULL "
                     "RETURN elementId(n) AS eid, n.text AS text").data()
    print(f"Chunks to embed: {len(rows)}")

    t0 = time.time()
    for i, row in enumerate(rows, 1):
        try:
            vec = embed(row["text"])
        except Exception as e:
            print(f"  [WARN] embed failed for {row['eid']}: {e}")
            continue
        with drv.session() as s:
            s.run("MATCH (n) WHERE elementId(n)=$eid SET n.embedding_bge=$v",
                  eid=row["eid"], v=vec)
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}  ({time.time()-t0:.0f}s)")
    print(f"Embedded {len(rows)} chunks in {time.time()-t0:.0f}s")

    # Create parallel vector index
    with drv.session() as s:
        s.run("DROP INDEX chunk_bge_index IF EXISTS")
        s.run(
            "CREATE VECTOR INDEX chunk_bge_index IF NOT EXISTS "
            "FOR (n:Chunk) ON (n.embedding_bge) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, "
            "`vector.similarity_function`: 'cosine'}}", dim=DIM)
    print("[DONE] index chunk_bge_index created")


if __name__ == "__main__":
    main()
