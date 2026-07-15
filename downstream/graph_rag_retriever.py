from neo4j import GraphDatabase
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
import os, sys, argparse, warnings
from dotenv import load_dotenv

# 載入環境變數 (.env 檔案位於 script 根目錄)
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dotenv_path)

# ==========================================
# 1. 系統設定與連線
# ==========================================
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

_GRAPH_AVAILABLE = True
_VECTOR_AVAILABLE = True

# 初始化 Graph 驅動
try:
    graph_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # 測試連線
    with graph_driver.session() as session:
        session.run("RETURN 1").single()
except Exception as e:
    print(f"⚠️  Neo4j 連線失敗，Graph 功能將無法使用: {e}")
    graph_driver = None
    _GRAPH_AVAILABLE = False

# 初始化 Vector 驅動 (使用本地 Ollama)
# RAG_EMBED=bge (default) → bge-m3 + chunk_bge_index (stronger retrieval, English corpus);
# RAG_EMBED=nomic → original nomic-embed-text + chunk_embedding_index (rollback).
_EMBED_CFG = {
    "bge":   ("bge-m3",          "chunk_bge_index"),
    "nomic": ("nomic-embed-text", "chunk_embedding_index"),
}
_embed_choice = os.getenv("RAG_EMBED", "bge").lower()
_embed_model, _embed_index = _EMBED_CFG.get(_embed_choice, _EMBED_CFG["bge"])
try:
    embeddings = OllamaEmbeddings(model=_embed_model)
    vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name=_embed_index,
        text_node_property="text"
    )
    print(f"[RAG] vector store: {_embed_model} / {_embed_index}")
except Exception as e:
    print(f"⚠️  Ollama/Vector Index 連線失敗，文獻檢索功能將無法使用: {e}")
    embeddings = None
    vector_store = None
    _VECTOR_AVAILABLE = False

# ==========================================
# 2. 病患節點 (Graph) 擷取
# ==========================================
def get_patient_graph_context(subject_id: str) -> str:
    """擷取病患的結構化人口統計脈絡，含 AI 校準機率"""
    if not _GRAPH_AVAILABLE:
        return "目前無該名病患的詳細人口統計學資料。"

    clean_id = subject_id.replace('sub-', '')
    query = """
    MATCH (p:Patient {subject_id: $id})
    OPTIONAL MATCH (p)-[:HAS_PREDICTION]->(pred:AI_Prediction)
    RETURN p.age AS age, p.sex AS sex, p.education AS education, p.clinical_dx AS dx,
           pred.calibrated_prob_AD_ncad   AS prob_ad_ncad,
           pred.calibrated_prob_MCI_ncmci AS prob_mci_ncmci,
           pred.calibrated_prob_AD_mciad  AS prob_ad_mciad,
           pred.routing_model             AS routing_model
    """
    with graph_driver.session() as session:
        result = session.run(query, id=clean_id).single()
        if result is None:
            return "目前無該名病患的詳細人口統計學資料。"

        if result["age"] is not None and result["age"] > 0:
            demo = (f"病患為 {result['age']} 歲 {result['sex']}，"
                    f"教育年限 {result['education']} 年，臨床診斷為 {result['dx']}。")
        else:
            demo = "目前無該名病患的詳細人口統計學資料。"

        if result["prob_ad_ncad"] is not None:
            fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
            pred_ctx = (
                f"AI 校準預測：NC vs AD 機率={fmt(result['prob_ad_ncad'])}，"
                f"NC vs MCI 機率={fmt(result['prob_mci_ncmci'])}，"
                f"MCI vs AD 機率={fmt(result['prob_ad_mciad'])}"
                f"（路由模型：{result['routing_model']}）。"
            )
            return f"{demo}\n{pred_ctx}"
        return demo


def get_similar_patients_context(subject_id: str, top_k: int = 3) -> str:
    """透過 SIMILAR_TO 邊查詢最相似病患，作為臨床參考佐證"""
    if not _GRAPH_AVAILABLE:
        return ""

    clean_id = subject_id.replace('sub-', '')
    query = """
    MATCH (p:Patient {subject_id: $id})-[s:SIMILAR_TO]->(p2:Patient)
    OPTIONAL MATCH (p2)-[:HAS_PREDICTION]->(pred:AI_Prediction)
    RETURN p2.subject_id AS sid,
           p2.clinical_dx AS dx,
           p2.age         AS age,
           p2.sex         AS sex,
           s.score        AS sim,
           pred.calibrated_prob_AD_ncad AS prob_ad
    ORDER BY s.score DESC
    LIMIT $k
    """
    with graph_driver.session() as session:
        rows = session.run(query, id=clean_id, k=top_k).data()

    if not rows:
        return ""

    lines = ["【腦部特徵相似病患參考（GNN embedding cosine similarity）】"]
    for r in rows:
        prob_str = f"，AD 校準機率={r['prob_ad']:.3f}" if r.get("prob_ad") is not None else ""
        age_str  = f"，{r['age']} 歲 {r['sex']}" if r.get("age") and r["age"] > 0 else ""
        lines.append(
            f"  - 病患 {r['sid']}（{r['dx']}{age_str}）"
            f"：相似度 {r['sim']:.3f}{prob_str}"
        )
    return "\n".join(lines)

# ==========================================
# 3. 醫學文獻 (Vector) 檢索
# ==========================================
def retrieve_medical_literature(patient_ctx: str, top_roi_names: list, query: str = None) -> tuple[str, list]:
    """
    動態生成 Query（或使用傳入的 query），並返回 (文獻文本字串, 引用陣列)。
    使用 MMR（Maximal Marginal Relevance）確保多樣性，並做 paper-level 去重。
    """
    if not _VECTOR_AVAILABLE:
        return "無法取得文獻支援。", []

    if query is None:
        # The literature corpus is in English; build an English biomedical query
        # so retrieval is not degraded by cross-lingual mismatch (a Chinese query
        # against English-embedded chunks retrieved markedly less relevant papers).
        _DIAG_EN = {
            "NC": "normal cognition", "MCI": "mild cognitive impairment",
            "AD": "Alzheimer's disease",
            "正常": "normal cognition", "輕度認知障礙": "mild cognitive impairment",
            "阿茲海默": "Alzheimer's disease",
        }
        diag_en = _DIAG_EN.get(str(patient_ctx).strip(), str(patient_ctx))
        roi_str = ", ".join(top_roi_names[:5]) if top_roi_names else "resting-state functional connectivity"
        query = (
            f"Clinical significance of {roi_str} abnormalities in {diag_en}: "
            f"resting-state functional connectivity, structural atrophy, "
            f"default mode network, cognitive reserve, biomarkers, and disease progression."
        )

    print(f"  🔍 [Vector Search] 正在檢索相關文獻（MMR, k=5）...")

    # MMR：從 20 個候選中選出 5 個多樣性最高的 chunk
    try:
        candidates = vector_store.max_marginal_relevance_search(query, k=5, fetch_k=20)
    except Exception as e:
        print(f"  ⚠️ MMR Search 失敗，退回 similarity_search: {e}")
        try:
            candidates = vector_store.similarity_search(query, k=20)
        except Exception as e2:
            print(f"  ⚠️ Vector Search 失敗: {e2}")
            return "無法取得文獻支援。", []

    # Paper-level 去重：每篇論文只保留最相關的一個 chunk
    seen_sources: set = set()
    results = []
    for doc in candidates:
        src = doc.metadata.get('source') or doc.metadata.get('title') or str(len(seen_sources))
        if src not in seen_sources:
            seen_sources.add(src)
            results.append(doc)
        if len(results) >= 5:
            break

    literature_context = ""
    citations = []

    for i, doc in enumerate(results, 1):
        title = doc.metadata.get('title', f'Reference Paper {i}')
        text = doc.page_content.replace('\n', ' ')
        # Numbered references so the generator can cite by [i] and stay grounded
        # in the retrieved set (discourages parametric / hallucinated citations).
        literature_context += f"[{i}] 《{title}》\n{text}\n\n"
        citations.append({
            "id": i,
            "title": title,
            "text": text
        })

    return literature_context, citations

# English prefixes: the literature corpus is English, so queries must be English
# to avoid cross-lingual retrieval degradation.
_CLASS_QUERY_PREFIX = {
    "AD":  "Alzheimer's disease brain atrophy, functional disconnection, amyloid pathology and neuroinflammation.",
    "MCI": "mild cognitive impairment early biomarkers, compensatory mechanisms, and conversion-to-AD risk.",
    "NC":  "normal aging neuroprotective factors, cognitive reserve, and healthy functional connectivity.",
}

# ==========================================
# 4. ROI Knowledge Graph 查詢
# ==========================================
def get_roi_clinical_context(roi_names: list) -> str:
    """
    查詢 Neo4j :ROIKnowledge 節點，回傳指定 ROI 的臨床意義描述。
    支援前綴比對（如 "Hippocampus" 可匹配 "Hippocampus_L"）。
    """
    if not _GRAPH_AVAILABLE or not graph_driver or not roi_names:
        return ""
    # Extract base names (strip _L/_R/_1/_2 suffix)
    import re
    base_names = list({re.sub(r'_[LR\d]+$', '', n) for n in roi_names})
    try:
        with graph_driver.session() as session:
            result = session.run("""
                MATCH (r:ROIKnowledge)
                WHERE any(b IN $bases WHERE r.name STARTS WITH b OR b STARTS WITH r.name)
                RETURN r.name AS name, r.relevance AS relevance, r.description AS desc
                ORDER BY
                  CASE r.relevance WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END
            """, bases=base_names)
            rows = result.data()
        if not rows:
            return ""
        lines = ["### [ROI 臨床背景知識（來自知識圖譜）]"]
        for row in rows:
            tag = {"high": "🔴 高度相關", "medium": "🟡 中度相關", "atypical": "⚪ 非典型"}.get(row["relevance"], "")
            lines.append(f"**{row['name']}** {tag}\n{row['desc']}\n")
        return "\n".join(lines)
    except Exception as e:
        print(f"  ⚠️ ROI knowledge query failed: {e}")
        return ""


# ==========================================
# 5. 多模態文獻檢索 (Modality-aware RAG)
# ==========================================
def retrieve_multimodal(fmri_findings: dict, smri_findings: dict, patient_ctx: str, predicted_class: str = None):
    """
    分別針對 fMRI 與 sMRI 的發現進行檢索，並合併上下文。
    Returns: (combined_context: str, all_citations: list)
    """
    combined_context = ""
    all_citations = []
    seen_titles = set()
    prefix = ""
    if predicted_class in _CLASS_QUERY_PREFIX:
        prefix = _CLASS_QUERY_PREFIX[predicted_class] + " "

    # ROI Knowledge Graph context (from :ROIKnowledge nodes)
    all_roi_names = []
    if fmri_findings:
        all_roi_names += [r["name"] for r in fmri_findings.get("top_regions", [])[:5]]
    if smri_findings:
        all_roi_names += [r["name"] for r in smri_findings.get("top_regions", [])[:5]]
    roi_kg_ctx = get_roi_clinical_context(all_roi_names)
    if roi_kg_ctx:
        combined_context += roi_kg_ctx + "\n"

    # fMRI Retrieval
    if fmri_findings:
        roi_names = [r["name"] for r in fmri_findings.get("top_regions", [])[:3]]
        roi_str = ", ".join(roi_names) if roi_names else "resting-state functional connectivity"
        fmri_query = (
            f"{prefix} Patient predicted as {predicted_class or 'cognitive impairment'}. "
            f"Resting-state functional connectivity shows salient model attention in {roi_str}. "
            f"Clinical significance, neural mechanisms, and prognostic biomarkers of these "
            f"brain regions in dementia progression."
        )
        ctx, cits = retrieve_medical_literature(patient_ctx, roi_names, query=fmri_query)
        if ctx:
            combined_context += f"### [fMRI 相關文獻證據]\n{ctx}\n"
        for c in cits:
            if c["title"] not in seen_titles:
                seen_titles.add(c["title"])
                all_citations.append(c)

    # sMRI Retrieval
    if smri_findings:
        atrophy_regions = smri_findings.get("atrophy_regions", [])
        atrophy_str = ", ".join(atrophy_regions) if atrophy_regions else "brain structure"
        smri_query = (
            f"{prefix} Patient predicted as {predicted_class or 'cognitive impairment'}. "
            f"Structural MRI shows model attention in {atrophy_str}. "
            f"Evidence for these structural abnormalities as early dementia biomarkers "
            f"and their association with cognitive decline."
        )
        ctx, cits = retrieve_medical_literature(patient_ctx, atrophy_regions, query=smri_query)
        if ctx:
            combined_context += f"### [sMRI 相關文獻證據]\n{ctx}\n"
        for c in cits:
            if c["title"] not in seen_titles:
                seen_titles.add(c["title"])
                all_citations.append(c)

    return combined_context, all_citations

