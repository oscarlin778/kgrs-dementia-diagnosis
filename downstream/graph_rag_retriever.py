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

# 初始化 Graph 驅動
graph_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# 初始化 Vector 驅動 (使用本地 Ollama)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Neo4jVector.from_existing_index(
    embedding=embeddings,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name="chunk_embedding_index", # 剛剛驗證成功的索引名稱
    text_node_property="text"
)

# ==========================================
# 2. 病患節點 (Graph) 擷取
# ==========================================
def get_patient_graph_context(subject_id: str) -> str:
    """擷取病患的結構化人口統計脈絡"""
    clean_id = subject_id.replace('sub-', '')
    query = """
    MATCH (p:Patient {subject_id: $id})
    RETURN p.age AS age, p.sex AS sex, p.education AS education, p.clinical_dx AS dx
    """
    with graph_driver.session() as session:
        result = session.run(query, id=clean_id).single()
        if result and result["age"] is not None and result["age"] > 0:
            return f"病患為 {result['age']} 歲 {result['sex']}，教育年限 {result['education']} 年，臨床診斷為 {result['dx']}。"
        return "目前無該名病患的詳細人口統計學資料。"

# ==========================================
# 3. 醫學文獻 (Vector) 檢索
# ==========================================
def retrieve_medical_literature(patient_ctx: str, top_roi_names: list, query: str = None) -> tuple[str, list]:
    """
    動態生成 Query（或使用傳入的 query），並返回 (文獻文本字串, 引用陣列)
    """
    if query is None:
        roi_str = "、".join(top_roi_names[:3]) if top_roi_names else "大腦功能性連結"
        query = f"探討 {patient_ctx} 的病患，其 {roi_str} 異常的臨床意義、認知儲備代償效應與預後發展。"

    print(f"  🔍 [Vector Search] 正在檢索相關文獻...")

    # 進行向量搜尋
    try:
        results = vector_store.similarity_search(query, k=3)
    except Exception as e:
        print(f"  ⚠️ Vector Search 失敗: {e}")
        return "無法取得文獻支援。", []

    literature_context = ""
    citations = []
    
    for i, doc in enumerate(results, 1):
        # 取出 metadata 中的論文標題，如果當初 ingest 沒存，就顯示預設
        title = doc.metadata.get('title', f'Reference Paper {i}')
        text = doc.page_content.replace('\n', ' ')
        
        literature_context += f"【文獻 {i}】來源《{title}》：\n{text}\n\n"
        citations.append({
            "id": i,
            "title": title,
            "text": text
        })
        
    return literature_context, citations

# ==========================================
# 4. 多模態文獻檢索 (Modality-aware RAG)
# ==========================================
def retrieve_multimodal(fmri_findings: dict, smri_findings: dict, patient_ctx: str) -> str:
    """
    分別針對 fMRI 與 sMRI 的發現進行檢索，並合併上下文。
    """
    combined_context = ""

    # fMRI Retrieval
    if fmri_findings:
        roi_names = [r["name"] for r in fmri_findings.get("top_regions", [])[:3]]
        roi_str = "、".join(roi_names) if roi_names else "大腦功能性連結"
        fmri_query = f"探討 {patient_ctx} 的病患，其大腦功能性連結異常（特別是 {roi_str}）的臨床意義。"
        ctx, _ = retrieve_medical_literature(patient_ctx, roi_names, query=fmri_query)
        if ctx:
            combined_context += f"### [fMRI 相關文獻證據]\n{ctx}\n"

    # sMRI Retrieval
    if smri_findings:
        atrophy_regions = smri_findings.get("atrophy_regions", [])
        atrophy_str = "、".join(atrophy_regions) if atrophy_regions else "腦部結構"
        smri_query = f"探討 {patient_ctx} 的病患，其腦部結構萎縮（如 {atrophy_str}）與認知衰退的關聯。"
        ctx, _ = retrieve_medical_literature(patient_ctx, atrophy_regions, query=smri_query)
        if ctx:
            combined_context += f"### [sMRI 相關文獻證據]\n{ctx}\n"

    return combined_context
