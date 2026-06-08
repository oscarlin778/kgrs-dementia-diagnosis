"""
seed_roi_knowledge.py
將 ROI-疾病關聯知識節點寫入 Neo4j。
每個節點為 :ROIKnowledge，屬性：name, relevance, conditions, description, source
並建立向量 embedding 供 RAG 查詢使用。
"""
import os, sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from neo4j import GraphDatabase
from langchain_community.embeddings import OllamaEmbeddings
import warnings
warnings.filterwarnings("ignore")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ── ROI 知識庫（基於文獻共識）──────────────────────────────────────────────────
ROI_KNOWLEDGE = [
    # 高度相關 — AD/MCI 核心病理區域
    {
        "name": "Hippocampus",
        "relevance": "high",
        "conditions": ["AD", "MCI"],
        "description": (
            "Hippocampus (海馬體) 是 AD 最早期萎縮的標誌性區域。海馬體體積縮小（hippocampal atrophy）"
            "與情節記憶損傷直接相關，是 AD 生物標記 cascade 的核心（Jack et al., 2010）。"
            "在 MCI 階段即可觀察到海馬體結構與功能異常，可作為 NC→MCI 轉換的預測因子。"
        ),
        "source": "Jack CR et al., Lancet Neurol 2010; Frisoni GB et al., Nat Rev Neurosci 2010"
    },
    {
        "name": "ParaHippocampal",
        "relevance": "high",
        "conditions": ["AD", "MCI"],
        "description": (
            "ParaHippocampal gyrus (海馬旁迴) 為內嗅皮質延伸區，是 Tau 蛋白最早累積的位置之一。"
            "與空間記憶和情景記憶密切相關，AD Braak stages I-II 即涵蓋此區。"
        ),
        "source": "Braak H & Braak E, Acta Neuropathol 1991"
    },
    {
        "name": "Amygdala",
        "relevance": "high",
        "conditions": ["AD", "MCI"],
        "description": (
            "Amygdala (杏仁核) 為 AD 早期 Tau 神經纖維糾結的好發區域，"
            "體積萎縮與情緒辨識障礙及記憶整合受損相關。"
            "MCI 患者已可觀察到杏仁核體積縮小。"
        ),
        "source": "Poulin SP et al., Alzheimers Dement 2011"
    },
    {
        "name": "Cingulum_Post",
        "relevance": "high",
        "conditions": ["AD", "MCI"],
        "description": (
            "Posterior cingulate cortex (後扣帶皮質, PCC) 是 default mode network (DMN) 的核心節點，"
            "也是 AD 最早出現 FDG-PET 代謝低下的區域之一。"
            "PCC 功能連結中斷是 NC→MCI→AD 認知衰退的早期生物標記。"
        ),
        "source": "Buckner RL et al., Ann NY Acad Sci 2005; Greicius MD et al., PNAS 2004"
    },
    {
        "name": "Precuneus",
        "relevance": "high",
        "conditions": ["AD", "MCI"],
        "description": (
            "Precuneus (楔前葉) 是澱粉樣蛋白 (Aβ) 沉積的熱點區域，"
            "在 AD 症狀出現前 15-20 年即可偵測到早期 Aβ 累積。"
            "屬於 DMN 的重要節點，其功能連結下降與認知儲備耗盡相關。"
        ),
        "source": "Mintun MA et al., Neurology 2006; Jagust W, Nat Rev Neurosci 2018"
    },
    {
        "name": "Angular",
        "relevance": "high",
        "conditions": ["AD", "MCI"],
        "description": (
            "Angular gyrus (角回) 位於頂葉聯合區，整合語言、空間與記憶功能。"
            "AD 患者此區域代謝顯著降低，與語言流暢性下降和空間認知退化相關。"
            "也是 DMN 的重要組成節點。"
        ),
        "source": "Buckner RL et al., J Neurosci 2005"
    },
    # 中度相關 — MCI 與認知功能
    {
        "name": "Cingulum_Mid",
        "relevance": "medium",
        "conditions": ["MCI", "AD"],
        "description": (
            "Middle cingulate cortex (中扣帶皮質) 參與注意力、執行功能與疼痛處理。"
            "MCI 與早期 AD 患者可觀察到此區域灰質體積縮小及功能連結改變。"
        ),
        "source": "Villain N et al., Brain 2008"
    },
    {
        "name": "Cingulum_Ant",
        "relevance": "medium",
        "conditions": ["MCI"],
        "description": (
            "Anterior cingulate cortex (前扣帶皮質) 為執行控制網路核心，"
            "與工作記憶和錯誤監控相關。MCI 早期可見此區域功能連結異常。"
        ),
        "source": "Tekin S & Cummings JL, Behav Neurol 2002"
    },
    {
        "name": "Temporal_Mid",
        "relevance": "medium",
        "conditions": ["AD", "MCI"],
        "description": (
            "Middle temporal gyrus (中顳葉) 儲存語義記憶，"
            "AD 患者顳葉萎縮是疾病進展的重要標誌，與詞彙提取困難相關。"
        ),
        "source": "Atran S, Science 1998; Gauthier I et al., Nat Neurosci 2000"
    },
    {
        "name": "Temporal_Sup",
        "relevance": "medium",
        "conditions": ["AD", "MCI"],
        "description": (
            "Superior temporal gyrus (顳上回) 負責語言處理與聽覺聯繫，"
            "也是鏡像神經元系統的一部分。AD 顳葉皮質萎縮影響溝通能力。"
        ),
        "source": "Thompson PM et al., Nat Neurosci 2003"
    },
    {
        "name": "Thalamus",
        "relevance": "medium",
        "conditions": ["AD"],
        "description": (
            "Thalamus (丘腦) 是大腦主要的訊息中繼站。"
            "AD 中期丘腦核群（特別是 mediodorsal nucleus）受累，"
            "影響皮質-丘腦-皮質迴路，與睡眠障礙和認知波動相關。"
        ),
        "source": "Braak H & Braak E, Neurobiol Aging 1998"
    },
    {
        "name": "Frontal_Med_Orb",
        "relevance": "medium",
        "conditions": ["MCI"],
        "description": (
            "Medial orbital frontal cortex (眼眶前額葉皮質) 負責決策、情緒調節與社交判斷。"
            "MCI 患者此區域灰質體積下降與執行功能障礙相關。"
        ),
        "source": "Rosen HJ et al., Neurology 2002"
    },
    # 非典型 AD 區域
    {
        "name": "Cerebelum",
        "relevance": "atypical",
        "conditions": [],
        "description": (
            "Cerebellum (小腦) 主要負責運動協調、平衡與精細動作控制，"
            "並非 AD 的主要病理標誌。AD Braak staging 不包含小腦為早期受累區域。"
            "影像分析中小腦 attention 高，可能源於個體掃描頭部姿勢差異、"
            "scanner site effect 或非疾病相關的個體影像特徵。"
            "解讀時應以 Hippocampus、Cingulum_Post、Precuneus 等典型區域為主要依據。"
        ),
        "source": "Braak & Braak staging; Villain et al., 2008"
    },
    {
        "name": "Occipital",
        "relevance": "atypical",
        "conditions": ["AD_late"],
        "description": (
            "Occipital cortex (枕葉) 為主要視覺皮質，AD 主要在晚期才出現枕葉萎縮。"
            "早期 AD 中枕葉通常相對保留（相較於顳頂葉）。"
            "例外：Posterior Cortical Atrophy（PCA，視覺型 AD 變體）以枕頂葉萎縮為主。"
            "若非 PCA 變體，早期分析中枕葉 attention 高需謹慎解讀。"
        ),
        "source": "Crutch SJ et al., Lancet Neurol 2012"
    },
    {
        "name": "Rolandic_Oper",
        "relevance": "atypical",
        "conditions": [],
        "description": (
            "Rolandic operculum (羅蘭蒂氏蓋) 位於初級感覺運動皮質交界，"
            "主要負責語音處理和軀體感覺，非 AD 的核心病理區域。"
            "分析中此區域 attention 高可能為 confound。"
        ),
        "source": "Standard neuroanatomy reference"
    },
]

def get_neo4j_password():
    """Try multiple ways to get NEO4J_PASSWORD"""
    pw = os.getenv("NEO4J_PASSWORD")
    if pw:
        return pw
    # Try reading from api_server env
    env_files = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "../.env"),
        os.path.expanduser("~/.neo4j_env"),
    ]
    for ef in env_files:
        if os.path.exists(ef):
            load_dotenv(ef, override=True)
            pw = os.getenv("NEO4J_PASSWORD")
            if pw:
                return pw
    return None


def seed_roi_knowledge(driver):
    """Create :ROIKnowledge nodes in Neo4j"""
    with driver.session() as session:
        # Create constraint if not exists
        try:
            session.run("CREATE CONSTRAINT roi_name IF NOT EXISTS FOR (r:ROIKnowledge) REQUIRE r.name IS UNIQUE")
        except Exception:
            pass

        created = 0
        for roi in ROI_KNOWLEDGE:
            result = session.run("""
                MERGE (r:ROIKnowledge {name: $name})
                SET r.relevance    = $relevance,
                    r.conditions   = $conditions,
                    r.description  = $description,
                    r.source       = $source,
                    r.text         = $text
                RETURN r.name AS name
            """,
                name=roi["name"],
                relevance=roi["relevance"],
                conditions=roi["conditions"],
                description=roi["description"],
                source=roi["source"],
                text=f"{roi['name']} ({roi['relevance']} relevance): {roi['description']} Source: {roi['source']}"
            )
            rec = result.single()
            if rec:
                created += 1
                print(f"  ✓ {rec['name']} ({roi['relevance']})")

        print(f"\n[seed] {created}/{len(ROI_KNOWLEDGE)} ROI knowledge nodes written to Neo4j")


def build_roi_vector_index(driver):
    """Add embedding to each ROIKnowledge node and create vector index"""
    try:
        emb = OllamaEmbeddings(model="nomic-embed-text")
    except Exception as e:
        print(f"⚠️  Ollama embeddings unavailable: {e}; skipping vector index")
        return

    with driver.session() as session:
        nodes = session.run("MATCH (r:ROIKnowledge) RETURN r.name AS name, r.text AS text").data()

    print(f"\n[embed] Generating embeddings for {len(nodes)} ROI nodes...")
    with driver.session() as session:
        for node in nodes:
            try:
                vec = emb.embed_query(node["text"])
                session.run(
                    "MATCH (r:ROIKnowledge {name: $name}) SET r.embedding = $embedding",
                    name=node["name"], embedding=vec
                )
                print(f"  ✓ {node['name']}")
            except Exception as e:
                print(f"  ✗ {node['name']}: {e}")

    # Create vector index for ROIKnowledge
    with driver.session() as session:
        try:
            session.run("""
                CREATE VECTOR INDEX roi_knowledge_index IF NOT EXISTS
                FOR (r:ROIKnowledge) ON r.embedding
                OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
            """)
            print("\n[index] roi_knowledge_index created (or already exists)")
        except Exception as e:
            print(f"[index] {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation")
    args = parser.parse_args()

    pw = get_neo4j_password()
    if not pw:
        print("❌ NEO4J_PASSWORD not set. Please export NEO4J_PASSWORD=<your_password>")
        sys.exit(1)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, pw))
    try:
        with driver.session() as s:
            s.run("RETURN 1").single()
        print(f"✓ Connected to Neo4j at {NEO4J_URI}")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        sys.exit(1)

    print("\n=== Step 1: Seeding ROI knowledge nodes ===")
    seed_roi_knowledge(driver)

    if not args.skip_embed:
        print("\n=== Step 2: Building vector embeddings ===")
        build_roi_vector_index(driver)

    driver.close()
    print("\n✅ Done.")
