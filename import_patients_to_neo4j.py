import pandas as pd
from neo4j import GraphDatabase

# 請替換成你的 Neo4j 連線資訊
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "nm6131027" # 替換成你的密碼

def import_patients(csv_path):
    print("⏳ 準備將病患資料寫入 Neo4j...")
    df = pd.read_csv(csv_path)
    
    # 將 DataFrame 轉為字典的 List，方便餵給 Neo4j
    patients = df.to_dict('records')
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    cypher_query = """
    UNWIND $patients AS row
    MERGE (p:Patient {subject_id: row.subject_id})
    SET p.age = toInteger(row.age),
        p.sex = row.sex,
        p.education = toInteger(row.education),
        p.clinical_dx = row.clinical_dx
    """
    
    with driver.session() as session:
        session.run(cypher_query, patients=patients)
        
    driver.close()
    print(f"✅ 成功將 {len(patients)} 位病患寫入 Neo4j 知識圖譜！")

if __name__ == "__main__":
    import_patients("final_patient_nodes.csv")