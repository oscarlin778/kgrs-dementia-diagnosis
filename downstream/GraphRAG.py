import os
import numpy as np
import warnings
import datetime

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 設定區 =================
API_KEY = os.getenv("GOOGLE_API_KEY")
DEMO_MODE = False

# ================= 初始化 Google Gemini =================
try:
    import google.generativeai as genai
    os.environ["GOOGLE_API_KEY"] = API_KEY
    genai.configure(api_key=API_KEY)
    
    # 自動選擇模型
    valid_model_name = None
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name: # 優先選快一點的
                    valid_model_name = m.name
                    break
        if not valid_model_name: valid_model_name = 'models/gemini-pro'
        print(f"✅ 連線成功，使用模型: {valid_model_name}")
    except:
        DEMO_MODE = True
except:
    DEMO_MODE = True

# ================= 1. 建立「文獻知識庫」 (Knowledge Corpus) =================
# 這裡模擬一個小型資料庫，存放我們整理好的論文片段 (Level 1 Database)
# 實務上這可以是一個 CSV 檔或 Vector DB
medical_documents = [
    {
        "id": "doc_001",
        "tags": ["Rectus", "Frontal", "Compensation"],
        "content": "研究 (Zhong et al., 2014) 指出，AD 患者的內側前額葉 (MPFC) 與直回 (Rectus Gyrus) 常出現功能性連接增強。這被認為是一種代償機制 (Compensatory Mechanism)，用以彌補後腦 DMN 節點的功能喪失。"
    },
    {
        "id": "doc_002",
        "tags": ["Insula", "Salience", "Disconnection"],
        "content": "腦島 (Insula) 是突顯網路 (Salience Network) 的核心樞紐。Han et al. (2024) 發現 AD 患者的腦島連接顯著降低，這導致了認知控制與注意力切換能力的衰退。"
    },
    {
        "id": "doc_003",
        "tags": ["Olfactory", "Early Biomarker", "NC"],
        "content": "嗅覺皮質 (Olfactory Cortex) 的萎縮是阿茲海默症最早期的病理特徵之一。若影像數據顯示該區域功能連接完整，則強烈支持正常老化 (Normal Control) 的診斷 (Clinical Consensus, 2023)。"
    },
    {
        "id": "doc_004",
        "tags": ["Fusiform", "Visual", "Recognition"],
        "content": "梭狀迴 (Fusiform Gyrus) 負責面孔識別。最新的 fMRI 研究顯示，AD 患者在梭狀迴的節點中心性 (Degree Centrality) 下降，這與臨床上的面孔失認症狀相關。"
    }
]

# ================= 2. 檢索系統 (Retrieval System) =================
def retrieve_knowledge(keywords, documents):
    """
    簡單的關鍵字檢索器 (模擬 Vector Search)
    """
    results = []
    print(f"🔍 正在資料庫中檢索關鍵字: {keywords} ...")
    
    for doc in documents:
        # 如果文獻的 tags 或內容包含關鍵字，就抓出來
        score = 0
        for kw in keywords:
            if kw in doc['tags'] or kw in doc['content']:
                score += 1
        
        if score > 0:
            results.append(doc['content'])
            
    if not results:
        return ["(查無相關文獻，建議參閱一般神經內科教科書)"]
    
    return list(set(results)) # 去重

# ================= 3. AI 報告生成器 (Generator) =================
def generate_patient_report(patient_data):
    """
    Input: patient_data (包含 ID, 預測結果, 關鍵腦區)
    """
    pid = patient_data['id']
    pred_str = "阿茲海默症 (AD)" if patient_data['prediction'] == 1 else "正常老化 (NC)"
    top_regions = patient_data['top_regions'] # 例如 ['Rectus_R', 'Insula_L']
    
    # 步驟 A: 去資料庫找知識 (RAG 的 R)
    # 我們把腦區名稱當作關鍵字去搜尋
    # 去除 _R, _L 這些後綴以增加搜尋命中率
    search_terms = [r.split('_')[0] for r in top_regions] 
    knowledge_context = retrieve_knowledge(search_terms, medical_documents)
    
    knowledge_text = "\n".join([f"• {k}" for k in knowledge_context])
    
    # 步驟 B: 組裝 Prompt (RAG 的 G)
    prompt = f"""
    【角色設定】
    你是一位資深神經內科醫師，請根據「GNN 影像分析結果」與「檢索到的醫學文獻」，撰寫一份專業的診斷報告。

    【病人資訊】
    - 編號: {pid}
    - AI 預測結果: {pred_str} (信心度 {patient_data['confidence']}%)
    - GNN 偵測到的關鍵腦區: {', '.join(top_regions)}

    【檢索到的醫學證據 (Reference Context)】
    {knowledge_text}

    【報告撰寫要求】
    1. **診斷摘要**：一句話說明診斷結果。
    2. **病理機制分析**：請務必結合上述「醫學證據」來解釋為什麼 GNN 會抓到這些腦區。
       - 如果是 AD，請強調「代償」或「斷網」。
       - 如果是 NC，請強調「功能完整」。
    3. **文獻引用**：請在解釋中提及 (e.g., Zhong et al.)。
    4. **臨床建議**：根據發現給予建議。

    請以繁體中文撰寫，語氣專業溫暖。
    """
    
    # 步驟 C: 呼叫 LLM
    if DEMO_MODE:
        return "Demo Mode: (API 連線失敗，無法生成真實報告，但流程如上所示)"
    else:
        try:
            print("🤖 正在呼叫 LLM 撰寫報告...")
            model = genai.GenerativeModel(valid_model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ 生成失敗: {e}"

# ================= 4. 執行演示 =================
if __name__ == "__main__":
    # 案例：我們之前的 GNN 抓到了 Rectus 和 Insula
    patient_case = {
        "id": "SUB_AD_023",
        "prediction": 1, # AD
        "confidence": 72.5,
        "top_regions": ["Rectus_R", "Insula_L"] 
    }
    
    print("="*60)
    report = generate_patient_report(patient_case)
    print("="*60)
    print(report)