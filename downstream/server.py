from mcp.server.fastmcp import FastMCP
import os
import sys
import config
from medical_ai_system import MedicalAISystem
from visualization_utils import BrainVisualizer

# 【新增】顯示目前正在使用的 Python 路徑 (Debugging)
# 如果看到 /envs/AD/bin/python 代表正確；如果看到 /miniconda3/bin/python 代表跑錯了
print(f"🐍 Server running on Python: {sys.executable}", file=sys.stderr)

# 初始化 MCP Server
mcp = FastMCP("Alzheimer's GNN Assistant")

# 【關鍵修正】這裡的 print 必須加 file=sys.stderr
print("🚀 初始化醫療 AI 系統中...", file=sys.stderr)

try:
    ai_system = MedicalAISystem(
        api_key=config.API_KEY, 
        model_path=config.MODEL_PATH, 
        dataset_csv=config.DATASET_CSV_PATH
    )
    visualizer = BrainVisualizer()
    print("✅ 系統載入完成！", file=sys.stderr)
except Exception as e:
    print(f"❌ 系統載入失敗: {e}", file=sys.stderr)
    ai_system = None

# 確保輸出目錄存在
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

@mcp.tool()
def list_patients() -> str:
    """
    列出資料庫中前 20 位病人的 ID 與診斷標籤。
    當使用者詢問有哪些病人可以分析時使用此工具。
    """
    if ai_system and ai_system.df is not None:
        return ai_system.df[['subject_id', 'label']].head(20).to_markdown(index=False)
    return "無法讀取資料庫或系統未初始化。"

@mcp.tool()
def analyze_patient(patient_id: str) -> str:
    """
    執行全套 AD 診斷流程：GNN 預測 + 3D 腦圖繪製 + GraphRAG 報告生成。
    輸入參數為病人 ID (例如: dswausub-027_S_6849_task-rest_bold)。
    """
    if not ai_system:
        return "系統初始化失敗，無法執行分析。"

    # Log 也要導向 stderr
    print(f"🔍 收到請求：分析病人 {patient_id}", file=sys.stderr)
    
    # 1. 獲取資料
    patient_data = ai_system.get_patient_data(patient_id)
    if not patient_data:
        return f"❌ 找不到病人 ID: {patient_id}，請檢查輸入是否正確。"

    # 2. 預測
    pred_class, confidence = ai_system.predict(patient_data)
    diag_str = "AD (阿茲海默症)" if pred_class == 1 else "NC (正常老化)"
    
    # 3. 視覺化
    img_name_3d = f"{patient_data['id']}_3d.png"
    img_path_3d = os.path.join(config.OUTPUT_DIR, img_name_3d)
    
    detected_regions = ["Rectus_R", "Insula_L"] if pred_class == 1 else ["Olfactory_R", "Rectus_L"]
    
    # 呼叫繪圖 (內部 print 已經改成 stderr 了，這裡安全)
    visualizer.show_3d_connectome(patient_data['adj'], detected_regions, output_file=img_path_3d)
    
    # 4. 生成報告
    report = ai_system.generate_report(patient_data['id'], pred_class, confidence, detected_regions)
    
    final_output = f"""
    【分析完成】
    - 病人 ID: {patient_data['id']}
    - GNN 預測: {diag_str} (信心度: {confidence:.2f}%)
    - 視覺化檔案位置: {img_path_3d}
    
    以下是 GraphRAG 生成的詳細報告：
    -----------------------------------
    {report}
    -----------------------------------
    """
    return final_output

if __name__ == "__main__":
    mcp.run()