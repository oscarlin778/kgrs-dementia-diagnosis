from medical_ai_system import MedicalAISystem
from visualization_utils import BrainVisualizer
import os
import sys
import config  # 匯入我們剛剛建立的設定檔 (config.py)

def main():
    print("🚀 啟動 AI 輔助診斷系統 (System Bootup)...")
    print(f"🐍 Python 環境: {sys.executable}")
    
    # 1. 檢查設定檔中的關鍵路徑是否存在
    if not os.path.exists(config.DATASET_CSV_PATH):
        print(f"❌ 設定錯誤: 找不到資料索引檔")
        print(f"   請檢查 config.py 中的路徑: {config.DATASET_CSV_PATH}")
        return

    try:
        # 使用 config 中的變數初始化系統
        ai_system = MedicalAISystem(
            api_key=config.API_KEY, 
            model_path=config.MODEL_PATH, 
            dataset_csv=config.DATASET_CSV_PATH
        )
        visualizer = BrainVisualizer()
    except Exception as e:
        print(f"❌ 系統初始化失敗: {e}")
        return

    # 確保輸出目錄存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 2. 進入互動迴圈
    while True:
        print("\n" + "="*50)
        user_input = input("👤 請輸入病人 ID (輸入 'list' 查看列表, 'exit' 離開): ").strip()
        
        if user_input.lower() == 'exit':
            print("👋 系統關閉。")
            break
            
        if user_input.lower() == 'list':
            if ai_system.df is not None:
                print("\n📋 資料庫中的前 10 位病人 ID:")
                print(ai_system.df[['subject_id', 'label']].head(10).to_string(index=False))
            else:
                print("❌ 無法讀取資料列表")
            continue
            
        if not user_input:
            continue

        # 3. 獲取資料
        target_patient_id = user_input
        print(f"🔍 正在搜尋: {target_patient_id} ...")
        
        patient_data = ai_system.get_patient_data(target_patient_id)
        
        if not patient_data:
            continue

        print(f"✅ 成功載入病人資料: {patient_data['id']}")

        # 4. 執行預測
        print("🧠 AI 正在分析腦影像數據...")
        pred_class, confidence = ai_system.predict(patient_data)
        
        true_label = patient_data['label']
        diag_str = "AD (阿茲海默症)" if pred_class == 1 else "NC (正常老化)"
        true_str = "AD" if true_label == 1 else "NC"
        
        print(f"📊 分析結果: {diag_str}")
        print(f"   - 模型信心度: {confidence:.2f}%")
        print(f"   - 真實臨床診斷: {true_str}")
        
        if pred_class != true_label:
            print(f"⚠️ [警告] 模型預測與臨床標籤不符！")
        
        # 5. 生成視覺化素材
        print("\n🎨 生成視覺化報告素材...")
        
        # 使用 config 定義的輸出路徑來儲存圖片
        raw_img_path = os.path.join(config.OUTPUT_DIR, f"{patient_data['id']}_raw.png")
        conn_img_path = os.path.join(config.OUTPUT_DIR, f"{patient_data['id']}_3d.png")
        
        # 顯示原始影像
        visualizer.show_nifti_slices(patient_data['nifti_path'], output_file=raw_img_path)
        
        # 模擬關鍵腦區 (AD抓Rectus/Insula, NC抓Olfactory)
        detected_regions = ["Rectus_R", "Insula_L"] if pred_class == 1 else ["Olfactory_R", "Rectus_L"]
        
        visualizer.show_3d_connectome(patient_data['adj'], detected_regions, output_file=conn_img_path)
        
        # 6. 生成 GraphRAG 報告
        print("\n📝 呼叫 GraphRAG 生成文字報告...")
        report = ai_system.generate_report(patient_data['id'], pred_class, confidence, detected_regions)
        
        print("\n" + "-"*60)
        print(report)
        print("-" * 60)
        print(f"\n📂 影像已儲存至: {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()