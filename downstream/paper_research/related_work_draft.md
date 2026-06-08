# Related Work (相關工作) 論文段落 Draft

本草案針對論文中的「Related Work」章節進行撰寫，字數約為 800 字，採用學術繁體中文表述，並保留專有名詞、方法名稱及文獻引用（使用標準學術格式）。

---

### Section 2: Related Work

**Multi-modal AD classification using fMRI and sMRI**
多模態神經影像融合已成為提高阿茲海默症（Alzheimer's Disease, AD）及其早期階段輕度認知障礙（Mild Cognitive Impairment, MCI）診斷準確度的關鍵路徑。近年來，基於圖神經網路（Graph Neural Networks, GNNs）的腦網路分類方法取得了顯著進展。代表性工作如 Li 等人提出的 BrainGNN (2021) 利用池化機制實現了具備可解釋性的脑網路分類；Kawahara 等人提出的 BrainNetCNN (2017) 則透過特殊的卷積核設計處理神經連接矩陣；而 Jiang 等人設計的 Hi-GCN (2020) 則透過層級式圖卷積網路捕捉全局與局部的拓撲結構特徵。與此同時，結構性核磁共振影像（sMRI）的分類研究也經歷了從傳統 3D-CNN 向 3D Vision Transformers (ViTs) 的典範轉移。例如，在 sMRI 分類中，許多研究利用 ViT 機制來擷取全局解剖結構的萎縮特徵。為了結合兩者的優勢，多個研究嘗試進行多模態融合（fMRI + sMRI），例如利用後期融合（Late Fusion）或簡單特徵拼接（Concatenation）來整合功能與結構資訊。然而，現有方法普遍面臨兩個核心痛點：第一，缺乏精確的交叉注意力（Cross-Attention）機制來捕捉模態間的非線性互補性，導致融合流於表面；第二，現有融合框架大多忽略了多站點（Multi-site）數據間的掃描儀偏差，限制了其在外部數據集上的泛化表現。

**Multi-site brain imaging harmonization**
在跨中心的腦影像聯合分析中，消除由不同掃描儀硬體及掃描協定產生的站點效應（Site Effects）至關重要。最經典且被廣泛應用的統計諧波化方法是 Johnson 等人提出的 ComBat (2007)，該方法基於經驗貝氏估計（Empirical Bayes），能穩健地校正加性與乘性站點偏差。隨後，許多學者將其推廣至腦影像領域，例如 Fortin 等人將 ComBat 應用於皮質厚度 (2017) 與擴散張量影像（DTI）的白質分數校正 (2018)；在功能連接矩陣（FC Matrix）上，ComBat 亦被證實能有效消除虛假的連接偏差。為了應對特徵間相關性結構的偏差，Chen 等人進一步提出了 CovBat (2022) 以修正協方差站點效應。在深度學習領域，利用領域對抗神經網路（Domain-Adversarial Neural Networks, DANN）（Ganin et al., 2016）來學習站點無關特徵也逐漸興起。然而，目前多站點腦影像研究仍存在顯著缺口（Gap）：極少有文獻同時報告跨站點的「站點內（Within-site）AUC」與「跨站點（Cross-site）AUC」，且缺乏對「有標籤諧波化（With-label）」與「無標籤諧波化（No-label）」在真實臨床推理管線（Inference Pipeline）下泛化能力的系統性比較與防禦性討論。

**Medical knowledge graphs and RAG for radiology report generation**
隨著大型語言模型（LLMs）的發展，如何利用人工智慧自動生成結構化且具臨床價值的醫療報告成為研究熱點。在失智症領域，知識圖譜（Knowledge Graphs, KGs）被廣泛用於整合異質性的臨床數據與生醫文獻。例如，AlzKB (2023) 整合了數十個開放數據源以建構阿茲海默症藥物研發與病理關係圖譜；ADKG (2022) 則透過自動化文本挖掘技術從 PubMed 提取阿茲海默症關聯實體。為了克服 LLM 在醫學領域常見的「幻覺（Hallucination）」問題，檢索增強生成（Retrieval-Augmented Generation, RAG）被引入放射學報告生成。檢索增強系統能將患者特徵與醫學指南（如 Radiopaedia）或歷史報告庫進行檢索比對，為 LLM 提供可靠的上下文背景。諸如 Google 的 Med-PaLM 系列 (2023, 2024) 等通用醫學大型模型已證實，基於高質量臨床數據微調或外掛知識檢索後，LLM 能在醫療問答中達到接近專業醫師的水平。然而，在腦神經影像診斷領域，目前仍缺乏一套將神經元分類結果（分類機率與重要腦區）、醫學知識圖譜（KG）、RAG 以及 LLM 進行端到端（End-to-End）無縫整合的系統，且鮮有研究針對生成報告的臨床適用性進行嚴謹的量化評估。

**Summary of Our Contributions**
為了填補上述研究空白，本研究提出了一套創新的三階段端到端框架。首先，我們設計了結合 Cross-Attention 的 PCAG-ComBat 雙模態融合分類模型，並配合嚴謹的無標籤多站點諧波化策略，在有效對齊多中心數據的同時，在小樣本測試集上取得了領先的 AUC 表現。其次，我們系統性地論證了無標籤 ComBat 校正策略在臨床推理時的抗洩漏優勢。最後，我們首次建構了從 GNN 分類概率與特徵圖、檢索關聯阿茲海默症知識圖譜、到引導 LLM 生成結構化放射診斷報告的端到端管線，實現了從「黑盒特徵分類」到「白盒臨床解釋」的完整閉環，為臨床決策提供了強大的輔助支援。
