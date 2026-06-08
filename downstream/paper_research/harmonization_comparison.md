# 多站點 fMRI Harmonization 方法比較報告

在跨中心（Multi-site）神經影像研究中，消除站點效應（Site Effects / Scanner Effects）是確保機器學習模型具備泛化能力的首要任務。本報告針對六種主流的諧波化（Harmonization）方法進行比較，並探討「無標籤（No-label）ComBat」在臨床預測場景中的關鍵優勢。

---

## 1. 諧波化方法介紹

### ComBat (Johnson et al. 2007)
ComBat 原本設計於基因晶片數據去噪，後被廣泛應用於腦部影像。它採用線性回歸模型，將數據分解為全域基準、生物協變量、加性站點偏差（均值平移）與乘性站點偏差（標準差縮放）。ComBat 最核心的特色是引入**經驗貝氏估計（Empirical Bayes）**，將各個特徵（如腦區）的站點估計值向整體平均值收縮，這使其在小樣本站點上依然非常穩定且不易過度擬合。

### CovBat (Chen et al. 2022)
CovBat 是 ComBat 的直接延伸，旨在解決 ComBat 無法修正的**協方差站點效應（Covariance Batch Effects）**。ComBat 僅能對齊單一特徵的均值和方差，但不同特徵之間的相關性結構（Covariance）在不同掃描儀下仍可能存在偏差。CovBat 通過對 ComBat 的殘差進行主成分分析（PCA），並在主成分空間中再次調整特徵間的協方差，從而實現了更深層次的多元分布對齊，非常適合以相關性為基礎的腦網路分析。

### DANN (Domain-Adversarial Neural Networks) (Ganin et al. 2016)
DANN 是一種基於深度學習的**領域對抗遷移學習（Domain Adaptation）**方法。它在模型中設計了一個「特徵提取器」和兩個並行的分支：「分類器」與「領域判別器」。特徵提取器負責提取腦影像特徵，分類器負責預測疾病標籤，而領域判別器則試圖去辨識該特徵來自哪一個掃描站點。藉由將領域判別器的梯度取反（Gradient Reversal），強迫特徵提取器學到「站點無關（Domain-Invariant）」的深層特徵。

### fMRIPrep + AROMA (ICA-AROMA)
這屬於 fMRI 的**預處理（Preprocessing）階段去噪**，而非 downstream 的統計對齊。fMRIPrep 是一個標準化的影像前處理管線，整合了頭動校正、空間標準化等步驟。AROMA 則是利用獨立成分分析（ICA）自動識別並去除影像中的運動偽影（Motion Artifacts）與非神經性雜訊。這能從物理與生理層面減少數據本身的噪音，但它無法完全消除因不同儀器硬體產生的長期系統性偏差。

### Harmony / NeuroHarmonize (Pomponio et al. 2020)
NeuroHarmonize 是專為腦影像設計的 Python 開發套件，本質上是 ComBat 演算法的現代延伸。傳統的 ComBat 僅支持線性協變量（如線性老化），但 NeuroHarmonize 整合了**廣義加性模型（Generalized Additive Models, GAMs）**，允許用戶對年齡等生物協變量進行複雜的非線性建模（如倒 U 型的老化軌跡）。這能更精準地剝離混雜因素，並支持輕鬆導出校正參數以應用於新樣本。

### Conditional VAE Site Harmonization (CVAE)
這是近年興起的生成式對齊方法。模型利用變分自編碼器（VAE），將輸入的腦網路編碼至一個低維度的隱空間（Latent Space），並在解碼器（Decoder）重建影像時，加入目標站點的條件標籤（Condition）。通過對隱空間的解構（Disentanglement），將生物信號與站點特徵分離，進而能將來自 Site A 的患者數據「平移/重建」成看似由 Site B 掃描出來的數據。

---

## 2. Harmonization 方法對比表

| 方法名稱 | 核心原理 | 在 FC Matrix 的適用度 | 是否需要 Retraining | 代表文獻 |
| :--- | :--- | :--- | :--- | :--- |
| **ComBat** | 經驗貝氏加性/乘性回歸校正 | **高**。腦區固定且對齊，能穩定對齊連接值。 | 否 (直接套用公式估算參數) | Johnson et al. (2007) *Biostatistics* |
| **CovBat** | 基於 PCA 的殘差協方差對齊 | **極高**。專門修正特徵間的相關性結構。 | 否 (基於回歸與 PCA 的非迭代計算) | Chen et al. (2022) *NeuroImage* |
| **DANN** | 對抗式領域無關特徵學習 | **中**。FC 特徵高維，小樣本深度學習極易過擬合。 | 是 (需要端到端神經網路訓練) | Ganin et al. (2016) *JMLR* |
| **fMRIPrep + AROMA** | ICA 空間去噪與物理預處理 | **高**。作為前處理基石，但無法完全替代 downstream 對齊。 | 否 (預處理 pipeline 運行) | Pruim et al. (2015) *NeuroImage* |
| **NeuroHarmonize** | 基於 GAMs 的非線性協變量 ComBat | **高**。適合有複雜年齡曲線的大規模隊列。 | 否 (直接估算 GAM 參數) | Pomponio et al. (2020) *NeuroImage* |
| **Conditional VAE** | 隱空間解構與條件式重建生成 | **中**。樣本量小時 VAE 難以收斂，容易失真。 | 是 (需要預先在大數據集上訓練 VAE) | Moyer et al. (2020) *Medical Image Analysis* |

---

## 3. 防禦 Reviewer 提問：為什麼「無標籤（No-label）ComBat」比「有標籤」更好？

在投稿時，Reviewer 常會問：「為什麼在執行 ComBat 時，沒有把診斷標籤（AD/MCI/NC）作為協變量（Covariates）傳入以保護生物信號？」
我們可以從以下 4 個角度進行強力的學術防禦：

### 1. 避免測試階段的「標籤洩漏」（Label Leakage / Data Leakage）
如果 ComBat 在諧波化時需要利用診斷標籤，那麼在**真實臨床部署或獨立測試集測試時**就會遇到邏輯悖論：我們使用 AI 模型就是為了解測未知病患的標籤，此時我們根本不知道病患是 AD 還是 NC，因此無法執行「有標籤」的 ComBat。如果在測試集上使用真實標籤進行諧波化，會造成嚴重的資訊洩漏，誇大分類準確率。

### 2. 解決訓練與測試階段的「校正不一致性」（Calibration Mismatch）
如果在訓練時使用「有標籤 ComBat」，而在預測新病人時因為缺乏標籤而改用「無標籤 ComBat」，會導致訓練集與測試集經過不同的數學轉換。這種校正上的不對稱會直接破壞神經網路的輸入分布，導致模型在獨立測試集上的表現劇烈下滑。

### 3. 無標籤 ComBat 具有更強的「非監督式魯棒性」
本研究的少數類（AD）樣本量極小（測試集僅 11 人）。在極端不平衡且樣本稀缺的情況下，若強行在小樣本上估計疾病協變量效應 $\beta$，其估計方差會非常大，極易將站點噪音誤判為疾病特徵。使用不帶 Label 的 ComBat 進行非監督式諧波化，雖然略微保守，但能有效防止模型對特定疾病特徵的過度擬合。

### 4. 符合臨床篩檢管線的實用性（Clinical Feasibility）
在臨床落地場景中，醫院希望將新掃描的患者資料直接輸入系統進行自動化診斷。使用無標籤 ComBat，系統只需要使用從訓練集固定的站點參數（Reference ComBat）對新樣本進行即時標準化，即可直接送入 GNN 模型。這使得整個推理管線（Inference Pipeline）是完全端到端且獨立於標籤的，具備高度的實用價值。
