# PCAG-ComBat 模型改進計畫

目標：提升三個 OVO 任務的 AUC，重點在目前最弱的 MCI_vs_AD（0.672）。

**基準線（當前）**

| Task | AUC | 目標 |
|------|-----|------|
| NC_vs_AD | 0.791 | > 0.82 |
| NC_vs_MCI | 0.686 | > 0.72 |
| MCI_vs_AD | 0.672 | > 0.75 |

---

## Phase 1：Fusion 模組升級（最高 CP 值）

> 參考：Hybrid Attention for Multimodal MCI Progression Prediction (NeurIPS 2025) — 0.9403 AUC on sMCI/pMCI

**目標**：把 `fusion_dim=20` 的單層 cross-attention 換成雙向 cross-attention + FFN block。

- [x] **1-A** 在 `train_pcag_combat_fusion.py` 新增 `BidirectionalPCAGFusion` 模組
  - fMRI→sMRI cross-attention（現有方向，保留原始 PCAG 機制）
  - sMRI→fMRI cross-attention（新增，對稱設計）
  - Learnable scalar blend merge（`sigmoid(w)*C1 + (1-sigmoid(w))*C2`）
  - LayerNorm + Residual connection
  - **設計決策**：不用 FFN（7.7x 參數膨脹，n=204 過擬合），不用 alignment loss（破壞模態互補性）
- [x] **1-B** 保持 `fusion_dim=20`（參數量 2.2x，合理；64 導致 7.7x 嚴重 overfit）
- [x] **1-C** ~~FFN~~ → 改用 learnable weighted sum + LayerNorm（更適合小資料集）
- [x] **1-D** ~~alignment loss~~ → 實驗發現 cosine alignment 反而破壞模態互補性，移除
- [x] **1-E** 第一輪（fusion_dim=64）測試結果：NC_vs_AD 0.650、NC_vs_MCI 0.672、MCI_vs_AD **0.354**（低於隨機）→ 確認過擬合問題
- [x] **1-F** 重新訓練（fusion_dim=20, bidir, 無 alignment loss）→ 結果仍低於基準線
- [x] **1-G** 分析 merge_w：三個任務均收斂至 0.62/0.38（fMRI→sMRI/sMRI→fMRI）

### 結論：Phase 1 負向結果（Negative Result）

| Task | 基準線 | bidir_v1(dim=64) | bidir_v2(dim=20) |
|------|--------|-----------------|-----------------|
| NC_vs_AD | **0.759** | 0.650 | 0.600 |
| NC_vs_MCI | **~0.670** | 0.672 | 0.647 |
| MCI_vs_AD | **0.672** | 0.354 | 0.520 |

**根本原因**：sMRI→fMRI 反向 attention 讓高品質 frozen BrainIAC embedding 去 query 只有 n=204 scratch 訓練的 fMRI features，引入噪聲。原始單向設計（fMRI 查詢 sMRI）是正確的。  
**結論**：fusion 方向不是瓶頸，sMRI encoder 凍結才是主要限制 → 移至 Phase 2。

**預期時間**：完成  
**風險**：低

---

## Phase 2：Augmentation 擴展 + sMRI 瓶頸分析

> Phase 1 結論後重新評估：ablation 顯示 sMRI 對 NC_vs_AD 有害、對 MCI_vs_AD 幫助極小。
> LoRA 在此情況下效益存疑，改優先測試低風險改動。

**分析結果**：

| Branch | NC_vs_AD | NC_vs_MCI | MCI_vs_AD |
|--------|----------|-----------|-----------|
| fMRI-only | **0.768** | 0.683 | 0.646 |
| sMRI-only | 0.750 | 0.533 | 0.515 |
| PCAG fusion | 0.682 | **0.708** | 0.651 |

→ NC_vs_AD：sMRI 反而拉低 AUC（0.768→0.682），aug 才是關鍵（+0.077）  
→ MCI_vs_AD：**從未嘗試 augmentation**，是最明顯的遺漏

- [x] **2-A** 認識到 LoRA 效益有限（sMRI 對兩個 task 的貢獻為負或極小）
- [x] **2-B** 對 MCI_vs_AD 和 NC_vs_MCI 加入 augmentation → **負向結果**
  - MCI_vs_AD: test 0.505（val 最高 0.976 → 嚴重 overfit）
  - NC_vs_MCI: test 0.597（比 noaug seed=456 差）
  - 根本原因：MCI/AD 邊界信號微弱，mixup 建立的混合樣本模糊類別邊界

### 結論：Augmentation 只對信號強的任務有效（NC_vs_AD），細微分類用 noaug

**關鍵發現**：inference pipeline 目前每個 task 都只用 seed=42 單一模型，但：
- NC_vs_MCI 論文用 seed=456（最佳 single seed）
- MCI_vs_AD 論文用 5-seed noaug median ensemble（已有全部 5 個 checkpoint）

→ 不需重新訓練，只要更新 inference 就能逼近論文數字

- [x] **2-C** 更新 inference pipeline 使用正確的 ensemble 策略
  - NC_vs_MCI：symlink 已改指向 seed=456（論文 single_best 策略）
  - MCI_vs_AD：`load_pcag_models_multi_seed()` 載入 25 個模型（5 seeds × 5 folds），`_pcag_predict()` 取 median
  - `_pcag_predict()` 加入 tuple 安全處理（未來 bidir 相容）

**預期時間**：完成  
**風險**：低  
**依賴**：無

---

## Phase 3：Multi-task Joint Training

> 參考：MCAD (2023)；Hybrid Attention (2025) 的 shared encoder 設計

**目標**：三個 OVO 任務共用 encoder，透過任務間知識共享改善 MCI_vs_AD。

- [x] **3-A** 設計 shared backbone + 3 task-specific head 架構（`PCAGMultiTaskModel`）
  - 共享：fMRI GAT encoder（`FMRIEncoder`）
  - 各自：獨立 `PCAGFusion` head × 3
- [x] **3-B** Heterogeneous batch multi-task loss：`1.0×L_ncad + 0.5×L_ncmci + 1.5×L_mciad`
- [x] **3-C** 各任務獨立 class-balanced sampler + 各任務獨立 ComBat per fold
- [x] **3-D** 實驗完成：joint vs separate 比較

### ❌ Phase 3 負向結果

| Task | 基準線 | Multi-task test AUC | 變化 |
|------|--------|---------------------|------|
| NC_vs_AD | 0.791 | 0.723 | **−0.068** |
| NC_vs_MCI | 0.686 | 0.581 | **−0.105** |
| MCI_vs_AD | 0.672 | 0.571 | **−0.101** |

**根本原因：Negative Transfer**
- AD 表徵在 NC_vs_AD（最大化 AD↔NC 距離）和 MCI_vs_AD（細微 AD↔MCI 分離）之間梯度衝突
- NC_vs_MCI（n=111 最多）主導 encoder 更新，儘管 loss weight=0.5
- n=16 AD 訓練樣本在 heterogeneous batch 下信號極度嘈雜（每 batch 可能只有 1-2 個 AD）
- Val AUC 跨 fold 方差大（MCI_vs_AD: 0.47–0.74），無法可靠指導 early stopping

**結論**：Multi-task 適合大規模資料集（n>1000），n=204 的小樣本情境下 negative transfer 不可避免。

腳本：`downstream/train_pcag_multitask.py`  
結果：`results/pcag_multitask_v1_results.json`

**預期時間**：完成  
**風險**：已驗證

---

## Phase 4：Dynamic Functional Connectivity（fMRI 圖升級）

> 參考：STAGIN (2021)；Spatio-temporal dynamic fMRI network (2025)；BrainGFM (NeurIPS 2025)

**目標**：用 raw BOLD 時間序列計算動態 FC，取代現有靜態 FC matrix。

- [ ] **4-A** 確認 raw BOLD 格式與覆蓋率
  - 原始檔：`datasets/fMRI/*/dswausub-*_task-rest_bold.nii.gz`
  - 確認有多少受試者有完整 BOLD（目前 n=204，需全部覆蓋）
- [ ] **4-B** 實作滑動窗口動態 FC 計算
  - 窗口大小 = 30 TR，步長 = 10 TR
  - 每個受試者產生 T 個 FC snapshot → shape `(T, 116, 116)`
- [ ] **4-C** 修改 GAT 成 temporal GAT（處理 T 個時間點的 graph sequence）
  - 參考 STAGIN：spatial attention + temporal attention
- [ ] **4-D** 評估動態 FC vs 靜態 FC 的 AUC 差異
- [ ] **4-E**（Optional）嘗試接入 BrainGFM pretrained encoder（等開源後）

**預期時間**：1–2 週  
**風險**：高（架構改動大，訓練時間長）  
**依賴**：Phase 1、2、3 完成後再考慮

---

## 進度追蹤

| Phase | 狀態 | 開始 | 完成 | AUC 改變 |
|-------|------|------|------|----------|
| Phase 1：Fusion 升級 | **❌ 負向結果** | 2026-06-08 | 2026-06-08 | 雙向 fusion 無改善，原架構最佳 |
| Phase 2：Aug 擴展 + Ensemble 修正 | **✅ 完成** | 2026-06-08 | 2026-06-08 | Aug 對細微任務無效；改用正確 ensemble 策略（NC_vs_MCI seed=456, MCI_vs_AD 5-seed median） |
| Phase 3：Multi-task | **❌ 負向結果** | 2026-06-08 | 2026-06-08 | 全任務 AUC 均下降（negative transfer，n=204 樣本太小） |
| Phase 4：Dynamic FC | 未開始 | — | — | — |

---

## 每次訓練的記錄格式

```
實驗：Phase X-Y
日期：YYYY-MM-DD
設定：（改了哪些 hyperparameter）
結果：
  NC_vs_AD:  X.XXX（vs 基準 0.791）
  NC_vs_MCI: X.XXX（vs 基準 0.686）
  MCI_vs_AD: X.XXX（vs 基準 0.672）
結論：（是否有進步、下一步做什麼）
```
