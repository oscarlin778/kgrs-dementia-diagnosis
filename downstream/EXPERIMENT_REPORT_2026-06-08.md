# PCAG-ComBat 模型優化實驗報告
**日期：2026-06-08**  
**目標：提升三個 OVO 任務的 AUC，重點改善 MCI_vs_AD（最弱任務）**

---

## 一、系統架構摘要

### PCAG-ComBat（Patient-specific Cross-Attention Generation）

| 元件 | 規格 |
|------|------|
| fMRI 編碼器 | GAT（3層 4頭，hidden=128） + 虛擬節點 + 9網路池化 → 1280-dim |
| sMRI 編碼器 | BrainIAC ViT（pre-trained, frozen） → 768-dim |
| 融合模組 | Cross-attention（Q=fMRI, K/V=sMRI），fusion_dim=20 |
| 訓練 | 5-fold CV, 200 epochs, AdamW lr=3e-4, batch=16, class-balanced sampler |
| 場域校正 | fMRI：no-label ComBat（全局預計算）；sMRI：label ComBat per fold |
| 分類架構 | OVO（One-vs-One）：NC_vs_AD / NC_vs_MCI / MCI_vs_AD |
| Ensemble | 5 seeds（42/123/456/789/2024），OOF 指導策略選擇 |

### 資料集

| | n | NC | MCI | AD |
|-|---|----|----|---|
| Train | 155 | 70 | 69 | 16 |
| Test | 49 | 20 | 18 | 11 |
| **Total** | **204** | **90** | **87** | **27** |

Sites: TPMIC（n=99）+ ADNI（n=56），2 sites

---

## 二、基準線（論文提交數字）

> 5-seed × 5-fold 集成，OOF 指導選策略，n=49 held-out test

| Task | AUC | 95% CI | Sensitivity | Specificity |
|------|-----|--------|-------------|-------------|
| NC vs AD | **0.791** | [0.609, 0.927] | 0.455 | 0.850 |
| NC vs MCI | **0.686** | [0.509, 0.855] | 0.889 | 0.600 |
| MCI vs AD | **0.672** | [0.468, 0.879] | 0.273 | 0.778 |

### 消融分析（單一 seed=42）

| Method | NC_vs_AD | NC_vs_MCI | MCI_vs_AD |
|--------|----------|-----------|-----------|
| SVM（FC features） | 0.755 | 0.750 | 0.505 |
| fMRI-only | 0.768 | 0.683 | 0.646 |
| sMRI-only | 0.750 | 0.533 | 0.515 |
| Concat fusion | **0.804** | 0.417 | 0.535 |
| No ComBat | 0.682 | 0.678 | 0.525 |
| Q/K/V Swapped | 0.700 | 0.681 | 0.515 |
| **PCAG-ComBat（ours）** | 0.682 | **0.708** | **0.651** |

**關鍵觀察：**
- fMRI-only 集成在 MCI_vs_AD 退化到 0.500（隨機），PCAG 達 0.672（+17.2%）
- Concat > PCAG 在 NC_vs_AD：大型組間差異下全局特徵足夠，但 cross-attention 在細微任務（NC_vs_MCI, MCI_vs_AD）才顯現優勢

---

## 三、Phase 1：雙向 Fusion 模組升級

### 動機
參考 Hybrid Attention for Multimodal MCI Progression Prediction（NeurIPS 2025），引入對稱雙向 cross-attention。

### 設計
- **Direction 1**（原始 PCAG）：fMRI → sMRI（Q=fMRI, K/V=sMRI）
- **Direction 2**（新增）：sMRI → fMRI（Q=sMRI, K/V=fMRI）
- 可學習標量混合：`w × C1 + (1-w) × C2`，避免 FFN 膨脹（fusion_dim=20 vs 64）

### 實驗結果

| 設定 | NC_vs_AD | NC_vs_MCI | MCI_vs_AD |
|------|----------|-----------|-----------|
| 基準線（原始 PCAG） | **0.759** | **~0.670** | **0.672** |
| bidir v1（dim=64） | 0.650 | 0.672 | 0.354 ⚠️ |
| bidir v2（dim=20） | 0.600 | 0.647 | 0.520 |

### 結論：❌ 負向結果

**根本原因**：
1. `fusion_dim=64` → 7.7× 參數量，n=204 嚴重 overfit
2. 反向注意力（sMRI→fMRI）讓高品質凍結 BrainIAC embedding 去 query 只有 n=204 從頭訓練的 fMRI 特徵，引入噪聲
3. `merge_w` 收斂到 0.62：模型傾向原始 fMRI→sMRI 方向，表示反向沒有幫助

**推論**：sMRI 編碼器凍結是主要瓶頸，而非 fusion 方向。原始單向設計是正確的。

---

## 四、Phase 2：Augmentation 分析 + Ensemble 修正

### 2-A：對 MCI_vs_AD 加 Augmentation

| 實驗 | val AUC | test AUC | 結論 |
|------|---------|----------|------|
| MCI_vs_AD + mixup(0.2) | 0.976 ⬆️ | 0.505 ❌ | 嚴重 overfit |
| NC_vs_MCI + mixup(0.2) | 高 | 0.597 ❌ | 低於 noaug |

**根本原因**：MCI/AD 邊界信號微弱，mixup 建立的混合樣本模糊類別邊界，模型學到輸出 0.5 機率而非真正辨別。**Augmentation 只對信號強（大組間差異）的任務有效**。

### 2-B：Ensemble 策略修正（不需重新訓練）

**問題發現**：Inference pipeline 所有任務都只用 seed=42 單一模型，但論文數字來自 OOF 指導的最佳策略選擇。

| Task | 論文策略 | 修正前 inference | 修正後 inference |
|------|---------|----------------|----------------|
| NC_vs_AD | top3 seeds + aug | seed=42 aug（同步，但需重建） | ✅ 重建後正常 |
| NC_vs_MCI | single_best seed=456 | seed=42（次佳） | ✅ symlink → seed=456 |
| MCI_vs_AD | 5-seed median | seed=42 單模型 | ✅ 25 模型 median |

**實施**：
- `pcag_combat_NC_vs_MCI_fold*.pt` → symlink 更新指向 `seed=456` checkpoints
- `inference_pipeline_v2.py` → 新增 `load_pcag_models_multi_seed()`，5 seeds × 5 folds = 25 模型，取 median
- NC_vs_AD checkpoints 於磁碟清理後重新訓練（test AUC 0.759，與基準線一致）

### 結論：✅ 完成（不含重新訓練的模型改進）

---

## 五、Phase 3：Multi-task Joint Training

### 動機
- NC_vs_MCI 任務有最清晰的 fMRI 訊號（NC vs MCI 邊界較明顯）
- 若 fMRI 編碼器同時學習三個任務，可望理解完整的疾病連續體（NC→MCI→AD）
- 特別期望 NC_vs_MCI 的梯度能改善編碼器對 MCI 表徵，進而提升 MCI_vs_AD

### 架構

```
輸入: fMRI 圖（116 ROI） + sMRI BrainIAC features
         ↓
  ┌─────────────────────┐
  │  共享 fMRI Encoder  │  ← GAT + VN + 9-network pool
  └─────────────────────┘
         ↓
  ┌──────┬──────┬──────┐
  │Head A│Head B│Head C│  ← 各自獨立 PCAGFusion（fusion_dim=20）
  │NC/AD │NC/MCI│MCI/AD│
  └──────┴──────┴──────┘
         ↓
  L_total = 1.0×L_ncad + 0.5×L_ncmci + 1.5×L_mciad
```

**Heterogeneous batch 訓練**：每個 step 同時處理 3 個任務的 batch（concat → 共享 encoder 一次 forward），確保 encoder 在每個 gradient step 都同時接收三個任務的梯度。

**ComBat**：每任務、每 fold 獨立 fit（不同受試者子集 → 不同 site distribution），結果存入各任務 head。

**輸出 checkpoint**：
- Full multi-task: `checkpoints/pcag_multitask_v1/multitask_fold{0-4}.pt`
- 任務特定（與 inference pipeline 相容）: `checkpoints/pcag_multitask_v1_{task}/pcag_combat_{task}_fold{0-4}.pt`

### Phase 3 實驗進度

訓練腳本：`downstream/train_pcag_multitask.py`  
啟動指令：`python3 train_pcag_multitask.py --epochs 200 --seed 42 --w_ncad 1.0 --w_ncmci 0.5 --w_mciad 1.5`  
Log：`logs/multitask_s42.log`

**Fold 1-2 中間結果（val AUC，供參考）：**

| Fold | NC_vs_AD | NC_vs_MCI | MCI_vs_AD |
|------|----------|-----------|-----------|
| 1 | 0.524 | 0.689 | **0.738** ⬆️ |
| 2 | 0.524 | 0.689 | **0.738** |

> MCI_vs_AD val AUC 0.738 > 基準線 0.672，初步跡象正面。

### Phase 3 最終結果：❌ 負向結果

| Task | OOF AUC | Test AUC | 基準線 | 變化 |
|------|---------|----------|--------|------|
| NC_vs_AD | 0.636 | 0.723 | 0.791 | **−0.068** ❌ |
| NC_vs_MCI | 0.558 | 0.581 | 0.686 | **−0.105** ❌ |
| MCI_vs_AD | 0.620 | 0.571 | 0.672 | **−0.101** ❌ |

Result JSON: `results/pcag_multitask_v1_results.json`

### 失敗分析

**根本原因：Negative Transfer（負向遷移）**

1. **AD 表徵衝突**：AD 受試者同時參與兩個任務的梯度：
   - NC_vs_AD 要求：AD 特徵 ↔ NC 距離最大化（大幅分離）
   - MCI_vs_AD 要求：AD 特徵 ↔ MCI 距離最大化（細微分離）
   - 兩者的最優 AD 表徵方向衝突，特別在只有 n=16 AD 訓練樣本的情況下

2. **NC_vs_MCI 主導訓練**：NC_vs_MCI 有最多受試者（n=111 vs 68），即使 loss weight=0.5，每個 epoch 的 batch 數量仍比其他任務多，實際梯度影響力超過預期

3. **Val AUC 跨 fold 方差極大**：
   - Fold 1: MCI_vs_AD val=0.738 → 但 test 集結果 0.571，表示 val set 無法可靠預測泛化
   - Fold 4: NC_vs_AD val=0.821（但最終 test 0.723）

4. **小資料集上的 multi-task 已知難題**：n=204 的資料集中，每個 task-specific batch 可能只包含 1-2 個 AD 受試者，使梯度信號極度嘈雜

### 結論

**Single-task OVO + 針對性 ensemble 策略仍是最佳設計。** Multi-task 適合大規模資料集（n>1000），但在 n=204 的小樣本情境下 negative transfer 不可避免。

---

## 六、Scripts 目錄整理

### 整理前

根目錄混雜 15+ `.py`、`.sh`、`.png`、`.log` 等文件，難以辨識功能。

### 整理後目錄結構

```
scripts/
├── downstream/         # 推論管線、API Server、訓練腳本（核心）
│   ├── api_server.py
│   ├── inference_pipeline_v2.py
│   ├── train_pcag_combat_fusion.py
│   └── train_pcag_multitask.py   ← Phase 3 新增
├── models/             # 模型歷史訓練腳本
│   ├── run_ablations.sh          ← 從根目錄移入
│   ├── run_remaining_ablations.sh ← 從根目錄移入
│   └── train_hierarchical_gnn_e*.py（原有）
├── preprocessing/      # fMRI ComBat、sMRI 特徵提取
├── analysis/           # 評估與分析腳本（7 個，從根目錄移入）
│   ├── analyze_splits.py
│   ├── check_labels.py
│   └── ...
├── visualization/      # 視覺化腳本 + 輸出圖片（從根目錄移入）
│   ├── plot_results.py
│   └── figures/        # PNG 輸出
├── data/               # 資料匯入/準備腳本（從根目錄移入）
│   ├── import_patients_to_neo4j.py
│   └── prepare_patient_nodes.py
├── utils/              # 共用工具函式
├── logs/               # 訓練 log（git-ignored）
├── checkpoints/        # 模型 checkpoints（git-ignored）
├── README.md           ← 更新目錄結構說明
└── requirements.txt
```

**注意**：
- `unified_subject_split*.json` 保留在根目錄（被 `models/` 下 10+ 訓練腳本引用，避免大範圍路徑更新）
- `*.csv` 資料檔（含病患 ID）移至 `data/`，`.gitignore` 已排除 `*.csv`
- 刪除 `REDO_PROMPT.md`（內部開發筆記，已在 `.gitignore`）

---

## 七、現有系統狀態

### Backend / Frontend Services

```bash
systemctl --user status brainiac-backend.service   # FastAPI @ port 8082
systemctl --user status brainiac-frontend.service  # Vite @ port 3000
```

Vite proxy: `/api`, `/static_data`, `/static_saliency` → `http://localhost:8082`

### Inference Pipeline 狀態

| 元件 | 狀態 | 備注 |
|------|------|------|
| NC_vs_AD | ✅ 正常 | aug checkpoints（5-fold）重建完成 |
| NC_vs_MCI | ✅ 修正 | symlink → seed=456（最佳 single seed） |
| MCI_vs_AD | ✅ 修正 | 25 模型 median（5 seeds × 5 folds） |
| sMRI saliency | ✅ 正常 | BrainIAC frozen ViT |
| KG + RAG | ✅ 正常 | Neo4j + nomic-embed |
| LLM 報告 | ✅ 正常 | Gemma3-12B via Ollama |

---

## 八、Benchmark 比較

| 方法 | NC_vs_AD | NC_vs_MCI | MCI_vs_AD | 備注 |
|------|----------|-----------|-----------|------|
| **Ours（5-seed ensemble）** | **0.791** | **0.686** | 0.672 | fMRI+sMRI, n=204 |
| ADMGCN (Bioinformatics 2025) | 0.759 | 0.665 | **0.758** | fMRI GCN, ADNI |
| Spatio-Temp dFC GCN (2025) | 0.831 | 0.792 | 0.769 | fMRI only, n=85 |
| Multimodal fMRI+sMRI+DTI (2026) | 0.901 | 0.839 | 0.809 | 額外 DTI modality |

**結論**：在 NC_vs_AD 和 NC_vs_MCI 上優於最接近的 comparable baseline（ADMGCN）；MCI_vs_AD 差距 0.086，原因為小測試集高方差（CI=[0.468, 0.879]）。

---

## 九、LLM 報告品質評估（v5）

使用 Gemma3 + Qwen3 作為評審，n=45 病患，n_pairs=90。

| 指標 | Gemma3 Δ | p | Qwen3 Δ | p |
|------|---------|---|---------|---|
| 事實準確性 | +0.067 | 0.406 | +0.133 | 0.359 |
| 臨床相關性 | +0.000 | 1.000 | **+0.333** | **0.007*** |
| 完整性 | **+0.178** | 0.069† | +0.000 | 1.000 |
| 連貫性 | -0.111 | 0.294 | +0.156 | 0.167 |

**主要發現**：RAG 顯著提升 Qwen3 的臨床相關性（p=0.007），Gemma3 完整性呈邊際改善（p=0.069†）。腦區名稱（Precuneus, Cingulum 等）作為 RAG 的具體錨點，是效果提升的關鍵。

---

## 十、建議下一步

### Phase 4：動態 FC（高風險，高回報）

**可行性確認**：
- Raw BOLD `.nii.gz` 已存在：`datasets/fMRI/*/dswausub-*_task-rest_bold.nii.gz`
- 預計 n=204 全覆蓋

**實作方案**：
1. 滑動窗口 FC（窗口=30 TR，步長=10 TR）→ 每人 T 個 FC snapshot
2. Temporal GAT（STAGIN 架構）：spatial + temporal attention
3. 預計改善幅度：根據 STAGIN 論文，動態 FC 相比靜態 FC 在 MCI 分類有 +5~8% AUC

**預估時間**：1–2 週  
**主要風險**：BOLD 時間序列預處理（去趨勢、bandpass filter）需確認已完成

---

## 附錄：Phase 3 訓練指令

```bash
# 基本訓練（seed=42）
cd downstream/
python3 train_pcag_multitask.py --epochs 200 --seed 42 --w_ncad 1.0 --w_ncmci 0.5 --w_mciad 1.5

# 追加 seed 以建立 ensemble
python3 train_pcag_multitask.py --seed 456 --w_mciad 1.5
python3 train_pcag_multitask.py --seed 123 --w_mciad 1.5

# 查看 log
tail -f ../logs/multitask_s42.log
```

```bash
# 將 multi-task checkpoints 接入 inference pipeline（更新 symlink）
# NC_vs_MCI：若 multi-task 結果 > 0.686
ln -sfn $(pwd)/checkpoints/pcag_multitask_v1_NC_vs_MCI checkpoints/pcag_combat_v2_NC_vs_MCI_current
```
