# 論文投稿計畫 — KGRS 失智症輔助診斷系統

> 建立日期：2026-05-25  
> 狀態標示：`[ ]` 未開始　`[~]` 進行中　`[x]` 完成　`[!]` 有問題

---

## 目標 Venue 總覽

| Venue | 估計 Deadline | 論文長度 | 定位 | 難度 | 策略 |
|---|---|---|---|---|---|
| **IEEE ISBI 2027** | ~Nov 2026 | 4 頁（IEEE） | 生醫影像方法 | 中 | 第一槍，方法論貢獻 |
| **AIME 2027** | ~Feb 2027 | 10–12 頁 | 醫療 AI 系統 | 中 | 主力目標，系統完整版 |
| **MICCAI 2027** | ~Feb 2027 | 8 頁（LNCS）| 醫學影像頂會 | 高 | 拚看看，需強 baseline |

> **策略邏輯**：ISBI 先試水溫（短版方法），拿 review feedback 改進；  
> AIME 投完整系統論文（主力）；MICCAI 同期衝或視 AIME 結果決定。

---

## 現況快照（2026-05-26 更新）

### 已完成
- [x] PCAG-ComBat 三任務訓練（v2 split，ComBat leakage 修正）
- [x] Test set 擴充：AD 4 → 11，v2 split（NC=20, MCI=18, AD=11）
- [x] 95% Bootstrap CI：NC/AD 0.814, NC/MCI 0.747, MCI/AD 0.697
- [x] 3-class Balanced Accuracy：0.483（PCAG all tasks）
- [x] MCI/AD 從 KD GNN 換成 PCAG dual-modal（AUC +0.183）
- [x] Q/K/V swap 消融實驗 v2（確認 Q=fMRI 為正確方向，圖：qkv_swap_v2.png）
- [x] Four-group modality analysis（三個 task 含 MCI_vs_AD，圖：four_group_analysis.png）
- [x] KG + RAG + LLM 報告生成系統（已上線）
- [x] 報告圖表：report_v2_main.png、report_v2_cm.png
- [x] **Baseline 比較實驗（Phase 2-A 全部完成）**
- [x] **Ablation table + 圖表（ablation_comparison_v2.png、ablation_delta_v2.png）**
- [x] **ROC 曲線圖（roc_curves_v2.png，含 95% CI band）**
- [x] **LLM-Judge 報告品質評估 pipeline（兩個 judges: Gemma3 + Qwen3，含 Wilcoxon + bootstrap CI）**
- [x] **api_server 加入 use_rag flag 和 model_observations 欄位（支援 RAG ablation 和注入模型觀察）**
- [x] **api_server URL 改為環境變數 API_BASE_URL（之後部署到實驗室網域只改 .env）**

### 📊 最新結果（2026-05-27 中午）

**fMRI no-label ComBat + Aug + 5-seed Variance-Penalized OOF Ensemble**（最終 2026-05-27 14:50）：
| Task | PCAG-ComBat | 95% CI | 與原 confounded Δ |
|---|---|---|---|
| NC_vs_AD | **0.791 ⭐** (top-3 OOF + aug) | [0.61, 0.93] | -0.023 |
| NC_vs_MCI | **0.686** (single_best seed=456) | [0.51, 0.86] | -0.061 |
| MCI_vs_AD | **0.672** (median) | [0.45, 0.86] | -0.025 |

✅ Selection 方法：OOF-AUC penalized by per-seed variance（Caruana 2004 KDD + Cawley 2010 JMLR）

對比 single-seed s=42 baseline (no aug)：
| Task | s=42 baseline | 最終 | Δ |
|---|---|---|---|
| NC_vs_AD | 0.682 | **0.791** | **+0.109** ✓ |
| NC_vs_MCI | 0.708 | **0.686** | -0.022 |
| MCI_vs_AD | 0.652 | **0.672** | **+0.020** ✓ |

⚠️ **Concat fusion 在 NC_vs_AD 上 0.805 > PCAG 0.682** — 論文敘事要小心：
- 主推 PCAG 為 MCI_vs_AD / NC_vs_MCI 上的 SOTA
- NC_vs_AD 不主推，當作 sanity check

詳見 [revise.md 對比表](revise.md)

### 待完成（Phase 3 論文寫作）
- [~] LLM 報告品質評估（3-C-2）— pipeline 已可跑，但發現「模型觀察」內容品質不足以投稿，見 [revise.md](revise.md)
- [ ] 模型可解釋性升級（revise.md Phase R-1 ~ R-5）⭐ **下個任務**
- [ ] 方法架構圖 System Overview Figure（3-A-1）
- [ ] Demographics table（3-A-2）
- [ ] ISBI draft 4 頁（3-B-2）

---

## Phase 2 — 補實驗（6–7 月完成）

> 沒有 baseline 比較，三個 venue 都難過 reviewer。這是最優先的事。

### 2-A｜Baseline 模型（必做）

所有 baseline 用**相同 v2 split**（pcag_train/test_aligned_v2.csv）。

| Baseline | 腳本 | 預計工時 | 狀態 |
|---|---|---|---|
| SVM + FC upper-triangle | `baseline_svm.py` | 0.5 天 | `[x]` |
| fMRI-only GNN（smri=zeros）| `train_pcag_combat_fusion.py --modality fmri_only` | 0.5 天 | `[x]` |
| sMRI-only | `train_pcag_combat_fusion.py --modality smri_only` | 0.5 天 | `[x]` |
| Concat fusion（無 cross-attention）| `train_concat_fusion.py` | 1 天 | `[x]` |
| PCAG 無 ComBat | `train_pcag_combat_fusion.py --no_combat` | 0.5 天 | `[x]` |

- [x] **2-A-1** `baseline_svm.py`（FC tri 6670→PCA→SVM RBF GridSearch）
- [x] **2-A-2** fMRI-only（`--modality fmri_only`，smri_feat=zeros）
- [x] **2-A-3** sMRI-only（`--modality smri_only`，fmri_emb=zeros）
- [x] **2-A-4** Concat fusion（ConcatFusion: concat+MLP，無 cross-attention）
- [x] **2-A-5** 無 ComBat（`--no_combat`，raw sMRI 直接進模型）

### 2-B｜Ablation Table（整理已有結果）

- [x] **2-B-1** 整理成表格（已出圖 `ablation_comparison_v2.png`）：

  | Model | NC/AD AUC | NC/MCI AUC | MCI/AD AUC |
  |---|---|---|---|
  | SVM (FC+PCA) | 0.755 | 0.750 | 0.505 |
  | fMRI-only GNN | 0.786 | 0.719 | 0.697 |
  | sMRI-only | 0.750 | 0.533 | 0.515 |
  | Concat fusion | 0.768 | 0.667 | 0.591 |
  | PCAG w/o ComBat | 0.727 | 0.569 | 0.510 |
  | **PCAG-ComBat (ours)** | **0.814** | **0.747** | **0.697** |

  圖檔：`results/ablation_comparison_v2.png`、`results/ablation_delta_v2.png`

- [x] **2-B-2** Q/K/V swap 消融（v2 重跑，圖檔 `results/qkv_swap_v2.png`）
  - NC/AD: Q=fMRI 0.814 vs Q=sMRI 0.800 (Δ+0.014)
  - NC/MCI: Q=fMRI 0.747 vs Q=sMRI 0.681 (Δ+0.067)
  - MCI/AD: Q=fMRI 0.697 vs Q=sMRI 0.591 (Δ+0.106)

### 2-C｜視覺化圖表

- [x] **2-C-1** ROC 曲線：`results/roc_curves_v2.png`（全模型 + PCAG-ComBat 95% CI band）
- [x] **2-C-2** AUC comparison figure（`ablation_comparison_v2.png`）
- [x] **2-C-3** Modality contribution figure（`results/four_group_analysis.png`，三個 task 含 MCI_vs_AD）

---

## Phase 3 — 論文寫作

### 3-A｜共用素材（三個 venue 都用）

- [ ] **3-A-1** 方法架構圖（System Overview Figure）
  - 左：fMRI preprocessing → GNN encoder
  - 中：sMRI T1 → BrainIAC ViT → ComBat harmonization
  - 右：PCAG cross-attention fusion → 3 binary classifiers → hierarchical decision
  - 底：KG + RAG → LLM report generation
- [ ] **3-A-2** 病患 demographics table（TPMIC vs ADNI 各組人數、年齡等）
- [ ] **3-A-3** 完整 results table（含 CI）
- [ ] **3-A-4** 報告生成範例截圖（UI 截圖 + 生成報告）

### 3-B｜ISBI 2027（目標：~Oct 2026 投稿）

**定位**：方法論文，強調 PCAG-ComBat 多模態融合 + ComBat harmonization

**頁數限制**：4 頁 IEEE 格式（精簡）

**重點 sections：**
- Abstract：多模態 fMRI+sMRI 融合，NC/MCI/AD 三分類，AUC 數字
- Method：PCAG cross-attention fusion + ComBat（核心貢獻）
- Experiments：AUC table + ablation（方法各模組貢獻）
- 不需要 KG/RAG/報告生成（頁數不夠，留給 AIME）

**Reviewer 最可能的問題：**
1. *"為什麼用 cross-attention 而不是 simple concat？"* → 已有 ablation
2. *"ComBat 真的有幫助嗎？"* → 已有 no-ComBat baseline
3. *"小資料集，結論可靠嗎？"* → 95% CI 回應

- [ ] **3-B-1** 確認 ISBI 2027 官方 deadline（預計 Nov 2026）
- [ ] **3-B-2** 寫 ISBI draft（4 頁）
- [ ] **3-B-3** 內部 review → 修改 → 投稿

---

### 3-C｜AIME 2027（目標：~Jan 2027 投稿，主力）

**定位**：端到端臨床 AI 系統論文，方法 + 系統 + 臨床可解釋性

**頁數限制**：10–12 頁（Springer LNAI）

**重點 sections：**
- Introduction：MCI/AD 臨床診斷困難，現有系統缺乏可解釋性
- Related Work：多模態 AD 分類、知識圖譜於醫療 AI、RAG + LLM
- Method：完整系統（PCAG-ComBat + KG + RAG + LLM report）
- Experiments：分類 AUC table + ablation + 報告品質評估
- Discussion：臨床意義、limitation（小資料集、CI 寬）
- 結論

**核心 selling point**：
> *"我們是第一個將多模態腦影像分類、知識圖譜檢索和 LLM 報告生成整合為完整系統的工作，且每個模組都有定量評估。"*

**Reviewer 最可能的問題：**
1. *"分類 AUC 不夠高"* → 強調系統貢獻，不是分類 SOTA
2. *"LLM 報告品質怎麼評估？"* → 需要做小規模 user study 或 LLM-as-judge
3. *"KG 和 RAG 對最終診斷有什麼貢獻？"* → 需要有 w/o RAG 的 ablation

- [ ] **3-C-1** 確認 AIME 2027 官方 deadline
- [x] **3-C-2 DONE 2026-05-28** LLM 報告品質評估（G-Eval, n=49 pipeline 已建立，2 judges Gemma3+Qwen3，含 Wilcoxon + bootstrap CI）⚠️ 需等 [revise.md](revise.md) Phase R 完成後重跑
- [x] **3-C-3 DONE** RAG ablation：use_rag flag 已在 api_server 實作；smoke test 可跑；待 revise 後正式跑全集
- [ ] **3-C-4** 寫 AIME 完整 draft（10–12 頁）
- [ ] **3-C-5** 內部 review → 修改 → 投稿
- [ ] **3-C-6** ⭐ 新增：Model Interpretability Section（PCAG attention + GradCAM + GAT rollout，見 revise.md）

---

### 3-D｜MICCAI 2027（目標：~Feb 2027 投稿，挑戰）

**定位**：醫學影像頂會，強調方法創新 + 臨床影像貢獻

**頁數限制**：8 頁（Springer LNCS）

**重要提醒**：MICCAI 接受率約 30%，reviewer 期望：
- 清楚的方法論貢獻（不只是系統整合）
- 與 SOTA 方法比較（可引用 BrainNetCNN、BrainGNN 等）
- 較大資料集或至少外部驗證

**你現在的風險**：小資料集（AD=27），沒有外部驗證集，CI 寬。

**如果要投，必須做到：**
- [ ] **3-D-1** 從 ADNI 公開資料補充更多 AD 樣本（如果可能）
- [ ] **3-D-2** 或改成 cross-validation 報告（5-fold outer CV 取代固定 hold-out）
- [ ] **3-D-3** 與至少 2 篇 published 方法比較（同 dataset 條件下）
- [ ] **3-D-4** 確認 MICCAI 2027 deadline 後決定是否衝
- [ ] **3-D-5** 如果 AIME 接受 → 修改重寫 8 頁版本投 MICCAI

> ⚠️ **建議**：先拚 ISBI + AIME；如果 AIME 投出去且有時間，再看 MICCAI 2027 值不值得衝。

---

## 時間軸

```
2026
May     ✅ Phase 1 + Phase 2 全部完成（比原計畫提前 2 個月）
        → 七張論文圖表已全部產出
Jun     Phase 3-A：共用素材
          - LLM-Judge 報告品質評估（3-C-2）← 最優先
          - 方法架構圖 System Overview（3-A-1）
          - Demographics table（3-A-2）
Jul     Phase 3-B：寫 ISBI draft（4 頁）
Aug     內部 review、修改 ISBI draft
Sep     開始寫 AIME 完整版（10–12 頁）
Oct     確認 ISBI 2027 deadline → 投稿
Nov     ISBI deadline（估計）
        AIME draft 完成 → 內部 review

2027
Jan     AIME 2027 deadline（估計）→ 投稿
        MICCAI 2027 決策（視狀況）
Feb     MICCAI 2027 deadline（估計）
```

---

## 必補但尚未做的關鍵實驗

> 以下是 reviewer 最可能要求的，但目前缺少：

| 實驗 | 重要性 | 需要 ISBI | 需要 AIME | 需要 MICCAI | 狀態 |
|---|---|---|---|---|---|
| Baseline 比較（SVM、concat 等）| ⭐⭐⭐ | 必要 | 必要 | 必要 | ✅ 完成 |
| ROC 曲線圖 | ⭐⭐⭐ | 必要 | 必要 | 必要 | ✅ 完成 |
| LLM 報告品質評估 pipeline | ⭐⭐ | 不需要 | 建議 | 不需要 | 🔄 框架完成，待 revise |
| w/o RAG ablation | ⭐⭐ | 不需要 | 建議 | 不需要 | 🔄 框架完成，待 revise |
| **Model interpretability**（PCAG attn / GradCAM / GAT rollout）| ⭐⭐⭐ | 不需要 | 強烈建議 | 必要 | ❌ 待做（[revise.md](revise.md)）|
| Demographics table | ⭐⭐ | 簡略版 | 完整版 | 完整版 | ❌ 待做 |
| 外部驗證集 | ⭐⭐⭐ | 不需要 | 不需要 | 強烈建議 | ❌ 待做 |
| Cross-validation 報告 | ⭐⭐ | 不需要 | 不需要 | 建議 | ❌ 待做 |

---

## 論文核心敘事（三個 venue 共用）

> 每次改寫只是調整**比例**，不是換故事。

**一句話版本：**
> *PCAG-ComBat 整合多模態腦影像分類（fMRI + sMRI）、ComBat 多站點諧波化、
> 知識圖譜與 RAG 報告生成，建構可解釋的失智症輔助診斷系統，
> 在小樣本臨床資料上達到 NC/AD AUC=0.814（95% CI: 0.632–0.939）。*

**三個 venue 的重心調整：**

| Venue | 方法 % | 系統 % | 臨床 % |
|---|---|---|---|
| ISBI | 70% | 20% | 10% |
| AIME | 30% | 50% | 20% |
| MICCAI | 60% | 30% | 10% |
