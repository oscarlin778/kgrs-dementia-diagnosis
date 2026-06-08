# Gemini CLI 任務集（給論文準備用）

> 這些是不需要動到我們 codebase 的「離線研究 / 寫作」任務。
> Gemini 跑完後請把結果存到 `paper_research/` 目錄下。
> 你可以一次給 Gemini 一個 task，或全部一起給。

---

## 任務 1（最優先）：FC 矩陣 / GNN data augmentation 方法調查

**請以繁體中文回答，但保留專有名詞、方法名稱、引用為英文。**

**背景**：我有一個多模態 (fMRI functional connectivity + sMRI features) 失智症 (NC/MCI/AD) 三分類研究，
GNN 在 116×116 對稱 FC 矩陣上做 message passing。資料集小（test n=49, AD class 只有 11 人）。
我需要做 data augmentation 提升 AUC。

**請完成**：
1. 找 5-10 篇 2020-2025 的 paper，涵蓋以下方法在 functional connectivity 上的應用：
   - SMOTE / borderline-SMOTE 對 brain connectivity
   - MixUp / CutMix on graphs
   - Adversarial perturbation / Gaussian noise injection on FC matrix
   - Node/edge dropout for GNN
   - Synthetic patient generation via VAE / GAN
2. 每篇給：title, 第一作者, 年份, 方法 1-2 句話描述, AUC/accuracy 提升幅度（如果有）
3. 給一個 **「最適合我們小資料、AD 少數類問題」的推薦方案**（top 1-2）+ 理由
4. 推薦方案的 **PyTorch 實作伪代碼**（不要完整 code，30-50 行 pseudocode 即可）

**輸出**：`paper_research/augmentation_survey.md`

---

## 任務 2：多站點 fMRI harmonization 方法比較

**背景**：我用 no-label ComBat 諧波化 FC 矩陣，已成功降低 site effect。但 reviewer 可能會問：
「為什麼不用 [其他方法]？」需要 defensive。

**請完成**：
1. 對以下方法各寫 3-5 句話介紹：
   - **ComBat** (Johnson et al. 2007) — 我用的，need brief recap
   - **CovBat** (Chen et al. 2022) — covariance harmonization
   - **DANN (Domain-Adversarial Neural Networks)** (Ganin et al. 2016) — site adversarial
   - **fMRIPrep + AROMA** — preprocessing 階段做
   - **Harmony / NeuroHarmonize** — Python ports of ComBat
   - **Conditional VAE site harmonization** — newer approach
2. 比較表（rows = methods, columns = "原理", "在 FC matrix 適用度", "是否需要 retraining", "報導文獻")
3. **回答潛在 reviewer 問題：「為什麼 ComBat without label 比 with label 好？」**（給 3-5 點論證）

**輸出**：`paper_research/harmonization_comparison.md`

---

## 任務 3：相關工作 (Related Work) 段落 draft

**背景**：我們的論文有三大貢獻：
1. PCAG-ComBat 多模態融合（fMRI GNN + BrainIAC ViT sMRI）做 NC/MCI/AD 分類
2. 嚴謹的多站點 site harmonization 方法論
3. KG + RAG + LLM 自動病患報告生成系統（端到端）

**請用繁體中文撰寫**一份 ~800 字的 Related Work 章節 draft，分四段：

**段 1：Multi-modal AD classification (fMRI + sMRI)**
- 引用 3-5 篇 fMRI GNN AD classification 的代表作 (BrainGNN, BrainNetCNN, Hi-GCN 等)
- 引用 2-3 篇 sMRI ViT / CNN AD classification
- 引用 1-2 篇 multimodal fusion (fMRI + sMRI)
- 帶出 gap：缺乏 cross-attention 融合 + 多站點 harmonization

**段 2：Multi-site brain imaging harmonization**
- 引用 ComBat 原文 + 1-2 篇 ComBat 在 brain imaging 應用
- 引用 1 篇 DANN 在腦影像
- 帶出 gap：少有 paper 同時 (a) report within-site AUC、(b) 比較有/無 label 諧波化策略

**段 3：Medical knowledge graphs and RAG for radiology**
- 引用 2-3 篇 medical KG 在 dementia 領域
- 引用 2-3 篇 RAG for medical / radiology report
- 引用 1-2 篇 LLM medical report generation (e.g., MedPaLM, GPT-4 in medicine)
- 帶出 gap：少有完整 KG→RAG→LLM 端到端整合 + 量化評估

**段 4：簡短 summary**
- 我們的貢獻怎麼填補上述 gaps

**輸出**：`paper_research/related_work_draft.md`

---

## 任務 4：Reviewer 可能問的 30 個問題

**背景**：我們投稿目標 AIME 2027 或 ISBI 2027（也許 MICCAI）。
小資料集（test n=49, AD=11）、多站點、有 site confounding 風險、AUC 在中下段（0.65-0.71）。

**請預估 reviewer 可能問的 30 個犀利問題**，分四類：

1. **Methodology 質疑**（10 題）：例如 "small sample, why not external validation?"
2. **Statistical rigor**（5 題）：例如 "wide CIs, are results significant?"
3. **可解釋性 / 系統整合**（10 題）：例如 "GradCAM 對 BrainIAC ViT 是否合適？"
4. **臨床效益**（5 題）：例如 "0.68 AUC 真的對臨床有幫助嗎？"

每個問題給：
- 問題本身（一句話）
- 嚴重度（low / medium / high）
- **我們已經有的應對材料**（指 revise.md 內哪一段，或 "需要補做"）
- **建議回應方向**（2-3 句）

**輸出**：`paper_research/reviewer_questions.md`

---

## 任務 5：找 5 個跟我們最相似的 paper 做為主要競爭對手

**背景**：我們需要在 paper 裡跟 5-8 個 baseline 比較。已經比了 SVM + GNN-only + concat fusion 等內部 ablation，
但缺乏 **外部** baseline（已 published 的方法）。

**請推薦 5 個最適合做外部比較的 paper**：
- 條件：multi-modal (fMRI + sMRI) 或 graph-based, 對 AD / MCI 分類, 有 open-source code
- 排除：純醫學影像（不 ML）、純單模態 paper
- 每篇給：title, paper link, GitHub link, 該方法在 ADNI 的 reported AUC, **預估在我們 setup 上要花多久 reproduce**

**輸出**：`paper_research/external_baselines.md`

---

## 給 Gemini 的執行指令

```bash
# 一次跑全部（建議）
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
mkdir -p paper_research
gemini-cli read gemini_tasks.md --output paper_research/

# 或單獨跑某個任務
gemini-cli "請執行 gemini_tasks.md 中的任務 3"
```

**回來後請告訴我哪些 task 完成、結果在哪。**

---

## 我這邊同步在做的

- 等 LLM-judge smoke test 跑完
- 寫 Data Augmentation 程式碼（SMOTE + Mixup on FC matrix）
- 設計 ensemble training script

兩邊匯合後，會把 Gemini 找到的 augmentation 推薦方法跟我寫的程式碼對照，挑最好的下去訓練。
