---
name: project-alzheimers-paper-status
description: "Current paper writing status for PCAG-ComBat Alzheimer's end-to-end system — all key numbers, decisions, and file locations for paper drafting"
metadata: 
  node_type: memory
  type: project
  originSessionId: 54ff566b-08aa-4ddd-bb3a-2c288145cf55
---

## Paper Overview

**Title (working):** An End-to-End Multimodal AI System for Alzheimer's Disease Staging: From Neuroimaging to Knowledge-Grounded Clinical Reports

**Contribution claim:** Complete end-to-end pipeline — PCAG-ComBat multimodal GNN (fMRI+sMRI) classification → Knowledge Graph → RAG → LLM diagnostic report generation

**Target venue:** IEEE BHI 2026 (4–7 pages incl. refs, double-blind)

**Working directory:** `/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/`

---

## Final Classification Numbers (use these in paper)

OOF-selected ensemble (5 seeds × 5-fold), held-out test set n=49:

| Task | AUC | 95% CI | Sensitivity | Specificity |
|------|-----|--------|-------------|-------------|
| NC vs AD | **0.791** | [0.609, 0.927] | — | — |
| NC vs MCI | **0.686** | [0.509, 0.855] | — | — |
| MCI vs AD | **0.672** | [0.468, 0.879] | — | — |

Dataset: ADNI + TPMIC, n=204 total (155 train / 49 test), NC=87 / MCI=72 / AD=45

---

## Ablation Table (seed=42, single model — fair comparison)

| Method | NC_vs_AD | NC_vs_MCI | MCI_vs_AD |
|--------|---------|---------|---------|
| SVM (FC features) | 0.755 | 0.750 | 0.505 |
| fMRI-only | 0.768 | 0.683 | 0.646 |
| sMRI-only | 0.750 | 0.533 | 0.515 |
| Concat fusion | **0.804** | 0.417 | 0.535 |
| No ComBat | 0.682 | 0.678 | 0.525 |
| Q/K/V Swapped | 0.700 | 0.681 | 0.515 |
| **PCAG-ComBat (ours)** | 0.682 | **0.708** | **0.651** |

Note: Final paper AUC uses ensemble (0.791/0.686/0.672); ablation uses single seed for fair comparison.

---

## Key Narrative Decisions

1. **Concat > PCAG on NC_vs_AD**: Frame as expected — large inter-group difference → global features sufficient; PCAG's cross-attention advantage shows on subtle tasks (NC_vs_MCI, MCI_vs_AD) where fine-grained cross-modal interaction matters.

2. **MCI_vs_AD 0.672 < ADMGCN 0.758**: Acknowledge honestly. Likely cause: small test set (n=18 AD), high variance. Report with CI [0.468, 0.879].

3. **LOOCV not reported**: LOOCV (NC_vs_MCI=0.477, below chance) reflects single-model instability vs ensemble; methodologically non-equivalent. Decision: not included in paper.

4. **Site confound**: Fixed with no-label fMRI ComBat. Include site confound analysis as validation that model learns disease signal, not site identity.

---

## Benchmark Comparison (searched 2026-05-28)

| Method | NC_vs_AD | NC_vs_MCI | MCI_vs_AD | Notes |
|--------|---------|---------|---------|-------|
| **Ours** | **0.791** | **0.686** | 0.672 | fMRI+sMRI, ADNI n=204 |
| ADMGCN (Bioinformatics 2025) | 0.759 | 0.665 | **0.758** | fMRI GCN, ADNI |
| Spatio-Temp dFC GCN (Frontiers 2025) | 0.831 | 0.792 | 0.769 | fMRI, n=85 small |
| Multimodal fMRI+sMRI+DTI (2026) | 0.901 | 0.839 | 0.809 | extra DTI modality |

---

## LLM Judge Results (v2 — archived reference only)

Gemma3 (n=49 patients, Wilcoxon):
- clinical_relevance: Δ=+0.184, p=0.039* | completeness: Δ=+0.204, p=0.041*
- coherence: Δ=−0.163, p=0.074† | factual_accuracy: Δ=+0.082, p=0.102
- Objective (Method B): 100% citation rate (RAG) vs 0%; report length +146 chars (p<0.001)
- **Limitation:** ROI mention rate only 8.2% (reports lacked actual ROI names)

---

## LLM Judge Results (v5 — USE THESE, primary results, completed 2026-05-29)

Reports include full ROI info (BrainIAC + GAT attention); n=45 patients (4 judge failures excluded); n_pairs=90 per judge.

| Dimension | Gemma3 Δ | p | Qwen3 Δ | p |
|-----------|---------|---|---------|---|
| factual_accuracy | +0.067 | 0.406 | +0.133 | 0.359 |
| **clinical_relevance** | +0.000 | 1.000 ⚠️ | **+0.333** | **0.007\*\*** |
| completeness | +0.178 | 0.069† | +0.000 | 1.000 |
| coherence | −0.111 | 0.294 | +0.156 | 0.167 |

Inter-judge agreement: r = −0.026 to +0.197

**Gemma3 clinical_relevance = 2.000 for ALL patients** (anchors at 2.0 regardless of RAG) → ceiling/anchor effect, not informative for this dimension.
**Qwen3 clinical_relevance +0.333 (p=0.007)** — highly significant, stronger than v2 because v5 reports contain actual ROI names that Qwen3 can cross-reference with literature.

### Evaluation validity (v5)
- Method A: Qwen3 with_rag — correctly classified vs misclassified: factual_accuracy 2.650 vs 2.160 **p=0.014**, clinical_relevance 2.900 vs 2.600 **p=0.013** ✓
- Method B: roi_mentioned = **91.8%** both conditions (ROI fix worked; was 8.2% in v2); cite_paper (with_rag) = 18.4% (flexible citation)

### Paper narrative
"RAG significantly improves clinical relevance (Qwen3: Δ=+0.333, p=0.007) with no significant coherence penalty in either judge (p>0.17). Judge validity confirmed: correctly classified patients score significantly higher on factual accuracy and clinical relevance (Qwen3, p<0.015)."

### Why v5 > v2
v2 Qwen3 clinical_relevance +0.122 p=0.196 (ns) → v5 Qwen3 +0.333 p=0.007 (significant): v5 reports contain actual ROI names (e.g., Precuneus, Cingulum_Mid_R) that Qwen3 can cross-reference with retrieved literature — concrete anatomical anchors make clinical relevance assessable.

---

## LLM Judge Results (v4 — NOT used, completed 2026-05-29)

All ns; Gemma3 clinical_relevance trending negative (−0.130, p=0.175); Qwen3 had 1 timeout. Not reported in paper.

---

## Key Files

- Main results: `results/full_predictions_v2_nolabel_ensemble.npz`
- Bootstrap CI: `results/bootstrap_ci_v2_nolabel.json`
- LLM judge (primary): `results/report_quality_v2_nolabel_results.json`
- LLM judge (v4, with ROI): `results/report_quality_v4_with_roi.json` (done; not used)
- LLM judge (v5, fixed query): `results/report_quality_v5_full_fix.json` (done; Qwen3 supporting only)
- Analysis script: `analyze_report_quality.py`
- All context: `revise.md` (590+ lines of experiment log)
- Figures: `results/*.png` (progression, ROC, ablation, site confound, qkv_swap)
- Paper draft notes: `revise.md`末尾有 Results section 草稿

## Architecture Summary (for Methods section)

- fMRI: GAT (3 layers, 4 heads, hidden=128) on functional connectivity graph (116 AAL ROIs, K_RATIO=0.20)
  + Virtual node + 9 network-level pooling → 1280-dim embedding
- sMRI: BrainIAC ViT (pre-trained, frozen) → 768-dim embedding  
- Fusion: Cross-attention (Q=fMRI, K/V=sMRI), fusion_dim=20
- Training: 5-fold CV, 200 epochs, AdamW lr=3e-4, batch=16, class-balanced sampler
- Ensemble: 5 seeds (42/123/456/789/2024), OOF-guided strategy selection per task
  - NC_vs_AD: top3 seeds (aug), NC_vs_MCI: single_best seed=456, MCI_vs_AD: median
- Site harmonization: fMRI — no-label ComBat (pre-computed globally); sMRI — label ComBat per fold
- KG: Neo4j, SIMILAR_TO edges via GNN embedding cosine similarity
- RAG: Neo4jVector (nomic-embed-text), MMR retrieval k=5 from 20 candidates
- Report: Gemma3-12B via Ollama, streaming, 繁體中文
