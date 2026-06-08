# Paper Context — IEEE BHI 2026

**Paper title (working):** An End-to-End Multimodal AI Pipeline for Alzheimer's Disease Staging: Harmonized GNN Classification and Knowledge-Grounded Report Generation

**Target venue:** IEEE BHI 2026 (4–7 pages incl. refs, double-blind)

**Working directory:** `/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/`

---

## File Index

| File | What it contains |
|------|-----------------|
| `01_revise_experiment_log.md` | Full experiment log (691 lines). Contains all decisions, bug fixes, ablation results, ComBat analysis, LLM judge raw numbers. **Read this for the complete story.** |
| `02_execution_plan.md` | Execution plan for the revise phase (interpretability, ComBat fix, ensemble). |
| `03_gemini_tasks.md` | Task notes from experiment iteration. |
| `04_paper_status_numbers.md` | **All final numbers for the paper**: AUCs with 95% CI, ablation table, LLM judge results, architecture summary, key narrative decisions. **Start here for numbers.** |
| `05_benchmark_comparison.md` | Literature benchmark table. Explains which comparisons are fair and which are not. |
| `06_loocv_decision.md` | Why LOOCV is NOT reported in the paper. |

---

## Key Numbers (quick reference)

### Classification AUC (ensemble, test set n=49)
| Task | AUC | 95% CI |
|------|-----|--------|
| NC vs AD | **0.791** | [0.609, 0.927] |
| NC vs MCI | **0.686** | [0.509, 0.855] |
| MCI vs AD | **0.672** | [0.468, 0.879] |

### LLM Judge — PRIMARY: v5 Qwen3 (n=45, completed 2026-05-29)
| Dimension | Δ (RAG - no RAG) | p |
|-----------|-----------------|---|
| **clinical_relevance** | **+0.333** | **0.007\*\*** |
| completeness (Gemma3) | +0.178 | 0.069† |
| coherence | ns (both judges) | — |

v5 uses full ROI-informed reports; ROI mention rate 91.8%. Validity: correct vs misclassified patients p=0.013–0.014.
v5 Gemma3 clinical_relevance saturated (all 2.0, uninformative). v4 all ns, not used.

---

## Main Contributions (for Introduction)

1. **No-label ComBat** — solves site harmonization under site-class imbalance (label-aware ComBat fails)
2. **PCAG cross-attention fusion** — fMRI (GAT + virtual node + 9-network pooling) × sMRI (BrainIAC ViT)
3. **KG + RAG report generation** — Neo4j knowledge graph → MMR retrieval → Gemma3-12B clinical report
4. **Multi-layer evaluation framework** — LLM judge + validity check + objective metrics

## Key Narrative Decisions

- Main claim is **end-to-end pipeline**, not just classification AUC
- Concat > PCAG on NC_vs_AD: expected (large inter-group gap); PCAG wins on subtle tasks (NC_vs_MCI, MCI_vs_AD)
- MCI_vs_AD 0.672 < ADMGCN 0.758: acknowledge honestly; likely cause = small test set (n=18 AD)
- ComBat coherence tradeoff (−0.163, p=0.074): known RAG information-density vs fluency tradeoff

---

## Dataset

- ADNI + TPMIC, n=204 total (155 train / 49 test)
- NC=87 / MCI=72 / AD=45
- Two sites: site-class imbalance (ADNI ≈ NC-heavy; TPMIC ≈ disease-heavy)

## Architecture Summary

- **fMRI encoder:** GAT (3 layers, 4 heads, hidden=128), 116 AAL ROIs, K_RATIO=0.20 → virtual node + 9-network pooling → 1280-dim
- **sMRI encoder:** BrainIAC ViT (pre-trained, frozen) → 768-dim
- **Fusion:** PCAG cross-attention (Q=fMRI, K/V=sMRI, fusion_dim=20)
- **Training:** 5-fold CV, 200 epochs, AdamW lr=3e-4, 5-seed OOF ensemble
- **KG:** Neo4j, SIMILAR_TO edges by GNN embedding cosine similarity
- **RAG:** Neo4jVector (nomic-embed-text), MMR k=5 from 20 candidates
- **Report:** Gemma3-12B via Ollama, Traditional Chinese
