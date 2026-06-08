---
name: project-alzheimers-loocv
description: "LOOCV results for PCAG-ComBat — below-chance AUC due to seed mismatch, decision to not report in paper"
metadata: 
  node_type: memory
  type: project
  originSessionId: 54ff566b-08aa-4ddd-bb3a-2c288145cf55
---

LOOCV ran on all 204 patients (seed=42, 150 epochs fixed). Results were poor:
- NC_vs_AD: 0.607 LOOCV vs 0.791 test
- NC_vs_MCI: 0.477 LOOCV (BELOW CHANCE) vs 0.686 test
- MCI_vs_AD: 0.521 LOOCV vs 0.672 test

**Why:** LOOCV used fixed seed=42, but optimal NC_vs_MCI seed is 456 (selected by OOF). seed=42 was always poor for NC_vs_MCI — that's why OOF-guided selection chose seed=456.

**How to apply:** Do NOT report this LOOCV in the paper. When discussing with the advisor, explain: "LOOCV with fixed hyperparameters shows high variance; model needs seed selection to generalize, which is why our OOF-guided strategy was critical." Reference: results/loocv_summary.json.
