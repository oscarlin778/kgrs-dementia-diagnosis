---
name: project-alzheimers-benchmarks
description: "Literature benchmarks for AD/MCI/NC classification on ADNI — AUC values for all three binary tasks, for comparison to PCAG-ComBat results"
metadata: 
  node_type: memory
  type: project
  originSessionId: 54ff566b-08aa-4ddd-bb3a-2c288145cf55
---

## Our system (PCAG-ComBat, test set, ensemble)
- NC_vs_AD: 0.791
- NC_vs_MCI: 0.686
- MCI_vs_AD: 0.672
- Dataset: ADNI n=204 (155 train / 49 test)

## Key comparable papers (searched 2026-05-28)

**ADMGCN** (Bioinformatics 2025, fMRI GCN + meta-learning, ADNI)
- NC_vs_AD: 0.759 → we win +0.032
- NC_vs_MCI: 0.665 → we win +0.021
- MCI_vs_AD: 0.758 → we lose -0.086
- Most directly comparable: same ADNI, fMRI, GCN, all three tasks

**Spatio-Temporal dFC GCN** (Frontiers Neuroscience 2025, fMRI, ADNI n=85 small)
- NC_vs_AD: 0.831, NC_vs_MCI: 0.792, MCI_vs_AD: 0.769 — all higher, but tiny n

**Multimodal fMRI+sMRI+DTI** (Frontiers Aging Neuroscience 2026)
- NC_vs_AD: 0.901, NC_vs_MCI: 0.839, MCI_vs_AD: 0.809 — uses DTI (we don't have)

**ADMV-Net sMRI+PET** (2025)
- NC_vs_AD: 0.960, NC_vs_MCI: 0.768, MCI_vs_AD: 0.889 — PET-based, different modality

## Key context for paper writing
- NC_vs_MCI is universally hardest; 0.686 is within fMRI GNN range
- MCI_vs_AD 0.672 is our weakest; ADMGCN achieves 0.758 (honest limitation to acknowledge)
- High AUC papers (0.95+) use PET or 2000+ patient multicenter datasets — not fair comparison
- ADMGCN is the fairest comparison for all three tasks
