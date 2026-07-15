# KGRS: Multimodal Neuroimaging Dementia Diagnosis System

# KGRS：多模態神經影像失智症輔助診斷系統

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0+-61dafb.svg)](https://reactjs.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-008cc1.svg)](https://neo4j.com/)

---

## Overview

**KGRS** is a multimodal AI system for Alzheimer's Disease (AD) staging that integrates resting-state fMRI and structural MRI (sMRI) with a Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) pipeline. The system supports three-class classification (NC / MCI / AD) via a One-vs-One (OVO) ensemble and generates fact-grounded clinical reports using a local LLM.

---

## Architecture

### 1. Multimodal Encoders

| Modality                  | Encoder                                                         | Output             |
| ------------------------- | --------------------------------------------------------------- | ------------------ |
| fMRI (116×116 FC matrix) | 3-layer, 4-head Graph Attention Network (GAT) + network pooling | 1280-dim embedding |
| sMRI (3D MRI scan)        | Frozen BrainIAC ViT (pretrained on 32,015 scans)                | 768-dim embedding  |

### 2. PCAG-ComBat Fusion

- **Label-Free ComBat** harmonization on fMRI functional connectivity to remove multi-site batch effects without conditioning on diagnosis labels
- **Patient-specific Cross-Attention Generation (PCAG):** fMRI embedding acts as Query; sMRI embedding as Key/Value — generates a patient-specific gating vector for modality weighting via sigmoid attention

### 3. OVO Ensemble Classifier

Three binary classifiers (NC vs. AD, NC vs. MCI, MCI vs. AD) trained independently with 5-seed cross-validation; soft-voting aggregates predictions into a three-class OVO decision.

### 4. KG-RAG Report Generation

- **Neo4j Knowledge Graph** stores literature chunks (2,432 chunks indexed with BGE-M3 embeddings)
- **Maximal Marginal Relevance (MMR)** retrieval selects diverse, relevant literature (k=5)
- **Gemma3-12B** (via Ollama) generates structured clinical reports grounded in GNN evidence and retrieved literature

---

## Results

Test set AUC (5-seed ensemble, held-out n=62; ADNI + private TW cohort):

| Task       | PCAG-ComBat (Ours) |
| ---------- | ------------------ |
| NC vs. AD  | 0.865              |
| NC vs. MCI | 0.719              |
| MCI vs. AD | 0.806              |

Contextual reference against published methods (different datasets / modalities — not direct comparisons):

| Method                                                                     | NC/AD           | NC/MCI          | MCI/AD          | Notes                      |
| -------------------------------------------------------------------------- | --------------- | --------------- | --------------- | -------------------------- |
| PCAG-ComBat (Ours)                                                         | **0.865** | **0.719** | 0.806           | fMRI+sMRI, multi-site      |
| [ADMGCN (Sun et al., 2025)](https://doi.org/10.1093/bioinformatics/btaf580) | 0.761           | 0.665           | 0.758           | fMRI only, most comparable |
| [Yuan et al. (2026)](https://doi.org/10.3389/fnagi.2026.1794982)            | 0.902           | 0.673           | **0.811** | fMRI+sMRI+DTI              |
| [Wen et al. (2025)](https://doi.org/10.3389/fnins.2025.1597777)             | 0.871           | 0.653           | 0.823           | fMRI+sMRI, single-site     |

---

## Key Features

- Multi-site harmonization designed for severe site-class imbalance scenarios
- Asymmetric cross-modal fusion guided by functional connectivity state
- One-vs-One ensemble with out-of-fold (OOF) strategy selection
- Streaming clinical report generation with citation traceability
- FastAPI backend + React frontend with real-time inference

---

## Technology Stack

| Category        | Tools                               |
| --------------- | ----------------------------------- |
| Deep Learning   | PyTorch, PyTorch Geometric (PyG)    |
| GNN             | Graph Attention Network (GAT)       |
| sMRI Encoder    | BrainIAC ViT (frozen)               |
| Harmonization   | neuroCombat (label-free)            |
| Backend         | FastAPI, Uvicorn                    |
| Frontend        | React, Vite, Tailwind CSS, Recharts |
| Knowledge Graph | Neo4j, BGE-M3 embeddings            |
| LLM             | Gemma3-12B via Ollama               |
| Evaluation      | RAGAS, custom LLM-judge pipeline    |

---

## Directory Structure

```
scripts/
├── downstream/
│   ├── api_server.py               # FastAPI inference & report generation endpoint
│   ├── inference_pipeline_v2.py    # End-to-end inference pipeline
│   ├── graph_rag_retriever.py      # Neo4j KG retrieval (MMR)
│   ├── data_prep/                  # ComBat harmonization, ensemble OOF scripts
│   ├── evaluation/                 # Evaluation scripts (citation, RAGAS, ablation)
│   ├── features/                   # Feature extraction utilities
│   ├── knowledge_graph/            # KG indexing and embedding scripts
│   ├── kgrs-frontend/              # React frontend
│   └── training/                   # Model training scripts (PCAG, concat, swapped)
└── preprocessing/                  # fMRI matrix extraction, subject matching
```

---

## Data

This repository contains **code only**. Patient data (ADNI and private clinical cohort), model checkpoints, and preprocessing outputs are **not included** due to data use agreements and patient privacy requirements.

- ADNI data: available at [adni.loni.usc.edu](https://adni.loni.usc.edu/) (requires DUA)
- Model weights: available upon reasonable request

---

## Setup

```bash
# Backend
conda create -n AD python=3.10
conda activate AD
pip install -r requirements.txt

# Start Neo4j (Docker)
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<your_password> \
  neo4j:5

# Start Ollama + pull model
ollama pull gemma3:12b

# Start API server
uvicorn api_server:app --host 0.0.0.0 --port 8080

# Frontend
cd kgrs-frontend && npm install && npm run dev
```

Environment variables (`.env`):

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your_password>
```

---

## Author

**Wei-Chi Lin**
National Cheng Kung University
