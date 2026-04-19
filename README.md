# KGRS: Dual-Modal Neuroimaging Dementia Diagnosis Platform
# KGRS：雙模態神經影像失智症輔助診斷平台

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.0+-61dafb.svg)](https://reactjs.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-008cc1.svg)](https://neo4j.com/)


---

### 📝 Overview
The **Knowledge Graph-based Reporting System (KGRS)** is a comprehensive, dual-modal AI platform designed to assist in the clinical diagnosis of Dementia and Alzheimer's Disease (AD). By fusing functional Magnetic Resonance Imaging (fMRI) with structural MRI (sMRI), the system provides highly robust diagnostic probabilities, mitigates the effect of functional compensation during early disease stages, and generates explainable, LLM-powered clinical reports.

### ✨ Key Features
* **🧠 Dual-Modal AI Fusion:** Integrates a Graph Neural Network (FNP-GNNv8) for fMRI functional connectivity analysis and a 3D ResNet for sMRI structural atrophy detection.
* **🔍 Explainable AI (XAI):** Utilizes Gradient-based Saliency Attribution to pinpoint the most critical brain regions (AAL116) contributing to the diagnosis, moving beyond generic attention mechanisms.
* **🔗 Knowledge Graph (RAG) Integration:** Queries a Neo4j property graph containing neurological knowledge (brain networks, disease stages, ROI functions) to provide medical context.
* **🤖 LLM Clinical Reporting:** Leverages local LLMs (Ollama / Gemma 4) to stream concise or detailed clinical reports based on multi-modal findings.
* **🌐 Interactive 3D Dashboard:** A modern React frontend featuring real-time adjustable modality weights, dynamic radar charts for network saliency, and **NiiVue.js** for 3D NIfTI brain rendering.
* **🛡️ Fallback Mechanism:** Automatically detects missing structural data and degrades gracefully to an "fMRI-only" mode without system failure.

### 🛠️ Technology Stack
* **Deep Learning:** PyTorch, MONAI, PyTorch Geometric (PyG)
* **Backend:** FastAPI, Uvicorn, Python
* **Frontend:** React, Vite, Tailwind CSS, Recharts, NiiVue.js
* **Database & AI:** Neo4j (Knowledge Graph), Ollama (Local LLM Inference)

