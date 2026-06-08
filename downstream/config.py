import os

DATA_BASE  = os.getenv("KGRS_DATA_ROOT",  "/home/wei-chi/Alzheimers_Project/external_data")
MODEL_BASE = os.getenv("KGRS_MODEL_ROOT", "/home/wei-chi/Alzheimers_Project/external_models")

SMRI_ROOT       = os.path.join(DATA_BASE,  "datasets/ADNI_sMRI_Aligned_MNI")
TPMIC_SMRI_ROOT = os.path.join(MODEL_BASE, "sMRI_data_MultiModal_Aligned_MNI")
MATCHED_ROOT    = os.path.join(DATA_BASE,  "Matched_Dual_Modal_Dataset")
MATRIX_DIR      = os.path.join(MODEL_BASE, "processed_116_matrices")
ADNI_MATRIX_DIR = os.path.join(DATA_BASE,  "features/ADNI_processed_116_matrices")
SALIENCY_DIR    = os.path.join(DATA_BASE,  "scripts/results/saliency")
DATA_ROOT       = DATA_BASE
MODEL_ROOT      = MODEL_BASE
