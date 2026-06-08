import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

HYBRID_COMBAT_JSON  = RESULTS_DIR / "hybrid_system_combat_metrics.json"
GLOBAL_OOF_JSON     = RESULTS_DIR / "global_oof_hybrid_metrics.json"
CALIBRATED_JSON     = RESULTS_DIR / "calibrated_hybrid_metrics.json"

TASKS = ['NC_vs_AD', 'NC_vs_MCI', 'MCI_vs_AD']

def load_json(path):
    if not path.exists():
        print(f"Warning: {path} not found.")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def main():
    hybrid_data = load_json(HYBRID_COMBAT_JSON)
    global_data = load_json(GLOBAL_OOF_JSON)
    calib_data = load_json(CALIBRATED_JSON)

    # Data preparation
    methods = [
        "PCAG ComBat",
        "KD Student",
        "Global Hybrid (Raw)",
        "Global Hybrid (Calib)"
    ]
    
    # We want 3 tasks x 4 methods
    data = []
    for task in TASKS:
        # 1. PCAG ComBat
        pcag_auc = hybrid_data.get(task, {}).get("pcag_combat", {}).get("auc", 0.0)
        n_pcag = hybrid_data.get(task, {}).get("pcag_combat", {}).get("n", 0)
        if n_pcag == 0: pcag_auc = 0.0
        
        # 2. KD Student
        kd_auc = hybrid_data.get(task, {}).get("kd_only", {}).get("auc", 0.0)
        
        # 3. Global Hybrid Raw
        raw_auc = global_data.get(task, {}).get("auc", 0.0)
        
        # 4. Global Hybrid Calibrated
        calib_auc = calib_data.get(task, {}).get("auc", 0.0)
        
        data.append([pcag_auc, kd_auc, raw_auc, calib_auc])

    data = np.array(data) # Shape (3, 4)

    # Plotting
    plt.figure(figsize=(12, 6))
    x = np.arange(len(TASKS))
    width = 0.2
    
    colors = ['#90EE90', '#4682B4', '#FA8072', '#006400'] # lightgreen, steelblue, salmon, darkgreen
    
    for i in range(4):
        bars = plt.bar(x + (i - 1.5) * width, data[:, i], width, label=methods[i], color=colors[i], edgecolor='black', linewidth=0.5)
        
        # Add values on top
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xlabel('Diagnostic Tasks', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('Hybrid System AUC Comparison: PCAG / KD / Global OOF Hybrid', fontsize=14, pad=20)
    plt.xticks(x, TASKS, fontsize=11)
    plt.ylim(0.5, 1.0)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = FIGURES_DIR / "Final_System_Comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n[DONE] Comparison figure saved to {save_path}")
    
    print("\nNote: ROC Curves were not generated as the raw probabilities are not stored in the summary metrics JSONs.")
    print("To generate ROC curves, ensure that Task 1 saves 'all_probs' and 'all_labels' to an NPZ file, then update the plotting script.")

if __name__ == "__main__":
    main()
