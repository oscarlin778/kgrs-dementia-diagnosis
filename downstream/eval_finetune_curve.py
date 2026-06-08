#!/usr/bin/env python3
import os
import re
import json
import time
import subprocess
import pandas as pd

FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
CKPT_BASE = "/home/wei-chi/Alzheimers_Project/external_data/scripts/checkpoints"
PIPELINE_FILE = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/inference_pipeline.py"

CKPT_MAP = {
    0.2: f"{CKPT_BASE}/finetune_tpmic_frac02",
    0.4: f"{CKPT_BASE}/finetune_tpmic_frac04",
    0.6: f"{CKPT_BASE}/finetune_tpmic_frac06",
    0.8: f"{CKPT_BASE}/finetune_tpmic_frac08",
    1.0: f"{CKPT_BASE}/finetune_tpmic_full",
}

BASELINE_CKPT = f"{CKPT_BASE}/combined_gnn/gnn_checkpoints"

def update_ckpt_dir(path):
    with open(PIPELINE_FILE, "r") as f:
        content = f.read()
    
    # Pattern to match the CKPT_DIR definition from Task B.1
    pattern = r'(CKPT_DIR = os\.getenv\(\s*"KGRS_GNN_CKPT_DIR",\s*os\.path\.join\([\s\S]*?\)\s*\))'
    new_line = f'CKPT_DIR = "{path}"'
    
    new_content = re.sub(pattern, new_line, content)
    with open(PIPELINE_FILE, "w") as f:
        f.write(new_content)
    print(f"Updated CKPT_DIR to: {path}")

def restore_ckpt_dir():
    original = (
        'CKPT_DIR = os.getenv(\n'
        '    "KGRS_GNN_CKPT_DIR",\n'
        '    os.path.join(\n'
        '        os.getenv("KGRS_MODEL_ROOT", "/home/wei-chi/Alzheimers_Project/external_data/scripts/checkpoints"),\n'
        '        "combined_gnn", "gnn_checkpoints"\n'
        '    )\n'
        ')'
    )
    with open(PIPELINE_FILE, "r") as f:
        content = f.read()
    
    # This might be tricky if the replace pattern changed. 
    # Let's use a simpler marker if possible, but for now we'll try to find the direct string we wrote.
    # Actually, the most robust way is to just write the whole thing back.
    
    # Search for the line we injected: CKPT_DIR = "/path/to/..."
    new_content = re.sub(r'CKPT_DIR = "/home/wei-chi/Alzheimers_Project/external_data/scripts/checkpoints/.*"', original, content)
    with open(PIPELINE_FILE, "w") as f:
        f.write(new_content)
    print("Restored original CKPT_DIR definition.")

def run_eval(output_file):
    cmd = [
        "conda", "run", "-n", "AD", "python3", "evaluate_system.py",
        "--test-csv", "splits/tpmic_test.csv",
        "--output", output_file
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    results_summary = []

    # 0.0 Baseline
    baseline_json = "results/eval_combined_tpmic_test.json"
    if os.path.exists(baseline_json):
        with open(baseline_json, "r") as f:
            data = json.load(f)
        results_summary.append({
            "Fraction": "0.0 (Baseline)",
            "NC vs AD AUC": data["task_metrics"]["NC vs AD"].get("auc", 0),
            "NC vs MCI AUC": data["task_metrics"]["NC vs MCI"].get("auc", 0),
            "MCI vs AD AUC": data["task_metrics"]["MCI vs AD"].get("auc", 0),
            "OVO ACC": data["ovo_analysis"]["ovo_accuracy"],
            "Ties": data["ovo_analysis"]["n_111_tie"]
        })

    try:
        for frac in FRACTIONS:
            frac_str = str(frac).replace(".", "")
            if frac == 1.0: frac_str = "full"
            else: frac_str = f"frac{frac_str}"
            
            ckpt_path = CKPT_MAP[frac]
            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint path {ckpt_path} not found. Skipping fraction {frac}.")
                continue
                
            update_ckpt_dir(ckpt_path)
            print("Waiting 6 seconds for API server reload...")
            time.sleep(6)
            
            out_json = f"results/eval_finetune_{frac_str}_tpmic_test.json"
            run_eval(out_json)
            
            with open(out_json, "r") as f:
                data = json.load(f)
            
            results_summary.append({
                "Fraction": frac,
                "NC vs AD AUC": data["task_metrics"]["NC vs AD"].get("auc", 0),
                "NC vs MCI AUC": data["task_metrics"]["NC vs MCI"].get("auc", 0),
                "MCI vs AD AUC": data["task_metrics"]["MCI vs AD"].get("auc", 0),
                "OVO ACC": data["ovo_analysis"]["ovo_accuracy"],
                "Ties": data["ovo_analysis"]["n_111_tie"]
            })
    finally:
        restore_ckpt_dir()

    # Print Summary Table
    df = pd.DataFrame(results_summary)
    print("\n" + "="*80)
    print("  DATA EFFICIENCY CURVE SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
