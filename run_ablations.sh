#!/bin/bash
cd /home/wei-chi/Data/script
mkdir -p logs

# 確保輸出不被緩衝
export PYTHONUNBUFFERED=1

echo "Starting Ablation 1: High Align (Lambda=0.5)..."
python3 models/train_hierarchical_gnn_e2_ablation.py --lambda_align 0.5 --lambda_contra 0.1 > logs/abl_high_align.log 2>&1
sleep 5

echo "Starting Ablation 2: No Contra (Lambda_c=0.0)..."
python3 models/train_hierarchical_gnn_e2_ablation.py --lambda_align 0.1 --lambda_contra 0.0 > logs/abl_no_contra.log 2>&1
sleep 5

echo "Starting Ablation 3: Dynamic MCI Weight..."
python3 models/train_hierarchical_gnn_e2_ablation.py --lambda_align 0.1 --lambda_contra 0.1 --dynamic_mci > logs/abl_dynamic_mci.log 2>&1

echo "All ablation studies finished."
