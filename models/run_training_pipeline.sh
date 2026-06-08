#!/bin/bash
# Runner script for Alzheimer classification training pipeline

LOG_FILE="/home/wei-chi/Alzheimers_Project/external_data/scripts/models/pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"

{
    echo "=== Starting Pipeline at $(date) ==="
    cd /home/wei-chi/Alzheimers_Project/external_data/scripts/models || exit 1

    echo "[Task 1] fMRI GNN Pretraining..."
    conda run -n AD python3 train_hierarchical_gnn_e13_gsl.py \
        --train-csv /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits/adni_train.csv \
        --output-dir /home/wei-chi/Alzheimers_Project/external_data/scripts/checkpoints/adni_only_gnn/
    
    RET1=$?
    if [ $RET1 -ne 0 ]; then
        echo "Error: Task 1 failed with exit code $RET1"
        exit $RET1
    fi

    echo "[Task 2] sMRI ResNet Pretraining..."
    conda run -n AD python3 train_3d_resnet_teacher.py \
        --adni-only \
        --train-subjects /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/splits/adni_train.csv \
        --output-dir /home/wei-chi/Alzheimers_Project/external_data/scripts/checkpoints/adni_only_smri/
    
    RET2=$?
    if [ $RET2 -ne 0 ]; then
        echo "Error: Task 2 failed with exit code $RET2"
        exit $RET2
    fi

    echo "=== Pipeline Finished Successfully at $(date) ==="
} 2>&1 | tee -a "$LOG_FILE"
