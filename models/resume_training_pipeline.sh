#!/bin/bash
# Resume runner script for Alzheimer classification training pipeline (Skip Task 1)

LOG_FILE="/home/wei-chi/Alzheimers_Project/external_data/scripts/models/pipeline_resume_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"

{
    echo "=== Resuming Pipeline at $(date) ==="
    cd /home/wei-chi/Alzheimers_Project/external_data/scripts/models || exit 1

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
