#!/bin/bash
set -e
BASE=/home/wei-chi/Alzheimers_Project/external_data/scripts
SPLITS=$BASE/downstream/splits
CKPTS=$BASE/checkpoints

cd $BASE/models

for FRAC in 0.2 0.4 0.6 0.8; do
    FRAC_STR=$(echo $FRAC | tr -d '.')  # "02", "04", "06", "08"
    OUTDIR=$CKPTS/finetune_tpmic_frac${FRAC_STR}
    echo "=== Training fraction $FRAC → $OUTDIR ==="

    conda run -n AD python3 train_finetune_gnn_tpmic.py \
        --pretrain-ckpt-dir $CKPTS/combined_gnn/gnn_checkpoints \
        --tpmic-train-csv $SPLITS/tpmic_train.csv \
        --tpmic-fraction $FRAC \
        --output-ckpt-dir $OUTDIR

    # Create alias copies expected by inference_pipeline.py
    for TASK in NC_vs_AD NC_vs_MCI MCI_vs_AD; do
        SRC="$OUTDIR/${TASK}_finetune_frac${FRAC}.pt"
        DST="$OUTDIR/${TASK}.pt"
        if [ -f "$SRC" ] && [ ! -f "$DST" ]; then
            cp "$SRC" "$DST"
            echo "  Copied $TASK alias"
        fi
    done

    echo "=== Done fraction $FRAC ==="
done
echo "All fractions complete."
