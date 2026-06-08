# PCAG-ComBat Paper: Remaining Execution Tasks

## Context

All model training is DONE. The code changes from this session:
- `aggregate_ensemble_oof_selected.py`: added `single_best` as 4th strategy, forced it for NC_vs_MCI
- `extract_full_predictions_ensemble.py`: added `single_best` branch in aggregation logic
- `results/make_progression_figure.py`: updated NC_vs_MCI final to 0.686

**Final AUC numbers (confirmed, OOF-guided selection):**
- NC_vs_AD: **0.791** (top3 ensemble, seeds 42/123/2024, augmentation)
- NC_vs_MCI: **0.686** (single_best OOF, seed=456)
- MCI_vs_AD: **0.672** (median ensemble)

Working directory for ALL commands: `/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream`
Conda env: `AD`

---

## Step 1 — Full Predictions Extraction (~10-15 min, GPU)

Generates `results/full_predictions_v2_nolabel_ensemble.npz` (all 49 patients × 3 tasks).
Required by LLM judge and api_server.

```bash
cd /home/wei-chi/Alzheimers_Project/external_data/scripts/downstream
conda run -n AD python extract_full_predictions_ensemble.py \
    > logs/extract_full_predictions.log 2>&1
```

**Verify success:**
```bash
tail -20 logs/extract_full_predictions.log
# Should show: NC_vs_AD AUC≈0.791, NC_vs_MCI AUC≈0.686, MCI_vs_AD AUC≈0.672
# Should show: [SAVED] results/full_predictions_v2_nolabel_ensemble.npz
```

---

## Step 2 — PCAG Cross-Attention Extraction (~5 min, GPU)

Extracts per-patient cross-attention weights from the PCAG model.
Output: `results/pcag_attention_v2_nolabel.npz`

```bash
conda run -n AD python extract_pcag_attention.py \
    --ckpt_dir checkpoints/pcag_ensemble_noaug_v2_fmricombat_nolabel \
    --fmri_dir fmri_combat_v2_nolabel \
    --out_suffix _nolabel \
    > logs/extract_pcag_attn.log 2>&1
```

**Verify:**
```bash
tail -5 logs/extract_pcag_attn.log
# Should show: [SAVED] results/pcag_attention_v2_nolabel.npz
```

---

## Step 3 — GAT Attention Extraction (~5 min, GPU)

Extracts fold-0 raw GAT attention weights (NOT cross-fold averaged).
Output: `results/gat_attention_v2_nolabel.npz`

```bash
conda run -n AD python extract_gat_attention.py \
    --ckpt_dir checkpoints/pcag_ensemble_noaug_v2_fmricombat_nolabel \
    --fmri_dir fmri_combat_v2_nolabel \
    --out_suffix _nolabel \
    > logs/extract_gat_attn.log 2>&1
```

**Verify:**
```bash
tail -5 logs/extract_gat_attn.log
# Should show: [SAVED] results/gat_attention_v2_nolabel.npz
```

---

## Step 4 — BrainIAC ViT Attention Extraction (~10-15 min, GPU)

Extracts sMRI ViT attention rollout per patient and maps to AAL116 ROIs.
Output: `results/brainiac_roi_v2.npz`

```bash
conda run -n AD python extract_brainiac_attention.py \
    > logs/extract_brainiac_attn.log 2>&1
```

**Verify:**
```bash
tail -20 logs/extract_brainiac_attn.log
# Should show: [SAVED] results/brainiac_roi_v2.npz
```

**If it fails** (MONAI ViT attention attribute error), look at the error in the log:
- Check what attribute name the error suggests (e.g., `att_mat` vs `attention_maps`)
- Edit `extract_brainiac_attention.py` accordingly and re-run

---

## Step 5 — Regenerate All Figures (~2 min total)

Run all figure scripts in sequence:

```bash
conda run -n AD python results/make_progression_figure.py
conda run -n AD python results/make_site_confound_figure.py
conda run -n AD python results/make_roc_figure_nolabel.py
conda run -n AD python results/make_ablation_figure_nolabel.py
conda run -n AD python results/make_qkv_swap_figure_nolabel.py
```

**Verify** — check these files were updated (timestamp should be recent):
```bash
ls -lt results/*.png | head -10
```

Expected output files:
- `results/model_progression_v2_nolabel.png`
- `results/site_confound_analysis_v2.png`
- `results/roc_curves_v2_nolabel.png`
- `results/ablation_comparison_v2_nolabel.png`
- `results/qkv_swap_v2_nolabel.png`

---

## Step 6 — LLM Judge Evaluation (~3-4 hours)

### 6a. Check Ollama is running and models are loaded

```bash
curl -s http://localhost:11434/api/tags | python3 -c \
    "import sys,json; [print(m['name']) for m in json.load(sys.stdin)['models']]"
# Should list: gemma3:12b (or similar) and qwen3:... (or similar)
```

### 6b. Smoke test (1 patient, ~2 min)

```bash
conda run -n AD python eval_report_quality.py \
    --use_ensemble --n 1 \
    > logs/eval_report_quality_smoke.log 2>&1
tail -30 logs/eval_report_quality_smoke.log
# Should show scores from 2 judges without errors
```

If smoke test fails:
- Check that `results/full_predictions_v2_nolabel_ensemble.npz` exists (Step 1 must complete first)
- Check that `results/brainiac_roi_v2.npz` exists (Step 4 must complete first)
- Check Ollama is responsive (may need to restart: `ollama serve &`)

### 6c. Full evaluation run (n=49)

```bash
conda run -n AD python eval_report_quality.py \
    --use_ensemble \
    --out results/report_quality_v2_nolabel_results.json \
    > logs/eval_report_quality_full.log 2>&1
```

**Monitor progress:**
```bash
tail -f logs/eval_report_quality_full.log
# Should show patient-by-patient progress
```

**Verify:**
```bash
tail -50 logs/eval_report_quality_full.log
# Should show Wilcoxon test results and inter-judge agreement
python3 -c "import json; d=json.load(open('results/report_quality_v2_nolabel_results.json')); print(d.keys())"
```

---

## Step 7 — Update Documentation

After all above steps complete, update `revise.md` to mark tasks complete and add final AUC numbers:

**Final AUC table to add to revise.md:**
```
| Task      | Strategy        | OOF AUC | Test AUC | 95% CI           | ADNI  | TPMIC |
|-----------|-----------------|---------|----------|------------------|-------|-------|
| NC_vs_AD  | top3_oof        | 0.7152  | 0.7909   | [0.609, 0.927]   | 1.000 | 0.673 |
| NC_vs_MCI | single_best     | 0.6654  | 0.6861   | [0.509, 0.855]   | 0.444 | 0.631 |
| MCI_vs_AD | median          | 0.7563  | 0.6717   | [0.468, 0.879]   | 1.000 | 0.631 |
```

---

## Dependency Order

```
Step 1 (full predictions)
    └── Step 6 (LLM judge) — needs full_predictions npz

Step 2 + Step 3 + Step 4 (attention extraction — can run in parallel)
    └── Step 6 (LLM judge) — needs all 3 attention npz files

Step 5 (figures) — independent, can run anytime

Step 6 = requires Steps 1 + 2 + 3 + 4 all done first
```

## Notes

- Steps 1-5 must run in the `AD` conda env
- Steps 2-4 need GPU (CUDA); they will fail on CPU with CUDA OOM if wrong device
- Step 6 needs local Ollama running with Gemma3 and Qwen3 models loaded
- All logs go to `logs/` directory (already exists)
- If any step fails, read the log file for the error — most failures will be Python exceptions with clear messages
