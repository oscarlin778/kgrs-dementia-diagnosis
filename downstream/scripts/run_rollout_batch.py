"""
批次執行 BrainIAC attention rollout，對所有 210 筆 T1 生成 saliency NIfTI。
輸出目錄：results/smri_saliency/
跳過已存在的檔案，可中斷後重跑。
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_pipeline_v2 import _get_pipeline, _compute_smri_saliency, SMRI_SAL_DIR

ADNI_ROOT  = "/home/wei-chi/Alzheimers_Project/external_data/datasets/ADNI_sMRI_Aligned_MNI"
TPMIC_ROOT = "/home/wei-chi/Alzheimers_Project/external_models/sMRI_data_MultiModal_Aligned_MNI"

def collect_t1s():
    entries = []
    # ADNI: *_T1_MNI.nii.gz → subject_id = basename without _T1_MNI.nii.gz
    for root, _, files in os.walk(ADNI_ROOT):
        for f in files:
            if f.endswith("_T1_MNI.nii.gz"):
                sid = f.replace("_T1_MNI.nii.gz", "")
                entries.append((sid, os.path.join(root, f)))
    # TPMIC: *_T1.nii.gz → subject_id = basename without _T1.nii.gz
    for root, _, files in os.walk(TPMIC_ROOT):
        for f in files:
            if f.endswith("_T1.nii.gz"):
                sid = f.replace("_T1.nii.gz", "")
                entries.append((sid, os.path.join(root, f)))
    return sorted(entries)

def main():
    entries = collect_t1s()
    total   = len(entries)
    print(f"共找到 {total} 筆 T1。輸出目錄：{SMRI_SAL_DIR}\n")

    pipe = _get_pipeline()

    done = skipped = failed = 0
    t0 = time.time()

    for i, (sid, t1_path) in enumerate(entries, 1):
        safe_sid  = sid.replace("/", "_").replace(" ", "_")
        out_path  = os.path.join(SMRI_SAL_DIR, f"{safe_sid}_brainiac_rollout.nii.gz")

        if os.path.exists(out_path):
            skipped += 1
            print(f"[{i:3d}/{total}] SKIP  {sid}")
            continue

        result = _compute_smri_saliency(pipe, t1_path, sid)

        if result["saliency_path"] and os.path.exists(result["saliency_path"]):
            done += 1
            top = result["top_regions"][0]["name"] if result["top_regions"] else "?"
            elapsed = time.time() - t0
            eta = elapsed / (done + failed) * (total - i)
            print(f"[{i:3d}/{total}] OK    {sid:30s}  top={top}  ETA={eta/60:.1f}min")
        else:
            failed += 1
            print(f"[{i:3d}/{total}] FAIL  {sid}")

    elapsed = time.time() - t0
    print(f"\n完成：{done} OK / {skipped} skipped / {failed} failed — {elapsed/60:.1f} min")

if __name__ == "__main__":
    main()
