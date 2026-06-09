"""
batch_evaluate.py
=================
對所有病患跑 analyze + 報告生成，輸出：
  - results/batch_eval/reports/<subject_id>.txt   每份報告全文
  - results/batch_eval/summary.csv                OVO 預測 vs 真實標籤
  - results/batch_eval/accuracy.json              準確率統計

執行：python batch_evaluate.py
"""
import os, sys, json, csv, time, re, requests

API     = os.environ.get("BATCH_API_URL", "http://localhost:8081")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "batch_eval")
os.makedirs(os.path.join(OUT_DIR, "reports"), exist_ok=True)

# ── 從路徑判斷真實標籤 ─────────────────────────────────────────────────────────
def infer_label(t1_path: str, matrix_path: str) -> str:
    for p in [t1_path or "", matrix_path or ""]:
        for sep in ["/NC/", "/MCI/", "/AD/"]:
            if sep in p:
                return sep.strip("/")
    return "unknown"

# ── 取得病患清單 ───────────────────────────────────────────────────────────────
def get_patients():
    resp = requests.get(f"{API}/api/v1/patients", timeout=30)
    return resp.json()["patients"]

# ── 跑單一病患的 analyze ───────────────────────────────────────────────────────
def analyze(p: dict):
    resp = requests.post(f"{API}/api/v1/analyze",
        data={
            "subject_id":  p["id"],
            "matrix_path": p.get("matrix_path") or "",
            "t1_path":     p.get("t1_path") or "",
            "fmri_weight": "0.5",
        }, timeout=120)
    if resp.status_code != 200:
        return None
    return resp.json()

# ── 跑單一病患的報告生成（streaming） ─────────────────────────────────────────
def generate_report(sid: str, analyze_data: dict) -> str:
    payload = {
        "subject_id":   sid,
        "task_results": analyze_data.get("task_results", {}),
        "kg_context":   analyze_data.get("kg_context", {}),
        "ovo_result":   analyze_data.get("ovo_result", {}),
        "mode":         "fast",
    }
    resp = requests.post(f"{API}/api/v1/report/stream",
        json=payload, stream=True, timeout=180)
    text = ""
    for chunk in resp.iter_content(chunk_size=None):
        if chunk:
            text += chunk.decode("utf-8", errors="replace")
    return text

# ── 從報告文字抽取 OVO 診斷 ────────────────────────────────────────────────────
def extract_report_pred(report_text: str) -> str:
    m = re.search(r"OVO[^\n]*?(NC|MCI|AD)", report_text, re.IGNORECASE)
    if m: return m.group(1).upper()
    for label in ["NC", "MCI", "AD"]:
        if re.search(rf"\b{label}\b", report_text[:300]):
            return label
    return "unknown"

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"[batch_evaluate] 連線至 {API}...")
    patients = get_patients()
    total    = len(patients)
    print(f"  共 {total} 位病患\n")

    rows       = []
    t_start    = time.time()
    n_ok = n_fail = 0

    for i, p in enumerate(patients, 1):
        sid        = p["id"]
        t1_path    = p.get("t1_path", "")
        mat_path   = p.get("matrix_path", "")
        true_label = infer_label(t1_path, mat_path)

        print(f"[{i:3d}/{total}] {sid} (label={true_label}) ...", end=" ", flush=True)
        t0 = time.time()

        # 1. Analyze
        try:
            ad = analyze(p)
        except Exception as e:
            print(f"FAIL (analyze: {e})")
            n_fail += 1
            rows.append({"subject_id": sid, "true_label": true_label,
                         "ovo_pred": "ERROR", "is_borderline": False,
                         "modality": p.get("label",""), "correct": False,
                         "report_path": "", "error": str(e)})
            continue

        if not ad:
            print("FAIL (no analyze data)")
            n_fail += 1
            rows.append({"subject_id": sid, "true_label": true_label,
                         "ovo_pred": "ERROR", "is_borderline": False,
                         "modality": p.get("label",""), "correct": False,
                         "report_path": "", "error": "empty response"})
            continue

        ovo          = ad.get("ovo_result", {})
        ovo_pred     = ovo.get("predicted_class", "N/A")
        is_border    = ovo.get("is_borderline", False)
        modality_tag = p.get("label", "")

        # 2. Generate report
        try:
            report_text = generate_report(sid, ad)
        except Exception as e:
            report_text = f"[報告生成失敗：{e}]"

        # 3. Save report
        safe_sid    = sid.replace("/", "_").replace(" ", "_")
        report_path = os.path.join(OUT_DIR, "reports", f"{safe_sid}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        correct = (true_label == ovo_pred) if true_label != "unknown" else None
        elapsed = time.time() - t0
        status  = "✓" if correct else ("?" if correct is None else "✗")
        border_note = " [邊界]" if is_border else ""
        print(f"{status} ovo={ovo_pred} label={true_label}{border_note}  ({elapsed:.1f}s)")

        n_ok += 1
        rows.append({
            "subject_id":   sid,
            "true_label":   true_label,
            "ovo_pred":     ovo_pred,
            "is_borderline": is_border,
            "modality":     modality_tag,
            "correct":      correct,
            "report_path":  report_path,
            "error":        "",
        })

        # Save CSV incrementally
        csv_path = os.path.join(OUT_DIR, "summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # ── Accuracy summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_start
    known = [r for r in rows if r["true_label"] != "unknown" and r["ovo_pred"] not in ("N/A","ERROR")]
    correct_rows = [r for r in known if r["correct"]]
    border_rows  = [r for r in known if r["is_borderline"]]
    border_correct = [r for r in border_rows if r["correct"]]

    acc_by_class = {}
    for label in ["NC", "MCI", "AD"]:
        cls_rows = [r for r in known if r["true_label"] == label]
        cls_correct = [r for r in cls_rows if r["correct"]]
        acc_by_class[label] = {
            "n": len(cls_rows),
            "correct": len(cls_correct),
            "accuracy": round(len(cls_correct)/len(cls_rows), 4) if cls_rows else None,
        }

    summary = {
        "total_patients": total,
        "analyzed_ok":    n_ok,
        "failed":         n_fail,
        "labeled_patients": len(known),
        "overall_accuracy": round(len(correct_rows)/len(known), 4) if known else None,
        "borderline_count": len(border_rows),
        "borderline_accuracy": round(len(border_correct)/len(border_rows), 4) if border_rows else None,
        "per_class": acc_by_class,
        "total_time_min": round(total_time/60, 1),
    }

    acc_path = os.path.join(OUT_DIR, "accuracy.json")
    with open(acc_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"完成：{n_ok} OK / {n_fail} FAIL — {total_time/60:.1f} min")
    print(f"Overall accuracy: {summary['overall_accuracy']} ({len(correct_rows)}/{len(known)})")
    for label, v in acc_by_class.items():
        print(f"  {label}: {v['correct']}/{v['n']} = {v['accuracy']}")
    print(f"Borderline: {len(border_rows)} cases, accuracy={summary['borderline_accuracy']}")
    print(f"\n結果儲存於：{OUT_DIR}")

if __name__ == "__main__":
    main()
