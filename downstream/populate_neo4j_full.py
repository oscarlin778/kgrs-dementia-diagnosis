"""
populate_neo4j_full.py
補全 Neo4j 知識圖譜：Patient 人口統計、AI_Prediction 節點、
Brain_Feature 節點（GNN embedding）、SIMILAR_TO 邊。
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
RESULTS     = BASE / "results"
SPLITS      = BASE / "splits"
DEMO_CSV    = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/patient_demographics.csv")
PCAG_TR_CSV = BASE / "pcag_train_aligned.csv"

TRAIN_CSV   = SPLITS / "combined_train.csv"
TEST_CSV    = SPLITS / "combined_test.csv"

KD_NPZ = {
    "NC_vs_AD":  RESULTS / "kd_NC_vs_AD_a3_T3_probs.npz",
    "NC_vs_MCI": RESULTS / "kd_NC_vs_MCI_a3_T3_probs.npz",
    "MCI_vs_AD": RESULTS / "kd_MCI_vs_AD_a10_T3_probs.npz",
}
PCAG_NPZ = {
    "NC_vs_AD":  RESULTS / "pcag_combat_NC_vs_AD_probs.npz",
    "NC_vs_MCI": RESULTS / "pcag_combat_NC_vs_MCI_probs.npz",
}
TASK_CFG = {
    "NC_vs_AD":  {"classes": [0, 2], "pos": 2, "use_pcag": True},
    "NC_vs_MCI": {"classes": [0, 1], "pos": 1, "use_pcag": True},
    "MCI_vs_AD": {"classes": [1, 2], "pos": 2, "use_pcag": False},
}
GNN_FILES = sorted((BASE / "embeddings").glob("gnn_embeddings_*.npz"))

# ── Neo4j ──────────────────────────────────────────────────────────────────────
NEO4J_URI  = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "nm6131027"


# ══════════════════════════════════════════════════════════════════════════════
# Platt scaling helpers (same as eval_calibrated_hybrid.py)
# ══════════════════════════════════════════════════════════════════════════════
def fit_platt(oof_probs: np.ndarray, oof_labels: np.ndarray) -> LogisticRegression:
    eps = 1e-7
    logits = np.log(oof_probs + eps) - np.log(1 - oof_probs + eps)
    lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
    lr.fit(logits.reshape(-1, 1), oof_labels)
    return lr

def apply_platt(lr: LogisticRegression, probs: np.ndarray) -> np.ndarray:
    eps = 1e-7
    logits = np.log(probs + eps) - np.log(1 - probs + eps)
    return lr.predict_proba(logits.reshape(-1, 1))[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# Step 0 – Load master subject lists
# ══════════════════════════════════════════════════════════════════════════════
df_tr = pd.read_csv(TRAIN_CSV)   # 193 rows
df_te = pd.read_csv(TEST_CSV)    # 49 rows
df_all = pd.concat([df_tr, df_te], ignore_index=True)  # 242 rows

# Add split column
df_tr = df_tr.copy(); df_tr["split"] = "train"
df_te = df_te.copy(); df_te["split"] = "test"
df_all = pd.concat([df_tr, df_te], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Compute per-subject calibrated probs (Platt scaling)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Computing per-subject calibrated probabilities...")

df_pcag_tr = pd.read_csv(PCAG_TR_CSV)

# subject_id → {task: (calibrated_prob, routing_model)}
subj_preds: dict[str, dict] = {}

for task, cfg in TASK_CFG.items():
    classes = cfg["classes"]
    pos     = cfg["pos"]

    # --- Train subjects for this task ---
    df_tr_task = df_tr[df_tr["label"].isin(classes)].reset_index(drop=True)
    kd_tr_ids  = df_tr_task["subject_id"].tolist()
    kd_tr_labs = (df_tr_task["label"] == pos).astype(int).values

    # --- Test subjects for this task ---
    df_te_task = df_te[df_te["label"].isin(classes)].reset_index(drop=True)
    kd_te_ids  = df_te_task["subject_id"].tolist()

    # --- Load KD NPZ ---
    kd_npz     = np.load(KD_NPZ[task], allow_pickle=True)
    kd_scaler  = fit_platt(kd_npz["oof_probs"], kd_tr_labs)
    kd_tr_cal  = apply_platt(kd_scaler, kd_npz["oof_probs"])
    kd_te_cal  = apply_platt(kd_scaler, kd_npz["test_probs"])
    kd_tr_map  = dict(zip(kd_tr_ids, kd_tr_cal))
    kd_te_map  = dict(zip(kd_te_ids, kd_te_cal))

    # --- Load PCAG NPZ (if applicable) ---
    pcag_tr_map: dict = {}
    pcag_te_map: dict = {}
    if cfg["use_pcag"]:
        df_pcag_tr_task = df_pcag_tr[df_pcag_tr["label"].isin(classes)].reset_index(drop=True)
        pcag_tr_ids     = df_pcag_tr_task["subject_id"].tolist()
        pcag_tr_labs    = (df_pcag_tr_task["label"] == pos).astype(int).values

        pcag_npz     = np.load(PCAG_NPZ[task], allow_pickle=True)
        pcag_scaler  = fit_platt(pcag_npz["oof_probs"], pcag_tr_labs)
        pcag_tr_cal  = apply_platt(pcag_scaler, pcag_npz["oof_probs"])
        pcag_te_cal  = apply_platt(pcag_scaler, pcag_npz["test_probs"])
        pcag_tr_map  = dict(zip(pcag_tr_ids, pcag_tr_cal))
        pcag_te_map  = dict(zip(pcag_npz["test_subject_ids"].tolist(), pcag_te_cal))

    # --- Route & store per subject ---
    for sid, prob_kd in kd_tr_map.items():
        if cfg["use_pcag"] and sid in pcag_tr_map:
            prob   = float(pcag_tr_map[sid])
            model  = "PCAG_ComBat"
        else:
            prob   = float(prob_kd)
            model  = "KD_Student"
        subj_preds.setdefault(sid, {})[task] = {"prob": prob, "model": model}

    for sid, prob_kd in kd_te_map.items():
        if cfg["use_pcag"] and sid in pcag_te_map:
            prob   = float(pcag_te_map[sid])
            model  = "PCAG_ComBat"
        else:
            prob   = float(prob_kd)
            model  = "KD_Student"
        subj_preds.setdefault(sid, {})[task] = {"prob": prob, "model": model}

print(f"  Subjects with calibrated predictions: {len(subj_preds)}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Build global embedding dict (mean pooling for duplicates)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Merging GNN embeddings (mean pooling duplicates)...")

emb_accum: dict[str, list[np.ndarray]] = {}
for fpath in GNN_FILES:
    d    = np.load(fpath, allow_pickle=True)
    sids = d["subject_ids"].tolist()
    embs = d["embeddings"]          # (n, 1280)
    for sid, emb in zip(sids, embs):
        emb_accum.setdefault(sid, []).append(emb)

global_emb: dict[str, np.ndarray] = {}
n_merged = 0
for sid, emb_list in emb_accum.items():
    if len(emb_list) > 1:
        n_merged += 1
    global_emb[sid] = np.mean(emb_list, axis=0).astype(np.float32)

print(f"  Total subjects with embeddings: {len(global_emb)}")
print(f"  Subjects with merged (>1 NPZ) embeddings: {n_merged}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Load ADNI demographics
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Loading ADNI demographics...")

df_demo = pd.read_csv(DEMO_CSV)
df_demo = df_demo.dropna(subset=["PTID"]).copy()
df_demo = df_demo.drop_duplicates(subset=["PTID"], keep="first")
gender_map = {1.0: "Male", 0.0: "Female"}
demo_lookup: dict[str, dict] = {}
for _, row in df_demo.iterrows():
    ptid = str(row["PTID"]).strip()
    demo_lookup[ptid] = {
        "age":       int(row["Age_Baseline"]) if pd.notna(row["Age_Baseline"]) else 0,
        "sex":       gender_map.get(row["Gender"], "Unknown"),
        "education": int(row["RMT_Education"]) if pd.notna(row["RMT_Education"]) else 0,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Write to Neo4j
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Writing to Neo4j...")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Counters
n_patient_upsert   = 0
n_demo_updated     = 0
n_pred_created     = 0
n_feat_created     = 0
n_similar_created  = 0
n_tpmic = n_adni   = 0

with driver.session() as session:

    # ── Create constraints if missing ────────────────────────────────────────
    session.run("""
        CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient)
        REQUIRE p.subject_id IS UNIQUE
    """)
    session.run("""
        CREATE CONSTRAINT IF NOT EXISTS FOR (a:AI_Prediction)
        REQUIRE a.subject_id IS UNIQUE
    """)
    session.run("""
        CREATE CONSTRAINT IF NOT EXISTS FOR (f:Brain_Feature)
        REQUIRE f.subject_id IS UNIQUE
    """)

    # ── 4-A: Upsert Patient nodes ─────────────────────────────────────────────
    print("  → Upserting Patient nodes...")
    for _, row in df_all.iterrows():
        sid  = str(row["subject_id"])
        site = "TPMIC" if "sub_" in sid else "ADNI"
        dx   = str(row["diagnosis"])
        lbl  = int(row["label"])
        split_val = str(row["split"])

        demo = demo_lookup.get(sid, {})
        session.run("""
            MERGE (p:Patient {subject_id: $sid})
            SET p.site        = $site,
                p.clinical_dx = $dx,
                p.label       = $lbl,
                p.split       = $split,
                p.age         = CASE WHEN $age > 0 THEN $age ELSE p.age END,
                p.sex         = CASE WHEN $sex <> 'Unknown' THEN $sex ELSE coalesce(p.sex, 'Unknown') END,
                p.education   = CASE WHEN $edu > 0 THEN $edu ELSE p.education END
        """,
            sid=sid, site=site, dx=dx, lbl=lbl, split=split_val,
            age=demo.get("age", 0),
            sex=demo.get("sex", "Unknown"),
            edu=demo.get("education", 0),
        )
        n_patient_upsert += 1
        if site == "TPMIC":
            n_tpmic += 1
        else:
            n_adni += 1
        if demo:
            n_demo_updated += 1

    # ── 4-B: Create AI_Prediction nodes + HAS_PREDICTION edges ───────────────
    print("  → Creating AI_Prediction nodes...")
    for sid, task_data in subj_preds.items():
        # Determine dominant routing (most tasks using PCAG wins)
        models = [v["model"] for v in task_data.values()]
        routing = "PCAG_ComBat" if models.count("PCAG_ComBat") > models.count("KD_Student") else "KD_Student"

        ncad  = task_data.get("NC_vs_AD",  {}).get("prob", None)
        ncmci = task_data.get("NC_vs_MCI", {}).get("prob", None)
        mciad = task_data.get("MCI_vs_AD", {}).get("prob", None)

        session.run("""
            MERGE (a:AI_Prediction {subject_id: $sid})
            SET a.routing_model       = $routing,
                a.calibrated_prob_AD_ncad   = $ncad,
                a.calibrated_prob_MCI_ncmci = $ncmci,
                a.calibrated_prob_AD_mciad  = $mciad
            WITH a
            MATCH (p:Patient {subject_id: $sid})
            MERGE (p)-[:HAS_PREDICTION]->(a)
        """,
            sid=sid, routing=routing,
            ncad=ncad, ncmci=ncmci, mciad=mciad,
        )
        n_pred_created += 1

    # ── 4-C: Create Brain_Feature nodes + HAS_FEATURE edges ──────────────────
    print("  → Creating Brain_Feature nodes...")
    for sid, emb in global_emb.items():
        emb_list = emb.tolist()
        session.run("""
            MERGE (f:Brain_Feature {subject_id: $sid})
            SET f.embedding_dim = 1280,
                f.embedding = $emb
            WITH f
            MATCH (p:Patient {subject_id: $sid})
            MERGE (p)-[:HAS_FEATURE]->(f)
        """,
            sid=sid, emb=emb_list,
        )
        n_feat_created += 1

    # ── 4-D: SIMILAR_TO edges (Top-5 cosine similarity) ─────────────────────
    print("  → Computing SIMILAR_TO edges (Top-5 cosine sim)...")

    # Build matrix only for subjects that are in both Patient nodes AND global_emb
    all_pids = [row["subject_id"] for _, row in df_all.iterrows()]
    pids_with_emb = [sid for sid in all_pids if sid in global_emb]
    emb_matrix    = np.stack([global_emb[sid] for sid in pids_with_emb])  # (N, 1280)

    sim_matrix = cosine_similarity(emb_matrix)   # (N, N)
    K = 5

    # Batch write SIMILAR_TO edges
    similar_batch = []
    for i, sid_i in enumerate(pids_with_emb):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0   # exclude self
        top_k_idx = np.argsort(sims)[-K:][::-1]
        for j in top_k_idx:
            similar_batch.append({
                "sid1":  sid_i,
                "sid2":  pids_with_emb[j],
                "score": float(sims[j]),
            })

    # Write in chunks of 500
    chunk_size = 500
    for start in range(0, len(similar_batch), chunk_size):
        chunk = similar_batch[start:start + chunk_size]
        session.run("""
            UNWIND $rows AS row
            MATCH (p1:Patient {subject_id: row.sid1})
            MATCH (p2:Patient {subject_id: row.sid2})
            MERGE (p1)-[s:SIMILAR_TO {target: row.sid2}]->(p2)
            SET s.score = row.score
        """, rows=chunk)
        n_similar_created += len(chunk)

driver.close()

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅  Neo4j Population Complete")
print("=" * 60)
print(f"  Patient nodes upserted   : {n_patient_upsert}")
print(f"    ├─ ADNI                : {n_adni}")
print(f"    └─ TPMIC               : {n_tpmic}")
print(f"  Demographics populated   : {n_demo_updated}")
print(f"  Subjects with merged emb : {n_merged}")
print(f"  AI_Prediction nodes      : {n_pred_created}")
print(f"  Brain_Feature nodes      : {n_feat_created}")
print(f"  SIMILAR_TO edges created : {n_similar_created}")
print("=" * 60)
print("\n🔍  Test Cypher (Neo4j Browser):")
print("""
-- 某病患的校正後機率 + 最相似的 3 位 AD 病患:
MATCH (p:Patient {subject_id: 'YOUR_SUBJECT_ID'})
OPTIONAL MATCH (p)-[:HAS_PREDICTION]->(pred:AI_Prediction)
OPTIONAL MATCH (p)-[:SIMILAR_TO]->(p2:Patient)
  WHERE p2.clinical_dx = 'AD'
  WITH p, pred, p2
  ORDER BY [(p)-[s:SIMILAR_TO]->(p2) | s.score][0] DESC
  LIMIT 3
RETURN p.subject_id, p.clinical_dx, p.site,
       pred.calibrated_prob_AD_ncad,
       pred.calibrated_prob_MCI_ncmci,
       pred.routing_model,
       collect(p2.subject_id)[0..3] AS similar_AD_patients
""")
