"""
Phase 1-B: Rebuild train/test splits to maximize AD patients in test set.

Total data: NC=90, MCI=87, AD=27 (204 total)
Strategy:
  - AD:  11 in test (40%), 16 in train  — maximizes statistical power for AD evaluation
  - NC:  20 in test (22%), 70 in train
  - MCI: 18 in test (21%), 69 in train
  - Total test: 49, Total train: 155

Outputs:
  pcag_train_aligned_v2.csv, pcag_test_aligned_v2.csv
  brainiac_features_train_v2.csv, brainiac_features_combined_test_v2.csv
  kd_train_aligned_v2.csv, kd_test_aligned_v2.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

BASE = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")

# ── Load all 204 patients ─────────────────────────────────────────────────────
pcag_tr = pd.read_csv(BASE / "pcag_train_aligned.csv")
pcag_te = pd.read_csv(BASE / "pcag_test_aligned.csv")
kd_tr   = pd.read_csv(BASE / "kd_train_aligned.csv")
kd_te   = pd.read_csv(BASE / "kd_test_aligned.csv")

feat_tr = pd.read_csv(BASE / "brainiac_features_train.csv")
feat_te = pd.read_csv(BASE / "brainiac_features_combined_test.csv")

# Reconstruct full feature matrix in pcag order (sequential smri_feat_row)
feat_all_tr = feat_tr.loc[pcag_tr["smri_feat_row"]].reset_index(drop=True)
feat_all_te = feat_te.loc[pcag_te["smri_feat_row"]].reset_index(drop=True)
feat_all = pd.concat([feat_all_tr, feat_all_te], ignore_index=True)

# Merge pcag metadata (has smri_feat_row ← will be regenerated)
pcag_all = pd.concat([
    pcag_tr.drop(columns=["smri_feat_row"]),
    pcag_te.drop(columns=["smri_feat_row"]),
], ignore_index=True)

# Merge kd metadata (aligned to pcag via subject_id)
kd_all = pd.concat([kd_tr, kd_te], ignore_index=True)

# Make sure kd_all is in same order as pcag_all
kd_all = pcag_all[["subject_id"]].merge(kd_all, on="subject_id", how="left")

assert len(pcag_all) == len(feat_all) == len(kd_all) == 204, "Row count mismatch!"
assert (pcag_all["subject_id"] == kd_all["subject_id"]).all(), "subject_id mismatch!"

# ── Stratified split per class ────────────────────────────────────────────────
TEST_N = {0: 20, 1: 18, 2: 11}   # NC=20, MCI=18, AD=11

test_indices = []
train_indices = []

for label, n_test in TEST_N.items():
    cls_idx = pcag_all.index[pcag_all["label"] == label].tolist()
    shuffled = rng.permutation(cls_idx)
    test_indices.extend(shuffled[:n_test].tolist())
    train_indices.extend(shuffled[n_test:].tolist())

test_indices  = sorted(test_indices)
train_indices = sorted(train_indices)

print(f"Train: {len(train_indices)}  Test: {len(test_indices)}")
label_map = {0: "NC", 1: "MCI", 2: "AD"}
for split, idxs in [("Train", train_indices), ("Test", test_indices)]:
    counts = pcag_all.loc[idxs, "label"].value_counts().sort_index()
    print(f"  {split}: " + ", ".join(f"{label_map[l]}={n}" for l, n in counts.items()))

# ── Build pcag v2 CSVs ────────────────────────────────────────────────────────
def make_pcag(indices):
    df = pcag_all.loc[indices].reset_index(drop=True)
    df["smri_feat_row"] = range(len(df))
    return df[["subject_id", "matrix_path", "label", "smri_feat_row"]]

pcag_tr_v2 = make_pcag(train_indices)
pcag_te_v2 = make_pcag(test_indices)

# ── Build brainiac feature v2 CSVs ───────────────────────────────────────────
feat_tr_v2 = feat_all.loc[train_indices].reset_index(drop=True)
feat_te_v2 = feat_all.loc[test_indices].reset_index(drop=True)

# ── Build kd v2 CSVs ─────────────────────────────────────────────────────────
kd_tr_v2 = kd_all.loc[train_indices].reset_index(drop=True)
kd_te_v2 = kd_all.loc[test_indices].reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────────────
pcag_tr_v2.to_csv(BASE / "pcag_train_aligned_v2.csv", index=False)
pcag_te_v2.to_csv(BASE / "pcag_test_aligned_v2.csv",  index=False)
feat_tr_v2.to_csv(BASE / "brainiac_features_train_v2.csv", index=False)
feat_te_v2.to_csv(BASE / "brainiac_features_combined_test_v2.csv", index=False)
kd_tr_v2.to_csv(BASE / "kd_train_aligned_v2.csv", index=False)
kd_te_v2.to_csv(BASE / "kd_test_aligned_v2.csv",  index=False)

print("\n[SAVED] All v2 splits written.")

# ── Verification ─────────────────────────────────────────────────────────────
for fname, df in [
    ("pcag_train_v2", pcag_tr_v2), ("pcag_test_v2", pcag_te_v2),
    ("kd_train_v2",   kd_tr_v2),   ("kd_test_v2",   kd_te_v2),
]:
    counts = df["label"].value_counts().sort_index()
    print(f"  {fname}: " + " | ".join(f"{label_map[l]}={n}" for l, n in counts.items()))

# Sanity: no subject in both train and test
train_subs = set(pcag_tr_v2.subject_id)
test_subs  = set(pcag_te_v2.subject_id)
assert not train_subs & test_subs, f"Leakage! {train_subs & test_subs}"
print("\n[OK] No subject overlap between train and test.")
