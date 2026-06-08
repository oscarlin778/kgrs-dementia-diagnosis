"""
baseline_svm.py — SVM baseline using FC upper-triangle features + PCA.

Feature: fMRI connectivity matrix upper-triangle (116×116 → 6670 dims)
         reduced to 100 dims via PCA (fit on train only).
Classifier: SVM with RBF kernel, GridSearchCV over C and gamma.
Same v2 splits and 5-fold CV as all other baselines.

Usage:
    python baseline_svm.py --task NC_vs_AD
    python baseline_svm.py --task NC_vs_MCI
    python baseline_svm.py --task MCI_vs_AD
"""
import os, sys, argparse, json, random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, roc_curve

BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
RES_DIR  = BASE_DIR / "results"

N_ROIS = 116
TRIU_IDX = np.triu_indices(N_ROIS, k=1)   # 6670 features


def load_fc_matrix(path: str) -> np.ndarray:
    mat = np.load(path)
    if isinstance(mat, np.ndarray):
        fc = mat
    else:
        fc = mat[mat.files[0]]
    return fc.astype(np.float32)


def extract_upper_tri(paths: list) -> np.ndarray:
    feats = []
    for p in paths:
        fc = load_fc_matrix(p)
        # Handle (1, N, N) or (N, N)
        if fc.ndim == 3:
            fc = fc[0]
        feats.append(fc[TRIU_IDX])
    return np.array(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="NC_vs_MCI",
                        choices=["NC_vs_AD", "NC_vs_MCI", "MCI_vs_AD"])
    parser.add_argument("--n_components", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    TASK_CFG = {
        'NC_vs_AD':  {'classes': [0, 2], 'pos': 2},
        'NC_vs_MCI': {'classes': [0, 1], 'pos': 1},
        'MCI_vs_AD': {'classes': [1, 2], 'pos': 2},
    }
    cfg = TASK_CFG[args.task]

    df_tr = pd.read_csv(BASE_DIR / "pcag_train_aligned_v2.csv")
    df_te = pd.read_csv(BASE_DIR / "pcag_test_aligned_v2.csv")

    df_tr = df_tr[df_tr['label'].isin(cfg['classes'])].reset_index(drop=True)
    df_tr['bin_label'] = (df_tr['label'] == cfg['pos']).astype(int)
    df_te = df_te[df_te['label'].isin(cfg['classes'])].reset_index(drop=True)
    df_te['bin_label'] = (df_te['label'] == cfg['pos']).astype(int)

    print(f"[{args.task}] Train: {len(df_tr)}  Test: {len(df_te)}")
    print("Extracting FC upper-triangle features...")

    X_tr = extract_upper_tri(df_tr['matrix_path'].tolist())
    X_te = extract_upper_tri(df_te['matrix_path'].tolist())
    y_tr = df_tr['bin_label'].values
    y_te = df_te['bin_label'].values

    print(f"Feature shape — train: {X_tr.shape}, test: {X_te.shape}")

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros(len(df_tr))
    test_probs_folds = []

    # PCA is fit outside GridSearchCV to avoid n_components > inner-fold train size.
    # Inner 3-fold CV on a 5-fold outer split can have as few as ~45 samples,
    # which is less than n_components=100.
    param_grid = {
        'C':     [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
    }

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_f_tr, X_f_val = X_tr[tr_idx], X_tr[val_idx]
        y_f_tr, y_f_val = y_tr[tr_idx],  y_tr[val_idx]

        # Fit Scaler + PCA on outer fold train only
        n_comp = min(args.n_components, X_f_tr.shape[0] - 1, X_f_tr.shape[1])
        scaler = StandardScaler()
        pca    = PCA(n_components=n_comp, random_state=args.seed)
        X_f_tr_t  = pca.fit_transform(scaler.fit_transform(X_f_tr))
        X_f_val_t = pca.transform(scaler.transform(X_f_val))
        X_te_t    = pca.transform(scaler.transform(X_te))

        svm = SVC(kernel='rbf', probability=True,
                  class_weight='balanced', random_state=args.seed)
        gs = GridSearchCV(svm, param_grid, cv=3, scoring='roc_auc',
                          n_jobs=-1, refit=True)
        gs.fit(X_f_tr_t, y_f_tr)
        best = gs.best_estimator_

        val_p  = best.predict_proba(X_f_val_t)[:, 1]
        val_auc = roc_auc_score(y_f_val, val_p) if len(set(y_f_val)) > 1 else 0.5
        print(f"n_comp={n_comp}  Best params: {gs.best_params_}  Val AUC: {val_auc:.4f}")

        oof_probs[val_idx] = val_p
        test_probs_folds.append(best.predict_proba(X_te_t)[:, 1])

    # Evaluation
    oof_auc = roc_auc_score(y_tr, oof_probs)
    fpr, tpr, thresholds = roc_curve(y_tr, oof_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]

    avg_test_p = np.mean(test_probs_folds, axis=0)
    test_auc   = roc_auc_score(y_te, avg_test_p)
    test_pred  = (avg_test_p >= best_thresh).astype(int)
    cm = confusion_matrix(y_te, test_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auc":  float(test_auc),
        "sens": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        "spec": float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        "f1":   float(f1_score(y_te, test_pred)),
        "acc":  float(accuracy_score(y_te, test_pred)),
        "cm":   cm.tolist(),
    }
    print(f"\nSVM Baseline Test Metrics ({args.task}):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_file = RES_DIR / f"svm_{args.task}_results_v2.json"
    with open(out_file, "w") as f:
        json.dump({"oof_auc": float(oof_auc), "test_metrics": metrics}, f, indent=2)
    print(f"\n[SAVED] {out_file}")

    np.savez(str(RES_DIR / f"svm_{args.task}_probs_v2.npz"),
             test_probs=avg_test_p, test_labels=y_te,
             oof_probs=oof_probs, oof_labels=y_tr)


if __name__ == "__main__":
    main()
