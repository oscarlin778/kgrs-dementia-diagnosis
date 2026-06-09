import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

def run_ovo_task(train_df, test_df, label1, label2, task_name):
    print(f"\n>>> Running Task: {task_name} (Labels {label1} vs {label2})")
    
    # 1. Filter classes
    train_subset = train_df[train_df['label'].isin([label1, label2])].copy()
    test_subset = test_df[test_df['label'].isin([label1, label2])].copy()
    
    # Map labels to 0 and 1 for the specific task
    # We use label1 as 0 and label2 as 1
    y_train = (train_subset['label'] == label2).astype(int).values
    X_train = train_subset.filter(regex='^Feature_').values
    
    y_test = (test_subset['label'] == label2).astype(int).values
    X_test = test_subset.filter(regex='^Feature_').values
    
    # 2. 5-Fold OOF CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_train))
    oof_preds = np.zeros(len(y_train))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
        clf.fit(X_tr_scaled, y_tr)
        
        oof_probs[val_idx] = clf.predict_proba(X_val_scaled)[:, 1]
        oof_preds[val_idx] = clf.predict(X_val_scaled)
        
    oof_auc = roc_auc_score(y_train, oof_probs)
    oof_acc = accuracy_score(y_train, oof_preds)
    
    print(f"  OOF Results:  AUC={oof_auc:.4f}, ACC={oof_acc:.4f}, n={len(y_train)}")
    
    # 3. Full Training & Test Evaluation
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42)
    clf.fit(X_train_full_scaled, y_train)
    
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    test_preds = clf.predict(X_test_scaled)
    
    test_auc = roc_auc_score(y_test, test_probs)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"  Test Results: AUC={test_auc:.4f}, ACC={test_acc:.4f}, n={len(y_test)}")
    
    return {
        "oof": {"auc": float(oof_auc), "acc": float(oof_acc), "n": int(len(y_train))},
        "test": {"auc": float(test_auc), "acc": float(test_acc), "n": int(len(y_test))}
    }

def main():
    train_path = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/vitmci_features_train.csv"
    test_path = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/vitmci_features_combined_test.csv"
    output_json = "/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results/vitmci_linear_probing_combined_test.json"
    
    # Load data
    print(f"Loading training features from {train_path}...")
    df_train = pd.read_csv(train_path)
    print(f"Loading test features from {test_path}...")
    df_test = pd.read_csv(test_path)
    
    tasks = [
        (0, 2, "NC vs AD"),
        (0, 1, "NC vs MCI"),
        (1, 2, "MCI vs AD")
    ]
    
    results = {
        "oof": {},
        "test": {},
        "train_n": len(df_train),
        "test_n": len(df_test)
    }
    
    for l1, l2, name in tasks:
        task_res = run_ovo_task(df_train, df_test, l1, l2, name)
        results["oof"][name] = task_res["oof"]
        results["test"][name] = task_res["test"]
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[SUCCESS] All tasks completed. Results saved to {output_json}")

if __name__ == "__main__":
    main()
