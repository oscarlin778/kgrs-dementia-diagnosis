
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle

def train_and_evaluate():
    # Paths
    train_features_path = '/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/brainiac_features_train.csv'
    test_features_path = '/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/brainiac_features_test.csv'
    results_dir = '/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream/results'
    os.makedirs(results_dir, exist_ok=True)

    # Load training features
    train_df = pd.read_csv(train_features_path)
    # The columns are Feature_0, ..., Feature_767, GroundTruthClassLabel
    X_train_full = train_df.filter(regex='Feature_').values
    y_train_full = train_df['GroundTruthClassLabel'].values
    # Label mapping in generate_brainiac_csvs.py: 0=NC, 1=MCI, 2=AD

    tasks = {
        "NC vs AD": {"classes": [0, 2], "labels": {0: 0, 2: 1}},
        "NC vs MCI": {"classes": [0, 1], "labels": {0: 0, 1: 1}},
        "MCI vs AD": {"classes": [1, 2], "labels": {1: 0, 2: 1}}
    }

    oof_results = {}
    test_results = {}
    
    # Load test features
    test_df = pd.read_csv(test_features_path)
    X_test_full = test_df.filter(regex='Feature_').values
    y_test_full = test_df['GroundTruthClassLabel'].values

    for task_name, task_info in tasks.items():
        print(f"\n--- Processing task: {task_name} ---")
        classes = task_info["classes"]
        label_map = task_info["labels"]
        
        # Filter training data for this task
        mask_train = np.isin(y_train_full, classes)
        X_task_train = X_train_full[mask_train]
        y_task_train = np.array([label_map[l] for l in y_train_full[mask_train]])
        
        # Filter test data for this task
        mask_test = np.isin(y_test_full, classes)
        X_task_test = X_test_full[mask_test]
        y_task_test = np.array([label_map[l] for l in y_test_full[mask_test]])
        
        # 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y_task_train))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_task_train, y_task_train)):
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_task_train[train_idx], y_task_train[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X_task_train[val_idx])[:, 1]
        
        task_auc_oof = roc_auc_score(y_task_train, oof_probs)
        print(f"OOF AUC for {task_name}: {task_auc_oof:.4f}")
        
        oof_results[task_name] = {
            "auc": task_auc_oof,
            "predictions": oof_probs.tolist()
        }
        
        # Train final classifier
        final_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        final_clf.fit(X_task_train, y_task_train)
        
        # Save model
        model_filename = f"brainiac_clf_{task_name.replace(' ', '_').lower()}.pkl"
        with open(os.path.join(results_dir, model_filename), 'wb') as f:
            pickle.dump(final_clf, f)
        
        # Predict on test set
        if len(y_task_test) > 0:
            test_probs = final_clf.predict_proba(X_task_test)[:, 1]
            test_preds = (test_probs > 0.5).astype(int)
            
            task_auc_test = roc_auc_score(y_task_test, test_probs)
            task_acc_test = accuracy_score(y_task_test, test_preds)
            print(f"Test AUC for {task_name}: {task_auc_test:.4f}, ACC: {task_acc_test:.4f}")
            
            test_results[task_name] = {
                "auc": task_auc_test,
                "acc": task_acc_test,
                "predictions": test_probs.tolist(),
                "n_samples": len(y_task_test)
            }
        else:
            print(f"No samples for {task_name} in test set.")
            test_results[task_name] = {"auc": 0, "acc": 0, "predictions": [], "n_samples": 0}

    # Save results
    with open(os.path.join(results_dir, 'brainiac_oof_predictions.json'), 'w') as f:
        json.dump(oof_results, f, indent=2)
    
    with open(os.path.join(results_dir, 'brainiac_test_predictions.json'), 'w') as f:
        json.dump(test_results, f, indent=2)

    # Comparison Table
    baseline_oof_auc = {
        "NC vs AD": 0.771,
        "NC vs MCI": 0.731,
        "MCI vs AD": 0.775
    }
    
    print("\n" + "="*50)
    print("Comparison Table:")
    print("| Task | Current sMRI OOF AUC | BrainIAC OOF AUC | Improvement |")
    print("|------|:---:|:---:|:---:|")
    for task_name in tasks:
        current = baseline_oof_auc[task_name]
        brainiac = oof_results[task_name]["auc"]
        improvement = brainiac - current
        print(f"| {task_name} | {current:.3f} | {brainiac:.3f} | {improvement:+.3f} |")
    print("="*50)

    # Decision
    all_better = all(oof_results[t]["auc"] > baseline_oof_auc[t] for t in tasks)
    any_better = any(oof_results[t]["auc"] > baseline_oof_auc[t] for t in tasks)
    
    decision_text = ""
    if all_better:
        decision_text = "Decision: Integrate into inference_pipeline.py\nReasoning: BrainIAC outperforms baseline on all tasks."
    elif any_better:
        better_tasks = [t for t in tasks if oof_results[t]["auc"] > baseline_oof_auc[t]]
        decision_text = f"Decision: Selective integration\nReasoning: BrainIAC outperforms baseline on {', '.join(better_tasks)}."
    else:
        decision_text = "Decision: Skip integration (stay with current ResNet)\nReasoning: BrainIAC does not show clear improvement on OOF AUC."
    
    print("\n" + decision_text)
    with open(os.path.join(results_dir, 'brainiac_integration_decision.txt'), 'w') as f:
        f.write(decision_text)

if __name__ == "__main__":
    train_and_evaluate()
