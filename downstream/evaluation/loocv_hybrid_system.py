import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from neuroCombat import neuroCombat
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("/home/wei-chi/Alzheimers_Project/external_data/scripts/downstream")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --- Models ---
class PCAGFusion(nn.Module):
    def __init__(self, fmri_dim=1280, smri_dim=768, fusion_dim=20, num_classes=2):
        super().__init__()
        self.fmri_proj = nn.Linear(fmri_dim, fusion_dim)
        self.smri_proj_k = nn.Linear(smri_dim, fusion_dim)
        self.smri_proj_v = nn.Linear(smri_dim, fusion_dim)
        self.W_e = nn.Linear(fusion_dim, fusion_dim)
        self.W_g1 = nn.Linear(fusion_dim, fusion_dim)
        self.W_g2 = nn.Linear(fusion_dim, fusion_dim)
        self.ln_e = nn.LayerNorm(fusion_dim)
        self.ln_g = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim, num_classes))
        
    def forward(self, fmri_emb, smri_feat):
        Q = self.fmri_proj(fmri_emb); K = self.smri_proj_k(smri_feat); V = self.smri_proj_v(smri_feat)
        P = (torch.tanh(Q) * torch.tanh(K) + 1) / 2
        A_gated = (Q * K) * P
        S = torch.sigmoid(A_gated); V_hat = S * V
        E = F.relu(self.W_e(V_hat)); G = F.relu(self.W_g1(V_hat) + self.W_g2(Q))
        C = self.ln_e(E) * self.ln_g(G)
        return self.classifier(C + F.relu(Q))

def train_fusion_head(train_gnn, train_smri, train_labels, epochs=20):
    model = PCAGFusion().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    gnn_t = torch.tensor(train_gnn, dtype=torch.float32).to(DEVICE)
    smri_t = torch.tensor(train_smri, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(train_labels, dtype=torch.long).to(DEVICE)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(gnn_t, smri_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
    return model

def run_loocv(task_name, tpmic_data, adni_data, sid_to_feat):
    print(f"\n--- LOOCV for {task_name} ---", flush=True)
    
    t_sids = tpmic_data['subject_ids']
    t_embs = tpmic_data['embeddings']
    t_bin_labels = tpmic_data['bin_labels']
    
    a_sids = adni_data['subject_ids']
    a_embs = adni_data['embeddings']
    a_bin_labels = adni_data['bin_labels']
    
    # Identify which subjects have sMRI and their sites
    def get_meta(sid_list, site_name):
        feats, sites, has_smri = [], [], []
        for sid in sid_list:
            if sid in sid_to_feat:
                feats.append(sid_to_feat[sid])
                has_smri.append(True)
            else:
                feats.append(np.zeros(768)) # Placeholder
                has_smri.append(False)
            sites.append(site_name)
        return np.array(feats), np.array(sites), np.array(has_smri)

    t_feats, t_sites, t_has_smri = get_meta(t_sids, 'TPMIC')
    a_feats, a_sites, a_has_smri = get_meta(a_sids, 'ADNI')

    loocv_probs = []
    
    for i in range(len(t_sids)):
        if (i+1) % 2 == 0: print(f"  [{task_name}] Iteration: {i+1}/{len(t_sids)}", flush=True)
        
        # Split TPMIC
        test_sid = t_sids[i]
        test_emb = t_embs[i:i+1]
        test_label = t_bin_labels[i]
        test_feat = t_feats[i:i+1]
        test_site = t_sites[i:i+1]
        test_has_smri = t_has_smri[i]
        
        train_t_embs = np.delete(t_embs, i, axis=0)
        train_t_labels = np.delete(t_bin_labels, i)
        train_t_feats = np.delete(t_feats, i, axis=0)
        train_t_sites = np.delete(t_sites, i)
        train_t_has_smri = np.delete(t_has_smri, i)
        
        # Combine with ADNI for training
        train_pool_embs = np.concatenate([train_t_embs, a_embs], axis=0)
        train_pool_labels = np.concatenate([train_t_labels, a_bin_labels])
        train_pool_feats = np.concatenate([train_t_feats, a_feats], axis=0)
        train_pool_sites = np.concatenate([train_t_sites, a_sites])
        train_pool_has_smri = np.concatenate([train_t_has_smri, a_has_smri])

        # 1. ComBat
        if task_name != "MCI_vs_AD":
            idx_with_smri = np.where(train_pool_has_smri)[0]
            combat_pool_feats = train_pool_feats[idx_with_smri]
            combat_pool_sites = train_pool_sites[idx_with_smri]
            combat_pool_labels = train_pool_labels[idx_with_smri]
            s_map = {'TPMIC': 0, 'ADNI': 1}
            b_idx = [s_map[s] for s in combat_pool_sites]
            dat = combat_pool_feats.T
            covars = pd.DataFrame({'site': b_idx, 'label': combat_pool_labels})
            if test_has_smri:
                dat = np.concatenate([dat, test_feat.T], axis=1)
                covars = pd.concat([covars, pd.DataFrame({'site': [s_map[test_site[0]]], 'label': [test_label]})], ignore_index=True)
            
            # Run neuroCombat silently
            combat_out = neuroCombat(dat=dat, covars=covars, batch_col='site', categorical_cols=['label'])
            harmonized = combat_out['data'].T
            
            if test_has_smri:
                adj_train_smri = harmonized[:-1]
                adj_test_smri = harmonized[-1:]
            else:
                adj_train_smri = harmonized
                adj_test_smri = test_feat
            
            fusion_train_gnn = train_pool_embs[idx_with_smri]
            fusion_train_labels = train_pool_labels[idx_with_smri]
            fusion_train_smri = adj_train_smri
        
        # 2. Train Models
        lr = LogisticRegression(max_iter=1000)
        lr.fit(train_pool_embs, train_pool_labels)
        if task_name != "MCI_vs_AD":
            fusion_model = train_fusion_head(fusion_train_gnn, fusion_train_smri, fusion_train_labels)
        
        # 3. Prediction
        if task_name == "MCI_vs_AD":
            prob = lr.predict_proba(test_emb)[0, 1]
        else:
            if test_has_smri:
                fusion_model.eval()
                with torch.no_grad():
                    gnn_v = torch.tensor(test_emb, dtype=torch.float32).to(DEVICE)
                    smri_v = torch.tensor(adj_test_smri, dtype=torch.float32).to(DEVICE)
                    logits = fusion_model(gnn_v, smri_v)
                    prob = F.softmax(logits, dim=1)[0, 1].item()
            else:
                prob = lr.predict_proba(test_emb)[0, 1]
        loocv_probs.append(prob)
        
    loocv_probs = np.array(loocv_probs)
    auc = roc_auc_score(t_bin_labels, loocv_probs)
    fpr, tpr, thresholds = roc_curve(t_bin_labels, loocv_probs)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    preds = (loocv_probs >= best_thresh).astype(int)
    cm = confusion_matrix(t_bin_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": float(auc), "sens": float(tp / (tp + fn)), "spec": float(tn / (tn + fp)),
        "f1": float(f1_score(t_bin_labels, preds)), "n": len(t_sids)
    }

def main():
    print("🚀 LOOCV Script Started", flush=True)
    with open(BASE_DIR / "sid_to_smri_feat.pkl", "rb") as f:
        sid_to_feat = pickle.load(f)
    print("✅ sMRI Feature mapping loaded", flush=True)
    
    tasks = ['NC_vs_AD', 'NC_vs_MCI', 'MCI_vs_AD']
    loocv_results = {}
    for task in tasks:
        t_data = np.load(BASE_DIR / "embeddings" / f"gnn_embeddings_tpmic_{task}.npz")
        a_data = np.load(BASE_DIR / "embeddings" / f"gnn_embeddings_adni_{task}.npz")
        loocv_results[task] = run_loocv(task, t_data, a_data, sid_to_feat)
        print(f"✅ {task} LOOCV Done. AUC: {loocv_results[task]['auc']:.4f}", flush=True)
        
    with open(RESULTS_DIR / "loocv_metrics.json", "w") as f:
        json.dump(loocv_results, f, indent=2)
    print(f"📊 Metrics saved to {RESULTS_DIR / 'loocv_metrics.json'}", flush=True)
        
    # Plotting
    plot_data = []
    def get_8020_auc(task, model_type):
        try:
            if model_type == 'KD':
                with open(RESULTS_DIR / "kd_comprehensive_metrics.json", "r") as f:
                    return json.load(f)[task]['kd']['auc']
            elif model_type == 'PCAG':
                with open(RESULTS_DIR / f"pcag_combat_{task}_results.json", "r") as f:
                    return json.load(f)['test_metrics']['auc']
            elif model_type == 'Hybrid':
                with open(RESULTS_DIR / "hybrid_system_combat_metrics.json", "r") as f:
                    return json.load(f)[task]['hybrid']['auc']
        except: return 0.0
        return 0.0

    for task in tasks:
        plot_data.append({'Task': task, 'Method': 'Hybrid (LOOCV)', 'AUC': loocv_results[task]['auc']})
        plot_data.append({'Task': task, 'Method': 'KD (80/20)', 'AUC': get_8020_auc(task, 'KD')})
        plot_data.append({'Task': task, 'Method': 'PCAG (80/20)', 'AUC': get_8020_auc(task, 'PCAG')})
        plot_data.append({'Task': task, 'Method': 'Hybrid (80/20)', 'AUC': get_8020_auc(task, 'Hybrid')})
        
    df_plot = pd.DataFrame(plot_data)
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    sns.barplot(data=df_plot, x="Task", y="AUC", hue="Method")
    plt.title("LOOCV vs 80/20 Split AUC Comparison")
    plt.ylim(0, 1.1)
    save_path = FIGURES_DIR / "LOOCV_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"📈 Plot saved to {save_path}", flush=True)

if __name__ == "__main__":
    main()
