import os
import time
import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ================= 1. 路徑與參數設定 =================
BASE_DIR = "/home/wei-chi/Data"

# 【致勝關鍵：只讀取兩份「血統最純正」的乾淨資料】
CSV_PATHS = [
    os.path.join(BASE_DIR, "dataset_index_116_clean.csv"),      # 52 筆乾淨新資料
    os.path.join(BASE_DIR, "dataset_index_116_clean_old.csv")   # 32 筆剛洗好的乾淨舊資料
]

OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, "script", "best_merged_gnn.pth")

BATCH_SIZE = 8
EPOCHS = 80  

# ================= 2. Focal Loss (對抗類別不平衡) =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# ================= 3. 升級版模型: Brain-Aware Attention GCN =================
class BrainAttentionGCN(nn.Module):
    def __init__(self, num_nodes=116, num_features=116, hidden_dim=64, dropout=0.6):
        super(BrainAttentionGCN, self).__init__()
        
        # 多層圖卷積 (保持 Dense 結構擷取豐富特徵)
        self.gc1 = nn.Linear(num_features, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 【致勝武器：節點注意力層 (Node Attention Layer)】
        # 學習為 116 個腦區分別打分數，找出對 AD 最敏感的腦區
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.bn = nn.BatchNorm1d(hidden_dim * 3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        self.dropout = dropout

    def graph_conv(self, x, adj, linear_layer):
        x = linear_layer(x)      
        x = torch.bmm(adj, x)    
        return x

    def forward(self, x, adj):
        # 提取圖特徵
        h1 = F.dropout(F.relu(self.graph_conv(x, adj, self.gc1)), self.dropout, training=self.training)
        h2 = F.dropout(F.relu(self.graph_conv(h1, adj, self.gc2)), self.dropout, training=self.training)
        h3 = F.dropout(F.relu(self.graph_conv(h2, adj, self.gc3)), self.dropout, training=self.training)
        
        # Dense Connection: 結合不同感受野的特徵
        combined = torch.cat([h1, h2, h3], dim=2) # Shape: [Batch, 116_Nodes, Hidden*3]
        
        # BatchNorm
        combined = combined.permute(0, 2, 1) 
        combined = self.bn(combined)
        combined = combined.permute(0, 2, 1)
        
        # --- 【關鍵修改：Self-Attention Pooling 替換 Global Mean Pooling】 ---
        # 1. 計算每個腦區的注意力分數 (Attention Scores)
        attn_weights = self.attention_layer(combined) # Shape: [Batch, 116, 1]
        attn_weights = F.softmax(attn_weights, dim=1) # 確保 116 個腦區的分數加起來為 1
        
        # 2. 加權總和 (Weighted Sum)：重要的腦區特徵會被放大，雜訊腦區會被歸零
        graph_embedding = torch.sum(combined * attn_weights, dim=1) # Shape: [Batch, Hidden*3]
        
        # 最後的分類器
        logits = self.fc(graph_embedding)
        return logits

# ================= 4. 資料準備 (加入閾值過濾) =================
class fMRIDataset(Dataset):
    def __init__(self, dataframe, threshold=0.3):
        self.df = dataframe.reset_index(drop=True)
        self.data_cache = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            adj_raw = np.load(row['matrix_path'])
            
            x_feat = adj_raw.copy()
            np.fill_diagonal(x_feat, 1.0)
            
            adj_mask = adj_raw.copy()
            adj_mask[np.abs(adj_mask) < threshold] = 0
            np.fill_diagonal(adj_mask, 1.0) 
            
            adj_abs = np.abs(adj_mask)
            rowsum = np.array(adj_abs.sum(1))
            rowsum[rowsum == 0] = 1e-10
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_mat_inv_sqrt = np.diag(d_inv_sqrt)
            adj_norm = adj_mask.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            
            self.data_cache.append({
                'x': torch.FloatTensor(x_feat),
                'adj': torch.FloatTensor(adj_norm), 
                'label': torch.tensor(row['label'], dtype=torch.long)
            })

    def __len__(self):
        return len(self.data_cache)
        
    def __getitem__(self, idx):
        return self.data_cache[idx]

# ================= 5. 評估函數 =================
def evaluate_model(df, params, cv_type='kfold', device='cpu'):
    dataset = fMRIDataset(df, threshold=params['threshold'])
    cv = KFold(n_splits=5, shuffle=True, random_state=42) if cv_type == 'kfold' else LeaveOneOut()

    all_true_labels = []
    all_pred_labels = []
    total_splits = cv.get_n_splits(df)
    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(cv.split(df)):
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        test_ds = torch.utils.data.Subset(dataset, test_idx)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=1 if cv_type == 'loocv' else BATCH_SIZE, shuffle=False)

        # 改用我們全新的注意力 GNN
        model = BrainAttentionGCN(num_nodes=116, num_features=116, hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-3)
        
        train_labels = df.iloc[train_idx]['label'].values
        classes = np.unique(train_labels)
        if len(classes) == 2:
            class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        else:
            class_weights = np.array([1.0, 1.0])
            
        weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)

        test_preds_history = [] 
        
        for epoch in range(EPOCHS):
            model.train()
            for batch in train_loader:
                x = batch['x'].to(device)
                adj = batch['adj'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                out = model(x, adj)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            if epoch >= EPOCHS - 5:
                model.eval()
                epoch_preds = []
                with torch.no_grad():
                    for batch in test_loader:
                        x = batch['x'].to(device)
                        adj = batch['adj'].to(device)
                        out = model(x, adj)
                        pred = out.argmax(dim=1).cpu().numpy()
                        epoch_preds.extend(pred)
                test_preds_history.append(epoch_preds)

        test_preds_history = np.array(test_preds_history) 
        final_preds = [np.bincount(test_preds_history[:, i]).argmax() for i in range(test_preds_history.shape[1])]
        
        true_labels = df.iloc[test_idx]['label'].values
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(final_preds)
        
        if cv_type == 'loocv' and (fold + 1) % 15 == 0:
            print(f"    ▶ LOOCV 進度: {fold+1:02d}/{total_splits} 輪 | 耗時: {time.time() - start_time:.1f}s")

    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1])
    
    return accuracy, cm

# ================= 6. 主流程 (純淨資料大融合) =================
def main():
    print(f"🔍 正在讀取並合併【血統純正】的兩批資料集...")
    
    valid_records = []
    for csv_file in CSV_PATHS:
        if not os.path.exists(csv_file):
            print(f"  ⚠️ 找不到檔案: {csv_file}")
            continue
            
        print(f"  📥 讀取: {csv_file}")
        df_raw = pd.read_csv(csv_file)
        
        for i in range(len(df_raw)):
            row = df_raw.iloc[i]
            # 只保留 NC 和 AD
            if pd.isna(row.get('label')) or row['label'] not in [0, 1]: 
                continue 
            
            valid_path = row['matrix_path']
            if os.path.exists(valid_path):
                try:
                    adj = np.load(valid_path)
                    # 嚴格過濾確保都是 116x116
                    if adj.shape == (116, 116): 
                        valid_records.append({
                            'subject_id': str(row.get('subject_id')),
                            'diagnosis': 'NC' if row['label'] == 0 else 'AD',
                            'label': int(row['label']),
                            'matrix_path': valid_path
                        })
                except: pass

    if not valid_records:
        print("❌ 找不到任何有效的矩陣資料！")
        return
        
    df_combined = pd.DataFrame(valid_records)
    
    total_samples = len(df_combined)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print(f"🚀 啟動 BrainAttentionGCN - 【大腦注意力機制升級版】")
    print(f"📊 融合後資料分佈: {df_combined['diagnosis'].value_counts().to_dict()} (共 {total_samples} 筆)")
    print(f"💻 使用硬體: {device}")
    print("="*60)
    
    print(f"\n🔍 啟動 Hyperparameter Grid Search (5-Fold CV)...")
    param_grid = {
        'threshold': [0.3, 0.4, 0.5],
        'hidden_dim': [32, 64],
        'dropout': [0.5, 0.6, 0.7],
        'lr': [0.001, 0.005]
    }
    
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_acc = 0
    best_params = None
    
    for i, params in enumerate(experiments):
        print(f"  [{i+1}/{len(experiments)}] T={params['threshold']}, HD={params['hidden_dim']}, DP={params['dropout']}, LR={params['lr']} ...", end=" ")
        acc, _ = evaluate_model(df_combined, params, cv_type='kfold', device=device)
        print(f"Acc: {acc*100:.1f}%")
        if acc > best_acc:
            best_acc = acc
            best_params = params

    print("\n" + "★"*60)
    print(f"🏆 Grid Search 完成！最佳參數組合 (5-Fold Acc: {best_acc*100:.1f}%):")
    print(f"   {best_params}")
    print("★"*60)
    
    print(f"\n🚀 使用最佳參數啟動最嚴格的 LOOCV 驗證...")
    loocv_acc, cm = evaluate_model(df_combined, best_params, cv_type='loocv', device=device)
    
    print("\n" + "="*60)
    print(f"🎉 最終 LOOCV 驗證完成！準確率: {loocv_acc*100:.1f}%")
    print("="*60)
    
    print("\n📊 【混淆矩陣分析 (Confusion Matrix)】")
    print(f"                 預測 NC (0) | 預測 AD (1)")
    print(f"實際 NC (0) |      {cm[0,0]:<9} |      {cm[0,1]:<7}")
    print(f"實際 AD (1) |      {cm[1,0]:<9} |      {cm[1,1]:<7}")

    # 儲存模型
    dataset = fMRIDataset(df_combined, threshold=best_params['threshold'])
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = BrainAttentionGCN(num_nodes=116, num_features=116, hidden_dim=best_params['hidden_dim'], dropout=best_params['dropout']).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=1e-3)
    
    labels_all = df_combined['label'].values
    classes = np.unique(labels_all)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels_all)
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion_final = FocalLoss(alpha=weights_tensor, gamma=2.0)

    final_model.train()
    for epoch in range(EPOCHS):
        for batch in full_loader:
            x = batch['x'].to(device)
            adj = batch['adj'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            out = final_model(x, adj)
            loss = criterion_final(out, labels)
            loss.backward()
            optimizer.step()

    torch.save(final_model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"\n💾 模型已儲存至: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()