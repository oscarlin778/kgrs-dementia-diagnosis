import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

# ================= 1. 醫學先驗：AAL116 到 9 大功能網路 =================
NETWORK_MAP = {
    'DMN': [34, 35, 66, 67, 64, 65, 22, 23, 24, 25], 
    'SMN': [0, 1, 56, 57, 68, 69],                 
    'VN':  [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], 
    'SN':  [28, 29, 30, 31, 32, 33],                 
    'FPN': [6, 7, 58, 59, 60, 61],                   
    'LN':  [36, 37, 38, 39, 40, 41],                 
    'VAN': [10, 11, 14, 15],                         
    'BGN': [70, 71, 72, 73, 74, 75, 76, 77],         
    'CereN': list(range(90, 116))                    
}

POOLING_MAT = torch.zeros(116, 9)
for i, net in enumerate(list(NETWORK_MAP.keys())):
    for node_idx in NETWORK_MAP[net]:
        POOLING_MAT[node_idx, i] = 1.0

# ================= 2. 核心架構：功能網路池化 GNN =================
# ================= 2. 核心架構：功能網路池化 GNN (雙層深度版) =================
# ================= 2. 核心架構：功能網路池化 GNN (黃金單層版) =================
class FNPGNN(nn.Module):
    def __init__(self, input_dim=116, hidden_dim=64, dropout=0.3): # 🔥 提升到 64 維度
        super(FNPGNN, self).__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(116),
            nn.LeakyReLU(0.2)
        )
        
        # 退回單層 GNN，避免過度平滑 (Oversmoothing)
        self.gc1 = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer('pooling_mat', POOLING_MAT)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(9 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, adj):
        batch_size = x.size(0)
        
        h = self.node_encoder(x)
        h = F.leaky_relu(torch.bmm(adj, self.gc1(h)), 0.2) + h
        
        pooled_h = torch.matmul(h.transpose(1, 2), self.pooling_mat).transpose(1, 2)
        
        flat_h = pooled_h.reshape(batch_size, -1)
        logits = self.fc(flat_h)
        
        return logits, pooled_h

# ================= 3. 資料處理 (單位矩陣 + Top 30% 連線) =================
class fMRIDataset(Dataset):
    def __init__(self, dataframe): 
        self.data_cache = []

        for _, row in dataframe.iterrows():
            adj = np.load(row['matrix_path'])
            label = row['current_task_label']
            
            # 盲人摸骨法：單位矩陣
            x_feat = np.eye(116, dtype=np.float32) 
            
            adj_abs = np.abs(adj)
            np.fill_diagonal(adj_abs, 0)
            
            # 🔥 黃金閾值：保留前 30% 的連線
            k = int(116 * 0.30) 
            adj_mask = np.zeros_like(adj)
            for row_idx in range(116):
                top_indices = np.argsort(adj_abs[row_idx])[-k:]
                adj_mask[row_idx, top_indices] = adj[row_idx, top_indices]
            
            adj_mask = np.maximum(adj_mask, adj_mask.T)
            np.fill_diagonal(adj_mask, 1.0)
            
            rowsum = np.abs(adj_mask).sum(1)
            rowsum[rowsum == 0] = 1e-10
            d_mat = np.diag(np.power(rowsum, -0.5).flatten())
            adj_norm = d_mat.dot(adj_mask).dot(d_mat)
            
            self.data_cache.append({
                'x': torch.FloatTensor(x_feat),
                'adj': torch.FloatTensor(adj_norm),
                'label': torch.tensor(label, dtype=torch.long)
            })

    def __len__(self): return len(self.data_cache)
    def __getitem__(self, idx): return self.data_cache[idx]

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.0): # 🔥 關閉 Focal Loss 機制，回歸單純權重 CE
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

# ================= 4. 子任務執行器 =================
# ================= 4. 子任務執行器 (過採樣平衡版) =================
def run_task(df_full, task_pair, device):
    class_a, class_b = task_pair
    print(f"\n🎯 啟動 FNP-GNN 臨床任務: {class_a} (0) vs {class_b} (1) ...")
    
    df_task = df_full[df_full['diagnosis'].isin([class_a, class_b])].copy()
    df_task['current_task_label'] = df_task['diagnosis'].map({class_a: 0, class_b: 1})
    df_task = df_task.reset_index(drop=True)
    
    if len(df_task) < 10: return 0, None
    print(f"📊 原始資料分佈: {df_task['diagnosis'].value_counts().to_dict()}")
    
    params = {'hidden_dim': 64, 'lr': 0.002, 'dropout': 0.3}
    epochs = 70 
    
    loo = LeaveOneOut()
    all_true, all_pred = [], []
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(df_task)):
        train_df = df_task.iloc[train_idx].reset_index(drop=True)
        test_df = df_task.iloc[test_idx].reset_index(drop=True)
        
        # 🔥 終極大絕招：物理過採樣 (Oversampling)
        # 強迫少數類別複製，直到兩邊數量完全一致 1:1
        max_count = train_df['current_task_label'].value_counts().max()
        train_df_balanced = train_df.groupby('current_task_label').sample(n=max_count, replace=True).reset_index(drop=True)
        
        train_loader = DataLoader(fMRIDataset(train_df_balanced), batch_size=16, shuffle=True)
        test_loader = DataLoader(fMRIDataset(test_df), batch_size=1)
        
        model = FNPGNN(input_dim=116, hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)
        
        # 因為資料變多了，我們稍微加一點 L2 Regularization (weight_decay) 防止複製的資料造成過擬合
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 資料已經 1:1 完美平衡，不需要再用 weight 懲罰了！
        criterion = FocalLoss(alpha=None, gamma=0.0)

        model.train()
        for epoch in range(epochs): 
            for b in train_loader:
                out, _ = model(b['x'].to(device), b['adj'].to(device))
                loss = criterion(out, b['label'].to(device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()
        
        model.eval()
        with torch.no_grad():
            b = next(iter(test_loader))
            out, _ = model(b['x'].to(device), b['adj'].to(device))
            all_pred.append(out.argmax(dim=1).item())
            all_true.append(b['label'].item())
        
        if (fold + 1) % 30 == 0: 
            print(f"    ▶ 進度: {fold+1}/{len(df_task)} | 當前 Acc: {accuracy_score(all_true, all_pred)*100:.1f}%")

    acc = accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)
    return acc, cm

# ================= 5. 主程式 =================
CSV_PATHS = [
    "/home/wei-chi/Model/_dataset_mapping.csv", 
    "/home/wei-chi/Data/dataset_index_116_clean_old.csv"
]
MATRIX_DIR = "/home/wei-chi/Model/processed_116_matrices"

def main():
    print("🚀 啟動 FNP-GNN (拓撲結構強制學習版) ...")
    
    label_dict = {'NC': 0, 'MCI': 1, 'AD': 2}
    valid_data = []
    
    for csv_path in CSV_PATHS:
        if not os.path.exists(csv_path): continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            label_str = row.get('diagnosis')
            if label_str not in label_dict: continue
            
            matrix_path = None
            if 'matrix_path' in row and pd.notna(row['matrix_path']):
                matrix_path = row['matrix_path']
            elif 'new_id_base' in row and pd.notna(row['new_id_base']):
                matrix_path = os.path.join(MATRIX_DIR, f"{row['new_id_base']}_matrix_116.npy")
            elif 'Subject' in row and pd.notna(row['Subject']):
                matrix_path = os.path.join(MATRIX_DIR, f"{row['Subject']}_matrix_116.npy")

            if matrix_path and not any(d['matrix_path'] == matrix_path for d in valid_data):
                if os.path.exists(matrix_path):
                    try:
                        if np.load(matrix_path).shape == (116, 116):
                            valid_data.append({'matrix_path': matrix_path, 'label': label_dict[label_str], 'diagnosis': label_str})
                    except: pass
                
    df_full = pd.DataFrame(valid_data)
    print(f"📦 成功合併載入 {len(df_full)} 筆有效矩陣資料！")
    if len(df_full) == 0: return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tasks = [('NC', 'AD'), ('NC', 'MCI'), ('MCI', 'AD')]
    results = {}
    for task in tasks:
        acc, cm = run_task(df_full, task, device)
        if cm is not None:
            results[f"{task[0]} vs {task[1]}"] = (acc, cm)

    print("\n" + "="*60)
    print("🏆 功能網路池化 GNN 效能排行榜 (破壁版)")
    print("="*60)
    for task_name, (acc, cm) in results.items():
        print(f"📌 {task_name:<12} | 準確率: {acc*100:.1f}%")
        print(f"   混淆矩陣: \n{cm}\n")
    print("="*60)

if __name__ == "__main__":
    main()