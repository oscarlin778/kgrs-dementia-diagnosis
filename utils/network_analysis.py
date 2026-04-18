import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets, plotting
import os
import sys

# ================= 設定區 (根據您的截圖結構優化) =================
# 自動抓取目前使用者的家目錄
HOME_DIR = os.path.expanduser("~")

# 假設資料結構: ~/Data/fMRI/...
BASE_DIR = os.path.join(HOME_DIR, "Data", "fMRI")

# 指定 AAL 圖譜路徑
CUSTOM_AAL_DIR = os.path.join(BASE_DIR, "nilearn_data")

# 指定 CSV 路徑 (嘗試多種可能性)
POSSIBLE_CSV_PATHS = [
    os.path.join(BASE_DIR, "processed_data", "dataset_index.csv"), # 標準絕對路徑
    "../processed_data/dataset_index.csv", # 相對於 script 資料夾
    "processed_data/dataset_index.csv"     # 相對於 fMRI 資料夾
]

# ================= 1. 定義 AAL 腦區到功能網路的映射 (Network Mapping) =================
# 這是核心邏輯：定義誰屬於哪個網路
NETWORK_MAPPING = {
    'DMN': [ # 預設模式網路 (記憶、自我意識)
        'Frontal_Sup_Medial', 'Frontal_Med_Orb', 'Cingulum_Ant', 'Cingulum_Post', 
        'Precuneus', 'Angular', 'Hippocampus', 'ParaHippocampal'
    ],
    'Salience': [ # 突顯網路 (注意力切換、情緒)
        'Insula', 'Cingulum_Mid', 'Frontal_Sup_Orb', 'Frontal_Mid_Orb'
    ],
    'CEN': [ # 中央執行網路 (決策、工作記憶)
        'Frontal_Sup', 'Frontal_Mid', 'Frontal_Inf_Oper', 'Frontal_Inf_Tri', 
        'Parietal_Sup', 'Parietal_Inf'
    ],
    'Sensorimotor': [ # 感覺運動網路
        'Precentral', 'Postcentral', 'Rolandic_Oper', 'Supp_Motor_Area', 'Paracentral_Lobule'
    ],
    'Visual': [ # 視覺網路
        'Calcarine', 'Cuneus', 'Lingual', 'Occipital_Sup', 'Occipital_Mid', 'Occipital_Inf', 'Fusiform'
    ],
    'Auditory': [ # 聽覺網路
        'Heschl', 'Temporal_Sup'
    ],
    'Subcortical': [ # 皮質下區域
        'Amygdala', 'Caudate', 'Putamen', 'Pallidum', 'Thalamus'
    ],
    'Cerebellum': [ # 小腦
        'Cerebelum', 'Vermis'
    ]
}

# ================= 2. 資料讀取與處理 =================
def load_data(index_file):
    print(f"📥 正在讀取索引檔: {index_file}...")
    df = pd.read_csv(index_file)
    csv_dir = os.path.dirname(os.path.abspath(index_file))
    
    # 載入 AAL 標籤
    try:
        print(f"🗺️ 嘗試載入 AAL 圖譜: {CUSTOM_AAL_DIR}")
        aal = datasets.fetch_atlas_aal(version='SPM12', data_dir=CUSTOM_AAL_DIR)
        aal_labels = [str(l) for l in aal.labels]
        print("✅ AAL 圖譜載入成功！")
    except Exception as e:
        print(f"⚠️ 無法載入 AAL: {e}")
        return None, None, None

    ad_matrices = []
    nc_matrices = []
    
    # 路徑解析器
    def resolve_path(raw_path):
        filename = os.path.basename(raw_path)
        # 1. 嘗試在 CSV 同目錄找
        p1 = os.path.join(csv_dir, filename)
        if os.path.exists(p1): return p1
        # 2. 嘗試原本的路徑
        if os.path.exists(raw_path): return raw_path
        # 3. 嘗試在 processed_data 下找
        p3 = os.path.join(BASE_DIR, "processed_data", filename)
        if os.path.exists(p3): return p3
        return None

    print("🔄 正在聚合矩陣數據...")
    for i in range(len(df)):
        mat_path = resolve_path(df.iloc[i]['matrix_path'])
        if not mat_path: continue
            
        adj = np.load(mat_path)
        np.fill_diagonal(adj, 0)
        
        if df.iloc[i]['label'] == 1:
            ad_matrices.append(adj)
        else:
            nc_matrices.append(adj)
            
    print(f"✅ 載入完成: AD={len(ad_matrices)}, NC={len(nc_matrices)}")
    return np.array(ad_matrices), np.array(nc_matrices), aal_labels

def calculate_network_matrix(matrices, aal_labels):
    """將 116x116 矩陣濃縮為 8x8 網路矩陣"""
    networks = list(NETWORK_MAPPING.keys())
    n_nets = len(networks)
    
    # 【修正】取得矩陣實際維度 (通常是 116)
    if len(matrices) > 0:
        n_matrix_nodes = matrices.shape[1]
    else:
        n_matrix_nodes = 116 # Fallback
    
    # 找出每個網路對應的 Index
    net_indices = {}
    for net_name, keywords in NETWORK_MAPPING.items():
        indices = []
        for kw in keywords:
            for i, label in enumerate(aal_labels):
                # 【修正】加入邊界檢查，防止 Index Out of Bounds
                if i >= n_matrix_nodes: 
                    continue 
                
                if kw in label:
                    indices.append(i)
        net_indices[net_name] = indices

    # 計算全組平均矩陣
    mean_matrix = np.mean(matrices, axis=0) # [116, 116]
    
    net_matrix = np.zeros((n_nets, n_nets))
    
    for i, net1 in enumerate(networks):
        for j, net2 in enumerate(networks):
            idxs1 = net_indices[net1]
            idxs2 = net_indices[net2]
            
            if not idxs1 or not idxs2: continue
            
            # 提取子矩陣
            sub_mat = mean_matrix[np.ix_(idxs1, idxs2)]
            
            if i == j: # 網路內部 (Intra-network)
                vals = sub_mat[np.triu_indices_from(sub_mat, k=1)]
                val = np.mean(vals) if len(vals) > 0 else 0
            else: # 網路之間 (Inter-network)
                val = np.mean(sub_mat)
                
            net_matrix[i, j] = val
            
    return net_matrix, networks

# ================= 3. 視覺化分析 =================
def run_analysis(index_file):
    res = load_data(index_file)
    if res[0] is None: return
    ad_mats, nc_mats, labels = res
    
    # 計算網路矩陣
    ad_net, net_names = calculate_network_matrix(ad_mats, labels)
    nc_net, _ = calculate_network_matrix(nc_mats, labels)
    
    # --- 【新增】數值透視分析 (Data Insight) ---
    print("\n🔍 關鍵網路數值透視 (AD vs NC):")
    target_nets = ['DMN', 'Salience', 'Visual', 'Auditory']
    
    for net in target_nets:
        if net in net_names:
            idx = net_names.index(net)
            # 自連接 (Intra-network)
            val_ad = ad_net[idx, idx]
            val_nc = nc_net[idx, idx]
            diff = val_ad - val_nc
            # 計算變化百分比 (相對於 NC 的變化率)
            pct_change = (diff / val_nc) * 100 if val_nc != 0 else 0
            
            print(f"   🌐 {net} Network (Internal):")
            print(f"      AD Avg: {val_ad:.4f}")
            print(f"      NC Avg: {val_nc:.4f}")
            print(f"      Diff  : {diff:.4f}")
            print(f"      Change: {pct_change:+.1f}%  <-- 這是重點！")
            print("-" * 30)
            
    # ------------------------------------------
    
    # 計算差異 (AD - NC)
    diff_net = ad_net - nc_net
    
    # --- 圖 1: 網路差異熱力圖 ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_net, xticklabels=net_names, yticklabels=net_names,
                cmap='RdBu_r', center=0, annot=True, fmt=".3f", square=True)
    plt.title('Network-Level Alterations (AD - NC)\n(Red: Increased, Blue: Decreased)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('Network_Matrix.png', dpi=300)
    plt.close()
    print("✅ 已儲存矩陣圖: Network_Matrix.png")
    
    # --- 圖 2: 3D 網路拓樸圖 ---
    print("🎨 正在繪製 3D 網路圖...")
    
    # 定義每個網路的 "重心座標" (Approximate Centroids in MNI space)
    net_coords = [
        [0, -60, 30],   # DMN (PCC/Precuneus)
        [38, 20, -4],   # Salience (Right Insula)
        [40, 40, 30],   # CEN (DLPFC)
        [0, -30, 60],   # Sensorimotor
        [0, -85, 5],    # Visual
        [-50, -20, 10], # Auditory
        [0, -10, -15],  # Subcortical
        [0, -60, -30]   # Cerebellum
    ]
    
    # 只畫出差異最大的連線 (過濾雜訊)
    threshold = np.percentile(np.abs(diff_net), 80) # 修正: 使用 diff_net
    
    fig = plt.figure(figsize=(10, 6))
    plotting.plot_connectome(
        diff_net, # 修正: 使用 diff_net
        net_coords, 
        node_color=['red', 'orange', 'gold', 'green', 'blue', 'purple', 'grey', 'brown'],
        node_size=300,
        edge_cmap='RdBu_r', edge_vmin=-0.05, edge_vmax=0.05,
        edge_threshold=threshold,
        title='Network-Level Brain Alterations',
        display_mode='lzry',
        figure=fig
    )
    plt.savefig('Network_Brain3D.png', dpi=300)
    plt.close()
    print("✅ 已儲存 3D 圖: Network_Brain3D.png")

if __name__ == "__main__":
    target_csv = None
    for p in POSSIBLE_CSV_PATHS:
        if os.path.exists(p):
            target_csv = p
            break
    
    if target_csv:
        run_analysis(target_csv)
    else:
        print("❌ 找不到 dataset_index.csv，請確認路徑。")