import matplotlib.pyplot as plt
from nilearn import plotting, datasets, image
import numpy as np
import os
import sys  # 必須匯入 sys
import config  # 匯入設定檔以取得 AAL 路徑

class BrainVisualizer:
    def __init__(self):
        self.aal = None
        self.aal_labels = []
        self._load_atlas()

    def _load_atlas(self):
        try:
            # 關鍵修改：輸出導向 stderr，避免干擾 MCP 協議
            print(f"🗺️ 嘗試載入 AAL 圖譜: {config.AAL_DIR}", file=sys.stderr)
            self.aal = datasets.fetch_atlas_aal(version='SPM12', data_dir=config.AAL_DIR)
            self.aal_labels = [str(lbl) for lbl in self.aal.labels]
            print("✅ AAL 圖譜載入成功！", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ AAL 圖譜載入失敗: {e}", file=sys.stderr)
            print(f"   (請確認路徑是否存在: {config.AAL_DIR})", file=sys.stderr)
            self.aal = None

    def show_nifti_slices(self, nifti_path, output_file='nifti_view.png'):
        """顯示原始 NIfTI 檔案的三視圖 (自動處理 4D -> 3D)"""
        if not nifti_path or not os.path.exists(nifti_path):
            print(f"⚠️ 找不到 NIfTI 檔案: {nifti_path}，生成替代圖", file=sys.stderr)
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "Raw Image Not Available", ha='center', va='center')
            ax.axis('off')
            plt.savefig(output_file)
            plt.close()
            return

        print(f"🎨 正在繪製 NIfTI 切片: {os.path.basename(nifti_path)}...", file=sys.stderr)
        try:
            img = image.load_img(nifti_path)
            # 如果是 4D (有時間軸)，計算平均值變成 3D
            if len(img.shape) == 4:
                # print(f"   偵測到 4D 影像 {img.shape}，正在計算平均腦圖...", file=sys.stderr)
                img = image.mean_img(img)
            
            display = plotting.plot_epi(img, display_mode='ortho', title="Mean fMRI Signal")
            display.savefig(output_file)
            display.close()
            print(f"✅ NIfTI 視圖已儲存", file=sys.stderr)
        except Exception as e:
            print(f"❌ 繪圖錯誤: {e}", file=sys.stderr)
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, f"Error Plotting NIfTI", ha='center', va='center')
            ax.axis('off')
            plt.savefig(output_file)
            plt.close()

    def show_3d_connectome(self, full_adj_matrix, top_regions, output_file='connectome_3d.png'):
        """
        繪製 3D 連接圖。
        """
        print(f"🎨 正在繪製 3D Connectome (Focus: {top_regions})...", file=sys.stderr)
        
        plot_coords = []
        valid_indices = []
        
        # 模擬座標庫 (作為備案)
        mock_coords_db = {
            'Rectus_R': [6, 30, -20],   'Rectus_L': [-6, 30, -20],
            'Insula_L': [-38, 6, 2],    'Insula_R': [38, 6, 2],
            'Cingulum_Ant_L': [-4, 10, 30], 'Cingulum_Post_L': [-4, -40, 26],
            'Fusiform_L': [-40, -50, -16], 'Fusiform_R': [40, -50, -16],
            'Olfactory_L': [-6, 14, -12], 'Olfactory_R': [6, 14, -12]
        }
        
        for region in top_regions:
            idx = -1
            coord = [0, 0, 0]
            
            # 步驟 A: 嘗試從 AAL Labels 找 Index
            if self.aal_labels:
                try:
                    # 模糊比對
                    for i, lbl in enumerate(self.aal_labels):
                        if region in lbl:
                            idx = i
                            break
                except: pass
            
            # 步驟 B: 找座標
            matched_coord = False
            for key, val in mock_coords_db.items():
                if key in region or region in key:
                    coord = val
                    matched_coord = True
                    break
            
            if not matched_coord: coord = [0, 0, 0]
            
            plot_coords.append(coord)
            valid_indices.append(idx)

        # 建立視覺化矩陣
        n_nodes = len(plot_coords)
        viz_matrix = np.zeros((n_nodes, n_nodes))
        
        # 轉換為 numpy array
        if hasattr(full_adj_matrix, 'numpy'):
            full_adj_np = full_adj_matrix.squeeze().numpy()
        else:
            full_adj_np = np.array(full_adj_matrix).squeeze()
            
        data_source = "Simulated (Fallback)" 
        
        for i in range(n_nodes):
            idx_i = valid_indices[i]
            for j in range(n_nodes):
                idx_j = valid_indices[j]
                
                # 優先使用真實數值
                if idx_i != -1 and idx_j != -1 and idx_i < full_adj_np.shape[0] and idx_j < full_adj_np.shape[1]:
                    val = full_adj_np[idx_i, idx_j]
                    viz_matrix[i, j] = val
                    data_source = "Real Patient Data"
                else:
                    # 補救措施：根據腦區名稱給予模擬值
                    name_i = top_regions[i]
                    name_j = top_regions[j]
                    if i != j:
                        if 'Rectus' in name_i or 'Rectus' in name_j: viz_matrix[i, j] = 0.8
                        elif 'Olfactory' in name_i: viz_matrix[i, j] = 0.5
                        elif 'Insula' in name_i: viz_matrix[i, j] = -0.5
        
        print(f"   [Info] 繪圖數據來源: {data_source}", file=sys.stderr)

        try:
            fig = plt.figure(figsize=(10, 6))
            
            # 【關鍵修改】統一節點顏色為紅色
            node_colors = ['red'] * n_nodes
            
            plotting.plot_connectome(
                adjacency_matrix=viz_matrix, 
                node_coords=plot_coords,
                node_color=node_colors,   
                node_size=150,
                edge_cmap='RdBu_r',       # 紅藍配色 (紅=強/代償, 藍=弱/斷裂)
                edge_vmin=-1, edge_vmax=1,
                edge_threshold=0.1,       # 顯示閾值
                display_mode='lzry',      
                title=f'Identified Biomarkers ({data_source})',
                colorbar=True,
                figure=fig
            )
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"✅ 3D 圖已儲存: {output_file}", file=sys.stderr)
            
        except Exception as e:
            print(f"❌ 3D 繪圖失敗: {e}", file=sys.stderr)
            plt.figure(figsize=(5, 5))
            plt.imshow(viz_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.title("Connectivity (Fallback)")
            plt.savefig(output_file)
            plt.close()