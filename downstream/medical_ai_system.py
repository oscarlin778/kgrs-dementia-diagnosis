import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
import config

# 【關鍵修正】所有 print 都要加 file=sys.stderr，避免干擾 MCP
print(f"🐍 Current Python: {sys.executable}", file=sys.stderr)

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("⚠️ Critical Warning: 未偵測到 'google.generativeai' 套件。", file=sys.stderr)

from nilearn import datasets

# GNN 模型定義 (保持不變)
class PaperArchGCN(torch.nn.Module):
    def __init__(self, num_nodes=116, num_features=116, hidden_dim=64, dropout=0.5):
        super(PaperArchGCN, self).__init__()
        self.gc1 = torch.nn.Linear(num_features, hidden_dim)
        self.gc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.gc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.gc5 = torch.nn.Linear(hidden_dim * 3, hidden_dim)
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x))
        x = torch.bmm(adj, x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x))
        x = torch.bmm(adj, x)
        x = F.relu(self.gc3(x))
        x = torch.bmm(adj, x)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        embedding = torch.mean(x, dim=1)
        logits = self.fc(embedding)
        return logits

class MedicalAISystem:
    def __init__(self, api_key, model_path=None, dataset_csv=None):
        self.api_key = api_key
        self.dataset_csv = os.path.abspath(dataset_csv) if dataset_csv else None
        self.THRESHOLD = 0.3 
        
        # 1. 載入 AAL Labels
        self.aal_labels = []
        try:
            print(f"🗺️ 嘗試載入 AAL 圖譜: {config.AAL_DIR}", file=sys.stderr)
            aal = datasets.fetch_atlas_aal(version='SPM12', data_dir=config.AAL_DIR)
            self.aal_labels = [str(l) for l in aal.labels]
        except Exception as e:
            print(f"⚠️ AAL 標籤載入失敗: {e}", file=sys.stderr)

        if self.dataset_csv and os.path.exists(self.dataset_csv):
            self.df = pd.read_csv(self.dataset_csv)
            self.df['subject_id'] = self.df['subject_id'].astype(str)
            self.nc_baseline = self._calculate_nc_baseline()
        else:
            self.df = None
            self.nc_baseline = {}
        
        # 2. 初始化 LLM
        self.llm_model = None
        if HAS_GENAI:
            try:
                os.environ["GOOGLE_API_KEY"] = api_key
                genai.configure(api_key=api_key)
                self.llm_model = self._init_llm()
            except: pass
        
        # 3. 初始化 GNN
        self.gnn_model = PaperArchGCN(num_nodes=116, num_features=116)
        if model_path and os.path.exists(model_path):
            try:
                self.gnn_model.load_state_dict(torch.load(model_path))
                print(f"✅ 模型權重載入成功", file=sys.stderr)
            except: print("⚠️ 權重載入失敗，使用隨機初始化。", file=sys.stderr)
        self.gnn_model.eval()

        # 4. 知識庫
        self.knowledge_base = {
            "Rectus": {"desc": "直回", "pathology": "前額葉代償機制", "ref": "Zhong et al. (2014)"},
            "Insula": {"desc": "腦島", "pathology": "突顯網路崩潰", "ref": "Han et al. (2024)"},
            "Fusiform": {"desc": "梭狀迴", "pathology": "視覺網路退化", "ref": "Frontiers (2025)"},
            "Olfactory": {"desc": "嗅覺皮質", "pathology": "早期病理特徵", "ref": "Clinical Consensus"},
            "Cingulum": {"desc": "扣帶迴", "pathology": "DMN 功能衰退", "ref": "Zhong et al. (2014)"},
            "DMN": {"desc": "預設模式網路", "pathology": "內部連接性顯著下降，為 AD 核心病理特徵", "ref": "Network Analysis Result"},
            "Salience": {"desc": "突顯網路", "pathology": "功能性整合能力減弱，影響注意力切換", "ref": "Han et al."},
            "Auditory-Subcortical": {"desc": "聽覺-皮質下迴路", "pathology": "異常增強 (Hyper-connectivity)，顯示代償或抑制失控", "ref": "Network Analysis Result"}
        }

    def _init_llm(self):
        try:
            target_models = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-pro']
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            for target in target_models:
                for av in available_models:
                    if target in av:
                        print(f"✅ 自動選定 LLM: {av}", file=sys.stderr)
                        return genai.GenerativeModel(av)
            if available_models: return genai.GenerativeModel(available_models[0])
        except: pass
        return None

    def _calculate_nc_baseline(self):
        if self.df is None or not self.aal_labels: return {}
        nc_rows = self.df[self.df['label'] == 0]
        nc_matrices = []
        csv_dir = os.path.dirname(self.dataset_csv)
        for _, row in nc_rows.iterrows():
            fname = os.path.basename(row['matrix_path'])
            paths = [os.path.join(csv_dir, fname), row['matrix_path'], os.path.join(config.PROCESSED_DATA_DIR, fname)]
            for p in paths:
                if os.path.exists(p):
                    try:
                        mat = np.load(p)
                        np.fill_diagonal(mat, 0)
                        nc_matrices.append(mat)
                    except: pass
                    break
        if not nc_matrices: return {}
        mean_nc_matrix = np.mean(nc_matrices, axis=0)
        baseline_scores = {}
        for net_name, regions in config.NETWORK_MAPPING.items() if hasattr(config, 'NETWORK_MAPPING') else {}.items():
            indices = [i for i, label in enumerate(self.aal_labels) if i < 116 and any(r in label for r in regions)]
            if not indices: continue
            sub_mat = mean_nc_matrix[np.ix_(indices, indices)]
            val = np.mean(sub_mat[np.triu_indices_from(sub_mat, k=1)])
            baseline_scores[net_name] = val
        print(f"📊 NC 基準線建立完成 (N={len(nc_matrices)})", file=sys.stderr)
        return baseline_scores

    def _analyze_patient_networks(self, patient_adj):
        if not self.nc_baseline or not self.aal_labels: return []
        analysis_results = []
        patient_adj = patient_adj.squeeze()
        if torch.is_tensor(patient_adj): patient_adj = patient_adj.numpy()
        
        # 簡單定義網路映射 (防止 config 沒定義)
        NET_MAP = {
            'DMN': ['Frontal_Sup_Medial', 'Cingulum_Ant', 'Cingulum_Post', 'Precuneus', 'Angular', 'Hippocampus'],
            'Salience': ['Insula', 'Cingulum_Mid'],
            'Visual': ['Calcarine', 'Cuneus', 'Fusiform'],
            'Auditory': ['Heschl', 'Temporal_Sup']
        }
        
        for net_name, regions in NET_MAP.items():
            indices = [i for i, label in enumerate(self.aal_labels) if i < 116 and any(r in label for r in regions)]
            if not indices: continue
            sub_mat = patient_adj[np.ix_(indices, indices)]
            pat_val = np.mean(sub_mat[np.triu_indices_from(sub_mat, k=1)])
            nc_val = self.nc_baseline.get(net_name, 0)
            if nc_val == 0: continue
            diff_pct = (pat_val - nc_val) / nc_val * 100
            if diff_pct < -15:
                analysis_results.append(f"**{net_name}** 連結強度下降 {abs(diff_pct):.1f}% (相較於正常值)，提示功能衰退。")
            elif diff_pct > 15:
                analysis_results.append(f"**{net_name}** 連結強度上升 {diff_pct:.1f}%，可能涉及代償機制。")
        return analysis_results

    def get_patient_data(self, patient_id):
        if self.df is None: return None
        row = self.df[self.df['subject_id'] == patient_id]
        if row.empty:
            mask = self.df['subject_id'].str.contains(patient_id, case=False)
            row = self.df[mask]
            if row.empty:
                print(f"❌ 找不到 ID '{patient_id}'", file=sys.stderr)
                return None
            else:
                print(f"ℹ️ 自動匹配至: {row.iloc[0]['subject_id']}", file=sys.stderr)
        row = row.iloc[0]
        csv_dir = os.path.dirname(self.dataset_csv)
        def resolve_path(raw_path):
            filename = os.path.basename(raw_path)
            paths = [os.path.join(csv_dir, filename), raw_path, os.path.join(config.PROCESSED_DATA_DIR, filename)]
            for p in paths:
                if os.path.exists(p): return p
            return None
        ts_path = resolve_path(row['timeseries_path'])
        mat_path = resolve_path(row['matrix_path'])
        if not ts_path or not mat_path: return None
        try:
            raw_adj = np.load(mat_path)
            adj_processed = raw_adj.copy()
            adj_processed[np.abs(adj_processed) < self.THRESHOLD] = 0
            np.fill_diagonal(adj_processed, 1.0)
            node_features = adj_processed
            nifti_path = ts_path.replace('_timeseries.npy', '.nii')
            if not os.path.exists(nifti_path):
                if os.path.exists(nifti_path + ".gz"): nifti_path += ".gz"
            return {
                'id': row['subject_id'],
                'x': torch.FloatTensor(node_features).unsqueeze(0),
                'adj': torch.FloatTensor(adj_processed).unsqueeze(0),
                'nifti_path': nifti_path,
                'label': row['label']
            }
        except: return None

    def predict(self, patient_data):
        with torch.no_grad():
            logits = self.gnn_model(patient_data['x'], patient_data['adj'])
            probs = F.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)
            return pred_class.item(), confidence.item() * 100

    def generate_report(self, patient_id, pred, conf, top_regions):
        patient_data = self.get_patient_data(patient_id)
        net_analysis_text = ""
        if patient_data:
            net_findings = self._analyze_patient_networks(patient_data['adj'])
            if net_findings:
                net_analysis_text = "\n".join(net_findings)
            else:
                net_analysis_text = "各大腦網路連接強度在正常範圍內。"
        context_list = []
        for region in top_regions:
            matched = False
            for key, info in self.knowledge_base.items():
                if key in region:
                    context_list.append(f"- **{info['desc']}**: {info['pathology']} ({info['ref']})")
                    matched = True
                    break
            if not matched:
                context_list.append(f"- {region}")
        context_str = "\n".join(context_list)
        diag_str = "阿茲海默症 (AD)" if pred == 1 else "正常老化 (NC)"
        if self.llm_model:
            prompt = f"""
            你是一位資深神經內科醫師。請根據以下數據撰寫「AI 輔助診斷報告」。
            【病患數據】
            - 編號: {patient_id}
            - AI 預測: {diag_str}
            - 模型信心度: {conf:.1f}%
            - 關鍵異常腦區 (GNN Saliency): {', '.join(top_regions)}
            【網路層級分析 (Network Analysis)】
            {net_analysis_text}
            【醫學證據資料庫】
            {context_str}
            請用繁體中文撰寫。
            """
            try:
                print("🤖 呼叫 LLM 撰寫報告...", file=sys.stderr)
                return self.llm_model.generate_content(prompt).text
            except:
                return "❌ API Error"
        return "Demo Mode Report..."