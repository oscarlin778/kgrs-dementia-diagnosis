import React, { useState, useEffect, useRef } from 'react';
import { Activity, Brain, FileText, Loader2, SlidersHorizontal, Zap, Lock, AlertTriangle } from 'lucide-react';
import { Niivue } from '@niivue/niivue';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from 'recharts';

// ── 9 大功能網路定義 (AAL116, 0-based index) ──
const NETWORK_AAL_MAP = {
  'DMN': [22,23,24,25,26,27,32,33,34,35,36,37,38,39,64,65,66,67,84,85,88,89],
  'SMN': [0,1,16,17,18,19,56,57,58,59,68,69],
  'DAN': [8,9,58,59,60,61,64,65],
  'VAN': [10,11,12,13,14,15,28,29,62,63,80,81],
  'LIM': [20,21,36,37,38,39,40,41,82,83,86,87],
  'FPN': [4,5,6,7,48,49,50,51,60,61],
  'VIS': [42,43,44,45,46,47,48,49,50,51,52,53,54,55],
  'SUB': [70,71,72,73,74,75,76,77],
  'CER': Array.from({length: 26}, (_, i) => 90 + i)
};

// ── Saliency Radar Chart 元件 ──
function SaliencyRadarChart({ saliencyData, labels }) {
  if (!saliencyData || !labels) return null;

  // 將 116 維 saliency 聚合至 9 大網絡
  const chartData = labels.map(net => {
    const indices = NETWORK_AAL_MAP[net] || [];
    const values = indices.map(idx => saliencyData[idx] || 0);
    const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    return { subject: net, A: avg * 100 }; // 放大一點方便觀察
  });

  return (
    <div className="w-full h-40">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData}>
          <PolarGrid stroke="#475569" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10 }} />
          <Radar
            name="Saliency"
            dataKey="A"
            stroke="#10b981"
            fill="#10b981"
            fillOpacity={0.5}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 9×9 Network Connectivity 熱力圖元件 ──
function NetworkHeatmap({ matrix, labels }) {
  if (!matrix || !matrix.length || !labels) return null;

  const flat = matrix.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);

  const toColor = (val) => {
    const t = (val - min) / (max - min || 1);
    const r = Math.round(20 + t * 210);
    const g = Math.round(40 + t * 60);
    const b = Math.round(180 - t * 160);
    return `rgb(${r},${g},${b})`;
  };

  const cell = 26;

  return (
    <div className="overflow-auto">
      <div className="flex" style={{ marginLeft: cell + 36 }}>
        {labels.map((l, j) => (
          <div key={j} style={{ width: cell, fontSize: 9 }}
            className="text-slate-400 text-center truncate">{l}</div>
        ))}
      </div>
      {matrix.map((row, i) => (
        <div key={i} className="flex items-center">
          <span className="text-slate-400 text-right pr-1"
            style={{ width: 36, fontSize: 9 }}>{labels[i]}</span>
          {row.map((val, j) => (
            <div
              key={j}
              style={{ width: cell, height: cell, backgroundColor: toColor(val), flexShrink: 0 }}
              title={`${labels[i]} ↔ ${labels[j]}: ${val.toFixed(3)}`}
              className="border border-slate-900"
            />
          ))}
        </div>
      ))}
      <div className="flex items-center gap-2 mt-2 px-1">
        <span className="text-xs text-slate-500">弱</span>
        <div className="flex-1 h-2 rounded" style={{
          background: 'linear-gradient(to right, rgb(20,40,180), rgb(230,100,20))'
        }} />
        <span className="text-xs text-slate-500">強</span>
      </div>
    </div>
  );
}

// ── NiiVue 3D 腦圖渲染元件 ──
function NiiVueViewer({ t1Url, overlayUrl, showOverlay }) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);

  useEffect(() => {
    if (!t1Url || !canvasRef.current) return;
    const nv = new Niivue({
      backColor: [0.1, 0.1, 0.15, 1],
      show3Dcrosshair: false,
    });
    nvRef.current = nv;
    nv.attachToCanvas(canvasRef.current);

    const volumes = [{ url: t1Url, colorMap: 'gray', opacity: 1 }];
    if (overlayUrl && showOverlay) {
      volumes.push({ url: overlayUrl, colorMap: 'redyell', opacity: 0.5 });
    }
    nv.loadVolumes(volumes).catch(err =>
      console.error('[NiiVue] 載入影像失敗:', err)
    );
    return () => {
      nvRef.current = null;
    };
  }, [t1Url, overlayUrl, showOverlay]);

  if (!t1Url) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-4">
        <AlertTriangle className="w-10 h-10 text-slate-600 mb-3" />
        <p className="text-slate-500 text-sm leading-relaxed">
          查無結構影像 (sMRI)
          <br />
          <span className="text-slate-600 text-xs">切換至單模態分析模式</span>
        </p>
      </div>
    );
  }

  return (
    <canvas ref={canvasRef} className="w-full h-full rounded" style={{ minHeight: '180px' }} />
  );
}

export default function App() {
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('');
  const [reportMode, setReportMode] = useState('detailed');
  const [analyzeData, setAnalyzeData] = useState(null);
  const [reportText, setReportText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showSaliency, setShowSaliency] = useState(false);

  const currentPatient = patients.find(p => p.id === selectedPatientId) || null;
  const hasSmri = Boolean(currentPatient?.t1_path);
  const hasFmri = Boolean(currentPatient?.fmri_valid);
  const t1Url = currentPatient?.t1_url || '';
  const saliencyUrl = analyzeData?.smri_saliency_url || '';

  useEffect(() => {
    setAnalyzeData(null);
    setReportText('');
    setShowSaliency(false);
  }, [selectedPatientId]);

  useEffect(() => {
    fetch('http://localhost:8080/api/v1/patients')
      .then(res => res.json())
      .then(data => {
        setPatients(data.patients);
        if (data.patients.length > 0) setSelectedPatientId(data.patients[0].id);
      })
      .catch(err => console.error('無法載入病患清單:', err));
  }, []);

  const handleAnalyze = async () => {
    if (!currentPatient) return;
    setIsAnalyzing(true);
    setAnalyzeData(null);
    setReportText('');

    try {
      const formData = new FormData();
      formData.append('subject_id', currentPatient.id);
      formData.append('matrix_path', currentPatient.matrix_path);
      formData.append('t1_path', currentPatient.t1_path || '');
      // Removed manual weight, backend handles modality priority automatically

      const response = await fetch('http://localhost:8080/api/v1/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setAnalyzeData({ ...data, subject_id: currentPatient.id });
    } catch (error) {
      alert('分析失敗，請確認後端 API 伺服器狀態！\n' + error);
    }
    setIsAnalyzing(false);
  };

  const handleGenerateReport = async () => {
    if (!analyzeData) return;
    setIsGenerating(true);
    setReportText('');

    try {
      const response = await fetch('http://localhost:8080/api/v1/report/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject_id: analyzeData.subject_id,
          task_results: analyzeData.task_results,
          kg_context: analyzeData.kg_context,
          mode: reportMode,
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        setReportText(prev => prev + decoder.decode(value));
      }
    } catch {
      setReportText('⚠️ 報告生成失敗。');
    }
    setIsGenerating(false);
  };

  // ── 新增：計算最終判定結果 (One-vs-One Voting) ──
  const getFinalDiagnosis = () => {
    if (!analyzeData || !analyzeData.task_results) return null;
    const res = analyzeData.task_results;
    
    // 1. 初始化投票箱
    const votes = { NC: 0, MCI: 0, AD: 0 };

    // 2. 使用後端已套用 INFERENCE_THRESHOLDS 的 prediction 欄位投票
    if (res['NC vs AD'])  votes[res['NC vs AD'].prediction  === 1 ? 'AD'  : 'NC']++;
    if (res['NC vs MCI']) votes[res['NC vs MCI'].prediction === 1 ? 'MCI' : 'NC']++;
    if (res['MCI vs AD']) votes[res['MCI vs AD'].prediction === 1 ? 'AD'  : 'MCI']++;

    // 3. 找出最高票
    let maxVote = -1;
    let finalClass = '';
    for (const [cls, count] of Object.entries(votes)) {
      if (count > maxVote) {
        maxVote = count;
        finalClass = cls;
      }
    }

    // 4. 處理罕見的 1:1:1 平手狀況 (依據 NC vs AD 任務來決斷)
    if (maxVote === 1 && res['NC vs AD']) {
        finalClass = res['NC vs AD'].prediction === 1 ? 'AD' : 'NC';
    }

    // 5. 根據結果動態給予警示顏色
    const colorClass = 
      finalClass === 'AD' ? 'text-red-400' : 
      finalClass === 'MCI' ? 'text-yellow-400' : 'text-emerald-400';

    return { label: finalClass, color: colorClass };
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-6 font-sans">
      <header className="flex items-center mb-8 border-b border-slate-700 pb-4">
        <Brain className="w-8 h-8 text-blue-400 mr-3" />
        <h1 className="text-2xl font-bold tracking-wider">KGRS 雙模態神經影像診斷平台</h1>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* ================= 左側：控制面板 ================= */}
        <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 h-fit xl:col-span-1">
          <h2 className="text-lg font-semibold flex items-center mb-6 text-blue-300">
            <SlidersHorizontal className="w-5 h-5 mr-2" /> 分析參數設定
          </h2>
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">1. 選擇病患 (自動掃描)</label>
              <select
                value={selectedPatientId}
                onChange={e => setSelectedPatientId(e.target.value)}
                className="w-full bg-slate-900 border border-slate-600 rounded p-2 text-slate-200 outline-none focus:border-blue-500"
              >
                {patients.length === 0 && <option>載入中...</option>}
                {patients.map(p => (
                  <option key={p.id} value={p.id}>{p.label}</option>
                ))}
              </select>
              {currentPatient && (
                <div className="mt-2 flex flex-wrap gap-2">
                  <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1 ${hasFmri ? 'bg-emerald-900 text-emerald-300' : 'bg-red-900 text-red-300'}`}>
                    {hasFmri ? 'fMRI 有效' : 'fMRI 無效'}
                  </div>
                  <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1 ${hasSmri ? 'bg-blue-900 text-blue-300' : 'bg-slate-700 text-slate-400'}`}>
                    {hasSmri ? 'sMRI 有效' : 'sMRI 缺失'}
                  </div>
                </div>
              )}
            </div>

            <div className="pt-4 border-t border-slate-700">
              <label className="block text-sm font-medium mb-2 text-slate-400">2. 臨床報告模式</label>
              <div className="flex gap-2">
                <button onClick={() => setReportMode('detailed')} className={`flex-1 py-2 text-sm rounded border transition ${reportMode === 'detailed' ? 'bg-blue-600 border-blue-500 text-white' : 'bg-slate-900 border-slate-600 text-slate-400'}`}>詳盡分析</button>
                <button onClick={() => setReportMode('fast')} className={`flex-1 py-2 text-sm rounded border transition flex items-center justify-center ${reportMode === 'fast' ? 'bg-emerald-600 border-emerald-500 text-white' : 'bg-slate-900 border-slate-600 text-slate-400'}`}><Zap className="w-4 h-4 mr-1" /> 快速重點</button>
              </div>
            </div>

            <button onClick={handleAnalyze} disabled={isAnalyzing} className="w-full mt-4 bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded shadow-lg transition flex justify-center items-center">
              {isAnalyzing ? <Loader2 className="animate-spin w-5 h-5" /> : <><Activity className="w-5 h-5 mr-2" /> 執行深度分析</>}
            </button>
          </div>
        </div>

        {/* ================= 右側：視覺化與報告 ================= */}
        <div className="xl:col-span-3 space-y-6">
          {analyzeData && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

              {/* 機率與判定結果 */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 col-span-1 flex flex-col">
                
                {/* 🌟 核心升級：系統最終判定 */}
                <div className="mb-6 pb-6 border-b border-slate-700">
                  <h2 className="text-sm font-semibold text-slate-400 mb-3">系統綜合判定 (OVO Voting)</h2>
                  {(() => {
                    const diagnosis = getFinalDiagnosis();
                    return diagnosis ? (
                      <div className={`text-4xl font-bold tracking-wider flex items-center ${diagnosis.color}`}>
                        <Activity className="w-8 h-8 mr-3" />
                        {diagnosis.label}
                      </div>
                    ) : null;
                  })()}
                </div>

                <div className="flex flex-col gap-1 mb-6">
                  <div className="text-xs text-slate-400">分析模態：</div>
                  <div className="flex flex-wrap gap-2">
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1 ${hasFmri ? 'bg-emerald-900 text-emerald-300' : 'bg-red-900 text-red-300'}`}>
                      {hasFmri ? 'fMRI 功能分析' : 'fMRI 數據失效'}
                    </div>
                    <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1 ${hasSmri ? 'bg-blue-900 text-blue-300' : 'bg-slate-700 text-slate-400'}`}>
                      {hasSmri ? 'sMRI 結構分析' : 'sMRI 缺失'}
                    </div>
                  </div>
                </div>


                <h2 className="text-sm font-semibold mb-4 text-blue-300">各子任務機率 (Sub-tasks)</h2>
                <div className="space-y-4 mb-6">
                  {Object.entries(analyzeData.task_results).map(([task, res]) => {
                    const prob = (res.prob_fused * 100).toFixed(1);
                    const color = prob > 60 ? 'bg-red-500' : prob > 40 ? 'bg-yellow-500' : 'bg-emerald-500';
                    return (
                      <div key={task}>
                        <div className="flex justify-between text-xs mb-1 text-slate-300">
                          <span>{task}</span>
                          <span className="font-mono">{prob}%</span>
                        </div>
                        <div className="w-full bg-slate-900 rounded-full h-2">
                          <div className={`${color} h-2 rounded-full transition-all duration-500`} style={{ width: `${prob}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {analyzeData.radar_data && Object.keys(analyzeData.radar_data).length > 0 && (
                  <div className="pt-4 border-t border-slate-700">
                    <h2 className="text-xs font-semibold mb-2 text-slate-400 flex items-center gap-1">
                      <Zap className="w-3 h-3 text-emerald-400" />
                      fMRI 功能貢獻度 (Saliency)
                    </h2>
                    <SaliencyRadarChart
                      saliencyData={analyzeData.radar_data['NC vs AD'] || Object.values(analyzeData.radar_data)[0]}
                      labels={analyzeData.network_labels}
                    />
                  </div>
                )}
              </div>

              {/* 3D 大腦視覺化 */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 col-span-2 flex flex-col">
                <h2 className="text-lg font-semibold text-blue-300 mb-4">大腦特徵視覺化 (Brain Viewer)</h2>
                <div className="flex-1 grid grid-cols-2 gap-4">
                  <div className="bg-slate-900 rounded-lg border border-slate-700 flex flex-col overflow-hidden" style={{ minHeight: '200px' }}>
                    <div className="flex items-center justify-between px-3 pt-3 pb-1">
                      <div className="text-purple-400 text-xs font-semibold">sMRI 結構萎縮區域</div>
                      {saliencyUrl && (
                        <label className="flex items-center gap-1.5 cursor-pointer select-none">
                          <span className="text-xs text-slate-400">顯示 AI 關注區域</span>
                          <div
                            onClick={() => setShowSaliency(v => !v)}
                            className={`relative w-9 h-5 rounded-full transition-colors duration-200 ${showSaliency ? 'bg-orange-500' : 'bg-slate-600'}`}
                          >
                            <div className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${showSaliency ? 'translate-x-4' : 'translate-x-0'}`} />
                          </div>
                        </label>
                      )}
                    </div>
                    <div className="flex-1 p-2">
                      <NiiVueViewer t1Url={t1Url} overlayUrl={saliencyUrl} showOverlay={showSaliency} />
                    </div>
                  </div>
                  <div className="bg-slate-900 rounded-lg border border-slate-700 flex flex-col p-3">
                    <div className="text-emerald-400 text-xs mb-2 font-semibold">fMRI 功能網路連結強度 (9×9 FC)</div>
                    {hasFmri ? (
                      <NetworkHeatmap matrix={analyzeData.network_matrix} labels={analyzeData.network_labels} />
                    ) : (
                      <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
                        <Activity className="w-8 h-8 mb-2 opacity-20" />
                        <span className="text-xs">fMRI 數據無效，無法生成熱力圖</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* 報告生成區塊 */}
          <div className="bg-slate-800 p-6 rounded-xl shadow-lg border border-slate-700 min-h-[300px]">
            <div className="flex justify-between items-center mb-4 border-b border-slate-700 pb-2">
              <h2 className="text-lg font-semibold flex items-center text-blue-300">
                <FileText className="w-5 h-5 mr-2" />
                AI 臨床報告 (Gemma 4 — {reportMode === 'fast' ? '快速模式' : '詳盡模式'})
              </h2>
              {analyzeData && (
                <button onClick={handleGenerateReport} disabled={isGenerating} className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded text-sm font-semibold transition flex items-center">
                  {isGenerating ? <Loader2 className="animate-spin w-4 h-4 mr-2" /> : '開始生成報告'}
                </button>
              )}
            </div>
            <div className="bg-slate-900 p-4 rounded min-h-[200px] text-slate-300 font-serif leading-relaxed whitespace-pre-wrap">
              {reportText || (analyzeData ? '請點擊右上角按鈕開始生成報告。' : '請先在左側選擇病患並執行分析。')}
              {isGenerating && <span className="ml-1 animate-pulse border-r-2 border-slate-400">&nbsp;</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}