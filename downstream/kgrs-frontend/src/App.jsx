import React, { useState, useEffect, useRef } from 'react';
import { Activity, Brain, FileText, Loader2, SlidersHorizontal, Zap, Lock, AlertTriangle } from 'lucide-react';
import { Niivue } from '@niivue/niivue';

// ── 9×9 Network Connectivity 熱力圖元件 ──
function NetworkHeatmap({ matrix, labels }) {
  if (!matrix || !matrix.length || !labels) return null;

  const flat = matrix.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);

  const toColor = (val) => {
    const t = (val - min) / (max - min || 1);
    // 冷藍 → 暖紅
    const r = Math.round(20 + t * 210);
    const g = Math.round(40 + t * 60);
    const b = Math.round(180 - t * 160);
    return `rgb(${r},${g},${b})`;
  };

  const cell = 26; // px per cell

  return (
    <div className="overflow-auto">
      {/* 欄標籤 */}
      <div className="flex" style={{ marginLeft: cell + 36 }}>
        {labels.map((l, j) => (
          <div key={j} style={{ width: cell, fontSize: 9 }}
            className="text-slate-400 text-center truncate">{l}</div>
        ))}
      </div>

      {/* 列 */}
      {matrix.map((row, i) => (
        <div key={i} className="flex items-center">
          {/* 列標籤 */}
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

      {/* 色階說明 */}
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
function NiiVueViewer({ t1Url }) {
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
    nv.loadVolumes([{ url: t1Url, colorMap: 'gray', opacity: 1 }]).catch(err =>
      console.error('[NiiVue] 載入 T1 失敗:', err)
    );

    return () => {
      // NiiVue 目前無正式 destroy API，清空 ref 即可
      nvRef.current = null;
    };
  }, [t1Url]);

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
    <canvas
      ref={canvasRef}
      className="w-full h-full rounded"
      style={{ minHeight: '180px' }}
    />
  );
}

export default function App() {
  // ── 狀態管理 ──
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState('');

  const [fmriWeight, setFmriWeight] = useState(0.5);
  const [reportMode, setReportMode] = useState('detailed');

  const [analyzeData, setAnalyzeData] = useState(null);
  const [reportText, setReportText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  // ── 目前選中病患的 T1 資訊 ──
  const currentPatient = patients.find(p => p.id === selectedPatientId) || null;
  const hasSmri = Boolean(currentPatient?.t1_path);
  const t1Url = currentPatient?.t1_url || '';

  // 單模態防呆：fMRI only 時強制鎖定權重為 1.0
  const effectiveFmriWeight = hasSmri ? fmriWeight : 1.0;
  const smriWeightDisplay = Math.round((1 - effectiveFmriWeight) * 100);
  const fmriWeightDisplay = Math.round(effectiveFmriWeight * 100);

  // 切換病患時重設結果
  useEffect(() => {
    setAnalyzeData(null);
    setReportText('');
    if (!hasSmri) setFmriWeight(1.0);
  }, [selectedPatientId]);

  // ── 啟動時自動抓取病患清單 ──
  useEffect(() => {
    fetch('http://localhost:8080/api/v1/patients')
      .then(res => res.json())
      .then(data => {
        setPatients(data.patients);
        if (data.patients.length > 0) setSelectedPatientId(data.patients[0].id);
      })
      .catch(err => console.error('無法載入病患清單:', err));
  }, []);

  // ── 執行深度分析 ──
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
      formData.append('fmri_weight', effectiveFmriWeight);

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

  // ── 生成 LLM 報告 (streaming) ──
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
            {/* 1. 病患下拉選單 */}
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

              {/* 模態徽章 */}
              {currentPatient && (
                <div className={`mt-2 text-xs px-2 py-1 rounded inline-flex items-center gap-1 ${hasSmri ? 'bg-blue-900 text-blue-300' : 'bg-slate-700 text-slate-400'}`}>
                  {hasSmri ? '雙模態 (fMRI + sMRI)' : 'fMRI 單模態'}
                </div>
              )}
            </div>

            {/* 2. 雙模態權重 */}
            <div className="pt-4 border-t border-slate-700">
              <label className="block text-sm font-medium mb-2 text-slate-400 flex items-center gap-2">
                2. 雙模態決策權重
                {!hasSmri && <Lock className="w-3.5 h-3.5 text-amber-400" />}
              </label>

              <input
                type="range" min="0" max="1" step="0.1"
                value={effectiveFmriWeight}
                onChange={e => setFmriWeight(parseFloat(e.target.value))}
                disabled={!hasSmri}
                className={`w-full accent-blue-500 ${!hasSmri ? 'opacity-40 cursor-not-allowed' : ''}`}
              />

              <div className="flex justify-between text-xs mt-2 font-mono">
                <span className="text-emerald-400">fMRI (功能): {fmriWeightDisplay}%</span>
                <span className="text-purple-400">sMRI (結構): {smriWeightDisplay}%</span>
              </div>

              {!hasSmri && (
                <p className="mt-2 text-xs text-amber-400 flex items-center gap-1">
                  <Lock className="w-3 h-3" />
                  此病患僅有 fMRI，已自動鎖定權重
                </p>
              )}
            </div>

            {/* 3. LLM 模式 */}
            <div className="pt-4 border-t border-slate-700">
              <label className="block text-sm font-medium mb-2 text-slate-400">3. 臨床報告模式</label>
              <div className="flex gap-2">
                <button
                  onClick={() => setReportMode('detailed')}
                  className={`flex-1 py-2 text-sm rounded border transition ${reportMode === 'detailed' ? 'bg-blue-600 border-blue-500 text-white' : 'bg-slate-900 border-slate-600 text-slate-400'}`}
                >
                  詳盡分析
                </button>
                <button
                  onClick={() => setReportMode('fast')}
                  className={`flex-1 py-2 text-sm rounded border transition flex items-center justify-center ${reportMode === 'fast' ? 'bg-emerald-600 border-emerald-500 text-white' : 'bg-slate-900 border-slate-600 text-slate-400'}`}
                >
                  <Zap className="w-4 h-4 mr-1" /> 快速重點
                </button>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="w-full mt-4 bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded shadow-lg transition flex justify-center items-center"
            >
              {isAnalyzing
                ? <Loader2 className="animate-spin w-5 h-5" />
                : <><Activity className="w-5 h-5 mr-2" /> 執行深度分析</>
              }
            </button>
          </div>
        </div>

        {/* ================= 右側：視覺化與報告 ================= */}
        <div className="xl:col-span-3 space-y-6">

          {analyzeData && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

              {/* 機率結果 */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 col-span-1">
                <h2 className="text-lg font-semibold mb-4 text-blue-300">綜合診斷機率</h2>
                <div className="space-y-4">
                  {Object.entries(analyzeData.task_results).map(([task, res]) => {
                    const prob = (res.prob_fused * 100).toFixed(1);
                    const color = prob > 60 ? 'bg-red-500' : prob > 40 ? 'bg-yellow-500' : 'bg-emerald-500';
                    return (
                      <div key={task}>
                        <div className="flex justify-between text-sm mb-1">
                          <span>{task}</span>
                          <span className="font-mono">{prob}%</span>
                        </div>
                        <div className="w-full bg-slate-900 rounded-full h-2.5">
                          <div className={`${color} h-2.5 rounded-full transition-all duration-500`} style={{ width: `${prob}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* 3D 大腦視覺化 */}
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 col-span-2 flex flex-col">
                <h2 className="text-lg font-semibold text-blue-300 mb-4">大腦特徵視覺化 (Brain Viewer)</h2>
                <div className="flex-1 grid grid-cols-2 gap-4">

                  {/* sMRI — NiiVue */}
                  <div className="bg-slate-900 rounded-lg border border-slate-700 flex flex-col overflow-hidden" style={{ minHeight: '200px' }}>
                    <div className="text-purple-400 text-xs font-semibold px-3 pt-3 pb-1">
                      sMRI 結構萎縮區域
                    </div>
                    <div className="flex-1 p-2">
                      <NiiVueViewer t1Url={t1Url} />
                    </div>
                  </div>

                  {/* fMRI — Network Connectivity Heatmap */}
                  <div className="bg-slate-900 rounded-lg border border-slate-700 flex flex-col p-3">
                    <div className="text-emerald-400 text-xs mb-2 font-semibold">
                      fMRI 功能網路連結強度 (9×9 FC)
                    </div>
                    <NetworkHeatmap
                      matrix={analyzeData.network_matrix}
                      labels={analyzeData.network_labels}
                    />
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
                <button
                  onClick={handleGenerateReport}
                  disabled={isGenerating}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-2 rounded text-sm font-semibold transition flex items-center"
                >
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
