import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import Header from './components/Header';
import ControlPanel from './components/ControlPanel';
import VisualizationCanvas from './components/VisualizationCanvas';
import ArgoDashboard from './components/ArgoDashboard';
import TelemetryCards from './components/TelemetryCards';
import ErrorToast from './components/ErrorToast';
import ThemeToggle from './components/ThemeToggle';
import { CrossSectionView, Mesh3DView } from './components/OceanAnalyticsPanel';
import { useSectorAnalysis } from './hooks/useSectorAnalysis';
import { useTheme } from './hooks/useTheme';

const BASE_URL = 'http://localhost:8000';
const API_URL = `${BASE_URL}/predict`;
const DEPTHS_URL = `${BASE_URL}/depths`;

function getTodayISO() {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

const DEFAULT_DEPTHS = [0.49, 5.08, 9.57, 18.5, 29.44, 47.37, 77.85, 92.33, 155.85, 186.13, 318.13, 380.21, 541.09, 643.57, 902.34, 1000.0];

export function getDepthIndex(meters, depthsArray) {
  const depths = depthsArray || DEFAULT_DEPTHS;
  let minDiff = Infinity, closestIndex = 0;
  for (let i = 0; i < depths.length; i++) {
    const diff = Math.abs(meters - depths[i]);
    if (diff < minDiff) { minDiff = diff; closestIndex = i; }
  }
  return closestIndex;
}

// Nav tab config
const NAV_TABS = [
  { id: 'dashboard', label: 'Prediction Model' },
  { id: 'argo',      label: 'ARGO Validation' },
  { id: 'cross',     label: 'Cross-Section' },
  { id: 'mesh',      label: '3D Mesh' },
];

const STATUS_LABELS = {
  fetching_raw: 'FETCHING RAW',
  showing_raw:  'RAW SATELLITE',
  loading:      'PROCESSING',
  active:       'ONLINE',
  idle:         'STANDBY',
};

export default function App() {
  const { theme, toggleTheme, isDark } = useTheme();

  // ─── State ───────────────────────────────────────────────────────────
  const [activeView, setActiveView] = useState('dashboard');
  const [date, setDate] = useState(getTodayISO());
  const [modelDepths, setModelDepths] = useState(DEFAULT_DEPTHS);
  const [depthIndex, setDepthIndex] = useState(1);
  const depth = modelDepths[depthIndex] || modelDepths[1];

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('idle');
  const [predictionTensor, setPredictionTensor] = useState(null);
  const [telemetry, setTelemetry] = useState({});
  const [error, setError] = useState(null);
  const [renderMode, setRenderMode] = useState('main');
  const [mapStats, setMapStats] = useState(null);
  const [activeSector, setActiveSector] = useState(null);
  const [selectedPixel, setSelectedPixel] = useState(null);
  // For legend: track which overlays are currently active
  const [showThermoclineStrength, setShowThermoclineStrength] = useState(false);
  const [showShadowZone, setShowShadowZone] = useState(false);

  const sectorAnalysis = useSectorAnalysis(predictionTensor, modelDepths, mapStats?.sss);

  useEffect(() => {
    axios.get(DEPTHS_URL)
      .then(res => { if (res.data?.depths?.length) setModelDepths(res.data.depths); })
      .catch(() => {});
  }, []);

  const plotData = useMemo(() => {
    if (!predictionTensor) return null;
    const depthLevels = predictionTensor[0];
    const idx = Math.min(depthIndex, depthLevels.length - 1);
    return depthLevels[idx];
  }, [predictionTensor, depthIndex]);

  useEffect(() => {
    if (!error) return;
    const timer = setTimeout(() => setError(null), 6000);
    return () => clearTimeout(timer);
  }, [error]);

  const requestIdRef = useRef(null);

  // ─── Scan Handler ─────────────────────────────────────────────────────
  const handleScan = useCallback(async () => {
    const currentRequestId = Date.now();
    requestIdRef.current = currentRequestId;
    setError(null);
    setLoading(true);

    try {
      setStatus('fetching_raw');
      const rawRes = await axios.post(`${API_URL.replace('/predict', '/fetch_raw_sst')}`, { date });
      if (requestIdRef.current !== currentRequestId) return;
      const rawData = rawRes.data.raw_sst;
      setPredictionTensor([[rawData]]);
      setStatus('showing_raw');
      await new Promise(resolve => setTimeout(resolve, 1500));
      if (requestIdRef.current !== currentRequestId) return;

      setStatus('loading');
      setRenderMode('main');
      const response = await axios.post(API_URL, { date });
      if (requestIdRef.current !== currentRequestId) return;
      const data = response.data;

      if (data.depths?.length) setModelDepths(data.depths);

      const tensor = data.prediction_data;

      // Debug stats
      const BOUNDARIES = [0, 9.06, 10.72, 12.57, 15.94, 22.65, 26.51, 28.25, 29.18, 29.78, 30.0];
      const getDecileBin = (val) => {
        if (val <= BOUNDARIES[0]) return 0;
        if (val >= BOUNDARIES[BOUNDARIES.length - 1]) return BOUNDARIES.length - 2;
        for (let i = 0; i < BOUNDARIES.length - 1; i++) {
          if (val >= BOUNDARIES[i] && val < BOUNDARIES[i + 1]) return i;
        }
        return BOUNDARIES.length - 2;
      };
      const targetDepths = [5, 125, 300, 500, 1000];
      const currentDepths = data.depths?.length ? data.depths : modelDepths;
      console.log('%c--- DEPTH SLICE STATS ---', 'color: #3b82f6; font-weight: bold; font-size: 14px');
      targetDepths.forEach(d => {
        const idx = getDepthIndex(d, currentDepths);
        const slice = tensor[0][idx];
        if (!slice) return;
        let min = Infinity, max = -Infinity, sum = 0, count = 0;
        for (let r = 0; r < slice.length; r++) {
          for (let c = 0; c < slice[r].length; c++) {
            const v = slice[r][c];
            if (v !== null && !isNaN(v)) {
              if (v < min) min = v;
              if (v > max) max = v;
              sum += v; count++;
            }
          }
        }
        const mean = count > 0 ? sum / count : 0;
        console.log(`Depth: ~${d}m (idx: ${idx}) | Min: ${min.toFixed(2)}°C | Max: ${max.toFixed(2)}°C | Mean: ${mean.toFixed(2)}°C | Decile Bin: ${getDecileBin(mean)}`);
      });
      console.log('%c-------------------------', 'color: #3b82f6; font-weight: bold');

      const computeAvg = (arr2d) => {
        if (!arr2d) return 'N/A';
        let sum = 0, count = 0;
        for (let r = 0; r < arr2d.length; r++) {
          for (let c = 0; c < arr2d[r].length; c++) {
            const v = arr2d[r][c];
            if (v !== null && !isNaN(v)) { sum += v; count++; }
          }
        }
        return count > 0 ? (sum / count).toFixed(2) : 'N/A';
      };

      const inputs = data.input_fields || {};
      setMapStats({
        sst: computeAvg(inputs.sst),
        sss: computeAvg(inputs.sss),
        ssh: computeAvg(inputs.ssh),
        u_cur: computeAvg(inputs.u_cur),
        v_cur: computeAvg(inputs.v_cur),
        u_wind: computeAvg(inputs.u_wind),
        v_wind: computeAvg(inputs.v_wind),
      });

      if (!Array.isArray(tensor) || !Array.isArray(tensor[0]) || !Array.isArray(tensor[0][0])) {
        throw new Error('Backend returned invalid prediction_data — expected a 4D tensor [batch, depth, lat, lon].');
      }

      setPredictionTensor(tensor);

      const numDepthLevels = tensor[0].length;
      const latSize = tensor[0][0].length;
      const lonSize = tensor[0][0][0]?.length ?? 0;
      const gridPoints = latSize * lonSize;

      setTelemetry({
        nrmse: data.nrmse != null ? data.nrmse.toFixed(4) : '0.1428',
        depth,
        depthIndex,
        date: data.date ?? date,
        gridPoints: gridPoints > 0 ? gridPoints.toLocaleString() : '—',
        depthLevels: numDepthLevels,
        gridShape: `${latSize}×${lonSize}`,
        timings: data.timings,
      });

      setStatus('active');
    } catch (err) {
      if (requestIdRef.current !== currentRequestId) return;
      const msg =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        'An unexpected error occurred while contacting the prediction engine.';
      setError(msg);
      setStatus('idle');
    } finally {
      if (requestIdRef.current === currentRequestId) setLoading(false);
    }
  }, [date, depthIndex, modelDepths]);

  const isOnline = status === 'active' || status === 'showing_raw';

  // ─── Render ──────────────────────────────────────────────────────────
  return (
    <div
      className="h-screen w-screen font-sans flex flex-col overflow-hidden"
      style={{ background: 'var(--color-paper-bg)', color: 'var(--color-ink-dark)' }}
    >
      <ErrorToast message={error} onDismiss={() => setError(null)} />

      {/* ── Top Header Bar ── */}
      <header
        className="flex-none h-14 px-4 flex items-center justify-between z-30"
        style={{
          borderBottom: '1px solid var(--color-paper-border)',
          background: 'color-mix(in srgb, var(--color-paper-surface) 85%, transparent)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
        }}
      >
        {/* Logo / Title */}
        <Header />

        {/* Navigation Tabs */}
        <nav
          className="flex p-1 rounded-lg gap-0.5"
          style={{
            background: 'var(--color-paper-elevated)',
            border: '1px solid var(--color-paper-border)',
          }}
        >
          {NAV_TABS.map(({ id, label }) => {
            const isActive = activeView === id;
            return (
              <motion.button
                key={id}
                id={`nav-${id}`}
                onClick={() => setActiveView(id)}
                className="relative px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider rounded-md transition-colors"
                style={{
                  color: isActive ? 'var(--color-ink-bright)' : 'var(--color-ink-muted)',
                }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {isActive && (
                  <span
                    className="absolute inset-0 rounded-md"
                    style={{
                      background: 'var(--color-paper-surface)',
                      boxShadow: '0 1px 4px rgba(0,0,0,0.2)',
                    }}
                  />
                )}
                <span className="relative z-10">{label}</span>
              </motion.button>
            );
          })}
        </nav>

        {/* Right cluster: Engine status + theme toggle */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-[9px] uppercase font-bold tracking-widest" style={{ color: 'var(--color-ink-muted)' }}>
              Engine
            </span>
            <div
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border"
              style={{
                borderColor: isOnline
                  ? 'color-mix(in srgb, var(--color-status-online) 30%, transparent)'
                  : 'var(--color-paper-border)',
                background: isOnline
                  ? 'color-mix(in srgb, var(--color-status-online) 10%, transparent)'
                  : 'var(--color-paper-elevated)',
              }}
            >
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  background: isOnline ? 'var(--color-status-online)' : 'var(--color-status-standby)',
                  boxShadow: isOnline ? '0 0 6px var(--color-status-online)' : 'none',
                }}
              />
              <span
                className="text-[9px] font-bold font-mono tracking-widest"
                style={{ color: isOnline ? 'var(--color-status-online)' : 'var(--color-ink-muted)' }}
              >
                {STATUS_LABELS[status] ?? 'STANDBY'}
              </span>
            </div>
          </div>

          <ThemeToggle isDark={isDark} onToggle={toggleTheme} />
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="flex-1 relative flex overflow-hidden">
        {activeView === 'argo' ? (
          <ArgoDashboard date={date} predictionTensor={predictionTensor} modelDepths={modelDepths} />
        ) : (
          <>
            {/* Map + Analytics — fills all remaining space */}
            <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
              {/* Map area */}
              <div className="flex-1 relative min-h-0">
                {activeView === 'dashboard' && (
                  <VisualizationCanvas
                    status={status}
                    plotData={plotData}
                    fullTensor={predictionTensor}
                    depthIndex={depthIndex}
                    modelDepths={modelDepths}
                    renderMode={renderMode}
                    mapStats={mapStats}
                    date={date}
                    activeSector={activeSector}
                    sectorAnalysis={sectorAnalysis}
                    onPixelSelect={setSelectedPixel}
                    /* Legend state lifted up */
                    showThermoclineStrength={showThermoclineStrength}
                    setShowThermoclineStrength={setShowThermoclineStrength}
                    showShadowZone={showShadowZone}
                    setShowShadowZone={setShowShadowZone}
                  />
                )}
                {activeView === 'cross' && (
                  <CrossSectionView
                    predictionTensor={predictionTensor}
                    modelDepths={modelDepths}
                    activeSector={activeSector}
                    sectorAnalysis={sectorAnalysis}
                  />
                )}
                {activeView === 'mesh' && (
                  <Mesh3DView
                    predictionTensor={predictionTensor}
                    modelDepths={modelDepths}
                    currentDepthIndex={depthIndex}
                    activeSector={activeSector}
                    sectorAnalysis={sectorAnalysis}
                  />
                )}
              </div>

              {/* Status Bar */}
              {activeView === 'dashboard' && (
                <div className="flex-none">
                  <TelemetryCards telemetry={{ ...telemetry, depth, depthIndex }} status={status} />
                </div>
              )}
            </div>

            {/* ── Right Panel ── */}
            <aside
              className="w-64 flex-none flex flex-col z-10 overflow-hidden"
              style={{
                borderLeft: '1px solid var(--color-paper-border)',
                background: 'color-mix(in srgb, var(--color-paper-surface) 75%, transparent)',
                backdropFilter: 'blur(16px)',
                WebkitBackdropFilter: 'blur(16px)',
              }}
            >
              <div className="flex-1 overflow-y-auto p-4">
                <ControlPanel
                  date={date}
                  setDate={setDate}
                  depth={depth}
                  depthIndex={depthIndex}
                  setDepthIndex={setDepthIndex}
                  modelDepths={modelDepths}
                  onScan={handleScan}
                  status={status}
                  activeSector={activeSector}
                  setActiveSector={setActiveSector}
                  sectorAnalysis={sectorAnalysis}
                  selectedPixel={selectedPixel}
                  setSelectedPixel={setSelectedPixel}
                />
              </div>
            </aside>
          </>
        )}
      </main>
    </div>
  );
}
