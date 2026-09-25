/**
 * ControlPanel.jsx
 * ================
 * Right panel: Mission Parameters + Sector Focus.
 * Full dark/light theme variables. Sector tabs animate (sliding indicator).
 * Sector switch shows brief "Recalculating..." feedback (200ms).
 */
import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Calendar, Anchor, Activity, Loader2, Crosshair, Map } from 'lucide-react';
import { getDepthIndex } from '../App';
import SectorInsightsCard from './SectorInsightsCard';

const SECTORS = [
  { id: 'fisheries', label: 'Fish', fullLabel: 'Fisheries', color: 'var(--color-sector-fisheries)' },
  { id: 'naval',     label: 'Naval', fullLabel: 'Naval',     color: 'var(--color-sector-naval)' },
  { id: 'scuba',     label: 'Scuba', fullLabel: 'Scuba',     color: 'var(--color-sector-scuba)' },
];

const SECTOR_CONTEXT = {
  fisheries: 'Highlighting 18 °C isosurface & thermocline to locate pelagic fish aggregation boundaries.',
  naval:     'Rendering thermal cross-sections for acoustic shadow zone & Sonic Layer Depth analysis.',
  scuba:     'Isolating 0–40m layers for thermal shock warnings and wetsuit recommendations.',
};

export default function ControlPanel({
  date, setDate, depth, depthIndex, setDepthIndex, modelDepths,
  onScan, status, activeSector, setActiveSector, sectorAnalysis, selectedPixel, setSelectedPixel
}) {
  const [localDepth, setLocalDepth] = useState(depth);
  const [transitioning, setTransitioning] = useState(false);
  const debounceRef = useRef(null);

  useEffect(() => { setLocalDepth(depth); }, [depth]);

  const handleSliderChange = (e) => {
    const val = parseFloat(e.target.value);
    setLocalDepth(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setDepthIndex(getDepthIndex(val, modelDepths));
    }, 150);
  };

  const handleSectorClick = (id) => {
    if (activeSector === id) {
      setActiveSector(null);
      return;
    }

    // Brief transition feedback
    setTransitioning(true);
    setActiveSector(id);
    setTimeout(() => setTransitioning(false), 250);

    let targetDepthIndex = null;
    if (id === 'fisheries') targetDepthIndex = sectorAnalysis?.fisheries?.mapDepthIndex ?? getDepthIndex(50, modelDepths);
    else if (id === 'naval') targetDepthIndex = sectorAnalysis?.naval?.mapDepthIndex ?? getDepthIndex(50, modelDepths);
    else if (id === 'scuba') targetDepthIndex = sectorAnalysis?.scuba?.mapDepthIndex ?? 0;

    if (targetDepthIndex !== null) {
      setDepthIndex(targetDepthIndex);
      setLocalDepth(modelDepths[targetDepthIndex] ?? 0);
    }
  };

  const activeSectorConfig = SECTORS.find(s => s.id === activeSector);

  const loading = ['fetching_raw', 'showing_raw', 'loading'].includes(status);
  
  const loadingText = status === 'fetching_raw' ? 'FETCHING RAW DATA' :
                      status === 'showing_raw' ? 'SATELLITE SYNC' :
                      'PROCESSING AI MODEL';

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Title */}
      <div
        className="flex items-center gap-2 pb-3 border-b"
        style={{ borderColor: 'var(--color-paper-border)' }}
      >
        <Crosshair className="w-3.5 h-3.5" style={{ color: 'var(--color-ink-muted)' }} />
        <h2 className="text-[10px] font-bold tracking-widest uppercase" style={{ color: 'var(--color-ink-dark)' }}>
          Mission Parameters
        </h2>
      </div>

      {/* Date */}
      <div className="flex flex-col gap-1.5">
        <label className="flex items-center gap-1.5 stat-label">
          <Calendar className="w-3 h-3" />
          Observation Date
        </label>
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="w-full px-3 py-1.5 rounded-lg text-sm font-medium font-sans focus:outline-none focus:ring-1 transition-all"
          style={{
            background: 'var(--color-paper-elevated)',
            border: '1px solid var(--color-paper-border)',
            color: 'var(--color-ink-dark)',
            '--tw-ring-color': 'var(--color-accent)',
          }}
        />
      </div>

      {/* Depth Slider */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-1.5 stat-label">
            <Anchor className="w-3 h-3" />
            Target Depth
          </span>
          <div className="flex items-baseline gap-1">
            <span className="text-xl font-bold tabular-nums font-display" style={{ color: 'var(--color-ink-bright)' }}>
              {Number(localDepth).toFixed(1).replace(/\.0$/, '')}
            </span>
            <span className="text-[10px]" style={{ color: 'var(--color-ink-muted)' }}>m</span>
            <span
              className="text-[9px] font-bold font-mono ml-1"
              style={{ color: 'var(--color-accent)' }}
            >
              L{getDepthIndex(localDepth, modelDepths)}
            </span>
          </div>
        </div>

        {(getDepthIndex(localDepth, modelDepths) === 0 || getDepthIndex(localDepth, modelDepths) === modelDepths.length - 1) && (
          <div className="callout-warning">
            Limited training data at this depth — predictions may be less accurate
          </div>
        )}

        <div className="relative px-1 pb-1.5 mt-1">
          <input
            type="range"
            min={modelDepths[0] || 0}
            max={modelDepths[modelDepths.length - 1] || 1000}
            step="1"
            value={localDepth}
            onChange={handleSliderChange}
            className="w-full"
          />
          <div className="flex justify-between mt-1.5 px-0.5">
            <span className="text-[9px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
              {Math.round(modelDepths[0] || 0)}m
            </span>
            <span className="text-[9px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
              {Math.round(modelDepths[modelDepths.length - 1] || 1000)}m
            </span>
          </div>
        </div>
      </div>

      {/* Sector Focus */}
      <div className="flex flex-col gap-2">
        <label className="flex items-center gap-1.5 stat-label">
          <Map className="w-3 h-3" />
          Sector Focus
        </label>

        {/* Segmented Tabs with sliding indicator */}
        <div
          className="relative flex rounded-lg p-1 gap-1"
          style={{
            background: 'var(--color-paper-elevated)',
            border: '1px solid var(--color-paper-border)',
          }}
        >
          {SECTORS.map(({ id, label, color }) => {
            const isActive = activeSector === id;
            return (
              <motion.button
                key={id}
                onClick={() => handleSectorClick(id)}
                className="flex-1 relative text-[9px] py-1.5 rounded-md font-bold uppercase tracking-wider z-10 transition-colors"
                style={{
                  color: isActive ? '#fff' : 'var(--color-ink-muted)',
                }}
                whileHover={{ scale: 1.03, brightness: 1.1 }}
                whileTap={{ scale: 0.97 }}
              >
                {isActive && (
                  <motion.span
                    layoutId="sector-pill"
                    className="absolute inset-0 rounded-md z-[-1]"
                    style={{ background: color }}
                    transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                  />
                )}
                {label}
              </motion.button>
            );
          })}
        </div>

        {/* Transition feedback */}
        <AnimatePresence mode="wait">
          {transitioning && activeSector && (
            <motion.div
              key="recalc"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.15 }}
              className="overflow-hidden"
            >
              <div
                className="text-[9px] font-mono px-2 py-1 rounded flex items-center gap-1.5 animate-pulse"
                style={{
                  color: activeSectorConfig?.color ?? 'var(--color-accent)',
                  background: 'color-mix(in srgb, var(--color-accent) 8%, transparent)',
                }}
              >
                <Loader2 className="w-2.5 h-2.5 animate-spin" />
                Recalculating {activeSectorConfig?.fullLabel} Analytics…
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Context description */}
        <AnimatePresence mode="wait">
          {activeSector && !transitioning && (
            <motion.div
              key={activeSector + '_ctx'}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div
                className="text-[10px] leading-relaxed rounded-lg px-2.5 py-1.5"
                style={{
                  color: 'var(--color-ink-medium)',
                  background: 'var(--color-paper-elevated)',
                  border: '1px solid var(--color-paper-border)',
                }}
              >
                {SECTOR_CONTEXT[activeSector]}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Analytics card */}
        <SectorInsightsCard
          activeSector={activeSector}
          sectorAnalysis={sectorAnalysis}
          selectedPixel={selectedPixel}
          setSelectedPixel={setSelectedPixel}
          date={date}
          modelDepths={modelDepths}
        />
      </div>

      {/* Scan Button */}
      <div
        className="mt-auto pt-4 border-t flex flex-col gap-2"
        style={{ borderColor: 'var(--color-paper-border)' }}
      >
        <motion.button
          id="initialize-scan-btn"
          onClick={onScan}
          disabled={loading}
          className="w-full py-3 rounded-xl font-bold text-xs tracking-widest uppercase font-sans text-white relative overflow-hidden shadow-lg disabled:opacity-60 disabled:cursor-not-allowed"
          style={{
            background: loading
              ? 'var(--color-ink-muted)'
              : 'var(--color-accent)',
          }}
          whileHover={!loading ? { scale: 1.02, boxShadow: '0 0 20px var(--color-accent-glow)' } : {}}
          whileTap={!loading ? { scale: 0.98 } : {}}
          transition={{ type: 'spring', stiffness: 400, damping: 25 }}
        >
          {/* Shimmer effect on hover/loading */}
          {loading && (
            <motion.span
              className="absolute inset-0 opacity-20"
              style={{
                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
              }}
              animate={{ x: ['-100%', '200%'] }}
              transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
            />
          )}

          <AnimatePresence mode="wait">
            {loading ? (
              <motion.span
                key="loading"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center justify-center gap-2"
              >
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> {loadingText}
              </motion.span>
            ) : (
              <motion.span
                key="idle"
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="flex items-center justify-center gap-2"
              >
                <Activity className="w-3.5 h-3.5" /> INITIALIZE SCAN
              </motion.span>
            )}
          </AnimatePresence>
        </motion.button>
      </div>
    </div>
  );
}
