/**
 * TelemetryCards.jsx
 * ==================
 * The bottom status bar — trust-building evidence of real data and accuracy.
 * NRMSE gets a prominent colored badge. All values animate on change.
 */
import { motion, AnimatePresence } from 'framer-motion';
import { useRef, useEffect, useState } from 'react';
import { Activity } from 'lucide-react';

function AnimatedValue({ value }) {
  const [display, setDisplay] = useState(value);
  const [flash, setFlash] = useState(false);
  const prevRef = useRef(value);

  useEffect(() => {
    if (prevRef.current !== value) {
      prevRef.current = value;
      setDisplay(value);
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 600);
      return () => clearTimeout(t);
    }
  }, [value]);

  return (
    <span
      className={`stat-value tabular-nums transition-all rounded px-0.5 ${flash ? 'animate-value-flash' : ''}`}
    >
      {display}
    </span>
  );
}

function NRMSEBadge({ value }) {
  const [flash, setFlash] = useState(false);
  const prevRef = useRef(value);

  useEffect(() => {
    if (prevRef.current !== value && value && value !== '—') {
      prevRef.current = value;
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 800);
      return () => clearTimeout(t);
    }
  }, [value]);

  return (
    <span
      className={`nrmse-badge ${flash ? 'animate-value-flash' : ''}`}
      title="Normalized Root Mean Square Error — lower is better"
    >
      <Activity className="w-3 h-3" />
      {value ?? '—'}
    </span>
  );
}

function TelemetryItem({ label, value, subtext, children }) {
  return (
    <div className="flex-1 min-w-[90px] flex flex-col justify-center px-4 py-2 border-r border-[var(--color-paper-border)] last:border-r-0">
      <span className="stat-label mb-0.5">{label}</span>
      <div className="flex items-baseline gap-1">
        {children || <AnimatedValue value={value ?? '—'} />}
        {subtext && (
          <span className="text-[8px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
            {subtext}
          </span>
        )}
      </div>
    </div>
  );
}

export default function TelemetryCards({ telemetry, status }) {
  if (status !== 'active' && !telemetry.date) return null;

  return (
    <motion.div
      className="status-bar"
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
    >
      <div className="flex flex-wrap overflow-hidden">
        <TelemetryItem
          label="Depth"
          value={telemetry.depth != null ? telemetry.depth.toFixed(1) : '—'}
          subtext={telemetry.depth != null ? 'm' : null}
        />
        <TelemetryItem label="Layer" value={telemetry.depthIndex != null ? `L${telemetry.depthIndex}` : '—'} />

        {/* NRMSE — promoted with badge */}
        <div className="flex-1 min-w-[110px] flex flex-col justify-center px-4 py-2 border-r border-[var(--color-paper-border)]">
          <span className="stat-label mb-1">NRMSE Accuracy</span>
          <NRMSEBadge value={telemetry.nrmse ?? '—'} />
        </div>

        <TelemetryItem label="Date" value={telemetry.date ?? '—'} subtext="UTC" />
        <TelemetryItem
          label="Grid"
          value={telemetry.gridShape ?? '—'}
          subtext={telemetry.gridPoints ? `${telemetry.gridPoints} pts` : null}
        />
      </div>
    </motion.div>
  );
}
