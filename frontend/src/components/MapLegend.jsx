/**
 * MapLegend.jsx
 * ==============
 * Unified, dynamic map legend. Shows only currently-active overlays.
 * Entries animate in/out with fade+slide when toggled or sector changes.
 */
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';

// Static legend entries always present when data is active
const STATIC_ENTRIES = [
  {
    id: 'colorscale',
    swatch: 'linear-gradient(to right, #042333, #662479, #ab1a45, #da561a, #eff542)',
    label: 'Temperature (°C)',
    alwaysShow: true,
    swatchType: 'gradient',
  },
  {
    id: 'land',
    swatch: null,
    swatchStyle: { background: 'rgba(120,120,120,0.4)', border: '1px solid #555' },
    label: 'Land / no-data mask',
    alwaysShow: true,
  },
];

// Dynamic entries keyed by conditions
function buildEntries({ activeSector, showThermoclineStrength, showShadowZone, hasArgo }) {
  const entries = [];

  if (hasArgo) {
    entries.push({
      id: 'argo',
      swatchStyle: { background: '#22c55e', borderRadius: '50%' },
      label: 'ARGO Float (validation)',
    });
  }

  if (activeSector === 'fisheries') {
    entries.push({
      id: 'fish-agg',
      swatchStyle: { background: color('fisheries', 0.7), borderRadius: '3px' },
      label: 'Aggregation Band (20–29 °C)',
    });
    entries.push({
      id: 'fish-tc',
      swatchStyle: { background: 'transparent', border: `2px dashed ${color('fisheries', 1)}`, borderRadius: '3px' },
      label: 'Thermocline depth marker',
    });
  }

  if (activeSector === 'naval') {
    entries.push({
      id: 'naval-sld',
      swatchStyle: { background: color('naval', 0.7), borderRadius: '3px' },
      label: 'Sonic Layer Depth (SLD)',
    });
    if (showThermoclineStrength) {
      entries.push({
        id: 'naval-tc-strength',
        swatch: 'linear-gradient(to right, #9e9ac8, #54278f)',
        swatchType: 'gradient',
        label: 'Thermocline strength (purple = stronger)',
      });
    }
    if (showShadowZone) {
      entries.push({
        id: 'naval-shadow',
        swatchStyle: {
          background: 'transparent',
          backgroundImage: 'repeating-linear-gradient(45deg, rgba(59,130,246,0.6) 0, rgba(59,130,246,0.6) 1px, transparent 0, transparent 50%)',
          backgroundSize: '6px 6px',
          border: '1px solid rgba(59,130,246,0.4)',
          borderRadius: '3px',
        },
        label: 'Sonar shadow zone (x hatching)',
      });
    }
  }

  if (activeSector === 'scuba') {
    entries.push({
      id: 'scuba-surface',
      swatchStyle: { background: color('scuba', 0.7), borderRadius: '3px' },
      label: '0–40m surface layer',
    });
    entries.push({
      id: 'scuba-coldshock',
      swatchStyle: { background: '#ef4444', borderRadius: '50%', width: '8px', height: '8px' },
      label: 'Cold shock boundary (>0.2 °C/m)',
    });
  }

  return entries;
}

function color(sector, alpha = 1) {
  const map = {
    fisheries: `rgba(245,158,11,${alpha})`,
    naval:     `rgba(59,130,246,${alpha})`,
    scuba:     `rgba(6,182,212,${alpha})`,
  };
  return map[sector] ?? `rgba(200,200,200,${alpha})`;
}

const entryVariants = {
  initial: { opacity: 0, x: -8, height: 0 },
  animate: { opacity: 1, x: 0,  height: 'auto' },
  exit:    { opacity: 0, x: -8, height: 0 },
};

export default function MapLegend({ activeSector, showThermoclineStrength, showShadowZone, hasArgo, status }) {
  const [collapsed, setCollapsed] = useState(false);

  if (status !== 'active') return null;

  const dynamicEntries = buildEntries({ activeSector, showThermoclineStrength, showShadowZone, hasArgo });
  const allEntries = [...STATIC_ENTRIES, ...dynamicEntries];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="absolute bottom-4 left-4 z-20 min-w-[170px] max-w-[220px] rounded-xl border border-[var(--color-paper-border)] shadow-lg overflow-hidden"
      style={{
        background: 'color-mix(in srgb, var(--color-paper-surface) 88%, transparent)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
      }}
    >
      {/* Header */}
      <button
        onClick={() => setCollapsed(c => !c)}
        className="w-full flex items-center justify-between px-3 py-2 border-b border-[var(--color-paper-border)] hover:bg-[var(--color-paper-elevated)] transition-colors"
      >
        <span className="text-[9px] font-black uppercase tracking-widest text-[var(--color-ink-muted)]">
          Map Legend
        </span>
        <motion.span
          animate={{ rotate: collapsed ? -90 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="w-3 h-3 text-[var(--color-ink-muted)]" />
        </motion.span>
      </button>

      {/* Entries */}
      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            key="legend-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="px-3 py-2 flex flex-col gap-1">
              <AnimatePresence mode="popLayout">
                {allEntries.map((entry) => (
                  <motion.div
                    key={entry.id}
                    variants={entryVariants}
                    initial="initial"
                    animate="animate"
                    exit="exit"
                    transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
                    className="legend-entry overflow-hidden"
                  >
                    {/* Swatch */}
                    {entry.swatchType === 'gradient' ? (
                      <div
                        className="legend-swatch flex-shrink-0"
                        style={{ background: entry.swatch }}
                      />
                    ) : (
                      <div
                        className="legend-swatch flex-shrink-0"
                        style={entry.swatchStyle ?? { background: '#888' }}
                      />
                    )}
                    <span className="text-[var(--color-ink-medium)] leading-tight">{entry.label}</span>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
