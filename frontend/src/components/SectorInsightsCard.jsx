/**
 * SectorInsightsCard.jsx
 * ======================
 * Accordion-structured analytics cards for Fisheries / Naval / Scuba sectors.
 * Each "Section" is independently collapsible with animated height transition.
 * Switching sector resets all sections to default-collapsed state.
 * Typography hierarchy: one headline stat per section, supporting stats smaller.
 */
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { getWetsuitRecommendation } from '../hooks/useSectorAnalysis';


// ── Shared sub-components ────────────────────────────────────────────────────

function HeadlineStat({ value, unit, label, color }) {
  return (
    <div className="flex flex-col items-center py-2 px-1">
      <span className="text-[8px] font-bold uppercase tracking-widest mb-1" style={{ color }}>
        {label}
      </span>
      <div className="flex items-baseline gap-0.5">
        <span className="stat-headline" style={{ color }}>
          {value ?? '—'}
        </span>
        {unit && (
          <span className="text-xs font-medium text-[var(--color-ink-muted)] ml-0.5">
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

function Row({ label, value, unit = '', assumed = false, assumedNote = '' }) {
  return (
    <div className="flex items-baseline justify-between gap-2 py-0.5">
      <span className="stat-label flex items-center gap-1">
        {label}
        {assumed && (
          <span
            className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full text-[8px] font-bold cursor-help leading-none"
            title={assumedNote}
            style={{
              background: 'color-mix(in srgb, var(--color-status-warning) 15%, transparent)',
              border: '1px solid color-mix(in srgb, var(--color-status-warning) 35%, transparent)',
              color: 'var(--color-status-warning)',
            }}
          >
            ≈
          </span>
        )}
      </span>
      <span className="stat-value tabular-nums">
        {value !== null && value !== undefined
          ? `${value}${unit}`
          : <span style={{ color: 'var(--color-ink-muted)', fontWeight: 400 }}>—</span>}
      </span>
    </div>
  );
}

function Pill({ label, color }) {
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-wide"
      style={{
        background: color + '22',
        color,
        border: `1px solid ${color}55`,
      }}
    >
      {label}
    </span>
  );
}

function Callout({ children, type = 'warning' }) {
  return (
    <div className="callout-warning mt-1">
      {children}
    </div>
  );
}

function InsufficientData({ message = 'Run a prediction scan first.' }) {
  return (
    <p className="stat-label text-center py-3 italic">
      {message}
    </p>
  );
}

// ── Accordion Section ────────────────────────────────────────────────────────

function AccordionSection({ title, color, defaultOpen = true, children, resetKey }) {
  const [open, setOpen] = useState(defaultOpen);
  const prevResetKey = useRef(resetKey);

  // Reset to default state when sector changes
  useEffect(() => {
    if (resetKey !== prevResetKey.current) {
      prevResetKey.current = resetKey;
      setOpen(defaultOpen);
    }
  }, [resetKey, defaultOpen]);

  return (
    <div className="border-b border-[var(--color-paper-border)] last:border-b-0">
      <button
        className="w-full flex items-center justify-between py-2 px-0 text-left hover:opacity-80 active:opacity-100 transition-opacity"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
      >
        <span
          className="text-[9px] font-black uppercase tracking-widest"
          style={{ color }}
        >
          {title}
        </span>
        <motion.span
          animate={{ rotate: open ? 0 : -90 }}
          transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
        >
          <ChevronDown className="w-3 h-3" style={{ color }} />
        </motion.span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="pb-2.5">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Fisheries Card ───────────────────────────────────────────────────────────

// Published thermal-preference ranges for North Indian Ocean pelagic species.
// Sources cited explicitly — these are NOT model predictions.
// Lehodey et al. 1997 (Deep-Sea Res II) for skipjack;
// Pillai & Alagaraja 1988 (CMFRI Spec. Publ.) for Indian mackerel;
// Srinath et al. 2012 (IOTC Working Paper) for oil sardine.
const NIO_SPECIES = [
  { name: 'Skipjack Tuna', sci: 'Katsuwonus pelamis',  tMin: 20, tMax: 29, cite: 'Lehodey et al. 1997' },
  { name: 'Indian Mackerel', sci: 'Rastrelliger kanagurta', tMin: 22, tMax: 29, cite: 'Pillai & Alagaraja 1988' },
  { name: 'Indian Oil Sardine', sci: 'Sardinella longiceps', tMin: 22, tMax: 28, cite: 'Srinath et al. 2012' },
];

function FisheriesCard({ data, selectedPixel, setSelectedPixel, modelDepths, resetKey }) {
  const color = 'var(--color-sector-fisheries)';

  if (!data) return <InsufficientData />;

  // Per-pixel vs basin-average display
  const isLocal = selectedPixel?.temp !== undefined;
  const displayTemp = isLocal ? selectedPixel.temp : null; // surface temp at selected location

  return (
    <>
      {/* Headline — thermocline depth */}
      <div
        className="rounded-lg p-2 mb-2 flex gap-3"
        style={{
          background: 'color-mix(in srgb, var(--color-sector-fisheries) 10%, transparent)',
          border: '1px solid color-mix(in srgb, var(--color-sector-fisheries) 20%, transparent)',
        }}
      >
        <HeadlineStat
          value={data.thermoclineDepth ?? '—'}
          unit="m"
          label="Thermocline Depth"
          color={color}
        />
        <div className="w-px bg-[var(--color-paper-border)]" />
        <HeadlineStat
          value={data.thermoclineTemp ?? '—'}
          unit="°C"
          label="Temp @ TC"
          color={color}
        />
      </div>

      {/* Selected pixel info */}
      {isLocal && (
        <div
          className="rounded-lg px-2 py-1.5 mb-1.5 flex items-center justify-between"
          style={{
            background: 'color-mix(in srgb, var(--color-sector-fisheries) 8%, transparent)',
            border: '1px solid color-mix(in srgb, var(--color-sector-fisheries) 20%, transparent)',
          }}
        >
          <div className="flex flex-col">
            <span className="text-[8px] font-bold uppercase tracking-wider" style={{ color }}>
              Selected Point
            </span>
            <span className="text-[9px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
              {selectedPixel.lat.toFixed(2)}°N, {selectedPixel.lon.toFixed(2)}°E
            </span>
            <span className="text-[9px] font-mono" style={{ color: 'var(--color-ink-dark)' }}>
              Surface: {displayTemp !== null ? `${(+displayTemp).toFixed(2)}°C` : '—'}
            </span>
          </div>
          <button
            onClick={() => setSelectedPixel(null)}
            className="text-[8px] font-bold px-1.5 py-0.5 rounded hover:opacity-80 transition-opacity"
            style={{
              background: 'color-mix(in srgb, var(--color-sector-fisheries) 15%, transparent)',
              border: '1px solid color-mix(in srgb, var(--color-sector-fisheries) 30%, transparent)',
              color,
            }}
          >
            RESET
          </button>
        </div>
      )}
      {!isLocal && (
        <p className="text-[8px] italic mb-1" style={{ color: 'var(--color-ink-muted)' }}>
          Click any ocean pixel for location-specific data
        </p>
      )}

      <AccordionSection title="Thermocline Details" color={color} defaultOpen={true} resetKey={resetKey}>
        <Row label="Gradient" value={data.thermoclineGradPer100m} unit=" °C/100m" />
        <Row label="Basin-avg TC depth" value={data.thermoclineDepth} unit=" m" />
      </AccordionSection>

      <AccordionSection title="Aggregation Band (20–29 °C)" color={color} defaultOpen={true} resetKey={resetKey}>
        {data.aggregationBandDepths.length > 0 ? (
          <div className="flex flex-wrap gap-1 py-1">
            {data.aggregationBandDepths.map(d => (
              <Pill key={d} label={`${d}m`} color={color} />
            ))}
          </div>
        ) : (
          <p className="stat-label italic py-1">No aggregation band at 20–29 °C in 0–200m</p>
        )}
      </AccordionSection>

      {/* F3 — Species Reference Advisory */}
      <AccordionSection title="Species Thermal Reference" color={color} defaultOpen={false} resetKey={resetKey}>
        <p className="text-[8px] italic mb-1.5 leading-tight" style={{ color: 'var(--color-ink-muted)' }}>
          Published thermal-preference ranges for North Indian Ocean pelagic species.
          Cross-referenced against today’s real basin-average aggregation depths.
          <strong> Not a model prediction — reference data only.</strong>
        </p>
        <div className="flex flex-col gap-1.5">
          {NIO_SPECIES.map(sp => {
            // Use the real per-depth basin-mean temperatures from the aggregation band profile.
            // Each entry in aggregationBandProfile is { depth, meanTemp } where meanTemp is
            // the genuine basin-average temperature at that depth (from real tensor pixels).
            // A depth matches a species if its real mean temperature falls within [sp.tMin, sp.tMax].
            const profile = data.aggregationBandProfile ?? [];
            const matchEntries = profile.filter(({ meanTemp }) =>
              meanTemp >= sp.tMin && meanTemp <= sp.tMax
            );
            const matchDepths = matchEntries.map(({ depth }) => depth);
            const hasMatch = matchDepths.length > 0;
            const color2 = hasMatch ? '#22c55e' : 'var(--color-ink-muted)';
            return (
              <div
                key={sp.name}
                className="rounded px-2 py-1"
                style={{
                  background: hasMatch
                    ? 'color-mix(in srgb, #22c55e 7%, transparent)'
                    : 'color-mix(in srgb, var(--color-ink-muted) 5%, transparent)',
                  border: `1px solid ${hasMatch ? 'color-mix(in srgb, #22c55e 22%, transparent)' : 'var(--color-paper-border)'}`,
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[9px] font-bold" style={{ color: color2 }}>
                    {hasMatch ? '✓' : '—'} {sp.name}
                  </span>
                  <span className="text-[8px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
                    {sp.tMin}–{sp.tMax}°C
                  </span>
                </div>
                <div className="text-[8px]" style={{ color: 'var(--color-ink-muted)' }}>
                  <span className="italic">{sp.sci}</span>
                  {hasMatch && matchEntries.length > 0 && (
                    <span> — active at {matchEntries.map(e => `${e.depth}m (${e.meanTemp}°C)`).join(', ')}</span>
                  )}
                  {!hasMatch && <span> — no overlap with current basin temps</span>}
                </div>
                <div className="text-[7px] mt-0.5" style={{ color: 'var(--color-ink-muted)', opacity: 0.7 }}>
                  Ref: {sp.cite}
                </div>
              </div>
            );
          })}
        </div>
      </AccordionSection>
    </>
  );
}

// ── Naval Card ───────────────────────────────────────────────────────────────

function NavalCard({ data, resetKey }) {
  const color = 'var(--color-sector-naval)';

  if (!data) return <InsufficientData />;

  const shadowTier = data.shadowZone
    ? `${data.shadowZone[0]}–${data.shadowZone[1]} m`
    : null;

  const strengthTier = data.maxThermoclinePixel
    ? (
        data.maxThermoclinePixel.value > (data.thermoclineTiers?.moderate ?? 0.1)
          ? 'Strong'
          : data.maxThermoclinePixel.value > (data.thermoclineTiers?.weak ?? 0.05)
          ? 'Moderate'
          : 'Weak'
      )
    : null;

  return (
    <>
      {/* Headline trio */}
      <div
        className="rounded-lg p-2 mb-2 flex justify-around"
        style={{
          background: 'color-mix(in srgb, var(--color-sector-naval) 10%, transparent)',
          border: '1px solid color-mix(in srgb, var(--color-sector-naval) 20%, transparent)',
        }}
      >
        <HeadlineStat value={data.sld ?? '—'} unit="m" label="SLD" color={color} />
        <div className="w-px bg-[var(--color-paper-border)]" />
        <HeadlineStat value={shadowTier ?? '—'} unit="" label="Shadow Zone" color={color} />
        <div className="w-px bg-[var(--color-paper-border)]" />
        <HeadlineStat value={strengthTier ?? '—'} unit="" label="TC Strength" color={color} />
      </div>

      <AccordionSection title="Sonic Layer Depth" color={color} defaultOpen={true} resetKey={resetKey}>
        <Row
          label="Speed @ SLD"
          value={data.sldSpeed}
          unit=" m/s"
          assumed={!data.hasRealSss}
          assumedNote={`Salinity assumed ${data.assumedSalinity} PSU — affects accuracy in river-freshened (Bay of Bengal ~32 PSU) vs saline (Arabian Sea ~36 PSU) regions.`}
        />
        <Row label="Basin-avg Thermocline" value={data.thermoclineDepth} unit=" m" />
      </AccordionSection>

      <AccordionSection title="Sonar Shadow Zone" color={color} defaultOpen={true} resetKey={resetKey}>
        {data.shadowZone ? (
          <>
            <Row label="Depth Range" value={`${data.shadowZone[0]}–${data.shadowZone[1]}`} unit=" m" />
            <Row label="Threshold dc/dz" value={data.shadowThreshold?.toFixed(3)} unit=" s⁻¹" />
            {data.bestShadowPixel && data.topShadowPixels && (() => {
              // Convert tensor row/col indices back to geographic coordinates.
              // The tensor row/col space is: r=0 → LAT_MIN=-10, c=0 → LON_MIN=45.
              // The shadow zone rings are DEPTH-INVARIANT — they represent column-integrated
              // dc/dz across the full 30–200m shadow zone range, not a single depth level.
              // Moving the slider does not (and should not) change which locations appear.
              const numShadow = data.topShadowPixels.length;
              const bsp = data.bestShadowPixel;
              // Use bestShadowPixel.r and .c; we don't know numLat/numLon dynamically here
              // so we use the known model grid (161 × 221 for the IndoPacific domain)
              const numLat = 161, numLon = 221;
              const bLat = -10 + (bsp.r / (numLat - 1)) * 40;
              const bLon = 45 + (bsp.c / (numLon - 1)) * 55;
              return (
                <div
                  className="mt-1.5 rounded-lg p-1.5"
                  style={{
                    background: 'color-mix(in srgb, #22c55e 8%, transparent)',
                    border: '1px solid color-mix(in srgb, #22c55e 25%, transparent)',
                  }}
                >
                  <span className="text-[8px] font-bold uppercase tracking-wider" style={{ color: '#22c55e' }}>
                    ● Optimal Concealment Point ({numShadow} hotspot{numShadow > 1 ? 's' : ''} mapped)
                  </span>
                  <div className="flex justify-between mt-0.5">
                    <span className="stat-label">Best Location</span>
                    <span className="stat-value font-mono text-[9px]">
                      {bLat.toFixed(1)}°N, {bLon.toFixed(1)}°E
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="stat-label">dc/dz</span>
                    <span className="stat-value">{bsp.gradient} s⁻¹</span>
                  </div>
                  <p className="text-[8px] italic mt-0.5" style={{ color: 'var(--color-ink-muted)' }}>
                    Green rings on map. Depth-slider does not affect these — shadow zone is a
                    column property (30–{data.shadowZone[1]}m), not a single depth.
                  </p>
                </div>
              );
            })()}
            <p className="stat-label italic mt-1 leading-tight">
              Downward ray bending creates detection blind spots. Rings mark ocean locations
              with strongest negative sound-speed gradient — all confirmed ocean pixels.
            </p>
          </>
        ) : (
          <p className="stat-label italic py-1">No distinct shadow zone found.</p>
        )}
      </AccordionSection>

      <AccordionSection title="Thermocline Strength" color={color} defaultOpen={false} resetKey={resetKey}>
        {data.maxThermoclinePixel ? (
          <>
            <Row label="Basin Max Gradient" value={data.maxThermoclinePixel.value.toFixed(2)} unit=" °C/m" />
            <Row
              label="Max Location"
              value={`${(-10 + (data.maxThermoclinePixel.r / 160) * 40).toFixed(1)}°N, ${(45 + (data.maxThermoclinePixel.c / 220) * 55).toFixed(1)}°E`}
              unit=""
            />
            <div className="flex flex-wrap gap-1 mt-1.5">
              <Pill label={`Weak < ${data.thermoclineTiers?.weak?.toFixed(2)}`} color="#9e9ac8" />
              <Pill label={`Strong > ${data.thermoclineTiers?.moderate?.toFixed(2)}`} color="#54278f" />
            </div>
          </>
        ) : (
          <p className="stat-label italic py-1">No thermocline data.</p>
        )}
      </AccordionSection>

      <AccordionSection title="Sound Speed Profile" color={color} defaultOpen={false} resetKey={resetKey}>
        <div className="flex flex-col gap-0.5 max-h-24 overflow-y-auto pr-1 py-1">
          {data.soundSpeedProfile.slice(0, 15).map(({ depth, speed }) => (
            <div key={depth} className="flex justify-between text-[10px] font-mono">
              <span style={{ color: 'var(--color-ink-muted)' }}>{depth}m</span>
              <span className="stat-value">{speed} m/s</span>
            </div>
          ))}
        </div>
        <Callout>
          {data.hasRealSss
            ? `Per-pixel SSS available but SVP requires full 3D salinity. Mean ${data.assumedSalinity.toFixed(1)} PSU used for subsurface calculations.`
            : `≈ Salinity assumed ${data.assumedSalinity} PSU — not measured`}
        </Callout>
        {data.soundSpeedProfile.some(s => s.depth > 300) && (
          <Callout>Limited training data below 300m — acoustic profiles may be less reliable.</Callout>
        )}
      </AccordionSection>
    </>
  );
}

function ComfortScoreCard({ displayTemp, localShockZones, resetKey, color }) {
  let score = 'GREEN';
  let message = 'Stable thermal conditions — good for diving.';
  let scoreColor = '#22c55e'; // Green
  
  const maxGradient = localShockZones.length > 0 
    ? Math.max(...localShockZones.map(z => z.grad))
    : 0;

  if (maxGradient > 0.4 || displayTemp < 15) {
    score = 'RED';
    message = 'High thermal gradients or very cold surface — potentially unsafe.';
    scoreColor = '#ef4444';
  } else if (maxGradient > 0.2 || displayTemp < 22) {
    score = 'YELLOW';
    message = 'Moderate thermal gradients — standard precautions apply.';
    scoreColor = '#f59e0b';
  }

  return (
    <AccordionSection title="Comfort / Safety Score (Today's Reading)" color={color} defaultOpen={true} resetKey={resetKey}>
      <div className="flex flex-col gap-1 py-1">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] font-mono text-[var(--color-ink-muted)]">Current Conditions</span>
          <Pill label={score} color={scoreColor} />
        </div>
        <p className="stat-label italic leading-tight">{message}</p>
      </div>
    </AccordionSection>
  );
}

function ScubaCard({ data, selectedPixel, setSelectedPixel, date, activeSector, modelDepths, resetKey }) {
  const color = 'var(--color-sector-scuba)';

  if (!data) return <InsufficientData />;

  const displayTemp = selectedPixel?.temp !== undefined ? selectedPixel.temp : data.shallowTemp;
  const isLocal = selectedPixel?.temp !== undefined;
  const wetsuit = getWetsuitRecommendation(displayTemp);

  // Build 0–40m profile with 40m interpolation
  const baseProfile = [];
  const sourceDepths = modelDepths?.length ? modelDepths : [0, 5, 10, 20, 30, 50, 75, 100];
  let tempAt30 = null, tempAt50 = null;

  for (let i = 0; i < sourceDepths.length; i++) {
    const d = sourceDepths[i];
    if (d > 50) break;
    let t = null;
    if (isLocal && selectedPixel.fullProfile) {
      t = selectedPixel.fullProfile[i] !== null ? +selectedPixel.fullProfile[i].toFixed(2) : null;
    } else {
      const b = data.profile.find(p => p.depth === d);
      t = b ? b.temp : null;
    }
    if (d === 30) tempAt30 = t;
    if (d === 50) tempAt50 = t;
    if (d <= 30) baseProfile.push({ depth: d, temp: t, label: `${d}m` });
  }

  if (tempAt30 !== null && tempAt50 !== null) {
    baseProfile.push({ depth: 40, temp: +((tempAt30 + tempAt50) / 2).toFixed(2), label: '40m (interpolated)' });
  } else if (tempAt30 !== null) {
    baseProfile.push({ depth: 40, temp: tempAt30, label: '40m (est)' });
  }

  const localShockZones = [];
  for (let k = 0; k < baseProfile.length - 1; k++) {
    const p1 = baseProfile[k], p2 = baseProfile[k + 1];
    if (p1.temp === null || p2.temp === null) continue;
    const drop = p1.temp - p2.temp;
    const dz = p2.depth - p1.depth;
    if (dz === 0) continue;
    const grad = drop / dz;
    if (grad > 0.2) {
      localShockZones.push({
        fromDepth: p1.depth, toDepth: p2.depth,
        boundaryDepth: +(p1.depth + dz / 2).toFixed(1),
        drop: +drop.toFixed(2), grad: +grad.toFixed(2),
      });
    }
  }

  return (
    <>
      {/* Headline */}
      <div
        className="rounded-lg p-2 mb-2"
        style={{
          background: 'color-mix(in srgb, var(--color-sector-scuba) 10%, transparent)',
          border: '1px solid color-mix(in srgb, var(--color-sector-scuba) 20%, transparent)',
        }}
      >
        <div className="flex items-center gap-1.5 mb-1">
          <span
            className="text-[8px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded"
            style={{
              background: isLocal
                ? 'color-mix(in srgb, var(--color-sector-scuba) 20%, transparent)'
                : 'color-mix(in srgb, var(--color-ink-muted) 15%, transparent)',
              color: isLocal ? 'var(--color-sector-scuba)' : 'var(--color-ink-muted)',
            }}
          >
            {isLocal ? 'SELECTED LOCATION' : 'BASIN AVERAGE'}
          </span>
          {isLocal && (
            <div className="flex items-center gap-2">
              <span className="text-[8px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>
                {selectedPixel.lat.toFixed(2)}°N, {selectedPixel.lon.toFixed(2)}°E
              </span>
              <button
                onClick={() => setSelectedPixel(null)}
                className="text-[8px] font-bold underline hover:text-white transition-colors"
                style={{ color: 'var(--color-sector-scuba)' }}
              >
                [RESET]
              </button>
            </div>
          )}
        </div>
        <div className="flex gap-3">
          <HeadlineStat value={displayTemp !== null ? +displayTemp.toFixed(1) : '—'} unit="°C" label="Surface Temp" color={color} />
          <div className="w-px bg-[var(--color-paper-border)]" />
          <div className="flex flex-col items-center py-2 px-1 flex-1">
            <span className="text-[8px] font-bold uppercase tracking-widest mb-1" style={{ color }}>Wetsuit</span>
            <span className="text-[10px] font-bold text-center leading-tight" style={{ color: 'var(--color-ink-dark)' }}>
              {wetsuit.label ?? '—'}
            </span>
          </div>
        </div>
        {!isLocal && (
          <p className="text-[8px] italic mt-1" style={{ color: 'var(--color-ink-muted)' }}>
            Click map for location-specific data
          </p>
        )}
      </div>

      <ComfortScoreCard displayTemp={displayTemp} localShockZones={localShockZones} resetKey={resetKey} color={color} />

      <AccordionSection title="Cold Shock Zones (>0.2 °C/m)" color="#ef4444" defaultOpen={true} resetKey={resetKey}>
        {localShockZones.length > 0 ? (
          <div className="flex flex-col gap-1 py-1">
            {localShockZones.map((z, idx) => (
              <div
                key={idx}
                className="rounded px-2 py-1"
                style={{
                  background: 'color-mix(in srgb, #ef4444 8%, transparent)',
                  border: '1px solid color-mix(in srgb, #ef4444 20%, transparent)',
                }}
              >
                <div className="flex justify-between text-[10px] font-mono">
                  <span style={{ color: '#ef4444' }}>~{z.boundaryDepth}m</span>
                  <span className="font-bold" style={{ color: '#ef4444' }}>↓{z.drop}°C</span>
                </div>
                <div className="text-[9px] italic" style={{ color: 'var(--color-ink-muted)' }}>
                  {z.fromDepth}–{z.toDepth}m ({z.grad} °C/m)
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="stat-label italic py-1">No cold-shock layers found in 0–40m</p>
        )}
      </AccordionSection>

      <AccordionSection title="Depth Profile (0–40m)" color={color} defaultOpen={false} resetKey={resetKey}>
        <div className="flex flex-col gap-0.5 py-1">
          {baseProfile.map(({ depth, temp, label }) => (
            <div key={depth} className="flex justify-between text-[10px] font-mono">
              <span style={{ color: 'var(--color-ink-muted)' }}>{label}</span>
              <span className="stat-value">{temp !== null ? `${temp} °C` : '—'}</span>
            </div>
          ))}
        </div>
      </AccordionSection>
    </>
  );
}

// ── Main export ──────────────────────────────────────────────────────────────

export default function SectorInsightsCard({ activeSector, sectorAnalysis, selectedPixel, setSelectedPixel, date, modelDepths }) {
  if (!activeSector) return null;

  // Use date + depth as reset key so accordions collapse when data changes
  const resetKey = `${activeSector}-${date}`;

  const SECTOR_CONFIG = {
    fisheries: {
      title: 'Fisheries Analytics',
      color: 'var(--color-sector-fisheries)',
      card: (
        <FisheriesCard
          data={sectorAnalysis?.fisheries}
          selectedPixel={selectedPixel}
          setSelectedPixel={setSelectedPixel}
          modelDepths={modelDepths}
          resetKey={resetKey}
        />
      ),
    },
    naval: {
      title: 'Naval Analytics',
      color: 'var(--color-sector-naval)',
      card: <NavalCard data={sectorAnalysis?.naval} resetKey={resetKey} />,
    },
    scuba: {
      title: 'Scuba Dive Analytics',
      color: 'var(--color-sector-scuba)',
      card: (
        <ScubaCard
          data={sectorAnalysis?.scuba}
          selectedPixel={selectedPixel}
          setSelectedPixel={setSelectedPixel}
          date={date}
          activeSector={activeSector}
          modelDepths={modelDepths}
          resetKey={resetKey}
        />
      ),
    },
  };

  const content = SECTOR_CONFIG[activeSector];
  if (!content) return null;

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={activeSector}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
        className="mt-2 rounded-xl border overflow-hidden"
        style={{
          borderColor: 'var(--color-paper-border)',
          background: 'var(--color-paper-surface)',
        }}
      >
        {/* Card header */}
        <div
          className="px-3 py-2 border-b"
          style={{
            borderColor: 'var(--color-paper-border)',
            background: 'color-mix(in srgb, var(--color-paper-elevated) 60%, transparent)',
          }}
        >
          <span
            className="text-[9px] font-black uppercase tracking-widest"
            style={{ color: content.color }}
          >
            {content.title}
          </span>
        </div>

        {/* Card body */}
        <div className="px-3 py-1">
          {content.card}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
