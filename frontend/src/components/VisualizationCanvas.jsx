import React, { useMemo, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Globe, ChevronDown, X } from 'lucide-react';
import Plot from 'react-plotly.js';
import LoadingScreen from './LoadingScreen';
import ProfilePopup from './ProfilePopup';
import MapLegend from './MapLegend';
import { mackenzie, isValidOceanPixel } from '../hooks/useSectorAnalysis';

// Pick one satellite image at random per session — locked on module init
import sat1 from '../assets/satellite-1.png';
import sat2 from '../assets/satellite-2.png';
import sat3 from '../assets/satellite-3.png';
import sat4 from '../assets/satellite-4.png';
import sat5 from '../assets/satellite-5.png';
import sat6 from '../assets/satellite-6.png';
const _SATS = [sat1, sat2, sat3, sat4, sat5, sat6];
const RANDOM_SAT = _SATS[Math.floor(Math.random() * _SATS.length)];


class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-red-50 rounded-xl border border-red-200 p-6 text-center">
          <h2 className="text-red-800 font-bold mb-2 font-display text-lg">Visualization Render Error</h2>
          <p className="text-red-600 text-sm max-w-md font-mono">{this.state.error?.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

// Geographic bounds for the expanded prediction grid
const LON_MIN = 45.0;
const LON_MAX = 100.0;
const LAT_MIN = -10.0;
const LAT_MAX = 30.0;

function AnimatedValue({ value, suffix = '' }) {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, scale: 0.8, color: 'var(--color-accent)' }}
      animate={{ opacity: 1, scale: 1, color: 'var(--color-ink-dark)' }}
      transition={{ duration: 0.3 }}
      className="stat-value"
    >
      {value}{suffix}
    </motion.span>
  );
}

function IdleState() {
  return (
    <motion.div
      className="absolute inset-0 flex flex-col items-center justify-center wireframe-grid rounded-xl"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div
        className="animate-pulse-soft mb-6"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <div className="relative p-6 rounded-full bg-[var(--color-paper-surface)] border border-[var(--color-paper-border)] shadow-sm">
          <Globe className="w-12 h-12 text-[var(--color-ink-light)] relative z-10" strokeWidth={1.5} />
        </div>
      </motion.div>

      <h3 className="text-[var(--color-ink-dark)] text-xl font-bold font-display mb-1">
        Awaiting Scan Initialization
      </h3>
      <p className="text-[var(--color-ink-light)] text-sm font-medium">
        Configure parameters and launch AI prediction
      </p>

      {['top-6 left-6', 'top-6 right-6', 'bottom-6 left-6', 'bottom-6 right-6'].map((pos, i) => (
        <div
          key={i}
          className={`absolute ${pos} w-4 h-4 border-[var(--color-ink-light)]/40 ${
            i === 0 ? 'border-t-2 border-l-2'
            : i === 1 ? 'border-t-2 border-r-2'
            : i === 2 ? 'border-b-2 border-l-2'
            : 'border-b-2 border-r-2'
          }`}
        />
      ))}
    </motion.div>
  );
}

// ── Naval SVP Panel ────────────────────────────────────────────────────────
// Floating Plotly line-chart showing per-point Sound Speed Profile.
// Displayed on Naval tab when a map pixel is selected.
// Depth > 300m points receive a warning marker (limited training data).
function NavalSvpPanel({ svpData, lat, lon, assumedSalinity, hasRealSss, onClose }) {
  if (!svpData || svpData.length === 0) return null;

  const depths = svpData.map(p => p.depth);
  const speeds = svpData.map(p => p.speed);

  // Identify uncertain depths (> 300m — limited training data per project notes)
  const reliableX = [], reliableY = [], uncertainX = [], uncertainY = [];
  for (let i = 0; i < svpData.length; i++) {
    if (svpData[i].depth <= 300) {
      reliableX.push(svpData[i].speed);
      reliableY.push(svpData[i].depth);
    } else {
      uncertainX.push(svpData[i].speed);
      uncertainY.push(svpData[i].depth);
    }
  }

  const allSpeeds = speeds.filter(Boolean);
  const sMin = allSpeeds.length ? Math.min(...allSpeeds) - 2 : 1480;
  const sMax = allSpeeds.length ? Math.max(...allSpeeds) + 2 : 1510;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.25 }}
      className="absolute top-4 right-4 z-30 rounded-xl shadow-2xl overflow-hidden"
      style={{
        width: 240,
        background: 'color-mix(in srgb, var(--color-paper-surface) 94%, transparent)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        border: '1px solid var(--color-paper-border)',
      }}
    >
      {/* Header */}
      <div
        className="px-3 py-2 flex items-center justify-between"
        style={{
          background: 'var(--color-paper-elevated)',
          borderBottom: '1px solid var(--color-paper-border)',
        }}
      >
        <div>
          <span className="text-[8px] font-black uppercase tracking-widest" style={{ color: 'var(--color-sector-naval)' }}>
            SVP — Selected Point
          </span>
          <p className="text-[9px] font-mono mt-0.5" style={{ color: 'var(--color-ink-muted)' }}>
            {lat.toFixed(2)}°N, {lon.toFixed(2)}°E
          </p>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:opacity-70 transition-opacity"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          <X size={12} />
        </button>
      </div>

      {/* Chart */}
      <Plot
        data={[
          // Reliable segment (0–300m) — solid line
          {
            type: 'scatter',
            x: reliableX,
            y: reliableY,
            mode: 'lines+markers',
            name: '0–300m',
            line: { color: '#3b82f6', width: 2 },
            marker: { size: 5, color: '#3b82f6' },
            showlegend: false,
          },
          // Uncertain segment (>300m) — dashed, different colour
          ...(uncertainX.length > 0 ? [{
            type: 'scatter',
            x: uncertainX,
            y: uncertainY,
            mode: 'lines+markers',
            name: '>300m (uncertain)',
            line: { color: '#f59e0b', width: 2, dash: 'dash' },
            marker: { size: 5, color: '#f59e0b', symbol: 'diamond-open' },
            showlegend: false,
          }] : []),
        ]}
        layout={{
          width: 240,
          height: 220,
          margin: { l: 46, r: 8, t: 6, b: 30 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { family: 'Inter', size: 10, color: 'var(--color-ink-medium)' },
          xaxis: {
            title: { text: 'c (m/s)', font: { size: 9 }, standoff: 4 },
            range: [sMin, sMax],
            gridcolor: 'rgba(100,100,100,0.15)',
            tickfont: { size: 8 },
          },
          yaxis: {
            title: { text: 'Depth (m)', font: { size: 9 }, standoff: 4 },
            autorange: 'reversed', // 0 at top, depth increases downward
            gridcolor: 'rgba(100,100,100,0.15)',
            tickfont: { size: 8 },
          },
          showlegend: false,
        }}
        config={{ displayModeBar: false, responsive: false, staticPlot: true }}
      />

      {/* Footnote */}
      <div className="px-3 pb-2">
        {uncertainX.length > 0 && (
          <p className="text-[8px] italic" style={{ color: '#f59e0b' }}>
            ◆ Dashed = limited training data (&gt;300m)
          </p>
        )}
        {!hasRealSss && (
          <p className="text-[8px] italic mt-0.5" style={{ color: 'var(--color-ink-muted)' }}>
            Salinity assumed {assumedSalinity} PSU
          </p>
        )}
      </div>
    </motion.div>
  );
}

function ActiveState({ plotData, fullTensor, depthIndex, modelDepths, renderMode, activeSector, sectorAnalysis, onPixelSelect, showThermoclineStrength, showShadowZone, date }) {
  const [popupData, setPopupData] = useState(null);
  const [isFading, setIsFading] = useState(false);
  // Naval: per-point SVP selection
  const [navalPixel, setNavalPixel] = useState(null); // { lat, lon, svp: [{depth, speed}] }

  // Clear naval pixel selection when sector changes away from naval
  React.useEffect(() => {
    if (activeSector !== 'naval') setNavalPixel(null);
  }, [activeSector]);

  // A3: Heatmap Cross-fade transition
  React.useEffect(() => {
    setIsFading(true);
    const timer = setTimeout(() => setIsFading(false), 250);
    return () => clearTimeout(timer);
  }, [plotData, date]);

  const { scaledZ, customDataZ, lonArray, latArray, dataZmin, dataZmax, smoothedImageUrl } = useMemo(() => {
    if (!plotData || !plotData.length || !plotData[0].length) return {};

    const numLat = plotData.length;
    const numLon = plotData[0].length;

    const latArray = Array.from({ length: numLat }, (_, i) => LAT_MIN + (i / (numLat - 1)) * (LAT_MAX - LAT_MIN));
    const lonArray = Array.from({ length: numLon }, (_, i) => LON_MIN + (i / (numLon - 1)) * (LON_MAX - LON_MIN));

    // Smooth continuous colorscale for CNN mode
    const SMOOTH_CMOCEAN_THERMAL = [
      [0.00, '#042333'], [0.10, '#0a3560'], [0.20, '#1b3472'],
      [0.30, '#3f3183'], [0.40, '#662479'], [0.50, '#8a1761'],
      [0.60, '#ab1a45'], [0.70, '#c6312a'], [0.80, '#da561a'],
      [0.90, '#e8841a'], [1.00, '#eff542']
    ];
    // Colorscale for gradients
    const GRADIENT_SCALE = [
      [0.0, '#ffffff'],
      [0.3, '#f2f0f7'],
      [0.6, '#cbc9e2'],
      [0.8, '#9e9ac8'],
      [0.9, '#756bb1'],
      [1.0, '#54278f']
    ];
    const isThermoclineMap = showThermoclineStrength && activeSector === 'naval' && sectorAnalysis?.naval?.thermoclineMap;
    const sourceData = isThermoclineMap ? sectorAnalysis.naval.thermoclineMap : plotData;
    const activeColorStops = isThermoclineMap ? GRADIENT_SCALE : SMOOTH_CMOCEAN_THERMAL;

    // Calculate robust dynamic min/max using percentiles (Active Scale with clipping)
    let validValues = [];
    for (let y = 0; y < numLat; y++) {
      for (let x = 0; x < numLon; x++) {
        const val = sourceData[y][x];
        if (val !== null && !isNaN(val)) {
          validValues.push(val);
        }
      }
    }
    
    let min = 0, max = 1;
    if (validValues.length > 0) {
      validValues.sort((a, b) => a - b);
      // Use 2nd and 98th percentile to dynamically ignore coastal bleed and padding edge artifacts
      const p2 = Math.floor(validValues.length * 0.02);
      const p98 = Math.floor(validValues.length * 0.98);
      min = validValues[p2];
      max = validValues[p98];
    }

    const hexToRgb = (hex) => {
      const r = parseInt(hex.slice(1,3), 16);
      const g = parseInt(hex.slice(3,5), 16);
      const b = parseInt(hex.slice(5,7), 16);
      return [r, g, b];
    };
    const colorStops = activeColorStops.map(s => ({ val: s[0], rgb: hexToRgb(s[1]) }));

    const getColor = (norm) => {
      if (norm <= 0) return colorStops[0].rgb;
      if (norm >= 1) return colorStops[colorStops.length - 1].rgb;
      for (let i = 0; i < colorStops.length - 1; i++) {
        if (norm >= colorStops[i].val && norm <= colorStops[i+1].val) {
          const t = (norm - colorStops[i].val) / (colorStops[i+1].val - colorStops[i].val);
          const c1 = colorStops[i].rgb;
          const c2 = colorStops[i+1].rgb;
          return [
            Math.round(c1[0] + t * (c2[0] - c1[0])),
            Math.round(c1[1] + t * (c2[1] - c1[1])),
            Math.round(c1[2] + t * (c2[2] - c1[2]))
          ];
        }
      }
      return [0,0,0];
    };

    // 1. Temporary/offscreen canvas for raw data mapping
      const offCanvas = document.createElement('canvas');
      offCanvas.width = numLon;
      offCanvas.height = numLat;
      const offCtx = offCanvas.getContext('2d');
      const imgData = offCtx.createImageData(numLon, numLat);

      for (let y = 0; y < numLat; y++) {
        for (let x = 0; x < numLon; x++) {
          // Flip Y because Plotly geo image mapping draws from top to bottom
          const flippedY = numLat - 1 - y;
          const idx = (flippedY * numLon + x) * 4;
          const val = sourceData[y][x];

          if (val === null || isNaN(val)) {
            // Fill with land color (#E8E2D5) so interpolation blends ocean with land naturally
            imgData.data[idx] = 232;
            imgData.data[idx + 1] = 226;
            imgData.data[idx + 2] = 213;
            imgData.data[idx + 3] = 255;
          } else {
            const norm = (val - min) / (max - min || 1);
            const rgb = getColor(norm);
            imgData.data[idx] = rgb[0];
            imgData.data[idx + 1] = rgb[1];
            imgData.data[idx + 2] = rgb[2];
            imgData.data[idx + 3] = 255;
          }
        }
      }
      offCtx.putImageData(imgData, 0, 0);

      // 2. Main visualizer canvas for interpolated drawing
      const mainCanvas = document.createElement('canvas');
      const scale = 4; // Upscale for smoothness
      mainCanvas.width = numLon * scale;
      mainCanvas.height = numLat * scale;
      const mainCtx = mainCanvas.getContext('2d');
      
      // CRUCIAL: Native browser interpolation
      mainCtx.imageSmoothingEnabled = true;
      mainCtx.imageSmoothingQuality = 'high';
      mainCtx.drawImage(offCanvas, 0, 0, mainCanvas.width, mainCanvas.height);

      // Re-apply the land mask to cut off any bleeding from interpolation
      const mainImgData = mainCtx.getImageData(0, 0, mainCanvas.width, mainCanvas.height);
      for (let y = 0; y < mainCanvas.height; y++) {
        for (let x = 0; x < mainCanvas.width; x++) {
          const origX = Math.floor(x / scale);
          const origY = Math.floor(y / scale);
          const val = sourceData[numLat - 1 - origY][origX];
          if (val === null || isNaN(val)) {
            const idx = (y * mainCanvas.width + x) * 4;
            // Force land color (#E8E2D5) to keep coastline sharp
            mainImgData.data[idx] = 232;
            mainImgData.data[idx + 1] = 226;
            mainImgData.data[idx + 2] = 213;
            mainImgData.data[idx + 3] = 255;
          }
        }
      }
      mainCtx.putImageData(mainImgData, 0, 0);

      const dataUrl = mainCanvas.toDataURL('image/png');
      const customDataZ = sourceData.map(row => [...row]);

      return { customDataZ, lonArray, latArray, dataZmin: min, dataZmax: max, smoothedImageUrl: dataUrl, isThermoclineMap };
  }, [plotData, showThermoclineStrength, activeSector, sectorAnalysis]);

  // ── Sector overlay: scatter highlight over the zone of interest ──────────
  const sectorOverlayTrace = useMemo(() => {
    if (!activeSector || !sectorAnalysis || !fullTensor || !plotData) return null;

    const numLat = plotData.length;
    const numLon = plotData[0]?.length ?? 0;
    const latArr = Array.from({ length: numLat }, (_, i) => LAT_MIN + (i / (numLat - 1)) * (LAT_MAX - LAT_MIN));
    const lonArr = Array.from({ length: numLon }, (_, i) => LON_MIN + (i / (numLon - 1)) * (LON_MAX - LON_MIN));

    const STEP = activeSector === 'fisheries' ? 8 : 4; // subsample for performance
    const lats = [];
    const lons = [];
    const texts = [];

    const colorMap = { fisheries: '#c06d38', naval: '#1a4e7a', scuba: '#0e7490' };
    const markerColor = colorMap[activeSector] ?? '#888';

    // For each sector we select a specific depth slice and filter pixels
    let targetDepthIndex = depthIndex;
    let tempTest = () => false;
    let hoverLabel = '';
    let markerConfig = {
      size: 4,
      color: markerColor,
      opacity: 0.55,
      symbol: 'circle',
    };

    if (activeSector === 'fisheries') {
      targetDepthIndex = sectorAnalysis.fisheries?.mapDepthIndex ?? depthIndex;
      hoverLabel = 'Aggregation Band';
      tempTest = (t) => t !== null && !isNaN(t) && t >= 20 && t <= 29;
    } else if (activeSector === 'naval') {
      // Dot appearance is ALWAYS consistent regardless of shadow zone toggle.
      // The shadow zone is visualized separately via green ring markers.
      targetDepthIndex = sectorAnalysis.naval?.mapDepthIndex ?? depthIndex;
      hoverLabel = 'SLD Layer';
      tempTest = (t) => t !== null && !isNaN(t);
    } else if (activeSector === 'scuba') {
      targetDepthIndex = sectorAnalysis.scuba?.mapDepthIndex ?? 0;
      hoverLabel = 'Sample Location';
      tempTest = (t) => t !== null && !isNaN(t);
    }

    const sliceLayer = fullTensor?.[0]?.[targetDepthIndex];
    if (!sliceLayer) return null;

    let filteredCount = 0;
    for (let r = 0; r < numLat; r += STEP) {
      for (let c = 0; c < numLon; c += STEP) {
        const temp = sliceLayer[r]?.[c];
        
        // --- DIAGNOSTIC LOG FOR ROW 8 ---
        if (r === 8 && c >= 80 && c <= 96 && activeSector === 'fisheries') {
          console.log(`[Diagnostic r=8, c=${c}] Temp: ${temp} | Type: typeof ${typeof temp} | Test: ${tempTest(temp)}`);
        }
        
        if (tempTest(temp)) {
          lats.push(+latArr[r].toFixed(2));
          lons.push(+lonArr[c].toFixed(2));
          texts.push(`${hoverLabel}<br>${latArr[r].toFixed(1)}°N, ${lonArr[c].toFixed(1)}°E<br>${Number(temp).toFixed(2)}°C`);
        } else if (temp !== null && !isNaN(temp)) {
          filteredCount++;
        }
      }
    }
    
    if (activeSector === 'fisheries') {
      console.log(`[Fisheries Marker Filter] Checked layer at index ${targetDepthIndex}. Omitted ${filteredCount} points that fell outside 20-29°C.`);
    }

    if (lats.length === 0) return null;

    return {
      type: 'scattergeo',
      lat: lats,
      lon: lons,
      mode: 'markers',
      marker: activeSector === 'fisheries' ? {
        size: 2,
          color: 'transparent',
          line: { color: '#ff00ff', width: 1.5 }, // high contrast hollow magenta
        symbol: 'circle'
      } : activeSector === 'scuba' ? {
        size: 4,
        color: 'white',
        opacity: 0.8,
        symbol: 'circle'
      } : markerConfig,
      text: texts,
      hovertemplate: '%{text}<extra></extra>',
      name: hoverLabel,
      showlegend: false,
    };
  }, [activeSector, sectorAnalysis, fullTensor, plotData, depthIndex]);

  if (!customDataZ) return null;

  // Smooth continuous colorscale
  const SMOOTH_CMOCEAN_THERMAL = [
    [0.00, '#042333'],
    [0.10, '#0a3560'],
    [0.20, '#1b3472'],
    [0.30, '#3f3183'],
    [0.40, '#662479'],
    [0.50, '#8a1761'],
    [0.60, '#ab1a45'],
    [0.70, '#c6312a'],
    [0.80, '#da561a'],
    [0.90, '#e8841a'],
    [1.00, '#eff542']
  ];

  const activeColorscale = SMOOTH_CMOCEAN_THERMAL;
  // Pad the min/max slightly so the colorbar looks neat
  const activeZmin = dataZmin;
  const activeZmax = dataZmax;

  return (
    <motion.div
      className="absolute inset-0 rounded-xl overflow-hidden shadow-sm border border-[var(--color-paper-border)] bg-[var(--color-paper-surface)]"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <Plot
        className="js-plotly-plot"
        style={{
          width: '100%',
          height: '100%',
          opacity: isFading ? 0.6 : 1,
          transition: 'opacity 0.25s ease-out'
        }}
        onClick={(e) => {
          if (!e.points || !e.points.length || !fullTensor) return;
          const pt = e.points[0];
          // Heatmap clicks give pt.x (lon) / pt.y (lat)
          // Scattergeo / sector-dot clicks give pt.lon / pt.lat
          // We must handle both — earlier code only checked pt.x so dot clicks silently bailed
          const lon = pt.x ?? pt.lon;
          const lat = pt.y ?? pt.lat;
          if (lon == null || lat == null || !latArray || !lonArray) return;

          // Nearest-neighbor lookup — grid spacing is ~0.25°, exact match never works
          let latIdx = 0, lonIdx = 0;
          let bestLatDiff = Infinity, bestLonDiff = Infinity;
          for (let i = 0; i < latArray.length; i++) {
            const d = Math.abs(latArray[i] - lat);
            if (d < bestLatDiff) { bestLatDiff = d; latIdx = i; }
          }
          for (let i = 0; i < lonArray.length; i++) {
            const d = Math.abs(lonArray[i] - lon);
            if (d < bestLonDiff) { bestLonDiff = d; lonIdx = i; }
          }

          const profile = fullTensor[0].map(layer => layer[latIdx]?.[lonIdx] ?? null);
          const isOcean = profile[0] !== null && !isNaN(profile[0]) && profile[0] !== 0.0;

          if (activeSector === 'naval' && isOcean) {
            // Naval tab: compute per-pixel SVP and show floating line chart
            const assumedSalinity = sectorAnalysis?.naval?.assumedSalinity ?? 35;
            const svp = modelDepths
              .map((depth, di) => {
                const t = profile[di];
                if (!isValidOceanPixel(t)) return null;
                return { depth, speed: +mackenzie(t, assumedSalinity, depth).toFixed(1) };
              })
              .filter(Boolean);
            setNavalPixel({ lat: latArray[latIdx], lon: lonArray[lonIdx], svp });
          } else if (isOcean) {
            setPopupData({ lat, lon, profile, depths: modelDepths });
            if (onPixelSelect) onPixelSelect({ lat, lon, temp: profile[0], fullProfile: profile });
          } else {
            setPopupData(null);
            setNavalPixel(null);
            if (onPixelSelect) onPixelSelect(null);
          }
        }}
        data={[
          {
            type: 'scattergeo',
            lat: [null],
            lon: [null],
            showlegend: false,
            hoverinfo: 'none',
          },
          {
            // Dummy trace just to render the dynamic colorbar
            type: 'heatmap',
            z: [[activeZmin, activeZmax]],
            zmin: activeZmin,
            zmax: activeZmax,
            colorscale: activeColorscale,
            showscale: true,
            opacity: 0,
            colorbar: {
              title: {
                text: 'Potential Temp (°C)',
                font: { color: '#4A3428', size: 12, family: 'Inter', weight: 'bold' },
                side: 'right',
              },
              nticks: 12,
              tickfont: { color: '#8B8379', size: 11, family: 'Inter', weight: '500' },
              thickness: 16,
              len: 0.8,
              outlinewidth: 1,
              outlinecolor: '#DDD5C8',
              bgcolor: 'rgba(251,249,244,0.8)',
              borderwidth: 0,
              tickcolor: '#8B8379',
              xpad: 15,
              showticksuffix: 'last',
              ticksuffix: '°C'
            },
            hoverinfo: 'none'
          },
          // Invisible scatter trace over the image to capture tooltips
          {
            type: 'heatmap',
            z: customDataZ.map(row => row.map(v => (v === null || isNaN(v)) ? null : 0)), // Dummy 0s just for hover
            customdata: customDataZ,
            x: lonArray,
            y: latArray,
            opacity: 0,
            showscale: false,
            hovertemplate: 'Lat: %{y:.2f}°N<br>Lon: %{x:.2f}°E<br>Temp: %{customdata:.2f}°C<extra></extra>',
          },
          // Sector highlight overlay
          ...(sectorOverlayTrace ? [sectorOverlayTrace] : []),

          // Naval: shadow zone hotspot green ring markers
          // Renders up to 5 real hotspots from topShadowPixels (real tensor-derived locations).
          // Each hotspot gets concentric rings to match the "fish zone" visual style.
          // Only shown when Shadow Zone toggle is active.
          ...(activeSector === 'naval' && showShadowZone && sectorAnalysis?.naval?.topShadowPixels?.length && latArray && lonArray
            ? sectorAnalysis.naval.topShadowPixels.flatMap((px, idx) => {
                const pLat = LAT_MIN + (px.r / (latArray.length - 1)) * (LAT_MAX - LAT_MIN);
                const pLon = LON_MIN + (px.c / (lonArray.length - 1)) * (LON_MAX - LON_MIN);
                const isBest = idx === 0;
                const label = isBest
                  ? `🔵 Optimal Shadow Zone\n${pLat.toFixed(2)}°N, ${pLon.toFixed(2)}°E\ndc/dz = ${px.gradient} s⁻¹`
                  : `Shadow Zone Hotspot #${idx + 1}\n${pLat.toFixed(2)}°N, ${pLon.toFixed(2)}°E\ndc/dz = ${px.gradient} s⁻¹`;
                return [
                  // Outer ring (large, hollow)
                  {
                    type: 'scattergeo',
                    lat: [pLat], lon: [pLon],
                    mode: 'markers',
                    marker: {
                      size: isBest ? 28 : 22,
                      color: 'rgba(0,0,0,0)',
                      line: { color: isBest ? '#22c55e' : 'rgba(34,197,94,0.65)', width: isBest ? 3 : 2 },
                      symbol: 'circle',
                    },
                    text: [label],
                    hovertemplate: '%{text}<extra></extra>',
                    showlegend: false,
                  },
                  // Inner ring
                  {
                    type: 'scattergeo',
                    lat: [pLat], lon: [pLon],
                    mode: 'markers',
                    marker: {
                      size: isBest ? 16 : 12,
                      color: 'rgba(0,0,0,0)',
                      line: { color: isBest ? '#4ade80' : 'rgba(74,222,128,0.5)', width: isBest ? 2 : 1.5 },
                      symbol: 'circle',
                    },
                    hoverinfo: 'skip',
                    showlegend: false,
                  },
                  // Center dot
                  {
                    type: 'scattergeo',
                    lat: [pLat], lon: [pLon],
                    mode: 'markers',
                    marker: {
                      size: isBest ? 6 : 4,
                      color: isBest ? '#22c55e' : 'rgba(34,197,94,0.7)',
                      symbol: 'circle',
                    },
                    hoverinfo: 'skip',
                    showlegend: false,
                  },
                ];
              })
            : []),

          // Naval: selected pixel ring (clicked point — blue to distinguish from green shadow rings)
          ...(activeSector === 'naval' && navalPixel ? [{
            type: 'scattergeo',
            lat: [navalPixel.lat],
            lon: [navalPixel.lon],
            mode: 'markers',
            marker: {
              size: 14,
              color: 'rgba(96,165,250,0.15)',
              line: { color: '#60a5fa', width: 2.5 },
              symbol: 'circle',
            },
            hoverinfo: 'skip',
            showlegend: false,
          }] : []),
        ]}
        layout={{
          autosize: true,
          margin: { l: 0, r: 0, t: 0, b: 0 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          paper_bgcolor: 'rgba(0,0,0,0)',
          font: { family: 'Inter', color: 'var(--color-ink-medium)' },
          images: smoothedImageUrl ? [
            {
              source: smoothedImageUrl,
              x: LON_MIN,
              y: LAT_MAX,
              sizex: LON_MAX - LON_MIN,
              sizey: LAT_MAX - LAT_MIN,
              xref: 'x',
              yref: 'y',
              sizing: 'stretch',
              layer: 'below'
            }
          ] : [],
          // Shadow zone: use annotations instead of shapes — shapes on mixed geo/xy axes
          // have a coordinate-system mismatch that prevents them from rendering reliably.
          // The DOM overlay below (shadow-zone-overlay div) handles the visual rectangle.
          shapes: [],
          annotations: (activeSector === 'naval' && showShadowZone && sectorAnalysis?.naval?.shadowZone)
            ? [{
                x: (LON_MIN + LON_MAX) / 2,
                y: LAT_MIN + 1.5,
                xref: 'x',
                yref: 'y',
                text: `▦ SONAR SHADOW ZONE ${sectorAnalysis.naval.shadowZone[0]}–${sectorAnalysis.naval.shadowZone[1]}m`,
                showarrow: false,
                font: { color: '#60a5fa', size: 10, family: 'Inter' },
                bgcolor: 'rgba(15,23,42,0.75)',
                bordercolor: 'rgba(96,165,250,0.6)',
                borderwidth: 1,
                borderpad: 4,
              }]
            : [],
          xaxis: { range: [LON_MIN, LON_MAX], showgrid: false, zeroline: false, showticklabels: false, visible: false },
          yaxis: { range: [LAT_MIN, LAT_MAX], showgrid: false, zeroline: false, showticklabels: false, visible: false, scaleanchor: 'x' },
          geo: {
            projection: { type: 'equirectangular' },
            lonaxis: { range: [LON_MIN, LON_MAX] },
            lataxis: { range: [LAT_MIN, LAT_MAX] },
            showcoastlines: true,
            coastlinecolor: 'rgba(100, 100, 100, 0.6)',
            coastlinewidth: 1,
            showland: true,
            landcolor: 'rgba(30, 35, 45, 0.85)',
            showocean: true,
            oceancolor: 'transparent',
            bgcolor: 'transparent',
            showframe: false,
            resolution: 50,
          },
        }}
        config={{ displayModeBar: false, responsive: true, scrollZoom: false }}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
      />

      <AnimatePresence>
        {popupData && (
          <ProfilePopup 
            lat={popupData.lat} 
            lon={popupData.lon} 
            profile={popupData.profile} 
            depths={popupData.depths} 
            onClose={() => setPopupData(null)} 
          />
        )}
      </AnimatePresence>

      {/* Shadow zone DOM overlay — rendered as absolute div, bypassing Plotly shape coord issues */}
      {activeSector === 'naval' && showShadowZone && sectorAnalysis?.naval?.shadowZone && (
        <div
          className="absolute inset-0 pointer-events-none z-10"
          style={{
            border: '2px dashed rgba(96,165,250,0.75)',
            background: 'rgba(26,78,122,0.10)',
            borderRadius: 'inherit',
          }}
        >
          {/* Depth badge in bottom-center */}
          <div
            className="absolute bottom-3 left-1/2 -translate-x-1/2 px-2.5 py-1 rounded-md text-[9px] font-bold tracking-wider"
            style={{
              background: 'rgba(15,23,42,0.82)',
              border: '1px solid rgba(96,165,250,0.55)',
              color: '#93c5fd',
              letterSpacing: '0.05em',
            }}
          >
            ▦ SONAR SHADOW ZONE &nbsp;{sectorAnalysis.naval.shadowZone[0]}–{sectorAnalysis.naval.shadowZone[1]}m
          </div>
        </div>
      )}

      {/* Naval: per-point SVP floating panel */}
      <AnimatePresence>
        {activeSector === 'naval' && navalPixel && (
          <NavalSvpPanel
            key={`${navalPixel.lat}-${navalPixel.lon}`}
            svpData={navalPixel.svp}
            lat={navalPixel.lat}
            lon={navalPixel.lon}
            assumedSalinity={sectorAnalysis?.naval?.assumedSalinity ?? 35}
            hasRealSss={sectorAnalysis?.naval?.hasRealSss ?? false}
            onClose={() => setNavalPixel(null)}
          />
        )}
      </AnimatePresence>

      {/* Naval: land-click message */}
      <AnimatePresence>
        {activeSector === 'naval' && navalPixel === null && (
          <motion.div
            key="naval-hint"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full text-[9px] font-medium pointer-events-none"
            style={{
              background: 'color-mix(in srgb, var(--color-paper-surface) 85%, transparent)',
              border: '1px solid var(--color-paper-border)',
              color: 'var(--color-ink-muted)',
              backdropFilter: 'blur(8px)',
            }}
          >
            Click any ocean pixel to view its Sound Speed Profile
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default function VisualizationCanvas({ status, plotData, fullTensor, depthIndex, modelDepths, renderMode, mapStats, date, activeSector, sectorAnalysis, onPixelSelect, showThermoclineStrength, setShowThermoclineStrength, showShadowZone, setShowShadowZone }) {
  // Inspector collapsed state
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const currentDepthStats = useMemo(() => {
    if (!plotData) return null;
    let min = Infinity, max = -Infinity, sum = 0, count = 0;
    for (let y = 0; y < plotData.length; y++) {
      for (let x = 0; x < plotData[y].length; x++) {
        const v = plotData[y][x];
        if (v !== null && !isNaN(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
          sum += v;
          count++;
        }
      }
    }
    if (count === 0) return null;
    return {
      min: min.toFixed(2),
      max: max.toFixed(2),
      avg: (sum / count).toFixed(2)
    };
  }, [plotData]);

  return (
    <motion.div
      className="relative w-full h-full min-h-[500px]"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <ErrorBoundary>
        <AnimatePresence mode="wait">
          {(status === 'idle' || status === 'fetching_raw') && <IdleState key="idle" />}
        </AnimatePresence>
        
        <AnimatePresence>
          {status === 'showing_raw' && (
            <motion.div 
              key="raw-satellite" 
              className="absolute inset-0 z-0"
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
            >
              <img 
                src={RANDOM_SAT}
                className="w-full h-full object-cover block"
                alt=""
              />
            </motion.div>
          )}

          {(status === 'active' || status === 'loading') && (
            <motion.div key="active-layer" className="absolute inset-0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <ActiveState plotData={plotData} fullTensor={fullTensor} depthIndex={depthIndex} modelDepths={modelDepths} renderMode={renderMode} activeSector={activeSector} sectorAnalysis={sectorAnalysis} onPixelSelect={onPixelSelect} showThermoclineStrength={showThermoclineStrength} showShadowZone={showShadowZone} date={date} />
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence>
          {status === 'active' && (
            <motion.div
              key="sweep-overlay"
              className="absolute inset-0 pointer-events-none z-30 mix-blend-overlay"
              initial={{ left: '-100%', opacity: 1 }}
              animate={{ left: '100%', opacity: 0 }}
              transition={{ duration: 1.2, ease: [0.4, 0, 0.2, 1] }}
              style={{
                background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.6) 50%, transparent 100%)',
                width: '100%',
                transform: 'skewX(-20deg)',
              }}
            />
          )}
        </AnimatePresence>

        <AnimatePresence>
          {status === 'loading' && <LoadingScreen key="loading" />}
        </AnimatePresence>

        {/* Inspector Telemetry Card — collapsible, themed */}
        <AnimatePresence>
          {status === 'active' && currentDepthStats && mapStats && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
              className="absolute top-4 left-4 z-20 w-64 rounded-xl overflow-hidden shadow-xl"
              style={{
                background: 'color-mix(in srgb, var(--color-paper-surface) 88%, transparent)',
                backdropFilter: 'blur(16px)',
                WebkitBackdropFilter: 'blur(16px)',
                border: '1px solid var(--color-paper-border)',
              }}
            >
              {/* Inspector Header */}
              <button
                className="w-full flex items-center justify-between px-4 py-2.5 hover:opacity-80 transition-opacity"
                style={{
                  background: 'var(--color-paper-elevated)',
                  borderBottom: '1px solid var(--color-paper-border)',
                }}
                onClick={() => setInspectorCollapsed(c => !c)}
              >
                <span className="text-[9px] font-black uppercase tracking-widest" style={{ color: 'var(--color-ink-muted)' }}>
                  Depth Telemetry ({modelDepths?.[depthIndex] || 0}m)
                </span>
                <motion.span
                  animate={{ rotate: inspectorCollapsed ? -90 : 0 }}
                  transition={{ duration: 0.18 }}
                >
                  <ChevronDown className="w-3 h-3" style={{ color: 'var(--color-ink-muted)' }} />
                </motion.span>
              </button>

              {/* Inspector Body */}
              <AnimatePresence initial={false}>
                {!inspectorCollapsed && (
                  <motion.div
                    key="inspector-body"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 py-3 flex flex-col gap-3">
                      {/* Headline stats */}
                      <div className="flex justify-between items-center">
                        <span className="stat-label">Avg Temp</span>
                        <AnimatedValue value={currentDepthStats?.avg ?? '—'} suffix="°C" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="stat-label">Min Temp</span>
                        <AnimatedValue value={currentDepthStats?.min ?? '—'} suffix="°C" />
                      </div>
                      <div
                        className="flex justify-between items-center pb-3"
                        style={{ borderBottom: '1px solid var(--color-paper-border)' }}
                      >
                        <span className="stat-label">Max Temp</span>
                        <AnimatedValue value={currentDepthStats?.max ?? '—'} suffix="°C" />
                      </div>

                      {/* Input Variables */}
                      <div>
                        <span className="stat-label block mb-2">Input Variables (Domain Avg)</span>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                          {[['SST', `${mapStats.sst}°C`], ['SSS', `${mapStats.sss}psu`], ['SSH', `${mapStats.ssh}m`],
                            ['U_cur', mapStats.u_cur], ['V_cur', mapStats.v_cur], ['U_wnd', mapStats.u_wind], ['V_wnd', mapStats.v_wind]
                          ].map(([k, v]) => (
                            <div key={k} className="flex justify-between">
                              <span className="text-[9px] font-mono" style={{ color: 'var(--color-ink-muted)' }}>{k}</span>
                              <span className="stat-value">{v}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Naval overlays — themed */}
                      {activeSector === 'naval' && (
                        <div
                          className="pt-3 flex flex-col gap-2"
                          style={{ borderTop: '1px solid var(--color-paper-border)' }}
                        >
                          <span className="stat-label" style={{ color: 'var(--color-sector-naval)' }}>Naval Overlays</span>
                          <div className="flex gap-2 flex-wrap">
                            {[{label: 'Thermocline Map', active: showThermoclineStrength, toggle: () => { setShowThermoclineStrength(p => !p); if (!showThermoclineStrength) setShowShadowZone(false); }},
                              {label: 'Shadow Zone', active: showShadowZone, toggle: () => { setShowShadowZone(p => !p); if (!showShadowZone) setShowThermoclineStrength(false); }}
                            ].map(({ label, active, toggle }) => (
                              <button
                                key={label}
                                onClick={toggle}
                                className="px-2 py-1 text-[8px] font-bold uppercase rounded-md transition-all"
                                style={{
                                  background: active
                                    ? 'var(--color-sector-naval)'
                                    : 'color-mix(in srgb, var(--color-sector-naval) 10%, transparent)',
                                  color: active ? '#fff' : 'var(--color-sector-naval)',
                                  border: `1px solid color-mix(in srgb, var(--color-sector-naval) 40%, transparent)`,
                                }}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Unified Map Legend */}
        <MapLegend
          activeSector={activeSector}
          showThermoclineStrength={showThermoclineStrength}
          showShadowZone={showShadowZone}
          hasArgo={false}
          status={status}
        />
      </ErrorBoundary>
    </motion.div>
  );
}
