/**
 * OceanAnalyticsViews.jsx
 * ========================
 * Standalone, full-screen analytics components.
 * Wired to sectorAnalysis for depth-range clipping and marker overlays.
 */

import React, { useMemo, useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { Sliders } from 'lucide-react';

const LON_MIN = 45.0;
const LON_MAX = 100.0;
const LAT_MIN = -10.0;
const LAT_MAX = 30.0;
const STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

// Sector depth-range limits (m)
const SECTOR_DEPTH_RANGE = {
  fisheries: [0, 200],
  naval:     [0, 1000],
  scuba:     [0, 40],
};

// Sector marker colors
const SECTOR_COLOR = {
  fisheries: '#c06d38',
  naval:     '#1a4e7a',
  scuba:     '#0e7490',
};

function getLayer(tensor, depthIdx) {
  if (!tensor?.[0]?.[depthIdx]) return null;
  return tensor[0][depthIdx];
}

function cleanRow(arr) {
  if (!arr) return [];
  return arr.map(v => (v === null || v === undefined || Number.isNaN(v) ? null : Number(v)));
}

function buildLonAxis(nCols) {
  const step = (LON_MAX - LON_MIN) / Math.max(nCols - 1, 1);
  return Array.from({ length: nCols }, (_, i) => +(LON_MIN + i * step).toFixed(2));
}

function buildLatAxis(nRows) {
  const step = (LAT_MAX - LAT_MIN) / Math.max(nRows - 1, 1);
  return Array.from({ length: nRows }, (_, i) => +(LAT_MIN + i * step).toFixed(2));
}

// ── CrossSectionView ─────────────────────────────────────────────────────────
export function CrossSectionView({ predictionTensor, modelDepths, activeSector, sectorAnalysis }) {
  const depths = (modelDepths?.length === 15) ? modelDepths : STANDARD_DEPTHS;
  const nRows = predictionTensor?.[0]?.[0]?.length ?? 161;
  const [latSliderIndex, setLatSliderIndex] = useState(() => Math.floor(nRows / 2));

  const latAtSlider = useMemo(() => {
    const step = (LAT_MAX - LAT_MIN) / Math.max(nRows - 1, 1);
    return (LAT_MIN + latSliderIndex * step).toFixed(2);
  }, [latSliderIndex, nRows]);

  const handleSliderChange = useCallback(e => setLatSliderIndex(Number(e.target.value)), []);

  // Determine the depth range to display
  const [depthMin, depthMax] = activeSector
    ? SECTOR_DEPTH_RANGE[activeSector] ?? [0, 1000]
    : [0, 1000];

  // Filter depths array to those within range
  const filteredDepthEntries = depths
    .map((d, i) => ({ d, i }))
    .filter(({ d }) => d >= depthMin && d <= depthMax);

  const { zMatrix, lonAxis } = useMemo(() => {
    const firstLayer = getLayer(predictionTensor, 0);
    if (!firstLayer) return { zMatrix: null, lonAxis: [] };

    const nCols = firstLayer[0]?.length ?? 0;
    const latIdx = Math.max(0, Math.min(latSliderIndex, nRows - 1));
    const lonAx = buildLonAxis(nCols);

    const matrix = filteredDepthEntries.map(({ i }) => {
      const layer = getLayer(predictionTensor, i);
      if (!layer) return new Array(nCols).fill(null);
      return cleanRow(layer[latIdx] ?? []);
    });

    return { zMatrix: matrix, lonAxis: lonAx };
  }, [predictionTensor, filteredDepthEntries, latSliderIndex, nRows]);

  // Compute the marker line depth from sectorAnalysis
  const markerDepth = useMemo(() => {
    if (!activeSector || !sectorAnalysis) return null;
    if (activeSector === 'fisheries') return sectorAnalysis.fisheries?.thermoclineDepth ?? null;
    if (activeSector === 'naval')     return sectorAnalysis.naval?.sld ?? null;
    if (activeSector === 'scuba') {
      const zones = sectorAnalysis.scuba?.coldShockZones;
      return zones?.length ? zones[0].boundaryDepth : null;
    }
    return null;
  }, [activeSector, sectorAnalysis]);

  const markerLabel = useMemo(() => {
    if (!activeSector) return '';
    if (activeSector === 'fisheries') return `Basin-avg Thermocline ${markerDepth ?? '?'}m`;
    if (activeSector === 'naval')     return `Basin-avg SLD ${markerDepth ?? '?'}m`;
    if (activeSector === 'scuba')     return `Cold Shock Boundary ${markerDepth ?? '?'}m`;
    return '';
  }, [activeSector, markerDepth]);

  const displayedDepthValues = filteredDepthEntries.map(({ d }) => d);

  if (!zMatrix) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-[#f4f7fc] text-gray-400 text-sm italic h-full w-full">
        <p>Run a prediction to see the ocean cross-section.</p>
      </div>
    );
  }

  // Build extra marker shape if we have a computed depth
  const shapes = [];
  if (markerDepth !== null && displayedDepthValues.length > 0) {
    shapes.push({
      type: 'line',
      x0: LON_MIN,
      x1: LON_MAX,
      y0: markerDepth,
      y1: markerDepth,
      line: {
        color: SECTOR_COLOR[activeSector] ?? '#888',
        width: 2,
        dash: 'dash',
      },
    });
  }

  // Shadow zone shaded band for Naval
  if (activeSector === 'naval' && sectorAnalysis?.naval?.shadowZone) {
    shapes.push({
      type: 'rect',
      x0: LON_MIN,
      x1: LON_MAX,
      y0: sectorAnalysis.naval.shadowZone[0],
      y1: sectorAnalysis.naval.shadowZone[1],
      fillcolor: 'rgba(26, 78, 122, 0.15)', // transparent naval blue
      line: { width: 0 },
      layer: 'below'
    });
  }

  const annotations = markerDepth !== null ? [{
    x: LON_MAX,
    y: markerDepth,
    xref: 'x',
    yref: 'y',
    text: markerLabel,
    showarrow: false,
    font: { size: 11, color: SECTOR_COLOR[activeSector] ?? '#888', family: 'Inter, sans-serif' },
    xanchor: 'right',
    yanchor: 'bottom',
    bgcolor: 'rgba(255,255,255,0.75)',
    borderpad: 2,
  }] : [];

  // Dynamic trace config for CrossSection
  let crossTraceConfig = { colorscale: 'Jet', zmin: 3, zmax: 31, zauto: false };
  if (activeSector === 'fisheries') {
    const [zMin, zMax] = sectorAnalysis?.fisheries?.colorRange ?? [20, 29];
    crossTraceConfig = { colorscale: 'YlOrRd', zmin: zMin, zmax: zMax, zauto: false };
  } else if (activeSector === 'naval') {
    crossTraceConfig = { colorscale: 'Jet', zauto: true };
  } else if (activeSector === 'scuba') {
    crossTraceConfig = { colorscale: 'Blues', zmin: 26, zmax: 30, zauto: false };
  }

  return (
    <div className="relative w-full h-full" style={{ background: 'var(--color-paper-bg)' }}>
      {/* Floating Slider Control */}
      <div
        className="absolute top-6 right-6 z-10 backdrop-blur shadow-md rounded-xl p-3 flex items-center gap-3"
        style={{
          background: 'color-mix(in srgb, var(--color-paper-surface) 90%, transparent)',
          border: '1px solid var(--color-paper-border)',
        }}
      >
        <Sliders size={16} style={{ color: 'var(--color-accent)' }} />
        <div className="flex flex-col">
          <span className="text-[9px] font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--color-ink-muted)' }}>Select Latitude</span>
          <div className="flex items-center gap-2">
            <span className="text-sm font-mono font-bold w-16 text-right" style={{ color: 'var(--color-ink-dark)' }}>
              {latAtSlider}&deg;N
            </span>
            <input
              type="range"
              min={0}
              max={nRows - 1}
              value={latSliderIndex}
              onChange={handleSliderChange}
              className="w-48 cursor-pointer"
            />
          </div>
        </div>
      </div>

      <Plot
        data={[{
          type: 'heatmap',
          z: zMatrix,
          x: lonAxis,
          y: displayedDepthValues,
          zsmooth: false,
          connectgaps: false,
          hoverongaps: false,
          ...crossTraceConfig,
          colorbar: {
            title: { text: 'Temp (°C)', side: 'right', font: { size: 13, color: '#9ca3af', family: 'Inter, sans-serif' } },
            thickness: 18,
            len: 0.9,
            tickfont: { size: 12, color: '#9ca3af', family: 'Inter, sans-serif' },
            tickmode: crossTraceConfig.zauto ? 'auto' : 'linear',
            tick0: crossTraceConfig.zmin ?? 3,
            dtick: crossTraceConfig.zauto ? undefined : 4,
            outlinewidth: 1,
            outlinecolor: '#374151',
          },
          hovertemplate: '<b>%{x:.1f}°E  |  %{y}m</b><br>Temp: %{z:.2f}°C<extra></extra>',
        }]}
        layout={{
          autosize: true,
          margin: { l: 68, r: 20, t: 60, b: 56 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          shapes,
          annotations,
          xaxis: {
            title: { text: 'Longitude (°E)', font: { size: 13, color: '#9ca3af', family: 'Inter, sans-serif' } },
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.05)',
            tickfont: { size: 12, color: '#9ca3af', family: 'Inter, sans-serif' },
            linecolor: '#374151',
            tickcolor: '#374151',
          },
          yaxis: {
            title: { text: 'Depth (m)', font: { size: 13, color: '#9ca3af', family: 'Inter, sans-serif' } },
            autorange: 'reversed',
            range: [depthMax, depthMin],
            showgrid: true,
            gridcolor: 'rgba(255,255,255,0.05)',
            tickfont: { size: 12, color: '#9ca3af', family: 'Inter, sans-serif' },
            linecolor: '#374151',
            tickcolor: '#374151',
            tickmode: 'array',
            tickvals: displayedDepthValues,
            ticktext: displayedDepthValues.map(String),
          },
          font: { family: 'Inter, sans-serif', color: '#9ca3af' },
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}


// ── Mesh3DView ────────────────────────────────────────────────────────────────
export function Mesh3DView({ predictionTensor, modelDepths, currentDepthIndex, activeSector, sectorAnalysis }) {
  const depths = (modelDepths?.length === 15) ? modelDepths : STANDARD_DEPTHS;
  const STEP = 2;

  // Key depth for the highlighted slab annotation (plane at a specific depth)
  const highlightDepth = useMemo(() => {
    if (!activeSector || !sectorAnalysis) return null;
    if (activeSector === 'fisheries') return sectorAnalysis.fisheries?.thermoclineDepth ?? null;
    if (activeSector === 'naval')     return sectorAnalysis.naval?.sld ?? null;
    if (activeSector === 'scuba') {
      const zones = sectorAnalysis.scuba?.coldShockZones;
      return zones?.length ? zones[0].boundaryDepth : null;
    }
    return null;
  }, [activeSector, sectorAnalysis]);

  // Determine Z range from sector (flip for Plotly: [deep, shallow])
  const zAxisRange = useMemo(() => {
      if (!activeSector) return [1000, -10];
      if (activeSector === 'scuba') return [40, 0];
      const [dMin, dMax] = SECTOR_DEPTH_RANGE[activeSector] ?? [0, 1000];
      return [dMax + 20, dMin];
    }, [activeSector]);

  const { traces } = useMemo(() => {
    const firstLayer = getLayer(predictionTensor, 0);
    if (!firstLayer) return { traces: null };

    const nRows = firstLayer.length;
    const nCols = firstLayer[0]?.length ?? 0;

    // Depth range to render
    const [depthMin, depthMax] = activeSector
      ? SECTOR_DEPTH_RANGE[activeSector] ?? [0, 1000]
      : [0, 1000];

    const indicesToRender = Array.from(
      new Set([
        currentDepthIndex,
        depths.length - 1, // always include deepest
      ])
    )
      .filter(i => depths[i] >= depthMin && depths[i] <= depthMax)
      .sort((a, b) => a - b);

    // If currentDepthIndex is outside range, only render what's in range
    const renderList = indicesToRender.length > 0 ? indicesToRender : 
      depths
        .map((d, i) => ({ d, i }))
        .filter(({ d }) => d >= depthMin && d <= depthMax)
        .map(({ i }) => i)
        .slice(-2); // pick last two in range

    const rawLons = buildLonAxis(nCols);
    const rawLats = buildLatAxis(nRows);

    const surfaceTraces = renderList.map((depthIdx, ti) => {
      const rawTempMatrix = getLayer(predictionTensor, depthIdx);
      if (!rawTempMatrix) return null;

      const depthVal = depths[depthIdx];

      const z_matrix = [];
      const color_matrix = [];
      const x_vals = [];
      const y_vals = [];
      const text_matrix_transposed = [];

      for (let j = 0; j < rawLons.length; j += STEP) {
        text_matrix_transposed.push([]);
      }

      for (let i = 0; i < rawLats.length; i += STEP) {
        y_vals.push(rawLats[i]);
        const z_row = [];
        const color_row = [];
        let col_idx = 0;

        for (let j = 0; j < rawLons.length; j += STEP) {
          if (i === 0) x_vals.push(rawLons[j]);

          const temp = rawTempMatrix[i]?.[j];

          if (temp === null || temp === undefined || Number.isNaN(temp)) {
            z_row.push(null);
            color_row.push(null);
            text_matrix_transposed[col_idx].push('Landmass');
          } else {
            z_row.push(depthVal);
            color_row.push(temp);
            text_matrix_transposed[col_idx].push(Number(temp).toFixed(2) + '°C');
          }
          col_idx++;
        }
        z_matrix.push(z_row);
        color_matrix.push(color_row);
      }

      // Dynamic trace config — no data mutation, only colorscale/cmin/cmax
      let dynamicTraceConfig = { colorscale: 'Jet', cmin: 3, cmax: 31, cauto: false };
      if (activeSector === 'fisheries') {
        const [cMin, cMax] = sectorAnalysis?.fisheries?.colorRange ?? [20, 29];
        dynamicTraceConfig = { colorscale: 'YlOrRd', cmin: cMin, cmax: cMax, cauto: false };
      } else if (activeSector === 'naval') {
        dynamicTraceConfig = { colorscale: 'Jet', cauto: true };
      } else if (activeSector === 'scuba') {
        dynamicTraceConfig = { colorscale: 'Blues', cmin: 26, cmax: 30, cauto: false };
      }

      const label = `${Math.round(depthVal)}m`;

      return {
        type: 'surface',
        x: x_vals,
        y: y_vals,
        z: z_matrix,
        surfacecolor: color_matrix,
        text: text_matrix_transposed,
        ...dynamicTraceConfig,
        showscale: ti === 0,
        opacity: 0.9,
        name: label,
        colorbar: ti === 0 ? {
          title: { text: '°C', side: 'right', font: { size: 14 } },
          thickness: 15,
          len: 0.8,
          x: 1.02,
          tickfont: { size: 12 },
        } : undefined,
        hovertemplate: 'Lon: %{x}<br>Lat: %{y}<br>Depth: %{z}m<br>Temp: %{text}<extra></extra>',
        hoverinfo: 'all',
        lighting: { ambient: 0.8, diffuse: 0.9, specular: 0.2, roughness: 0.6 },
      };
    });

    // Add a semi-transparent highlight slab at the computed key depth
    if (highlightDepth !== null) {
      const slabLayer = getLayer(predictionTensor,
        depths.reduce((best, d, i) =>
          Math.abs(d - highlightDepth) < Math.abs(depths[best] - highlightDepth) ? i : best, 0)
      );

      if (slabLayer) {
        const slabZ = [];
        const slabColor = [];
        const slabX = [];
        const slabY = [];

        for (let i = 0; i < rawLats.length; i += STEP * 2) {
          slabY.push(rawLats[i]);
          const zRow = [];
          const cRow = [];
          for (let j = 0; j < rawLons.length; j += STEP * 2) {
            if (i === 0) slabX.push(rawLons[j]);
            const temp = slabLayer[i]?.[j];
            zRow.push(highlightDepth);
            cRow.push(temp !== null && temp !== undefined && !Number.isNaN(temp) ? temp : null);
          }
          slabZ.push(zRow);
          slabColor.push(cRow);
        }

        surfaceTraces.push({
          type: 'surface',
          x: slabX,
          y: slabY,
          z: slabZ,
          surfacecolor: slabColor,
          colorscale: [[0, SECTOR_COLOR[activeSector] + '44'], [1, SECTOR_COLOR[activeSector]]],
          showscale: false,
          opacity: 0.45,
          name: `Key Depth ${highlightDepth}m`,
          cauto: false,
          cmin: 3,
          cmax: 31,
          hovertemplate: `Key Layer: ${highlightDepth}m<br>Temp: %{text}<extra></extra>`,
          lighting: { ambient: 1, diffuse: 0, specular: 0 },
          contours: {
            x: { highlight: false },
            y: { highlight: false },
            z: { highlight: false },
          },
        });
      }
    }

    // Shadow Zone planes for Naval
    if (activeSector === 'naval' && sectorAnalysis?.naval?.shadowZone) {
      const [szStart, szEnd] = sectorAnalysis.naval.shadowZone;
      const createPlane = (depthVal, name) => {
        const pZ = []; const pC = []; const pX = []; const pY = [];
        for (let i = 0; i < rawLats.length; i += STEP * 4) {
          pY.push(rawLats[i]);
          const zR = []; const cR = [];
          for (let j = 0; j < rawLons.length; j += STEP * 4) {
            if (i === 0) pX.push(rawLons[j]);
            zR.push(depthVal);
            cR.push(1);
          }
          pZ.push(zR);
          pC.push(cR);
        }
        return {
          type: 'surface',
          x: pX, y: pY, z: pZ, surfacecolor: pC,
          colorscale: [[0, 'rgba(0, 0, 0, 0.4)'], [1, 'rgba(0, 0, 0, 0.4)']],
          showscale: false, opacity: 0.3, name,
          hoverinfo: 'name',
          contours: { x: { highlight: false }, y: { highlight: false }, z: { highlight: false } },
        };
      };
      surfaceTraces.push(createPlane(szStart, 'Shadow Zone Top'));
      surfaceTraces.push(createPlane(szEnd, 'Shadow Zone Bottom'));
    }

    return { traces: surfaceTraces.filter(Boolean) };
  }, [predictionTensor, currentDepthIndex, depths, activeSector, highlightDepth]);

  if (!traces || traces.length === 0) {
    return (
      <div
        className="flex-1 flex flex-col items-center justify-center text-sm italic h-full w-full"
        style={{ background: 'var(--color-paper-bg)', color: 'var(--color-ink-muted)' }}
      >
        <p>Run a prediction to see the 3D volumetric view.</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative" style={{ background: 'var(--color-paper-bg)' }}>
      <Plot
        revision={String(currentDepthIndex) + String(activeSector)}
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 0, r: 0, t: 20, b: 0 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          scene: {
            xaxis: {
              title: { text: 'Longitude (°E)', font: { size: 13, color: '#9ca3af' } },
              showgrid: true,
              gridcolor: 'rgba(255,255,255,0.08)',
              tickfont: { size: 11, color: '#9ca3af' },
              backgroundcolor: 'rgba(10,14,20,0.6)',
              showbackground: true,
            },
            yaxis: {
              title: { text: 'Latitude (°N)', font: { size: 13, color: '#9ca3af' } },
              showgrid: true,
              gridcolor: 'rgba(255,255,255,0.08)',
              tickfont: { size: 11, color: '#9ca3af' },
              backgroundcolor: 'rgba(10,14,20,0.6)',
              showbackground: true,
            },
            zaxis: {
              title: { text: 'Depth (m)', font: { size: 13, color: '#9ca3af' } },
              range: zAxisRange,
              showgrid: true,
              gridcolor: 'rgba(255,255,255,0.08)',
              tickfont: { size: 11, color: '#9ca3af' },
              backgroundcolor: 'rgba(16,20,28,0.7)',
              showbackground: true,
            },
            camera: activeSector === 'naval'
              ? { eye: { x: 0, y: -2, z: 0.1 }, up: { x: 0, y: 0, z: 1 } }
              : { eye: { x: 1.6, y: -1.9, z: 1.1 }, up: { x: 0, y: 0, z: 1 } },
            aspectmode: 'manual',
            aspectratio: { x: 2, y: 1.5, z: 0.8 },
          },
          font: { family: 'Inter, sans-serif', color: '#9ca3af' },
          showlegend: false,
        }}
        config={{ displayModeBar: true, modeBarButtonsToRemove: ['toImage'], responsive: true }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
