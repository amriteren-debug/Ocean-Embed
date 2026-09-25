import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, AlertCircle, Info } from 'lucide-react';
import ProfilePopup from './ProfilePopup';

const BASE_URL = 'http://localhost:8000';

// Geographic bounds for the map
const LON_MIN = 45.0;
const LON_MAX = 100.0;
const LAT_MIN = -10.0;
const LAT_MAX = 30.0;

export default function ArgoDashboard({ date, predictionTensor, modelDepths }) {
  const [argoData, setArgoData] = useState(null);
  const [skillMetrics, setSkillMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [popupData, setPopupData] = useState(null);


  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      setArgoData(null);
      setSkillMetrics(null);
      setPopupData(null);

      try {
        const [argoRes, skillRes] = await Promise.all([
          axios.get(`${BASE_URL}/argo/${date}`),
          axios.get(`${BASE_URL}/skill/${date}`)
        ]);

        if (argoRes.data.status === 'success') {
          setArgoData(argoRes.data);
        } else {
          setError(argoRes.data.reason || 'Failed to fetch ARGO data');
        }

        if (skillRes.data.status === 'success') {
          setSkillMetrics(skillRes.data);
        }
      } catch (err) {
        setError(err.message || 'Error communicating with backend.');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  // Re-fetch whenever date changes OR a new prediction lands
  }, [date, predictionTensor]);

  // Prepare plot data
  const plotData = useMemo(() => {
    if (!argoData || !argoData.floats || argoData.floats.length === 0) return [];

    const lats = argoData.floats.map(f => f.lat);
    const lons = argoData.floats.map(f => f.lon);

    return [
      {
        type: 'scattergeo',
        lat: lats,
        lon: lons,
        mode: 'markers',
        marker: {
          size: 6,
          color: '#1b3472',
          line: { width: 1, color: 'white' }
        },
        name: 'ARGO Floats',
        hovertemplate: 'ARGO Float<br>Lat: %{lat:.2f}°N<br>Lon: %{lon:.2f}°E<extra></extra>',
      }
    ];
  }, [argoData]);

  const handleMapClick = (e) => {
    if (!e.points || !e.points.length || !argoData) return;
    const pt = e.points[0];
    const lat = pt.lat || pt.y; // scattergeo might use lat/lon directly
    const lon = pt.lon || pt.x;
    const pointIndex = pt.pointIndex;

    const argoFloat = argoData.floats[pointIndex];
    if (!argoFloat) return;

    let predProfile = null;
    if (predictionTensor && predictionTensor.length > 0) {
      // Find indices based on grid
      const numLat = predictionTensor[0][0].length;
      const numLon = predictionTensor[0][0][0].length;
      const latStep = (LAT_MAX - LAT_MIN) / (numLat - 1);
      const lonStep = (LON_MAX - LON_MIN) / (numLon - 1);
      
      const latIdx = Math.round((lat - LAT_MIN) / latStep);
      const lonIdx = Math.round((lon - LON_MIN) / lonStep);

      if (latIdx >= 0 && latIdx < numLat && lonIdx >= 0 && lonIdx < numLon) {
        predProfile = predictionTensor[0].map(layer => layer[latIdx][lonIdx]);
      }
    }

    setPopupData({
      lat: argoFloat.lat,
      lon: argoFloat.lon,
      argoProfile: argoFloat.profile,
      profile: predProfile || Array(15).fill(null),
      depths: argoData.depths
    });
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[var(--color-paper-bg)] overflow-hidden">
      <div className="flex-1 flex min-h-0">
        
        {/* Map Area */}
        <div className="flex-1 relative border-r border-[var(--color-paper-border)] p-4 flex flex-col min-h-0 overflow-hidden">
          <div className="flex-none bg-white/50 backdrop-blur-md border border-[var(--color-paper-border)] p-4 rounded-xl mb-4 flex items-center justify-between shadow-sm">
            <div>
              <h2 className="font-display text-xl font-bold text-[var(--color-ink-dark)]">Independent Validation</h2>
              <p className="text-sm text-[var(--color-ink-light)]">
                ARGO Gridded Temperature Profiles for {date}
              </p>
            </div>
            <div className="flex items-center gap-3">
              {argoData && (
                <div className="px-3 py-1 bg-blue-50 text-blue-800 border border-blue-200 rounded-full text-xs font-bold tracking-wide">
                  SOURCE: {argoData.source}
                </div>
              )}
            </div>
          </div>

          <div className="flex-1 relative rounded-xl overflow-hidden border border-[var(--color-paper-border)] bg-[var(--color-paper-surface)] shadow-sm min-h-0">
            {/* ── Real ARGO floats (Plotly scattergeo) ── */}
            {loading ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-[var(--color-coffee-main)] mb-2" />
                <p className="text-[var(--color-ink-medium)] font-medium">Fetching ARGO Profiles...</p>
              </div>
            ) : error ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-red-50 p-6 text-center">
                <AlertCircle className="w-10 h-10 text-red-500 mb-2" />
                <h3 className="font-display font-bold text-red-800">Validation Unavailable</h3>
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            ) : argoData?.floats?.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-[var(--color-paper-bg)]/50 p-6 text-center">
                <Info className="w-10 h-10 text-gray-400 mb-2" />
                <h3 className="font-display font-bold text-gray-700">No Coverage</h3>
                <p className="text-gray-500 text-sm max-w-sm">There are no ARGO floats in this region for the selected date ({date}).</p>
              </div>
            ) : (
              <Plot
                onClick={handleMapClick}
                data={plotData}
                layout={{
                  autosize: true,
                  margin: { l: 0, r: 0, t: 0, b: 0 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  geo: {
                    projection: { type: 'equirectangular' },
                    lonaxis: { range: [LON_MIN, LON_MAX] },
                    lataxis: { range: [LAT_MIN, LAT_MAX] },
                    showcoastlines: true,
                    coastlinecolor: 'rgba(0,0,0,0.2)',
                    showland: true,
                    landcolor: '#E8E2D5',
                    showocean: true,
                    oceancolor: '#F0F8FF',
                  }
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
              />
            )}

            {/* Sensor Info Tile Overlay */}
            <AnimatePresence>
              {!loading && !error && argoData && (
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.4 }}
                  className="absolute top-4 left-4 z-10 w-80 bg-[var(--color-paper-surface)]/80 backdrop-blur-md border border-[var(--color-paper-border)] shadow-lg rounded-xl overflow-hidden pointer-events-none"
                >
                  <div className="bg-[#1e3a8a] px-4 py-2 text-white font-bold tracking-widest text-xs uppercase shadow-sm flex items-center justify-between">
                    <span>Argo Float Sensor Data</span>
                    <span className="opacity-75">In-Situ</span>
                  </div>
                  <div className="p-4 flex flex-col gap-3">
                    <p className="text-xs text-[var(--color-ink-medium)] leading-relaxed border-b border-[var(--color-paper-border)] pb-3">
                      This validation data is sourced from autonomous drifting profiling floats that measure water column properties as they rise to the surface from 2,000m depth.
                    </p>
                    <div>
                      <span className="text-[10px] font-bold text-[var(--color-ink-light)] uppercase tracking-widest block mb-2">Primary Sensor Array (CTD)</span>
                      <div className="mb-2 pb-2 border-b border-[var(--color-paper-border)] border-dashed">
                        <span className="text-[11px] font-mono text-[var(--color-ink-dark)] font-bold">Model: Sea-Bird SBE 41CP CTD</span>
                      </div>
                      <ul className="space-y-1.5">
                        <li className="flex justify-between items-center">
                          <span className="text-xs font-semibold text-[var(--color-ink-dark)]">Temperature</span>
                          <span className="text-[10px] text-gray-500 font-mono">Precision Thermistor</span>
                        </li>
                        <li className="flex justify-between items-center">
                          <span className="text-xs font-semibold text-[var(--color-ink-dark)]">Salinity</span>
                          <span className="text-[10px] text-gray-500 font-mono">Inductive Conductivity Cell</span>
                        </li>
                        <li className="flex justify-between items-center">
                          <span className="text-xs font-semibold text-[var(--color-ink-dark)]">Depth</span>
                          <span className="text-[10px] text-gray-500 font-mono">Strain Gauge Pressure</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {popupData && (
                <ProfilePopup 
                  lat={popupData.lat} 
                  lon={popupData.lon} 
                  profile={popupData.profile}
                  argoProfile={popupData.argoProfile}
                  depths={popupData.depths} 
                  onClose={() => setPopupData(null)} 
                />
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Sidebar for Metrics */}
        <div className="w-80 flex-none bg-[var(--color-paper-surface)] overflow-y-auto p-4 flex flex-col">
          <h3 className="font-display font-bold text-[var(--color-ink-dark)] mb-4 uppercase tracking-widest text-xs border-b border-[var(--color-paper-border)] pb-2">
            Skill Metrics
          </h3>
          
          {!skillMetrics ? (
            <p className="text-sm text-[var(--color-ink-light)] italic">Run a prediction to see metrics.</p>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-2 mb-4">
                <div className="bg-[var(--color-paper-elevated)] p-3 rounded-lg border border-[var(--color-paper-border)]">
                  <div className="stat-label mb-1">Coverage</div>
                  <div className="stat-headline text-lg" style={{ color: 'var(--color-accent)' }}>{skillMetrics.argo_coverage_pct}%</div>
                </div>
                <div className="bg-[var(--color-paper-elevated)] p-3 rounded-lg border border-[var(--color-paper-border)]">
                  <div className="stat-label mb-1">Floats</div>
                  <div className="stat-headline text-lg" style={{ color: 'var(--color-accent)' }}>{argoData?.floats?.length || 0}</div>
                </div>
              </div>

              <div className="bg-[var(--color-paper-elevated)] rounded-lg border border-[var(--color-paper-border)] overflow-hidden">
                <table className="w-full text-left">
                  <thead className="border-b border-[var(--color-paper-border)]" style={{ background: 'var(--color-paper-bg)' }}>
                    <tr>
                      <th className="px-3 py-2 stat-label">Depth (m)</th>
                      <th className="px-3 py-2 stat-label text-right">Corr</th>
                      <th className="px-3 py-2 stat-label text-right">RMSE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {skillMetrics.depths.map((d, i) => (
                      <tr key={d} className="border-b border-[var(--color-paper-border-subtle)] last:border-0 transition-colors hover:bg-black/5">
                        <td className="px-3 py-1.5 stat-value">{d}</td>
                        <td className="px-3 py-1.5 stat-value text-right">
                          {skillMetrics.correlation[i] !== null ? skillMetrics.correlation[i].toFixed(3) : '--'}
                        </td>
                        <td className="px-3 py-1.5 stat-value text-right">
                          {skillMetrics.rmse[i] !== null ? skillMetrics.rmse[i].toFixed(2) : '--'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
