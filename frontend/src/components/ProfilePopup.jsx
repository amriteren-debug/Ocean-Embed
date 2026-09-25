import React from 'react';
import { motion } from 'framer-motion';
import { X } from 'lucide-react';
import Plot from 'react-plotly.js';

export default function ProfilePopup({ lat, lon, profile, argoProfile, depths, onClose }) {
  if (!profile) return null;

  // Filter out NaNs if we want, or Plotly will just leave gaps
  
  const plotData = [
    {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Prediction',
      x: profile,
      y: depths,
      line: { color: '#ab1a45', width: 3 },
      marker: { color: '#ab1a45', size: 6 },
      hovertemplate: 'Temp: %{x:.2f}°C<br>Depth: %{y}m<extra></extra>',
    }
  ];

  if (argoProfile && argoProfile.some(v => v !== null && !isNaN(v))) {
    plotData.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: 'ARGO',
      x: argoProfile,
      y: depths,
      line: { color: '#1b3472', width: 2, dash: 'dot' },
      marker: { color: '#1b3472', size: 8, symbol: 'diamond' },
      hovertemplate: 'ARGO: %{x:.2f}°C<br>Depth: %{y}m<extra></extra>',
    });
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="absolute bottom-4 right-4 w-80 bg-white/95 backdrop-blur-md border border-[var(--color-paper-border)] shadow-xl rounded-xl z-[100] flex flex-col overflow-hidden"
    >
      <div className="flex items-center justify-between p-3 border-b border-[var(--color-paper-border)] bg-[var(--color-paper-surface)]">
        <div>
          <h4 className="font-display font-bold text-sm text-[var(--color-ink-dark)]">Depth Profile</h4>
          <p className="text-[10px] text-[var(--color-ink-light)] tracking-wide">
            {lat.toFixed(2)}°N, {lon.toFixed(2)}°E
          </p>
        </div>
        <button 
          onClick={onClose}
          className="p-1 rounded-md hover:bg-gray-200 text-gray-500 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="h-64 p-2 bg-white">
        <Plot
          data={plotData}
          layout={{
            autosize: true,
            margin: { l: 40, r: 10, t: 10, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            showlegend: argoProfile ? true : false,
            legend: { x: 0.5, y: 1.1, xanchor: 'center', orientation: 'h', bgcolor: 'rgba(255,255,255,0.7)' },
            xaxis: {
              title: 'Temperature (°C)',
              titlefont: { size: 10, color: '#8B8379' },
              tickfont: { size: 9 },
              gridcolor: '#f0f0f0',
              zeroline: false
            },
            yaxis: {
              title: 'Depth (m)',
              titlefont: { size: 10, color: '#8B8379' },
              tickfont: { size: 9 },
              gridcolor: '#f0f0f0',
              autorange: 'reversed', // Y-axis reversed (0 at top)
              zeroline: false
            }
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </div>
    </motion.div>
  );
}
