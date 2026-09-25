/**
 * useSectorAnalysis.js
 * ====================
 * Computes scientifically meaningful metrics from the raw predictionTensor
 * for each SECTOR FOCUS tab (Fisheries, Naval, Scuba).
 *
 * DATA ACCURACY CONTRACT:
 *  - Every returned number traces back to a real tensor value.
 *  - Assumed constants (salinity for SVP) are clearly flagged in the result.
 *  - If a computation can't run, the relevant field is null (UI shows "Insufficient data").
 */

import { useMemo } from 'react';

// Shared helper to guarantee consistent land masking across all components
export function isValidOceanPixel(val) {
  return val !== null && val !== undefined && !Number.isNaN(val);
}

/**
 * Mackenzie (1981) sound speed formula.
 * T = temperature (°C), S = salinity (PSU), D = depth (m)
 * Returns speed of sound in m/s.
 */
export function mackenzie(T, S, D) {
  return (
    1448.96 +
    4.591 * T -
    0.05304 * T * T +
    0.0002374 * T * T * T +
    1.340 * (S - 35) +
    0.01630 * D +
    1.675e-7 * D * D -
    0.01025 * T * (S - 35) -
    7.139e-13 * T * D * D * D
  );
}

/**
 * Compute a basin-wide mean temperature at a specific depth index.
 * Returns null if no valid ocean pixels exist.
 */
function basinMean(layer) {
  if (!layer) return null;
  let sum = 0, count = 0;
  for (let r = 0; r < layer.length; r++) {
    for (let c = 0; c < layer[r].length; c++) {
      const v = layer[r][c];
      if (isValidOceanPixel(v)) {
        sum += v;
        count++;
      }
    }
  }
  return count > 0 ? sum / count : null;
}

function findClosestIndex(depths, target) {
  let best = 0, bestDiff = Infinity;
  depths.forEach((d, i) => {
    const diff = Math.abs(d - target);
    if (diff < bestDiff) { bestDiff = diff; best = i; }
  });
  return best;
}

export function useSectorAnalysis(predictionTensor, modelDepths, inputsSss) {
  return useMemo(() => {
    if (!predictionTensor || !modelDepths || modelDepths.length < 2) {
      return { fisheries: null, naval: null, scuba: null };
    }

    const tensor = predictionTensor[0]; // shape [15, lat, lon]
    const depths = modelDepths;

    // Determine the salinity to use
    // inputsSss might be a string (like "34.50") or a number or "N/A"
    const parsedSss = Number(inputsSss);
    const hasRealSss = !Number.isNaN(parsedSss) && parsedSss > 0;
    const assumedSalinity = hasRealSss ? parsedSss : 35;

    // ── Basin-wide mean profile ─────────────────────────────────────────────
    const meanProfile = depths.map((d, di) => ({
      depth: d,
      mean: basinMean(tensor[di]),
    }));

    // ══════════════════════════════════════════════════════════════════════
    // FISHERIES (0–200m)
    // ══════════════════════════════════════════════════════════════════════
    const fisheriesEntries = depths
      .map((d, i) => ({ d, i }))
      .filter(({ d }) => d <= 200);

    let thermoclineDepth = null;
    let thermoclineTemp = null;
    let maxGrad = 0;

    // First collect all valid gradients
    const fisheriesGradients = [];
    for (let k = 0; k < fisheriesEntries.length - 1; k++) {
      const { d: d1, i: i1 } = fisheriesEntries[k];
      const { d: d2, i: i2 } = fisheriesEntries[k + 1];
      const t1 = meanProfile[i1].mean;
      const t2 = meanProfile[i2].mean;
      if (t1 === null || t2 === null || (d2 - d1) === 0) {
        fisheriesGradients.push(null);
      } else {
        fisheriesGradients.push((t1 - t2) / (d2 - d1));
      }
    }

    let maxFishIdx = -1;
    for (let k = 0; k < fisheriesGradients.length; k++) {
      if (fisheriesGradients[k] !== null && fisheriesGradients[k] > maxGrad) {
        maxGrad = fisheriesGradients[k];
        maxFishIdx = k;
      }
    }

    if (maxFishIdx !== -1) {
      const d1 = fisheriesEntries[maxFishIdx].d;
      const d2 = fisheriesEntries[maxFishIdx + 1].d;
      const dz = d2 - d1;
      
      const g0 = maxFishIdx > 0 && fisheriesGradients[maxFishIdx - 1] !== null 
        ? fisheriesGradients[maxFishIdx - 1] : maxGrad;
      const g1 = maxGrad;
      const g2 = maxFishIdx < fisheriesGradients.length - 1 && fisheriesGradients[maxFishIdx + 1] !== null 
        ? fisheriesGradients[maxFishIdx + 1] : maxGrad;

      let offset = 0;
      const denom = 2 * (g0 - 2 * g1 + g2);
      if (denom !== 0) {
        offset = (g0 - g2) / denom;
      }
      offset = Math.max(-0.5, Math.min(0.5, offset));

      thermoclineDepth = d1 + (dz / 2) + offset * dz;
      
      const t1 = meanProfile[fisheriesEntries[maxFishIdx].i].mean;
      const t2 = meanProfile[fisheriesEntries[maxFishIdx + 1].i].mean;
      const fraction = 0.5 + offset;
      thermoclineTemp = t1 - fraction * (t1 - t2);
    }

    // Aggregation band depths (mean temp 20–29°C) within 0–200m
    const aggregationBandDepths = fisheriesEntries
      .filter(({ i }) => {
        const m = meanProfile[i].mean;
        return m !== null && m >= 20 && m <= 29;
      })
      .map(({ d }) => d);

    // Compute dynamic min/max for 3D Mesh (pixels in the aggregation band)
    let minAggTemp = 20, maxAggTemp = 29;
    if (aggregationBandDepths.length > 0) {
      let actualMin = Infinity, actualMax = -Infinity;
      fisheriesEntries.forEach(({ i }) => {
        const m = meanProfile[i].mean;
        if (m !== null && m >= 20 && m <= 29) {
           // Scan tensor for min/max within this valid depth slice
           const slice = tensor[i];
           if (!slice) return;
           for (let r = 0; r < slice.length; r++) {
             for (let c = 0; c < slice[r].length; c++) {
               const v = slice[r][c];
               if (isValidOceanPixel(v) && v >= 20 && v <= 29) {
                 if (v < actualMin) actualMin = v;
                 if (v > actualMax) actualMax = v;
               }
             }
           }
        }
      });
      if (actualMin <= actualMax) {
        minAggTemp = actualMin;
        maxAggTemp = actualMax;
      }
    }

    // Per-depth basin-mean temperatures at aggregation-band depths.
    // Used by the species advisory to match species thermal ranges to actual current temps.
    const aggregationBandProfile = fisheriesEntries
      .filter(({ i }) => {
        const m = meanProfile[i].mean;
        return m !== null && m >= 20 && m <= 29;
      })
      .map(({ d, i }) => ({ depth: d, meanTemp: +meanProfile[i].mean.toFixed(2) }));

    const fisheries = {
      thermoclineDepth: thermoclineDepth !== null ? +thermoclineDepth.toFixed(1) : null,
      thermoclineTemp: thermoclineTemp !== null ? +thermoclineTemp.toFixed(2) : null,
      thermoclineGradPer100m: maxGrad > 0 ? +(maxGrad * 100).toFixed(2) : null,
      aggregationBandDepths,
      aggregationBandProfile, // [{depth, meanTemp}] — for species advisory
      colorRange: [minAggTemp, maxAggTemp], // Dynamic range for 3D mesh
      mapDepth: thermoclineDepth !== null ? thermoclineDepth : 50,
      mapDepthIndex: findClosestIndex(depths, thermoclineDepth !== null ? thermoclineDepth : 50),
    };

    // ══════════════════════════════════════════════════════════════════════
    // NAVAL (0–1000m full range)
    // ══════════════════════════════════════════════════════════════════════
    const soundSpeeds = meanProfile.map(({ depth, mean }) => ({
      depth,
      speed: mean !== null ? mackenzie(mean, assumedSalinity, depth) : null,
    }));

    // SLD = depth of max sound speed in upper 300m
    let sld = null;
    let sldSpeedVal = null;
    let maxSpeed = -Infinity;
    let maxSpeedIdx = -1;
    depths.forEach((d, i) => {
      if (d > 300) return;
      const sp = soundSpeeds[i].speed;
      if (sp !== null && sp > maxSpeed) {
        maxSpeed = sp;
        sld = d;
        sldSpeedVal = sp;
        maxSpeedIdx = i;
      }
    });

    // Interpolate SLD if possible
    if (maxSpeedIdx >= 0) {
      // Linear interpolation of depth between the peak and the higher of its two neighbors.
      let neighborIdx = -1;
      let neighborSpeed = -Infinity;
      if (maxSpeedIdx > 0) {
        const prevSp = soundSpeeds[maxSpeedIdx - 1].speed;
        if (prevSp !== null && prevSp > neighborSpeed) {
          neighborSpeed = prevSp;
          neighborIdx = maxSpeedIdx - 1;
        }
      }
      if (maxSpeedIdx < soundSpeeds.length - 1 && depths[maxSpeedIdx + 1] <= 300) {
        const nextSp = soundSpeeds[maxSpeedIdx + 1].speed;
        if (nextSp !== null && nextSp > neighborSpeed) {
          neighborSpeed = nextSp;
          neighborIdx = maxSpeedIdx + 1;
        }
      }
      if (neighborIdx !== -1) {
        // Take the midpoint of the interval where the peak is occurring.
        sld = (depths[maxSpeedIdx] + depths[neighborIdx]) / 2;
        sldSpeedVal = (maxSpeed + neighborSpeed) / 2; // Approximate speed at midpoint
      }
    }

    // Steepest temp gradient over full range
    let navalThermoclineDepth = null;
    let navalMaxGrad = 0;
    const navalGradients = [];
    for (let k = 0; k < meanProfile.length - 1; k++) {
      const t1 = meanProfile[k].mean;
      const t2 = meanProfile[k + 1].mean;
      const dz = depths[k + 1] - depths[k];
      if (t1 === null || t2 === null || dz === 0) {
        navalGradients.push(null);
      } else {
        navalGradients.push((t1 - t2) / dz);
      }
    }

    let maxNavalIdx = -1;
    for (let k = 0; k < navalGradients.length; k++) {
      if (navalGradients[k] !== null && navalGradients[k] > navalMaxGrad) {
        navalMaxGrad = navalGradients[k];
        maxNavalIdx = k;
      }
    }

    if (maxNavalIdx !== -1) {
      const d1 = depths[maxNavalIdx];
      const d2 = depths[maxNavalIdx + 1];
      const dz = d2 - d1;
      
      const g0 = maxNavalIdx > 0 && navalGradients[maxNavalIdx - 1] !== null 
        ? navalGradients[maxNavalIdx - 1] : navalMaxGrad;
      const g1 = navalMaxGrad;
      const g2 = maxNavalIdx < navalGradients.length - 1 && navalGradients[maxNavalIdx + 1] !== null 
        ? navalGradients[maxNavalIdx + 1] : navalMaxGrad;

      let offset = 0;
      const denom = 2 * (g0 - 2 * g1 + g2);
      if (denom !== 0) {
        offset = (g0 - g2) / denom;
      }
      offset = Math.max(-0.5, Math.min(0.5, offset));

      navalThermoclineDepth = d1 + (dz / 2) + offset * dz;
    }

    // Shadow Zone calculation
    let shadowZone = null;
    let shadowThreshold = null;
    const negativeGradients = [];
    const gradientProfile = [];
    if (maxSpeedIdx >= 0) {
      for (let k = maxSpeedIdx; k < soundSpeeds.length - 1; k++) {
        const s1 = soundSpeeds[k].speed;
        const s2 = soundSpeeds[k + 1].speed;
        if (s1 === null || s2 === null) continue;
        const dz = depths[k + 1] - depths[k];
        if (dz === 0) continue;
        const grad = (s2 - s1) / dz;
        gradientProfile.push({ grad, d1: depths[k], d2: depths[k + 1] });
        if (grad < 0) negativeGradients.push(grad);
      }
    }
    if (negativeGradients.length > 0) {
      negativeGradients.sort((a, b) => a - b); // Most negative first
      const medianIdx = Math.floor(negativeGradients.length / 2);
      shadowThreshold = negativeGradients[medianIdx];
      
      let shadowStart = null, shadowEnd = null;
      for (const p of gradientProfile) {
        if (p.grad <= shadowThreshold) {
          if (shadowStart === null) shadowStart = p.d1;
          shadowEnd = p.d2;
        } else {
          if (shadowStart !== null) break; // Capture the first continuous block
        }
      }
      if (shadowStart !== null) shadowZone = [shadowStart, shadowEnd];
    }

    // Thermocline Strength Map (Per-pixel)
    const numLat = tensor[0].length;
    const numLon = tensor[0][0].length;
    const thermoclineMap = Array(numLat).fill(null).map(() => Array(numLon).fill(null));
    let maxGradValue = -Infinity;
    let maxGradPixel = null;
    const allGradients = [];
    
    // Subsample step for performance if tensor is huge, but we can do dense for accuracy.
    for (let r = 0; r < numLat; r++) {
      for (let c = 0; c < numLon; c++) {
        let maxColGrad = -Infinity;
        for (let k = 0; k < depths.length - 1; k++) {
          if (!tensor[k] || !tensor[k + 1] || !tensor[k][r] || !tensor[k + 1][r]) continue;
          
          const t1 = tensor[k][r][c];
          const t2 = tensor[k + 1][r][c];
          if (isValidOceanPixel(t1) && isValidOceanPixel(t2)) {
             const dz = depths[k + 1] - depths[k];
             if (dz > 0) {
               const grad = (t1 - t2) / dz;
               if (grad > maxColGrad) maxColGrad = grad;
             }
          }
        }
        if (maxColGrad > -Infinity) {
          thermoclineMap[r][c] = maxColGrad;
          allGradients.push(maxColGrad);
          if (maxColGrad > maxGradValue) {
             maxGradValue = maxColGrad;
             maxGradPixel = { r, c, value: maxColGrad };
          }
        }
      }
    }

    let thermoclineTiers = { weak: 0, moderate: 0 };
    if (allGradients.length > 0) {
      allGradients.sort((a, b) => a - b);
      thermoclineTiers.weak = allGradients[Math.floor(allGradients.length * 0.33)];
      thermoclineTiers.moderate = allGradients[Math.floor(allGradients.length * 0.66)];
    }

    const naval = {
      sld: sld !== null ? +sld.toFixed(1) : null,
      sldSpeed: sldSpeedVal !== null ? +sldSpeedVal.toFixed(1) : null,
      assumedSalinity,
      hasRealSss,
      thermoclineDepth: navalThermoclineDepth !== null ? +navalThermoclineDepth.toFixed(1) : null,
      soundSpeedProfile: soundSpeeds
        .filter(s => s.speed !== null)
        .map(s => ({ depth: s.depth, speed: +s.speed.toFixed(1) })),
      mapDepth: sld !== null ? sld : 50,
      mapDepthIndex: findClosestIndex(depths, sld !== null ? sld : 50),
      shadowZone,
      shadowThreshold,
      thermoclineMap,
      thermoclineTiers,
      maxThermoclinePixel: maxGradPixel,
      // bestShadowPixel computed below
      bestShadowPixel: null,
    };

    // ── Shadow Zone Hotspots (per-pixel, dense scan) ────────────────────
    // For each ocean pixel below the SLD, compute the mean dc/dz (sound-speed gradient)
    // across the shadow zone depth pairs using real per-pixel Mackenzie temperatures.
    // "Best" = most negative gradient = steepest downward ray bending (Snell's law).
    // We expose the top-5 spatially-spread hotspots so the UI can show green zone rings,
    // not just a single pin. Minimum pixel separation of MIN_SEP enforces spatial spread.
    const MIN_SEP = 15; // minimum grid-cell separation between reported hotspots
    const allShadowCandidates = []; // will hold { r, c, gradient }

    if (maxSpeedIdx >= 0 && shadowZone !== null) {
      const shadowDepthPairs = [];
      for (let k = maxSpeedIdx; k < depths.length - 1; k++) {
        if (depths[k] >= shadowZone[0] && depths[k + 1] <= shadowZone[1] + 200) {
          shadowDepthPairs.push(k);
        }
      }

      if (shadowDepthPairs.length > 0) {
        for (let r = 0; r < numLat; r++) {
          for (let c = 0; c < numLon; c++) {
            let totalGrad = 0, pairCount = 0;
            for (const k of shadowDepthPairs) {
              if (!tensor[k] || !tensor[k + 1] || !tensor[k][r] || !tensor[k + 1][r]) continue;
              const t1 = tensor[k][r][c];
              const t2 = tensor[k + 1][r][c];
              if (!isValidOceanPixel(t1) || !isValidOceanPixel(t2)) continue;
              const dz = depths[k + 1] - depths[k];
              if (dz === 0) continue;
              const sp1 = mackenzie(t1, assumedSalinity, depths[k]);
              const sp2 = mackenzie(t2, assumedSalinity, depths[k + 1]);
              totalGrad += (sp2 - sp1) / dz;
              pairCount++;
            }
            if (pairCount === 0) continue;
            const meanGrad = totalGrad / pairCount;
            if (meanGrad < 0) { // only actual negative gradients qualify
              allShadowCandidates.push({ r, c, gradient: +meanGrad.toFixed(4) });
            }
          }
        }
      }
    }

    // Sort by most-negative gradient (strongest concealment first)
    allShadowCandidates.sort((a, b) => a.gradient - b.gradient);

    // Greedily pick top-5 with spatial spread: reject any candidate within MIN_SEP of
    // an already-selected hotspot to guarantee distinct geographic regions.
    const topShadowPixels = [];
    for (const cand of allShadowCandidates) {
      if (topShadowPixels.length >= 5) break;
      const tooClose = topShadowPixels.some(
        sel => Math.abs(sel.r - cand.r) < MIN_SEP && Math.abs(sel.c - cand.c) < MIN_SEP
      );
      if (!tooClose) topShadowPixels.push(cand);
    }

    // Expose both the single best and the ranked list
    naval.bestShadowPixel = topShadowPixels[0] ?? null;
    naval.topShadowPixels = topShadowPixels;


    const scubaEntries = depths.map((d, i) => ({ d, i })).filter(({ d }) => d <= 40);

    const coldShockZones = [];
    for (let k = 0; k < scubaEntries.length - 1; k++) {
      const { d: d1, i: i1 } = scubaEntries[k];
      const { d: d2, i: i2 } = scubaEntries[k + 1];
      const t1 = meanProfile[i1].mean;
      const t2 = meanProfile[i2].mean;
      if (t1 === null || t2 === null) continue;
      const drop = t1 - t2;
      const dz = d2 - d1;
      if (dz === 0) continue;
      const grad = drop / dz; // gradient in °C/m

      // Threshold is 0.2 °C/m
      if (grad > 0.2) {
        // Interpolate boundary to midpoint
        const boundaryDepth = d1 + dz / 2;
        coldShockZones.push({ 
          fromDepth: d1, 
          toDepth: d2, 
          boundaryDepth: +boundaryDepth.toFixed(1),
          drop: +drop.toFixed(2),
          grad: +grad.toFixed(2)
        });
      }
    }

    const shallowIdx = scubaEntries[0]?.i ?? 0;
    const shallowTemp = meanProfile[shallowIdx]?.mean ?? null;

    const scuba = {
      shallowTemp: shallowTemp !== null ? +shallowTemp.toFixed(2) : null, // Basin mean fallback
      coldShockZones,
      mapDepth: depths[scubaEntries[0]?.i ?? 0] ?? 5,
      mapDepthIndex: scubaEntries[0]?.i ?? 0,
      profile: scubaEntries.map(({ d, i }) => ({
        depth: d,
        temp: meanProfile[i].mean !== null ? +meanProfile[i].mean.toFixed(2) : null,
      })),
    };

    return { fisheries, naval, scuba };
  }, [predictionTensor, modelDepths, inputsSss]);
}

export function getWetsuitRecommendation(temp) {
  if (temp === null || temp === undefined || Number.isNaN(temp)) return { rec: null, label: null };
  if (temp > 26) return { rec: '3mm', label: 'Shortie / 3mm Wetsuit' };
  if (temp >= 20) return { rec: '5mm', label: '5mm Full Wetsuit' };
  return { rec: '7mm+hood', label: '7mm Drysuit + Hood' };
}
