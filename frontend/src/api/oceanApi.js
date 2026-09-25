/**
 * oceanApi.js — Async utility for the OceanPredictorNet backend (port 8000).
 *
 * Provides a clean, reusable function to call the 36-channel CNN /predict
 * endpoint.  Keeps all network logic out of React components.
 */

import axios from 'axios';

const CNN_BASE_URL = 'http://localhost:8000';

/**
 * Generate a mock 7-channel input tensor for testing.
 *
 * Returns a nested array of shape [7, H, W] filled with plausible
 * synthetic ocean values.
 *
 * @param {number} H  - Grid height  (default 161)
 * @param {number} W  - Grid width   (default 221)
 * @returns {number[][][]}  Shape [7, H, W]
 */
export function createMockInput(H = 161, W = 221) {
  const channels = [];
  // Channel ranges: SST ~20-30, SSS ~33-37, SSH ~-0.5-0.5,
  // U_cur ~-0.5-0.5, V_cur ~-0.5-0.5, U_wind ~-5-5, V_wind ~-5-5
  const ranges = [
    [20, 30],    // SST
    [33, 37],    // SSS
    [-0.5, 0.5], // SSH
    [-0.5, 0.5], // U_cur
    [-0.5, 0.5], // V_cur
    [-5, 5],     // U_wind
    [-5, 5],     // V_wind
  ];

  for (let c = 0; c < 7; c++) {
    const [lo, hi] = ranges[c];
    const grid = [];
    for (let i = 0; i < H; i++) {
      const row = [];
      for (let j = 0; j < W; j++) {
        row.push(lo + Math.random() * (hi - lo));
      }
      grid.push(row);
    }
    channels.push(grid);
  }
  return channels;
}

/**
 * Call the OceanPredictorNet /predict endpoint.
 *
 * @param {number[][][]} inputData  - Shape [7, H, W]
 * @returns {Promise<{status: string, shape: number[], prediction: number[][][]}>}
 * @throws {Error} with a human-readable message on failure.
 */
export async function predictCNN(inputData) {
  try {
    const response = await axios.post(`${CNN_BASE_URL}/predict`, {
      data: inputData,
    });
    return response.data;
  } catch (err) {
    // Extract the most useful error string
    const detail =
      err.response?.data?.detail ||
      err.response?.data?.message ||
      err.message ||
      'Network error — could not reach the prediction backend.';
    throw new Error(detail);
  }
}

/**
 * Health-check — hit GET / and return the payload.
 */
export async function healthCheck() {
  try {
    const res = await axios.get(`${CNN_BASE_URL}/`);
    return res.data;
  } catch {
    return { model_loaded: false, error: 'Backend unreachable' };
  }
}
