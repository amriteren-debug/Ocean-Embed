import xarray as xr
import numpy as np
import json

ds = xr.open_dataset('stats_sample.nc')
thetao = ds['thetao'].isel(time=0).values # shape: [depth, lat, lon]
depths = ds.depth.values
target_depths_meters = [0.5, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 925.0]

target_indices = []
for d in target_depths_meters:
    idx = (np.abs(depths - d)).argmin()
    target_indices.append(idx)

# Extract data for all 15 depths
combined_valid_data = []
for i, idx in enumerate(target_indices):
    layer_data = thetao[idx]
    valid_data = layer_data[~np.isnan(layer_data)]
    combined_valid_data.append(valid_data)

all_valid = np.concatenate(combined_valid_data)

# We want roughly 11 color stops, so deciles: 0, 10, 20... 100
percentiles = np.linspace(0, 100, 11)
results = np.percentile(all_valid, percentiles)

print("Deciles:")
for p, val in zip(percentiles, results):
    # normalize relative to 0-30 max range
    pos = val / 30.0
    print(f"[{pos:.3f}, color_{int(p)}], // {val:.2f}°C ({p}th percentile)")

