import copernicusmarine
import xarray as xr
import numpy as np
import os

dates = ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01']
target_depths_meters = [0.5, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 700.0, 925.0]
combined_valid_data = []

for date in dates:
    print(f"Downloading data for {date}...")
    filename = f"stats_sample_{date}.nc"
    copernicusmarine.subset(
        dataset_id='cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m',
        variables=['thetao'],
        start_datetime=f'{date}T00:00:00', end_datetime=f'{date}T00:00:00',
        minimum_longitude=45.0, maximum_longitude=100.0,
        minimum_latitude=-10.0, maximum_latitude=30.0,
        minimum_depth=0.0, maximum_depth=1000.0,
        output_filename=filename,
        force_download=True
    )
    
    ds = xr.open_dataset(filename)
    thetao = ds['thetao'].isel(time=0).values # shape: [depth, lat, lon]
    depths = ds.depth.values
    
    target_indices = []
    for d in target_depths_meters:
        idx = (np.abs(depths - d)).argmin()
        target_indices.append(idx)
        
    for i, idx in enumerate(target_indices):
        layer_data = thetao[idx]
        valid_data = layer_data[~np.isnan(layer_data)]
        combined_valid_data.append(valid_data)
        
    # Clean up file to save space
    ds.close()
    os.remove(filename)

all_valid = np.concatenate(combined_valid_data)

percentiles = np.linspace(0, 100, 11)
results = np.percentile(all_valid, percentiles)

print("\n--- Multi-Season Deciles ---")
for p, val in zip(percentiles, results):
    pos = val / 30.0
    print(f"[{pos:.3f}, color_{int(p)}], // {val:.2f}°C ({p}th percentile)")
