import os
import json
import numpy as np
import xarray as xr

NATIVE_DEPTHS_M = [
    0.494, 5.078, 9.573, 18.496, 29.445,
    47.374, 77.854, 92.326, 155.851, 186.126,
    318.127, 380.213, 541.089, 643.567, 902.339,
]

def compute_stats(data_dir):
    thetao_path = os.path.join(data_dir, "thetao_mock.nc")
    
    if not os.path.exists(thetao_path):
        print(f"Error: {thetao_path} not found.")
        return
        
    ds = xr.open_dataset(thetao_path)
    depths = ds.depth.values
    
    target_indices = []
    for d in NATIVE_DEPTHS_M:
        target_indices.append(int(np.abs(depths - d).argmin()))
        
    stats = []
    print("Computing depth-wise statistics for target normalization...")
    
    for i, idx in enumerate(target_indices):
        data = ds['thetao'].isel(depth=idx).values
        # Only compute stats over valid ocean pixels (non-zero)
        valid = data[data != 0]
        
        if len(valid) > 0:
            mean = float(valid.mean())
            std = float(valid.std())
            # Ensure std is never exactly zero to prevent div/0
            if std == 0:
                std = 1.0
        else:
            mean = 0.0
            std = 1.0
            
        stats.append({
            "level": i,
            "depth_m": float(depths[idx]),
            "mean": mean,
            "std": std
        })
        print(f"Depth {depths[idx]:.1f}m -> mean: {mean:.2f}, std: {std:.2f}")
        
    out_path = "target_depth_stats.json"
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Saved stats to {out_path}")
    ds.close()

if __name__ == "__main__":
    compute_stats("training_data")
