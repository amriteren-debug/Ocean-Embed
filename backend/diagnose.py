import requests
import json
import numpy as np

DATE = "2026-09-06"
BASE_URL = "http://localhost:8001"

print(f"Fetching prediction for {DATE}...")
try:
    pred_res = requests.post(f"{BASE_URL}/predict", json={"date": DATE})
    pred_data = pred_res.json()
except Exception as e:
    print(f"Failed to hit /predict: {e}")
    exit(1)

print(f"Fetching ARGO data for {DATE}...")
try:
    argo_res = requests.get(f"{BASE_URL}/argo/{DATE}")
    argo_data = argo_res.json()
except Exception as e:
    print(f"Failed to hit /argo: {e}")
    exit(1)

# Extract prediction grid
# pred_data["prediction_data"] is shape (1, 15, lat, lon)
pred_tensor = np.array(pred_data["prediction_data"][0])
lats = np.array(pred_data["lat"])
lons = np.array(pred_data["lon"])

# Find indices for 500m, 700m, 1000m
depths = pred_data["depths"]
depth_idx_500 = depths.index(500)
depth_idx_700 = depths.index(700)
depth_idx_1000 = depths.index(1000)

target_depths = {
    500: depth_idx_500,
    700: depth_idx_700,
    1000: depth_idx_1000
}

# Function to get nearest grid cell index
def get_nearest_idx(val, arr):
    return (np.abs(arr - val)).argmin()

print("\n=======================================================")
print("             ARGO vs MODEL DEEP LAYER COMPARISON         ")
print("=======================================================\n")

for d_label, d_idx in target_depths.items():
    print(f"--- DEPTH: {d_label}m ---")
    print(f"{'Float_Lat':>10} | {'Float_Lon':>10} | {'ARGO (°C)':>10} | {'Model (°C)':>10} | {'Diff':>10}")
    print("-" * 60)
    
    for float_pt in argo_data["floats"]:
        lat = float_pt["lat"]
        lon = float_pt["lon"]
        profile = float_pt["profile"]
        
        argo_val = profile[d_idx]
        if argo_val is None:
            continue
            
        lat_idx = get_nearest_idx(lat, lats)
        lon_idx = get_nearest_idx(lon, lons)
        
        model_val = pred_tensor[d_idx, lat_idx, lon_idx]
        
        if np.isnan(model_val) or model_val is None:
            model_val_str = "NaN"
            diff_str = "NaN"
        else:
            model_val_str = f"{model_val:.2f}"
            diff = abs(argo_val - model_val)
            diff_str = f"{diff:.2f}"
            
        print(f"{lat:>10.2f} | {lon:>10.2f} | {argo_val:>10.2f} | {model_val_str:>10} | {diff_str:>10}")
    
    print("\n")
