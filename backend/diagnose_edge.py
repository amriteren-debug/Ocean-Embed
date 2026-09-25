import numpy as np
import requests
from preprocessing import preprocess_date

DATE = "2026-09-06"
BASE_URL = "http://localhost:8000"

print(f"Loading preprocessing for {DATE}...")
prep = preprocess_date(DATE, data_dir=".")
lats = prep["lat"]
lons = prep["lon"]
raw_fields = prep["raw_fields"]
land_mask = prep["land_mask"]

def get_nearest_idx(val, arr):
    return (np.abs(arr - val)).argmin()

# The anomaly point
target_lat = 23.25
target_lon = 61.00
lat_idx = get_nearest_idx(target_lat, lats)
lon_idx = get_nearest_idx(target_lon, lons)

print(f"\n=======================================================")
print(f"ANOMALY POINT: {target_lat}°N, {target_lon}°E (Grid index: lat={lat_idx}, lon={lon_idx})")
print(f"Actual grid coordinates: {lats[lat_idx]:.2f}°N, {lons[lon_idx]:.2f}°E")
print(f"=======================================================\n")

# Check land mask
print(f"Is this pixel masked as land? {land_mask[lat_idx, lon_idx]}")

# Print raw inputs
print("\nRaw Input Channels:")
for name, field in raw_fields.items():
    val = field[lat_idx, lon_idx]
    print(f"  {name:>8}: {val:.4f}")

# Fetch ARGO and Prediction data to do a wider sample comparison
print(f"\nFetching prediction for {DATE}...")
pred_res = requests.post(f"{BASE_URL}/predict", json={"date": DATE}).json()
pred_tensor = np.array(pred_res["prediction_data"][0])

print(f"Fetching ARGO data for {DATE}...")
argo_data = requests.get(f"{BASE_URL}/argo/{DATE}").json()

depths = pred_res["depths"]
depth_idx_500 = depths.index(500)
depth_idx_700 = depths.index(700)
depth_idx_1000 = depths.index(1000)

print("\n=======================================================")
print("        WIDER SAMPLE COMPARISON (COASTAL VS OPEN)       ")
print("=======================================================\n")

# Pick a few specific points from the ARGO data based on their lat/lon
# We will identify if they are edge/coastal (near land mask or boundary) or open ocean.
print(f"{'Lat':>8} | {'Lon':>8} | {'Type':>10} | {'500m (A/M)':>15} | {'700m (A/M)':>15} | {'1000m (A/M)':>15}")
print("-" * 90)

for float_pt in argo_data["floats"]:
    lat = float_pt["lat"]
    lon = float_pt["lon"]
    profile = float_pt["profile"]
    
    idx_y = get_nearest_idx(lat, lats)
    idx_x = get_nearest_idx(lon, lons)
    
    # Check if it's near the edge or near land
    is_edge = (idx_y < 5 or idx_y > len(lats)-5 or idx_x < 5 or idx_x > len(lons)-5)
    
    # Check land proximity (is there land within 2 pixels?)
    y_min = max(0, idx_y - 2)
    y_max = min(len(lats), idx_y + 3)
    x_min = max(0, idx_x - 2)
    x_max = min(len(lons), idx_x + 3)
    near_land = np.any(land_mask[y_min:y_max, x_min:x_max])
    
    if is_edge and near_land:
        pt_type = "Edge+Coast"
    elif is_edge:
        pt_type = "Edge"
    elif near_land:
        pt_type = "Coastal"
    else:
        pt_type = "Open Ocean"
        
    a500 = profile[depth_idx_500]
    a700 = profile[depth_idx_700]
    a1000 = profile[depth_idx_1000]
    
    if a500 is None and a700 is None and a1000 is None:
        continue
        
    m500 = pred_tensor[depth_idx_500, idx_y, idx_x]
    m700 = pred_tensor[depth_idx_700, idx_y, idx_x]
    m1000 = pred_tensor[depth_idx_1000, idx_y, idx_x]
    
    # Handle None from JSON nulls
    m500 = np.nan if m500 is None else float(m500)
    m700 = np.nan if m700 is None else float(m700)
    m1000 = np.nan if m1000 is None else float(m1000)
    
    s_500 = f"{a500 if a500 else 'NaN':>5}/{m500 if np.isfinite(m500) else 'NaN':>5.2f}"
    s_700 = f"{a700 if a700 else 'NaN':>5}/{m700 if np.isfinite(m700) else 'NaN':>5.2f}"
    s_1000 = f"{a1000 if a1000 else 'NaN':>5}/{m1000 if np.isfinite(m1000) else 'NaN':>5.2f}"
    
    print(f"{lat:>8.2f} | {lon:>8.2f} | {pt_type:>10} | {s_500:>15} | {s_700:>15} | {s_1000:>15}")
