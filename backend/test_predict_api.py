import requests
import json
import numpy as np

res = requests.post("http://localhost:8001/predict", json={"date": "2026-08-31"})
if res.status_code == 200:
    data = res.json()
    tensor = data["prediction_data"][0]  # Should be 15, lat, lon
    print("Shape:", len(tensor), len(tensor[0]), len(tensor[0][0]))
    
    # Check for None values (which represent null in JSON)
    layer = tensor[0]
    has_null = False
    max_val = -np.inf
    min_val = np.inf
    
    for row in layer:
        for val in row:
            if val is None:
                has_null = True
            else:
                max_val = max(max_val, val)
                min_val = min(min_val, val)
                
    print(f"Has nulls? {has_null}")
    print(f"Min: {min_val}, Max: {max_val}")
else:
    print(f"Error {res.status_code}: {res.text}")
