import asyncio
import numpy as np
from main import predict_subsurface, PredictRequest

async def test():
    req = PredictRequest(date="2026-08-31")
    res = await predict_subsurface(req)
    
    tensor = res["prediction_data"][0]
    layer14 = np.array(tensor[14], dtype=float)
    
    valid_mask = ~np.isnan(layer14)
    if not valid_mask.any():
        print("No valid pixels at Depth 14")
        return
        
    valid_pixels = layer14[valid_mask]
    min_val = valid_pixels.min()
    max_val = valid_pixels.max()
    
    print(f"Depth 14 min: {min_val:.4f}, max: {max_val:.4f}")
    
    # Check for near-identical low-variance values
    # Let's count how many pixels fall within 0.1 of the median
    median_val = np.median(valid_pixels)
    narrow_range = (valid_pixels >= median_val - 0.05) & (valid_pixels <= median_val + 0.05)
    narrow_pct = narrow_range.mean() * 100
    
    print(f"Depth 14 variance: {valid_pixels.var():.6f}")
    print(f"Depth 14 standard deviation: {valid_pixels.std():.6f}")
    print(f"Percentage of valid pixels within 0.1 range of median ({median_val:.4f}): {narrow_pct:.2f}%")

asyncio.run(test())
