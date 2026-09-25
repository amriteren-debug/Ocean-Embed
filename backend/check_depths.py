import asyncio
from main import predict_subsurface, PredictRequest

async def test():
    req = PredictRequest(date="2026-08-31")
    res = await predict_subsurface(req)
    
    tensor = res["prediction_data"][0]
    import numpy as np
    layer14 = np.array(tensor[14], dtype=float)
    
    valid_mask = ~np.isnan(layer14)
    if valid_mask.any():
        print(f"Depth 14 min: {layer14[valid_mask].min():.2f}, max: {layer14[valid_mask].max():.2f}")
        
    layer0 = np.array(tensor[0], dtype=float)
    valid_mask0 = ~np.isnan(layer0)
    if valid_mask0.any():
        print(f"Depth 0 min: {layer0[valid_mask0].min():.2f}, max: {layer0[valid_mask0].max():.2f}")
        
asyncio.run(test())
