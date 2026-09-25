import asyncio
from main import predict_subsurface, PredictRequest

async def test():
    req = PredictRequest(date="2026-08-31")
    res = await predict_subsurface(req)
    
    tensor = res["prediction_data"][0]
    import numpy as np
    
    for i in range(15):
        layer = np.array(tensor[i], dtype=float)
        valid_mask = ~np.isnan(layer)
        if valid_mask.any():
            print(f"Depth index {i} min: {layer[valid_mask].min():.2f}, max: {layer[valid_mask].max():.2f}")
        
asyncio.run(test())
