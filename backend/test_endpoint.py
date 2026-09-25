import asyncio
from main import predict_subsurface, PredictRequest

async def test():
    req = PredictRequest(date="2026-08-31")
    res = await predict_subsurface(req)
    
    tensor = res["prediction_data"][0]
    import numpy as np
    layer = np.array(tensor[0], dtype=float)
    
    null_count = np.isnan(layer).sum()
    valid_mask = ~np.isnan(layer)
    if valid_mask.any():
        valid_min = layer[valid_mask].min()
        valid_max = layer[valid_mask].max()
    else:
        valid_min = valid_max = float('nan')
        
    print(f"Depth 0 summary: shape={layer.shape}, nulls={null_count}, min={valid_min:.2f}, max={valid_max:.2f}")

asyncio.run(test())
