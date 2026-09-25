import sys
import numpy as np
import json

sys.path.append('c:/Ocean-Predictor(first)/backend')
from main import preprocess_date, run_inference, model, device, TARGET_STATS, MASK_3D
import torch
import scipy.ndimage

def test():
    date = "2026-09-10"
    prep = preprocess_date(date, data_dir="c:/Ocean-Predictor(first)/backend")
    input_data = prep["input_tensor"].squeeze(0).cpu().numpy()
    
    norm_inputs = []
    for ch in range(7):
        ch_data = input_data[ch]
        mask = (ch_data != 0)
        if mask.any():
            mean = ch_data[mask].mean()
            std = ch_data[mask].std() + 1e-6
            indices = scipy.ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            infilled = ch_data[tuple(indices)]
            norm_inputs.append((infilled - mean) / std)
        else:
            norm_inputs.append(ch_data)
    
    norm_data = np.stack(norm_inputs, axis=0)
    prediction = run_inference(norm_data, device, model)
    pred_numpy = prediction.squeeze(0).cpu().numpy()
    
    means = np.array([s.get("mean", 0.0) for s in TARGET_STATS]).reshape(15, 1, 1)
    stds = np.array([s.get("std", 1.0) for s in TARGET_STATS]).reshape(15, 1, 1)
    pred_numpy = (pred_numpy * stds) + means
    
    surface_land = (input_data[0] == 0.0)
    for j in range(15):
        if MASK_3D is not None:
            pred_numpy[j, ~MASK_3D[j]] = np.nan
        else:
            pred_numpy[j, surface_land] = np.nan
            
    layer = pred_numpy[7].tolist() # 100m
    
    # Replace nan with None
    for r in range(len(layer)):
        for c in range(len(layer[r])):
            if np.isnan(layer[r][c]):
                layer[r][c] = None

    with open('layer100.json', 'w') as f:
        json.dump(layer, f)
    
    print("Saved layer100.json")

if __name__ == "__main__":
    test()
