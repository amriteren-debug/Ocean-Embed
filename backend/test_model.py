import os
import sys
import numpy as np
import torch
import scipy.ndimage

sys.path.append('c:/Ocean-Predictor(first)/backend')
from main import model, device, run_inference, preprocess_date, TARGET_STATS, MASK_3D, MODEL_DEPTHS

def compute_thermocline(pred_numpy):
    # compute mean profile
    mean_profile = []
    for d in range(15):
        slice_data = pred_numpy[d]
        valid_mask = ~np.isnan(slice_data)
        if valid_mask.any():
            mean_profile.append(slice_data[valid_mask].mean())
        else:
            mean_profile.append(None)
    
    # 0-200m is up to index 10 (200m). MODEL_DEPTHS[10] = 200
    fisheries_entries = [(d, i) for i, d in enumerate(MODEL_DEPTHS) if d <= 200]
    
    thermocline_depth = None
    max_grad = 0
    
    fisheries_gradients = []
    for k in range(len(fisheries_entries) - 1):
        d1, i1 = fisheries_entries[k]
        d2, i2 = fisheries_entries[k + 1]
        t1 = mean_profile[i1]
        t2 = mean_profile[i2]
        if t1 is None or t2 is None or (d2 - d1) == 0:
            fisheries_gradients.append(None)
        else:
            fisheries_gradients.append((t1 - t2) / (d2 - d1))
            
    max_fish_idx = -1
    for k in range(len(fisheries_gradients)):
        if fisheries_gradients[k] is not None and fisheries_gradients[k] > max_grad:
            max_grad = fisheries_gradients[k]
            max_fish_idx = k
            
    if max_fish_idx != -1:
        d1 = fisheries_entries[max_fish_idx][0]
        d2 = fisheries_entries[max_fish_idx + 1][0]
        dz = d2 - d1
        
        g0 = fisheries_gradients[max_fish_idx - 1] if (max_fish_idx > 0 and fisheries_gradients[max_fish_idx - 1] is not None) else max_grad
        g1 = max_grad
        g2 = fisheries_gradients[max_fish_idx + 1] if (max_fish_idx < len(fisheries_gradients) - 1 and fisheries_gradients[max_fish_idx + 1] is not None) else max_grad
        
        offset = 0
        denom = 2 * (g0 - 2 * g1 + g2)
        if denom != 0:
            offset = (g0 - g2) / denom
        offset = max(-0.5, min(0.5, offset))
        
        thermocline_depth = d1 + (dz / 2) + offset * dz
        
    return thermocline_depth, max_grad, mean_profile, g0, g1, g2, offset

def run_test():
    dates = ["2026-09-10", "2026-04-15", "2026-01-14"]
    
    for i, date in enumerate(dates):
        prep = preprocess_date(date, data_dir="c:/Ocean-Predictor(first)/backend")
        input_data = prep["input_tensor"].squeeze(0).cpu().numpy()
        
        # apply normalization as in main.py
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
        
        if MASK_3D is not None:
            surface_land = ~MASK_3D[0]
        else:
            surface_land = (input_data[0] == 0.0)
            
        prediction = run_inference(norm_data, device, model)
        
        pred_numpy = prediction.squeeze(0).cpu().numpy()
        
        # Denormalize
        if TARGET_STATS is not None:
            means = np.array([s.get("mean", 0.0) for s in TARGET_STATS]).reshape(15, 1, 1)
            stds = np.array([s.get("std", 1.0) for s in TARGET_STATS]).reshape(15, 1, 1)
            pred_numpy = (pred_numpy * stds) + means
            
        # Mask
        for j in range(15):
            if MASK_3D is not None:
                pred_numpy[j, ~MASK_3D[j]] = np.nan
            else:
                pred_numpy[j, surface_land] = np.nan
                
        tc_depth, max_grad, mp, g0, g1, g2, offset = compute_thermocline(pred_numpy)
        print(f"Date: {date}")
        print(f"  Gradients/100m -> g0(shallower): {g0*100:.3f}, g1(peak): {g1*100:.3f}, g2(deeper): {g2*100:.3f}")
        print(f"  Offset: {offset:.4f} | Interpolated TC Depth: {tc_depth:.3f}m | Max Grad/100m: {max_grad * 100:.3f}")

if __name__ == "__main__":
    run_test()
