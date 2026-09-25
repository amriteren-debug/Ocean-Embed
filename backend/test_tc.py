import sys, json, numpy as np
sys.path.append('c:/Ocean-Predictor(first)/backend')
from main import preprocess_date, run_inference, model, device, TARGET_STATS, MASK_3D
import torch, scipy.ndimage

def test():
    date = "2026-09-10"
    prep = preprocess_date(date, data_dir="c:/Ocean-Predictor(first)/backend")
    input_data = prep["input_tensor"].squeeze(0).cpu().numpy()
    norm_inputs = []
    for ch in range(7):
        ch_data = input_data[ch]
        mask = (ch_data != 0)
        if mask.any():
            mean, std = ch_data[mask].mean(), ch_data[mask].std() + 1e-6
            indices = scipy.ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
            infilled = ch_data[tuple(indices)]
            norm_inputs.append((infilled - mean) / std)
        else: norm_inputs.append(ch_data)
    norm_data = np.stack(norm_inputs, axis=0)
    prediction = run_inference(norm_data, device, model).squeeze(0).cpu().numpy()
    means = np.array([s.get("mean", 0.0) for s in TARGET_STATS]).reshape(15, 1, 1)
    stds = np.array([s.get("std", 1.0) for s in TARGET_STATS]).reshape(15, 1, 1)
    prediction = (prediction * stds) + means
    
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    means_profile = []
    for j in range(15):
        if MASK_3D is not None: prediction[j, ~MASK_3D[j]] = np.nan
        else: prediction[j, input_data[0] == 0.0] = np.nan
        valid = prediction[j][~np.isnan(prediction[j])]
        means_profile.append(valid.mean() if len(valid) > 0 else None)
        
    fish_entries = []
    for i in range(11): # up to 200m
        fish_entries.append({'d': depths[i], 'i': i})
        
    grads = []
    for k in range(len(fish_entries)-1):
        d1, i1 = fish_entries[k]['d'], fish_entries[k]['i']
        d2, i2 = fish_entries[k+1]['d'], fish_entries[k+1]['i']
        t1, t2 = means_profile[i1], means_profile[i2]
        grads.append((t1 - t2) / (d2 - d1))
        
    max_idx = np.argmax(grads)
    print(f"Max grad idx: {max_idx}")
    print(f"Thermocline interval: {fish_entries[max_idx]['d']} to {fish_entries[max_idx+1]['d']}")

if __name__ == "__main__":
    test()
