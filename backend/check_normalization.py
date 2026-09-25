import torch
import json
import numpy as np
import xarray as xr
from preprocessing import preprocess_date
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=7,
    classes=15,
)
state_dict = torch.load("OceanEmbed_Final_Weights.pth", map_location=device)
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
model.to(device)
model.eval()

# Load stats
with open("depth_stats.json", "r") as f:
    stats = json.load(f)
stat_means = np.array(stats["mean"])
stat_stds = np.array(stats["std"])
stat_depths = np.array(stats["depths"])

# Get sample input
prep = preprocess_date("2026-09-06", data_dir=".")
input_tensor = prep["input_tensor"].to(device)
raw_sst_channel = input_tensor[0, 0, :, :].cpu().numpy()
land_mask = (raw_sst_channel == 0.0)

# Dummy wind
dummy_wind = torch.zeros((1, 2, input_tensor.shape[2], input_tensor.shape[3]), dtype=torch.float32, device=device)
# Normalization of input as in main.py
real_channels = input_tensor[:, :5, :, :]
normalized_real = torch.zeros_like(real_channels)
mask = (real_channels != 0.0)
if mask.any():
    mean = real_channels[mask].mean()
    std = real_channels[mask].std() + 1e-6
    normalized_real[mask] = (real_channels[mask] - mean) / std

model_input = torch.cat([normalized_real, dummy_wind], dim=1)

import torch.nn.functional as F
h, w = model_input.shape[2], model_input.shape[3]
pad_h = (32 - (h % 32)) % 32
pad_w = (32 - (w % 32)) % 32
if pad_h > 0 or pad_w > 0:
    model_input = F.pad(model_input, (0, pad_w, 0, pad_h), mode='replicate')

with torch.no_grad():
    prediction = model(model_input)

if pad_h > 0 or pad_w > 0:
    prediction = prediction[:, :, :h, :w]

raw_orig = prediction.squeeze(0).cpu().numpy()

print(f"\n  OceanEmbed_Final_Weights.pth RAW outputs:")
print(f"  {'Ch':>3} | {'Raw Mean':>10} | {'Raw Std':>10} | {'Stats Mean':>10} | {'Stats Std':>10} | {'Diagnosis':>20}")
print(f"  " + "-" * 80)

normalized_channels = []
unnormalized_channels = []

for ch in range(15):
    layer = raw_orig[ch].copy()
    layer[land_mask] = np.nan
    valid = layer[np.isfinite(layer)]
    raw_mean = np.mean(valid)
    raw_std = np.std(valid)
    
    # We compare with stat_means[ch]. Note that stat_means has 16 items!
    # For now, let's just compare index for index
    sm = stat_means[ch]
    ss = stat_stds[ch]
    
    near_zero = abs(raw_mean) < 2.0 and abs(raw_std - 1.0) < 1.0
    near_stat = abs(raw_mean - sm) < (2 * ss)
    
    if near_zero and not near_stat:
        diagnosis = "NORMALIZED"
        normalized_channels.append(ch)
    elif near_stat:
        diagnosis = "RAW TEMPERATURE"
        unnormalized_channels.append(ch)
    else:
        diagnosis = "AMBIGUOUS"
    
    print(f"  {ch:3d} | {raw_mean:10.3f} | {raw_std:10.3f} | {sm:10.3f} | {ss:10.3f} | {diagnosis:>20}")

