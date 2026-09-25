"""
OceanEmbed Proof-of-Concept Demonstration (R12)
=================================================
Standalone script that runs a prediction for a specific date and produces
detailed output for the Bay of Bengal and Arabian Sea regions.

Usage:
    python poc_demo.py --date 2026-08-20

This script:
1. Runs the preprocessing pipeline (R1-R4)
2. Loads the model and runs inference (R6)
3. Interpolates to 15 standard depths (R7)
4. Extracts Bay of Bengal and Arabian Sea subregions
5. Prints per-depth temperature statistics for each region
6. Computes and displays validation metrics if ARGO data available (R8)
"""
import os
import sys
import json
import time
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess_date, TARGET_LATS, TARGET_LONS
from argo_loader import load_argo_for_date, STANDARD_DEPTHS


def run_poc_demo(date_str, data_dir="."):
    """Run the full PoC demonstration pipeline."""

    REGIONS = {
        "Bay of Bengal": {"lat_min": 5.0, "lat_max": 23.0,
                          "lon_min": 80.0, "lon_max": 95.0},
        "Arabian Sea":   {"lat_min": 5.0, "lat_max": 25.0,
                          "lon_min": 50.0, "lon_max": 75.0},
    }

    print("=" * 70)
    print("  OceanEmbed - Proof of Concept Demonstration")
    print("  Satellite Embedding-Based Subsurface Temperature Reconstruction")
    print("=" * 70)
    print(f"\n  Date: {date_str}")
    print(f"  Domain: North Indian Ocean (5N-30N, 45E-105E)")
    print(f"  Output grid: 0.25 deg x 0.25 deg, daily")
    print(f"  Standard depths: {STANDARD_DEPTHS}")

    # ── Step 1: Preprocessing ──
    print("\n--- Step 1: Preprocessing Pipeline ---")
    t0 = time.time()
    try:
        prep = preprocess_date(date_str, data_dir=data_dir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print(f"  Ensure you have downloaded data for {date_str}.")
        print(f"  Run the backend server and make a /predict request first.")
        return False

    input_tensor = prep["input_tensor"]
    land_mask = prep["land_mask"]
    print(f"  Input tensor shape: {input_tensor.shape}")
    print(f"  Grid: {len(prep['lat'])} lat x {len(prep['lon'])} lon (0.25 deg)")
    print(f"  Land fraction: {land_mask.mean():.1%}")
    print(f"  Channels: {prep['channel_order']}")
    print(f"  Time: {time.time()-t0:.2f}s")

    # ── Step 2: Load model ──
    print("\n--- Step 2: Model Loading ---")
    from main import ResNet50_UNet_Pro
    device = torch.device("cpu")

    model_path = os.path.join(data_dir, "BEST_ocean_model_1YR_7CH.pth")
    if not os.path.exists(model_path):
        print(f"  ERROR: Model checkpoint not found at {model_path}")
        return False

    model = ResNet50_UNet_Pro(in_channels=7, out_channels=15).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Model: ResNet50_UNet_Pro(in=7, out=15)")
    print(f"  Checkpoint: {model_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Step 3: Load denormalization stats ──
    stats_path = os.path.join(data_dir, "depth_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    model_depths = stats["depths"]
    means_t = torch.tensor(stats["mean"][:15], dtype=torch.float32).view(1, 15, 1, 1)
    stds_t = torch.tensor(stats["std"][:15], dtype=torch.float32).view(1, 15, 1, 1)

    # ── Step 4: Run inference ──
    print("\n--- Step 3: Model Inference ---")
    t0 = time.time()
    with torch.no_grad():
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        inp = input_tensor.to(device)
        if pad_h > 0 or pad_w > 0:
            inp = F.pad(inp, (0, pad_w, 0, pad_h), mode='replicate')

        prediction, embedding = model.forward_with_embedding(inp)

        if pad_h > 0 or pad_w > 0:
            prediction = prediction[:, :, :h, :w]

        # Denormalize channels 0 and 14
        for c in [0, 14]:
            prediction[:, c:c+1] = prediction[:, c:c+1] * stds_t[0, c:c+1] + means_t[0, c:c+1]

        # Derive 1000m by extrapolation
        d13, d14 = model_depths[13], model_depths[14]
        slope = (prediction[:, 14:15] - prediction[:, 13:14]) / (d14 - d13)
        pred_1000 = prediction[:, 14:15] + slope * (1000.0 - d14)
        prediction = torch.cat([prediction, pred_1000], dim=1)

    pred_np = prediction.squeeze(0).cpu().numpy()  # (16, lat, lon)
    emb_np = embedding.squeeze(0).cpu().numpy()
    print(f"  Raw prediction shape: {pred_np.shape}")
    print(f"  Embedding shape: {embedding.shape} (satellite embedding, R5)")
    print(f"  Embedding summary: mean={emb_np.mean():.4f}, std={emb_np.std():.4f}")
    print(f"  Time: {time.time()-t0:.2f}s")

    # ── Step 5: Interpolate to standard depths ──
    print("\n--- Step 4: Depth Interpolation (R7) ---")
    interpolator = interp1d(model_depths, pred_np, axis=0, fill_value="extrapolate")
    pred_standard = interpolator(STANDARD_DEPTHS)  # (15, lat, lon)
    pred_standard[:, land_mask] = np.nan
    print(f"  Output shape: {pred_standard.shape}")
    print(f"  Standard depths: {STANDARD_DEPTHS}")

    # ── Step 6: Regional analysis ──
    print("\n--- Step 5: Regional Analysis (R12) ---")
    lat_arr = prep["lat"]
    lon_arr = prep["lon"]

    for region_name, bounds in REGIONS.items():
        print(f"\n  === {region_name} ===")
        print(f"  Bounds: {bounds['lat_min']}N-{bounds['lat_max']}N, "
              f"{bounds['lon_min']}E-{bounds['lon_max']}E")

        lat_idx = np.where((lat_arr >= bounds["lat_min"]) & (lat_arr <= bounds["lat_max"]))[0]
        lon_idx = np.where((lon_arr >= bounds["lon_min"]) & (lon_arr <= bounds["lon_max"]))[0]

        if len(lat_idx) == 0 or len(lon_idx) == 0:
            print("  No grid points in this region!")
            continue

        region_data = pred_standard[:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        print(f"  Grid points: {region_data.shape[1]} x {region_data.shape[2]}")

        print(f"\n  {'Depth(m)':>10s} {'Mean(C)':>10s} {'Std(C)':>10s} "
              f"{'Min(C)':>10s} {'Max(C)':>10s} {'N_valid':>10s}")
        print(f"  {'-'*60}")

        for d, depth in enumerate(STANDARD_DEPTHS):
            layer = region_data[d]
            valid = layer[np.isfinite(layer)]
            if len(valid) > 0:
                print(f"  {depth:10d} {valid.mean():10.2f} {valid.std():10.2f} "
                      f"{valid.min():10.2f} {valid.max():10.2f} {len(valid):10d}")
            else:
                print(f"  {depth:10d} {'--':>10s} {'--':>10s} "
                      f"{'--':>10s} {'--':>10s} {'0':>10s}")

    # ── Step 7: Validation against ARGO ──
    print("\n--- Step 6: Validation Against ARGO (R8/R10) ---")
    try:
        argo_result = load_argo_for_date(date_str)
        argo_data = argo_result["data"]
        print(f"  ARGO source: {argo_result['source']}")
        print(f"  ARGO coverage: {argo_result['coverage_pct']}%")

        # Compute skill metrics
        from main import compute_skill_metrics
        # Use only ocean points for validation
        metrics = compute_skill_metrics(pred_standard, argo_data, STANDARD_DEPTHS)

        print(f"\n  {'Depth(m)':>10s} {'Corr':>10s} {'RMSE(C)':>10s} {'Bias(C)':>10s}")
        print(f"  {'-'*42}")
        for i, depth in enumerate(STANDARD_DEPTHS):
            corr = metrics["correlation"][i]
            rmse = metrics["rmse"][i]
            bias = metrics["bias"][i]
            corr_str = f"{corr:.4f}" if corr is not None else "--"
            rmse_str = f"{rmse:.4f}" if rmse is not None else "--"
            bias_str = f"{bias:.4f}" if bias is not None else "--"
            print(f"  {depth:10d} {corr_str:>10s} {rmse_str:>10s} {bias_str:>10s}")

    except Exception as e:
        print(f"  ARGO validation failed: {e}")

    print("\n" + "=" * 70)
    print("  PoC Demonstration Complete")
    print("=" * 70)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OceanEmbed PoC Demonstration")
    parser.add_argument("--date", default="2026-08-20", help="Date (YYYY-MM-DD)")
    parser.add_argument("--data-dir", default=".", help="Data directory")
    args = parser.parse_args()

    success = run_poc_demo(args.date, data_dir=args.data_dir)
    sys.exit(0 if success else 1)
