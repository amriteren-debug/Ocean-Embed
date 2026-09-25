import json
import time
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import scipy.ndimage
import math
import segmentation_models_pytorch as smp
import copernicusmarine
import xarray as xr
from preprocessing import preprocess_date, regrid_output_to_025, TARGET_LATS, TARGET_LONS, CHANNEL_ORDER

# ==========================================
# 1. INITIALIZE FASTAPI APP
# ==========================================
app = FastAPI(title="Ocean AI Backend - Live Fetch")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    date: str  # Format: "YYYY-MM-DD"
    allow_download: bool = True

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class DecoderBlockPro(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        )
        self.conv = nn.Sequential(
            nn.Conv2d((in_channels // 2) + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet50_UNet_Pro(nn.Module):
    # in_channels=7: SST, SSS, SSH, U_cur, V_cur, U_wind, V_wind
    # out_channels=15: 15 native GLORYS depth levels (0.5m – 902m)
    def __init__(self, in_channels=7, out_channels=15):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Seed all in_channels slots cyclically from the 3 pretrained RGB channels.
            # In practice load_state_dict() overwrites conv1.weight with the trained checkpoint.
            for c in range(in_channels):
                self.conv1.weight[:, c:c + 1, :, :] = resnet.conv1.weight[:, c % 3:c % 3 + 1, :, :]
        self.bn1 = resnet.bn1; self.relu = resnet.relu; self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1; self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3; self.layer4 = resnet.layer4
        self.dec4 = DecoderBlockPro(2048, 1024, 512); self.dec3 = DecoderBlockPro(512, 512, 256)
        self.dec2 = DecoderBlockPro(256, 256, 128); self.dec1 = DecoderBlockPro(128, 64, 64)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        original_size = (x.shape[2], x.shape[3])
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        skip1 = self.layer1(x1); skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2); bottleneck = self.layer4(skip3)
        d4 = self.dec4(bottleneck, skip3); d3 = self.dec3(d4, skip2)
        d2 = self.dec2(d3, skip1); d1 = self.dec1(d2, x0)
        out = self.final_up(d1)
        if out.shape[2:] != original_size: out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=True)
        return self.final_conv(out)

    def forward_embedding(self, x):
        """Extract the satellite embedding (R5) -- the encoder bottleneck.

        Returns the output of ResNet50's layer4 (the deepest encoder block),
        which is a compact latent representation of the surface state.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 7, H, W)

        Returns
        -------
        torch.Tensor, shape (B, 2048, H/32, W/32) -- the satellite embedding
        """
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        skip1 = self.layer1(x1)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        bottleneck = self.layer4(skip3)
        return bottleneck

    def forward_with_embedding(self, x):
        """Run full forward pass and also return the satellite embedding.

        Returns
        -------
        tuple of (prediction, embedding)
            prediction : torch.Tensor, shape (B, out_channels, H, W)
            embedding  : torch.Tensor, shape (B, 2048, H/32, W/32)
        """
        original_size = (x.shape[2], x.shape[3])
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        skip1 = self.layer1(x1); skip2 = self.layer2(skip1)
# 3. LOAD MODEL 
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting Backend... Using device: {device}")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model_1YR_7CH_BEST_OVERALL.pth")

try:
    print(f"Loading model on {device}...")
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=7,
        classes=15,
    )
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    clean_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval() 
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")


# All 15 depth levels served to the frontend
MODEL_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# Load the true 3D Bathymetry Mask
MASK_3D_PATH = os.path.join(os.path.dirname(__file__), "bathymetry_mask.npy")
try:
    MASK_3D = np.load(MASK_3D_PATH)
    print(f"Loaded 3D bathymetry mask: {MASK_3D.shape}")
except Exception as e:
    print(f"Warning: Could not load 3D bathymetry mask. Error: {e}")
    MASK_3D = None

# ==========================================
# 3.5 LOAD DENORMALIZATION STATS
# ==========================================
TARGET_STATS_PATH = os.path.join(os.path.dirname(__file__), "models", "target_depth_stats.json")
try:
    with open(TARGET_STATS_PATH, 'r') as f:
        TARGET_STATS = json.load(f)
    print(f"Loaded target normalization stats from {TARGET_STATS_PATH}")
except Exception as e:
    print(f"Warning: Could not load target normalization stats. Model outputs will NOT be denormalized. Error: {e}")
    TARGET_STATS = None

# ==========================================
# 3.6 PREDICTION CACHE
# ==========================================
# In-memory cache keyed by date string.
# Each entry is a dict with:
#   "prediction_data": the serialized prediction (nested list),
#   "input_fields":    dict of 7 serialized 2D arrays (nested lists)
# This avoids repeated Copernicus downloads for the same date.
_prediction_cache = {}
_MAX_CACHE_ENTRIES = 5  # Keep up to 5 dates cached


def _cache_key(date_str):
    """Generate a cache key from the date string."""
    return date_str


def _evict_if_needed():
    """Evict oldest entry if cache exceeds max size."""
    while len(_prediction_cache) > _MAX_CACHE_ENTRIES:
        oldest_key = next(iter(_prediction_cache))
        del _prediction_cache[oldest_key]


# ==========================================
# 4. API ROUTES & LIVE PROCESSING
# ==========================================
@app.get("/")
def read_root():
    return {"message": "Ocean Predictor Live Backend is UP!"}

@app.get("/depths")
def get_depths():
    """Return the model's native depth levels so the frontend can stay in sync."""
    return {"depths": MODEL_DEPTHS}

@app.post("/fetch_raw_sst")
async def fetch_raw_sst(request: PredictRequest):
    """
    Downloads only the SST file from Copernicus (thetao) and regrids it.
    Returns the raw 2D array (with NaNs preserved) to show Step A in the frontend.
    """
    from preprocessing import _load_and_regrid_ocean_var
    target_date = request.date
    file_thetao = f"temp_thetao_{target_date}.nc"
    
    # Download thetao if missing
    if not os.path.exists(file_thetao) or os.path.getsize(file_thetao) < 1000:
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
            minimum_longitude=45.0, maximum_longitude=100.0,
            minimum_latitude=-10.0, maximum_latitude=30.0,
            minimum_depth=0.49402499198913574,
            maximum_depth=0.49402499198913574,
            coordinates_selection_method="nearest",
            output_filename=file_thetao,
            force_download=True
        )
        
    ds = xr.open_dataset(file_thetao)
    raw_sst = _load_and_regrid_ocean_var(ds, "thetao")
    ds.close()
    
    # We replace NaNs with None so JSON serialization works
    raw_list = np.where(np.isnan(raw_sst), None, raw_sst).tolist()
    
    return {
        "status": "success",
        "date": target_date,
        "raw_sst": raw_list
    }

def run_inference(input_data, device, model):
    """Run model inference on raw input data (no in-painting).
    
    The model was trained with zero-padded landmasses, so we must preserve
    that convention to keep activations mathematically stable.
    
    Parameters
    ----------
    input_data : np.ndarray, shape (7, H, W)
        Raw 7-channel input. Land/cloud pixels are 0.0.
    device : torch.device
    model : nn.Module
    
    Returns
    -------
    prediction : torch.Tensor, shape (1, 15, H, W)
    """
    # 1. NO INPUT IN-PAINTING. Pass the raw input directly to maintain scale!
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. Pad to multiple of 32 for UNet compatibility
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    # 3. Run Inference
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # 4. Remove padding
    if pad_h > 0 or pad_w > 0:
        prediction = prediction[:, :, :h, :w]
    
    return prediction

@app.post("/predict")
async def predict_subsurface(request: PredictRequest):
    target_date = request.date
    timings = {}
    t_start = time.time()
    
    # The 15 standard depth levels required by the Problem Statement (R7)
    STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    
    # ── Cache check ──────────────────────────────────────
    cache_key = _cache_key(target_date)
    if cache_key in _prediction_cache:
        cached = _prediction_cache[cache_key]
        timings["cache_hit"] = True
        timings["total_s"] = round(time.time() - t_start, 3)
        print(f"Cache HIT for {target_date} - returning in {timings['total_s']}s")
        
        return {
            "status": "success",
            "date": target_date,
            "depths": STANDARD_DEPTHS,
            "lat": cached["lat"],
            "lon": cached["lon"],
            "prediction_data": [cached["prediction_data"]], # Wrap in list to make it [1, 15, lat, lon] for frontend
            "input_fields": cached["input_fields"],
            "timings": timings,
        }
    
    timings["cache_hit"] = False
    
    # Temporary filenames for Copernicus downloads
    file_thetao = f"temp_thetao_{target_date}.nc"
    file_so = f"temp_so_{target_date}.nc"
    file_cur = f"temp_cur_{target_date}.nc"
    file_zos = f"temp_zos_{target_date}.nc"
    file_wind = f"temp_wind_{target_date}.nc"
    
    t_fetch_start = time.time()
    
    try:
        # Check if files already exist in cache (skip download if they do)
        cache_hit_disk = True
        for f in [file_thetao, file_so, file_cur, file_zos, file_wind]:
            if not os.path.exists(f) or os.path.getsize(f) < 1000:
                cache_hit_disk = False
                break
                
        if cache_hit_disk:
            print(f"Using cached Copernicus NetCDF files for {target_date}...")
        else:
            if getattr(request, 'allow_download', True) is False:
                return {"status": "unavailable", "reason": "Data not cached and download not allowed."}
                
            print(f"Downloading live satellite data for {target_date} (5 streams in parallel)...")

            def fetch_thetao():
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
                    variables=["thetao"],
                    start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                    minimum_longitude=45.0, maximum_longitude=100.0,
                    minimum_latitude=-10.0, maximum_latitude=30.0,
                    minimum_depth=0.49402499198913574,
                    maximum_depth=0.49402499198913574,
                    coordinates_selection_method="nearest",
                    output_filename=file_thetao,
                    force_download=True
                )

            def fetch_so():
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
                    variables=["so"],
                    start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                    minimum_longitude=45.0, maximum_longitude=100.0,
                    minimum_latitude=-10.0, maximum_latitude=30.0,
                    minimum_depth=0.49402499198913574,
                    maximum_depth=0.49402499198913574,
                    coordinates_selection_method="nearest",
                    output_filename=file_so,
                    force_download=True
                )

            def fetch_cur():
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
                    variables=["uo", "vo"],
                    start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                    minimum_longitude=45.0, maximum_longitude=100.0,
                    minimum_latitude=-10.0, maximum_latitude=30.0,
                    minimum_depth=0.49402499198913574,
                    maximum_depth=0.49402499198913574,
                    coordinates_selection_method="nearest",
                    output_filename=file_cur,
                    force_download=True
                )

            def fetch_zos():
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                    variables=["zos"],
                    start_datetime=f"{target_date}T00:00:00", end_datetime=f"{target_date}T23:59:59",
                    minimum_longitude=45.0, maximum_longitude=100.0,
                    minimum_latitude=-10.0, maximum_latitude=30.0,
                    output_filename=file_zos,
                    force_download=True
                )

            def fetch_wind():
                """Try requested date, fall back to previous day if NRT lag."""
                for attempt_date in [target_date, (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')]:
                    try:
                        copernicusmarine.subset(
                            dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
                            variables=["eastward_wind", "northward_wind"],
                            start_datetime=f"{attempt_date}T00:00:00", end_datetime=f"{attempt_date}T23:59:59",
                            minimum_longitude=45.0, maximum_longitude=100.0,
                            minimum_latitude=-10.0, maximum_latitude=30.0,
                            coordinates_selection_method="nearest",
                            output_filename=file_wind,
                            force_download=True
                        )
                        if attempt_date != target_date:
                            print(f"  [WARN] Wind data used fallback date {attempt_date}")
                        return
                    except Exception:
                        continue
                # If both dates fail, write a sentinel so preprocessing knows
                with open(file_wind, 'w') as f:
                    f.write('NO_WIND_DATA')
                print("  [WARN] Wind data unavailable -- using zero-fill")

            # Run all 5 downloads concurrently
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(fetch_thetao): "thetao",
                    executor.submit(fetch_so): "salinity",
                    executor.submit(fetch_cur): "currents",
                    executor.submit(fetch_zos): "ssh",
                    executor.submit(fetch_wind): "wind",
                }
                for future in as_completed(futures):
                    name = futures[future]
                    future.result()  # Raise if any download failed
                    print(f"  [OK] {name} downloaded")
        
        timings["fetch_s"] = round(time.time() - t_fetch_start, 3)

        # ── Preprocessing: harmonize to 0.25 deg grid (R1, R2, R3, R4) ──
        t_preprocess = time.time()
        
        prep = preprocess_date(target_date, data_dir=".")
        
        timings["preprocess_s"] = round(time.time() - t_preprocess, 3)
        
        # ── Model inference (MATCH TRAINING NORMALIZATION EXACTLY) ─────────────
        t_inference = time.time()
        print("Running AI Prediction...")
        input_data = prep["input_tensor"].squeeze(0).cpu().numpy()  # Shape: (7, H, W)
        
        import scipy.ndimage
        norm_inputs = []
        for i in range(7):
            ch_data = input_data[i]
            mask = (ch_data != 0)
            if mask.any():
                mean = ch_data[mask].mean()
                std = ch_data[mask].std() + 1e-6
                # Nearest neighbor coastal infilling (extrapolation)
                indices = scipy.ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
                infilled = ch_data[tuple(indices)]
                # Standardize strictly exactly like train_7ch.py
                norm_inputs.append((infilled - mean) / std)
            else:
                norm_inputs.append(ch_data)
                
        # Re-stack into normalized input tensor
        input_data = np.stack(norm_inputs, axis=0)
        
        # Define surface land based on the original un-infilled data for post-processing later
        if MASK_3D is not None:
            surface_land = ~MASK_3D[0]
        else:
            surface_land = (prep["input_tensor"].squeeze(0).cpu().numpy()[0] == 0.0)
            
        prediction = run_inference(input_data, device, model)
        timings["inference_s"] = round(time.time() - t_inference, 3)
        
        # ── Postprocessing + Coastal Healer ──────────────────────────────
        t_post = time.time()
        pred_numpy = prediction.squeeze(0).cpu().numpy()  # Shape: (15, lat, lon)
        
        # 1. DENORMALIZE PREDICTIONS
        if TARGET_STATS is not None:
            # Extract means and stds sequentially for all 15 layers
            means = np.array([s.get("mean", 0.0) for s in TARGET_STATS]).reshape(15, 1, 1)
            stds = np.array([s.get("std", 1.0) for s in TARGET_STATS]).reshape(15, 1, 1)
            
            # Strict broadcasting: (15, lat, lon) * (15, 1, 1) + (15, 1, 1)
            pred_numpy = (pred_numpy * stds) + means
                
        # 2. APPLY TRUE LAND MASK (Target masking to avoid land hallucinations)
        for i in range(15):
            if MASK_3D is not None:
                pred_numpy[i, ~MASK_3D[i]] = np.nan
            else:
                pred_numpy[i, surface_land] = np.nan
        
        pred_rounded = np.round(pred_numpy, 2)
        pred_rounded = np.where(np.isnan(pred_rounded), None, pred_rounded)
        
        # ── Serialize the 7 raw input fields ──
        land_mask = surface_land # Sync variable for below
        _input_field_names = ["sst", "sss", "ssh", "u_cur", "v_cur", "u_wind", "v_wind"]
        input_fields = {}
        for name in _input_field_names:
            arr = prep["raw_fields"][name].copy().astype(float)
            arr[land_mask] = np.nan
            arr_rounded = np.round(arr, 2)
            arr_safe = np.where(np.isnan(arr_rounded), None, arr_rounded)
            input_fields[name] = arr_safe.tolist()
        
        # Coordinate arrays for frontend
        lat_list = [round(float(v), 2) for v in prep["lat"]]
        lon_list = [round(float(v), 2) for v in prep["lon"]]
        
        timings["postprocess_s"] = round(time.time() - t_post, 3)
        # ── Cache the FULL 3D result ────────────────────────
        _prediction_cache[cache_key] = {
            "prediction_data": pred_rounded.tolist(), # Now caching the 3D array directly
            "input_fields": input_fields,
            "lat": lat_list,
            "lon": lon_list,
        }
        _evict_if_needed()
        
        timings["total_s"] = round(time.time() - t_start, 3)
        print(f"Prediction complete in {timings['total_s']}s "
              f"(fetch={timings['fetch_s']}s, preprocess={timings['preprocess_s']}s, "
              f"inference={timings['inference_s']}s, postprocess={timings['postprocess_s']}s)")
        
        return {
            "status": "success",
            "date": target_date,
            "depths": STANDARD_DEPTHS,
            "lat": lat_list,
            "lon": lon_list,
            "prediction_data": [pred_rounded.tolist()], # Shape [1, 15, lat, lon]
            "input_fields": input_fields,
            "timings": timings,
        }
        
    except HTTPException:
        raise  # Re-raise explicit HTTP exceptions (e.g. 401 Unauthorized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 5. INPUT TRANSPARENCY ENDPOINT
# ==========================================
@app.get("/inputs/{date}")
def get_inputs(date: str):
    """Return the 7 raw input fields for a previously-predicted date.
    Only works for dates already in the in-memory cache — does NOT
    trigger a new Copernicus download.
    """
    cache_key = _cache_key(date)
    if cache_key not in _prediction_cache:
        return {
            "status": "not_found",
            "reason": f"No cached prediction for date '{date}'. Run /predict first.",
        }
    cached = _prediction_cache[cache_key]
    return {
        "status": "success",
        "date": date,
        "fields": ["sst", "sss", "ssh", "u_cur", "v_cur", "u_wind", "v_wind"],
        "input_fields": cached["input_fields"],
    }


# ==========================================
# 6. ARGO VALIDATION / SKILL METRICS (R8, R10)
# ==========================================
from argo_loader import load_argo_for_date as _argo_load, STANDARD_DEPTHS as ARGO_DEPTHS


def compute_skill_metrics(prediction, argo_obs, depths):
    """Compute per-depth Pearson correlation, RMSE, and Bias.

    Parameters
    ----------
    prediction : np.ndarray
        Model prediction, shape (n_depths, lat, lon).
    argo_obs : np.ndarray
        ARGO observations, same shape. NaN where no observation.
    depths : list[float]
        Depth values corresponding to axis-0.

    Returns
    -------
    dict with keys "depths", "correlation", "rmse", "bias" -- each a
    list of len(depths) floats (or None where insufficient data).
    """
    n_depths = len(depths)
    correlations = []
    rmses = []
    biases = []

    for d in range(n_depths):
        pred_slice = prediction[d].ravel().astype(float)
        obs_slice = argo_obs[d].ravel().astype(float)

        # Keep only cells where BOTH pred and obs are valid (non-NaN)
        valid = np.isfinite(pred_slice) & np.isfinite(obs_slice)
        n_valid = int(np.sum(valid))

        if n_valid < 2:
            # Not enough points for meaningful statistics
            correlations.append(None)
            rmses.append(None)
            biases.append(None)
            continue

        p = pred_slice[valid]
        o = obs_slice[valid]
        diff = p - o

        # Bias = mean(pred - obs)
        bias = float(np.mean(diff))

        # RMSE = sqrt(mean((pred - obs)^2))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        # Pearson correlation -- guard against zero-variance
        p_std = np.std(p)
        o_std = np.std(o)
        if p_std < 1e-12 or o_std < 1e-12:
            corr = None
        else:
            corr_matrix = np.corrcoef(p, o)
            r = float(corr_matrix[0, 1])
            corr = None if (math.isnan(r) or math.isinf(r)) else round(r, 6)

        correlations.append(corr)
        rmses.append(round(rmse, 4))
        biases.append(round(bias, 4))

    return {
        "depths": depths,
        "correlation": correlations,
        "rmse": rmses,
        "bias": biases,
    }


@app.get("/skill/{date}")
def get_skill_metrics_endpoint(date: str):
    """Compute skill metrics (correlation, RMSE, bias) per depth level
    between the model's cached prediction and independent ARGO observations.
    """
    STANDARD_DEPTHS_LOCAL = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    # 1. Check that we have a cached prediction for this date
    cache_key = _cache_key(date)
    if cache_key not in _prediction_cache:
        return {
            "status": "unavailable",
            "reason": f"No cached prediction for date '{date}'. Run /predict first.",
        }

    # 2. Load ARGO observations via the real loader
    try:
        argo_result = _argo_load(date)
        argo_obs = argo_result["data"]
        argo_source = argo_result["source"]
        argo_coverage = argo_result["coverage_pct"]
    except Exception as e:
        return {
            "status": "unavailable",
            "reason": f"Failed to load ARGO data: {e}",
        }

    # 3. Recover the prediction as a numpy array from the cache
    try:
        cached = _prediction_cache[cache_key]
        pred_nested = cached["prediction_data"]  # list of 15 depth layers
        pred_array = np.array(pred_nested, dtype=float)  # NaN where None was
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to deserialize cached prediction: {e}",
        }

    # 4. Compute metrics at the 15 standard depths
    try:
        metrics = compute_skill_metrics(pred_array, argo_obs, STANDARD_DEPTHS_LOCAL)
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Metric computation failed: {e}",
        }

    return {
        "status": "success",
        "date": date,
        "argo_source": argo_source,
        "argo_coverage_pct": argo_coverage,
        **metrics,
    }


@app.get("/argo/{date}")
def get_argo_floats(date: str):
    """Return ARGO float profiles for a given date as discrete points.
    
    Used for visualization and map overlays.
    """
    try:
        argo_result = _argo_load(date)
        argo_data = argo_result["data"]
        lat_arr = argo_result["lat"]
        lon_arr = argo_result["lon"]
        argo_source = argo_result["source"]
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to load ARGO data: {e}",
        }

    # Extract columns that have at least one non-NaN value as discrete float profiles
    valid_mask = np.any(~np.isnan(argo_data), axis=0)
    lats_idx, lons_idx = np.where(valid_mask)
    
    floats = []
    for i, j in zip(lats_idx, lons_idx):
        profile = argo_data[:, i, j]
        
        # Round and convert to list, keeping null for NaN
        profile_clean = [
            round(float(v), 2) if np.isfinite(v) else None 
            for v in profile
        ]
        
        floats.append({
            "lat": float(lat_arr[i]),
            "lon": float(lon_arr[j]),
            "profile": profile_clean
        })

    return {
        "status": "success",
        "date": date,
        "source": argo_result["source"],
        "coverage_pct": argo_result["coverage_pct"],
        "depths": ARGO_DEPTHS,
        "floats": floats
    }


# ==========================================
# 7. SATELLITE EMBEDDING ENDPOINT (R5)
# ==========================================

@app.get("/embedding/{date}")
def get_embedding(date: str):
    """Extract the satellite embedding (compact latent representation) for
    a previously predicted date.

    The embedding is the output of the ResNet50 encoder's deepest block
    (layer4), a 2048-channel feature map that compactly represents the
    surface ocean state.

    Returns
    -------
    JSON with:
        - shape: list of ints (e.g. [1, 2048, 6, 7])
        - embedding_summary: dict with per-channel statistics
        - embedding_data: the full tensor as nested list (optional, for small sizes)
    """
    cache_key = _cache_key(date)
    if cache_key not in _prediction_cache:
        return {
            "status": "unavailable",
            "reason": f"No cached prediction for date '{date}'. Run /predict first.",
        }

    # Re-run preprocessing to get the input tensor
    try:
        prep = preprocess_date(date, data_dir=".")
        input_tensor = prep["input_tensor"].to(device)
    except Exception as e:
        return {"status": "error", "reason": f"Preprocessing failed: {e}"}

    # Extract embedding
    with torch.no_grad():
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        if pad_h > 0 or pad_w > 0:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='replicate')

        embedding = model.forward_embedding(input_tensor)

    emb_np = embedding.squeeze(0).cpu().numpy()  # (2048, H', W')

    # Summary statistics (full tensor is too large to serialize)
    return {
        "status": "success",
        "date": date,
        "shape": list(embedding.shape),
        "n_channels": int(embedding.shape[1]),
        "spatial_size": [int(embedding.shape[2]), int(embedding.shape[3])],
        "total_elements": int(embedding.numel()),
        "summary": {
            "mean": round(float(emb_np.mean()), 6),
            "std": round(float(emb_np.std()), 6),
            "min": round(float(emb_np.min()), 6),
            "max": round(float(emb_np.max()), 6),
            "nonzero_frac": round(float((emb_np != 0).mean()), 4),
        },
        # Top-5 most active channels by mean activation
        "top_channels": sorted(
            [{"channel": int(c), "mean_activation": round(float(emb_np[c].mean()), 4)}
             for c in range(emb_np.shape[0])],
            key=lambda x: abs(x["mean_activation"]),
            reverse=True
        )[:10],
    }


# ==========================================
# 8. PROOF-OF-CONCEPT DEMO ENDPOINTS (R12)
# ==========================================

# Named regions for PoC
POC_REGIONS = {
    "bob": {
        "name": "Bay of Bengal",
        "lat_min": 5.0, "lat_max": 23.0,
        "lon_min": 80.0, "lon_max": 95.0,
    },
    "arabian": {
        "name": "Arabian Sea",
        "lat_min": 5.0, "lat_max": 25.0,
        "lon_min": 50.0, "lon_max": 75.0,
    },
}


@app.get("/poc/{region}/{date}")
def get_poc_demo(region: str, date: str):
    """Proof-of-Concept demonstration for a named region (R12).

    Extracts model predictions for Bay of Bengal ('bob') or Arabian Sea
    ('arabian'), computes regional statistics and returns a cropped
    prediction grid for visualization.

    Parameters
    ----------
    region : str -- 'bob' or 'arabian'
    date : str -- 'YYYY-MM-DD'
    """
    STANDARD_DEPTHS_LOCAL = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    if region not in POC_REGIONS:
        return {
            "status": "error",
            "reason": f"Unknown region '{region}'. Use 'bob' or 'arabian'.",
            "available_regions": list(POC_REGIONS.keys()),
        }

    cache_key = _cache_key(date)
    if cache_key not in _prediction_cache:
        return {
            "status": "unavailable",
            "reason": f"No cached prediction for date '{date}'. Run /predict first.",
        }

    reg = POC_REGIONS[region]
    cached = _prediction_cache[cache_key]

    # Get the full prediction grid
    pred_nested = cached["prediction_data"][0]  # 15 depth layers
    pred_array = np.array(pred_nested, dtype=float)

    # Get coordinate arrays
    lat_arr = np.array(cached.get("lat", TARGET_LATS.tolist()))
    lon_arr = np.array(cached.get("lon", TARGET_LONS.tolist()))

    # Find region indices
    lat_mask = (lat_arr >= reg["lat_min"]) & (lat_arr <= reg["lat_max"])
    lon_mask = (lon_arr >= reg["lon_min"]) & (lon_arr <= reg["lon_max"])

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return {"status": "error", "reason": "Region has no grid points in domain."}

    # Crop prediction
    region_pred = pred_array[:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    region_lats = lat_arr[lat_idx[0]:lat_idx[-1]+1]
    region_lons = lon_arr[lon_idx[0]:lon_idx[-1]+1]

    # Compute per-depth statistics for the region
    depth_stats = []
    for d, depth_m in enumerate(STANDARD_DEPTHS_LOCAL):
        layer = region_pred[d]
        valid = layer[np.isfinite(layer)]
        if len(valid) > 0:
            depth_stats.append({
                "depth_m": depth_m,
                "mean": round(float(valid.mean()), 2),
                "std": round(float(valid.std()), 2),
                "min": round(float(valid.min()), 2),
                "max": round(float(valid.max()), 2),
                "n_valid": int(len(valid)),
            })
        else:
            depth_stats.append({
                "depth_m": depth_m,
                "mean": None, "std": None, "min": None, "max": None,
                "n_valid": 0,
            })

    # Serialize cropped region prediction
    region_safe = np.where(np.isnan(region_pred), None, np.round(region_pred, 2))

    return {
        "status": "success",
        "date": date,
        "region": reg["name"],
        "region_key": region,
        "lat_range": [round(float(region_lats[0]), 2), round(float(region_lats[-1]), 2)],
        "lon_range": [round(float(region_lons[0]), 2), round(float(region_lons[-1]), 2)],
        "grid_shape": [len(region_lats), len(region_lons)],
        "depths": STANDARD_DEPTHS_LOCAL,
        "depth_statistics": depth_stats,
        "prediction_data": [region_safe.tolist()],
        "lat": [round(float(v), 2) for v in region_lats],
        "lon": [round(float(v), 2) for v in region_lons],
    }


@app.get("/poc/regions")
def list_poc_regions():
    """List available PoC demonstration regions."""
    return {"regions": POC_REGIONS}

