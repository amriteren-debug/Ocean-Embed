"""
OceanEmbed Preprocessing & Harmonization Pipeline
===================================================
Standalone, callable module that ingests raw multi-source satellite/ocean
NetCDF files and outputs standardised, analysis-ready tensors.

Satisfies requirements R1 (pipeline), R2 (0.25° spatial), R3 (daily temporal),
R4 (7-channel input order).

Channel order (fixed, documented):
    0: SST    — Sea Surface Temperature        (°C)
    1: SSS    — Sea Surface Salinity            (PSU)
    2: SSH    — Sea Surface Height / SLA        (m)
    3: U_cur  — Eastward surface current        (m/s)
    4: V_cur  — Northward surface current       (m/s)
    5: U_wind — Eastward 10m wind               (m/s)
    6: V_wind — Northward 10m wind              (m/s)

Usage:
    from preprocessing import preprocess_date
    result = preprocess_date("2026-08-20", data_dir=".")
    # result["input_tensor"]  — torch.Tensor (1, 7, lat_025, lon_025)
    # result["land_mask"]     — np.ndarray   (lat_025, lon_025)
    # result["lat"]           — np.ndarray   of 0.25° latitudes
    # result["lon"]           — np.ndarray   of 0.25° longitudes

CLI:
    python preprocessing.py --date 2026-08-20
"""
import os
import argparse
import numpy as np
import torch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

# Problem Statement domain: North Indian Ocean
DOMAIN_LAT_MIN = -10.0
DOMAIN_LAT_MAX = 30.0
DOMAIN_LON_MIN = 45.0
DOMAIN_LON_MAX = 100.0

# Target spatial resolution (R2)
TARGET_RES = 0.25  # degrees

# Standard 0.25° grid
TARGET_LATS = np.arange(DOMAIN_LAT_MIN, DOMAIN_LAT_MAX + TARGET_RES / 2, TARGET_RES)
TARGET_LONS = np.arange(DOMAIN_LON_MIN, DOMAIN_LON_MAX + TARGET_RES / 2, TARGET_RES)

# Channel order (R4) — must match training AND inference
CHANNEL_ORDER = ["SST", "SSS", "SSH", "U_cur", "V_cur", "U_wind", "V_wind"]


def _regrid_to_025(data_2d, src_lats, src_lons):
    """Regrid a 2D (lat, lon) field to the standard 0.25° grid.

    Uses bilinear interpolation via scipy RegularGridInterpolator.
    NaN-safe: replaces NaN with 0 before interpolation, then restores
    a land mask afterward based on the nearest-neighbour NaN pattern.

    Parameters
    ----------
    data_2d : np.ndarray, shape (n_lat, n_lon)
    src_lats : np.ndarray, shape (n_lat,)  — must be monotonically increasing
    src_lons : np.ndarray, shape (n_lon,)  — must be monotonically increasing

    Returns
    -------
    np.ndarray, shape (len(TARGET_LATS), len(TARGET_LONS))
    """
    # Ensure source coordinates are ascending
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        data_2d = data_2d[::-1, :]
    if src_lons[0] > src_lons[-1]:
        src_lons = src_lons[::-1]
        data_2d = data_2d[:, ::-1]

    # Build NaN mask at source resolution, then use nearest-neighbour to
    # propagate it onto the target grid
    nan_mask_src = np.isnan(data_2d).astype(float)
    nn_interp = RegularGridInterpolator(
        (src_lats, src_lons), nan_mask_src,
        method="nearest", bounds_error=False, fill_value=1.0
    )
    target_pts = np.meshgrid(TARGET_LATS, TARGET_LONS, indexing="ij")
    target_pts = np.stack(target_pts, axis=-1)
    nan_mask_target = nn_interp(target_pts) > 0.5

    # Bilinear interpolation on the data (fill NaN with 0 first)
    data_filled = np.where(np.isnan(data_2d), 0.0, data_2d)
    bilinear = RegularGridInterpolator(
        (src_lats, src_lons), data_filled,
        method="linear", bounds_error=False, fill_value=0.0
    )
    result = bilinear(target_pts)

    # Restore NaN for land
    result[nan_mask_target] = np.nan
    return result


def _load_and_regrid_ocean_var(ds, var_name, depth_sel=True):
    """Load a single variable from an xarray Dataset, select surface + first
    time step, and regrid to 0.25°.

    Parameters
    ----------
    ds : xr.Dataset
    var_name : str  — e.g. "thetao", "so", "uo", etc.
    depth_sel : bool — if True, selects depth index 0

    Returns
    -------
    np.ndarray, shape (len(TARGET_LATS), len(TARGET_LONS))
    """
    da = ds[var_name]
    # Select first time step
    if "time" in da.dims:
        da = da.isel(time=0)
    # Select surface depth
    if depth_sel and "depth" in da.dims:
        da = da.isel(depth=0)

    raw = da.values.astype(np.float64)
    lats = da.coords["latitude"].values.astype(np.float64)
    lons = da.coords["longitude"].values.astype(np.float64)

    return _regrid_to_025(raw, lats, lons)


def _load_wind_daily(ds_wind, var_name, ref_lats, ref_lons):
    """Load wind variable, average across all hourly timesteps (R3),
    and regrid to 0.25°.

    Parameters
    ----------
    ds_wind : xr.Dataset
    var_name : str — "eastward_wind" or "northward_wind"
    ref_lats, ref_lons : ignored (regridding uses TARGET grid directly)

    Returns
    -------
    np.ndarray, shape (len(TARGET_LATS), len(TARGET_LONS))
    """
    da = ds_wind[var_name]
    # Average across all time steps → daily mean (R3)
    if "time" in da.dims:
        da = da.mean(dim="time")

    raw = da.values.astype(np.float64)
    lats = da.coords["latitude"].values.astype(np.float64)
    lons = da.coords["longitude"].values.astype(np.float64)

    return _regrid_to_025(raw, lats, lons)


def preprocess_date(date_str, data_dir="."):
    """Run the full preprocessing pipeline for a single date.

    Parameters
    ----------
    date_str : str — "YYYY-MM-DD"
    data_dir : str — directory containing temp_*.nc files

    Returns
    -------
    dict with keys:
        "input_tensor"  : torch.Tensor (1, 7, H, W)  — 7 channels on 0.25° grid
        "input_array"   : np.ndarray   (7, H, W)      — same as numpy
        "land_mask"     : np.ndarray   (H, W)          — True = land
        "lat"           : np.ndarray   (H,)            — 0.25° latitudes
        "lon"           : np.ndarray   (W,)            — 0.25° longitudes
        "channel_order" : list[str]                    — channel names
        "raw_fields"    : dict[str, np.ndarray]        — individual 2D fields
    """
    # ── File paths ──
    f_thetao = os.path.join(data_dir, f"temp_thetao_{date_str}.nc")
    f_so     = os.path.join(data_dir, f"temp_so_{date_str}.nc")
    f_cur    = os.path.join(data_dir, f"temp_cur_{date_str}.nc")
    f_zos    = os.path.join(data_dir, f"temp_zos_{date_str}.nc")
    f_wind   = os.path.join(data_dir, f"temp_wind_{date_str}.nc")

    # Check existence
    for label, path in [("thetao", f_thetao), ("so", f_so),
                        ("cur", f_cur), ("zos", f_zos)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {path}")

    # ── Open datasets ──
    ds_thetao = xr.open_dataset(f_thetao)
    ds_so     = xr.open_dataset(f_so)
    ds_cur    = xr.open_dataset(f_cur)
    ds_zos    = xr.open_dataset(f_zos)

    try:
        # ── Load and regrid ocean variables to 0.25° (R2) ──
        sst   = _load_and_regrid_ocean_var(ds_thetao, "thetao", depth_sel=True)
        sss   = _load_and_regrid_ocean_var(ds_so, "so", depth_sel=True)
        u_cur = _load_and_regrid_ocean_var(ds_cur, "uo", depth_sel=True)
        v_cur = _load_and_regrid_ocean_var(ds_cur, "vo", depth_sel=True)
        ssh   = _load_and_regrid_ocean_var(ds_zos, "zos", depth_sel=False)

        # ── Wind: load, daily-average (R3), regrid to 0.25° ──
        wind_available = False
        try:
            if os.path.exists(f_wind) and os.path.getsize(f_wind) > 1000:
                ds_wind = xr.open_dataset(f_wind)
                wind_available = True
        except Exception:
            pass

        if wind_available:
            u_wind = _load_wind_daily(ds_wind, "eastward_wind", None, None)
            v_wind = _load_wind_daily(ds_wind, "northward_wind", None, None)
            ds_wind.close()
        else:
            u_wind = np.zeros_like(sst)
            v_wind = np.zeros_like(sst)
            print(f"  [WARN] Wind data unavailable for {date_str} -- using zero-fill")

        # ── Land mask from SST NaN pattern ──
        land_mask = np.isnan(sst)

        # ── Fill NaN with 0 for model input ──
        sst_filled   = np.nan_to_num(sst, nan=0.0)
        sss_filled   = np.nan_to_num(sss, nan=0.0)
        ssh_filled   = np.nan_to_num(ssh, nan=0.0)
        u_cur_filled = np.nan_to_num(u_cur, nan=0.0)
        v_cur_filled = np.nan_to_num(v_cur, nan=0.0)
        u_wind_filled = np.nan_to_num(u_wind, nan=0.0)
        v_wind_filled = np.nan_to_num(v_wind, nan=0.0)

        # ── Stack in documented channel order (R4) ──
        # Channel 0: SST, 1: SSS, 2: SSH, 3: U_cur, 4: V_cur, 5: U_wind, 6: V_wind
        input_array = np.stack([
            sst_filled, sss_filled, ssh_filled,
            u_cur_filled, v_cur_filled,
            u_wind_filled, v_wind_filled
        ], axis=0)  # Shape: (7, H, W)

        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)

        raw_fields = {
            "sst": sst, "sss": sss, "ssh": ssh,
            "u_cur": u_cur, "v_cur": v_cur,
            "u_wind": u_wind, "v_wind": v_wind,
        }

        return {
            "input_tensor": input_tensor,
            "input_array": input_array,
            "land_mask": land_mask,
            "lat": TARGET_LATS.copy(),
            "lon": TARGET_LONS.copy(),
            "channel_order": list(CHANNEL_ORDER),
            "raw_fields": raw_fields,
        }

    finally:
        ds_thetao.close()
        ds_so.close()
        ds_cur.close()
        ds_zos.close()


def regrid_output_to_025(pred_numpy, src_lats, src_lons):
    """Regrid model output (n_depths, lat, lon) from native resolution to 0.25°.

    Parameters
    ----------
    pred_numpy : np.ndarray, shape (n_depths, lat, lon)
    src_lats : np.ndarray — source latitudes
    src_lons : np.ndarray — source longitudes

    Returns
    -------
    np.ndarray, shape (n_depths, len(TARGET_LATS), len(TARGET_LONS))
    """
    n_depths = pred_numpy.shape[0]
    result = np.empty((n_depths, len(TARGET_LATS), len(TARGET_LONS)), dtype=np.float64)
    for d in range(n_depths):
        result[d] = _regrid_to_025(pred_numpy[d], src_lats, src_lons)
    return result


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed Preprocessing Pipeline")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--data-dir", default=".", help="Directory with temp_*.nc files")
    args = parser.parse_args()

    print(f"Running preprocessing pipeline for {args.date}...")
    print(f"  Domain: {DOMAIN_LAT_MIN} to {DOMAIN_LAT_MAX} N, "
          f"{DOMAIN_LON_MIN} to {DOMAIN_LON_MAX} E")
    print(f"  Target resolution: {TARGET_RES} deg")
    print(f"  Target grid: {len(TARGET_LATS)} lat x {len(TARGET_LONS)} lon")
    print(f"  Channel order: {CHANNEL_ORDER}")

    result = preprocess_date(args.date, data_dir=args.data_dir)

    print(f"\n[OK] Preprocessing complete!")
    print(f"  input_tensor shape: {result['input_tensor'].shape}")
    print(f"  land_mask shape:    {result['land_mask'].shape}")
    print(f"  lat range:          {result['lat'][0]:.2f} to {result['lat'][-1]:.2f}")
    print(f"  lon range:          {result['lon'][0]:.2f} to {result['lon'][-1]:.2f}")
    print(f"  land fraction:      {result['land_mask'].mean():.1%}")

    for i, name in enumerate(CHANNEL_ORDER):
        ch = result["input_array"][i]
        ocean = ch[~result["land_mask"]]
        if len(ocean) > 0:
            print(f"  Ch{i} ({name:6s}): min={ocean.min():.4f}  max={ocean.max():.4f}  "
                  f"mean={ocean.mean():.4f}  std={ocean.std():.4f}")
        else:
            print(f"  Ch{i} ({name:6s}): ALL LAND")
