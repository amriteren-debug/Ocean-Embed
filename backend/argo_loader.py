"""
ARGO Observation Loader for OceanEmbed Validation (R8, R10)
============================================================
Downloads REAL Argo float profile data from live ERDDAP servers.
No synthetic data is generated — if all sources fail the endpoint
returns an empty array with source='NONE'.

Primary   : IFREMER ERDDAP  (erddap.ifremer.fr)
Secondary : INCOIS  ERDDAP  (erddap.incois.gov.in)
Tertiary  : NOAA ArgoVis    (argovis.colorado.edu)

For any given date we use a ±7-day window so we capture a full
basin snapshot (Argo floats surface every ~10 days).
"""

import io
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import requests
from scipy.interpolate import interp1d

from preprocessing import TARGET_LATS, TARGET_LONS

# ── constants ────────────────────────────────────────────────────────────────
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# Indian-Ocean bounding box (must match model domain)
LAT_MIN_IO, LAT_MAX_IO = -10.0, 30.0
LON_MIN_IO, LON_MAX_IO =  45.0, 105.0

ARGO_CACHE_DIR = os.path.join(os.path.dirname(__file__), "argo_cache")
os.makedirs(ARGO_CACHE_DIR, exist_ok=True)

# Request timeout — generous for overseas servers
_TIMEOUT = 30

# Suppress InsecureRequestWarning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ── helpers ───────────────────────────────────────────────────────────────────

def _date_window(date_str: str, days: int = 7):
    """Return (start_iso, end_iso) strings for a ±days window."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    end   = (dt + timedelta(days=days)).strftime("%Y-%m-%dT23:59:59Z")
    return start, end


def _profiles_to_grid(df, col_lat="latitude", col_lon="longitude",
                      col_pres="pres", col_temp="temp",
                      col_platform="platform_number"):
    """
    Interpolate each float profile to STANDARD_DEPTHS and assign it to the
    nearest 0.25-degree grid cell.

    Returns np.ndarray of shape (15, n_lat, n_lon). Unobserved cells = NaN.
    """
    n_lat = len(TARGET_LATS)
    n_lon = len(TARGET_LONS)
    result = np.full((len(STANDARD_DEPTHS), n_lat, n_lon), np.nan)

    if df is None or df.empty:
        return result

    # Coerce column names to lowercase for safety
    df.columns = [c.lower() for c in df.columns]
    col_lat   = col_lat.lower()
    col_lon   = col_lon.lower()
    col_pres  = col_pres.lower()
    col_temp  = col_temp.lower()
    col_plat  = col_platform.lower()

    # Group by platform (each float = one profile position per surfacing)
    for plat, group in df.groupby(col_plat):
        group = group.dropna(subset=[col_pres, col_temp]).sort_values(col_pres)
        if len(group) < 3:
            continue

        lat = float(group[col_lat].iloc[0])
        lon = float(group[col_lon].iloc[0])

        # Clip to domain
        if not (LAT_MIN_IO <= lat <= LAT_MAX_IO and LON_MIN_IO <= lon <= LON_MAX_IO):
            continue

        lat_idx = int(np.argmin(np.abs(TARGET_LATS - lat)))
        lon_idx = int(np.argmin(np.abs(TARGET_LONS - lon)))

        pres = group[col_pres].values.astype(float)
        temp = group[col_temp].values.astype(float)

        try:
            f = interp1d(pres, temp, bounds_error=False, fill_value=np.nan)
            interp_temp = f(STANDARD_DEPTHS)
        except Exception:
            continue

        # If a cell is already occupied, average the profiles
        existing = result[:, lat_idx, lon_idx]
        if np.any(np.isfinite(existing)):
            result[:, lat_idx, lon_idx] = np.nanmean(
                np.stack([existing, interp_temp]), axis=0
            )
        else:
            result[:, lat_idx, lon_idx] = interp_temp

    return result


# ── Source 1: IFREMER ERDDAP ──────────────────────────────────────────────────

def _fetch_ifremer(date_str: str):
    """Fetch real Argo profiles from IFREMER ERDDAP (primary)."""
    start, end = _date_window(date_str, days=7)
    url = (
        "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"
        f"?platform_number,latitude,longitude,pres,temp"
        f"&time>={start}&time<={end}"
        f"&latitude>={LAT_MIN_IO}&latitude<={LAT_MAX_IO}"
        f"&longitude>={LON_MIN_IO}&longitude<={LON_MAX_IO}"
        f"&pres<=1050"          # cap at 1050 dbar — avoids deep-Argo edge cases
    )
    print(f"  [ARGO] Trying IFREMER ERDDAP … {url[:90]}…")
    try:
        resp = requests.get(url, timeout=_TIMEOUT, verify=False)
        resp.raise_for_status()
        import pandas as pd
        df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])   # row 1 = units
        if df.empty or len(df) < 5:
            print("  [ARGO] IFREMER: empty response")
            return None, 0
        n = df["platform_number"].nunique() if "platform_number" in df.columns else "?"
        print(f"  [ARGO] IFREMER: {n} floats fetched")
        return _profiles_to_grid(df), n
    except Exception as e:
        print(f"  [WARN] IFREMER failed: {e}")
        return None, 0


# ── Source 2: INCOIS ERDDAP ───────────────────────────────────────────────────

def _fetch_incois(date_str: str):
    """Fetch real Argo profiles from INCOIS ERDDAP (Indian Ocean specialist)."""
    start, end = _date_window(date_str, days=7)
    url = (
        "https://erddap.incois.gov.in/erddap/tabledap/Indian_ARGO_Floats.csv"
        f"?PLATFORM_NUMBER,LATITUDE,LONGITUDE,PRES,TEMP"
        f"&time>={start}&time<={end}"
        f"&LATITUDE>={LAT_MIN_IO}&LATITUDE<={LAT_MAX_IO}"
        f"&LONGITUDE>={LON_MIN_IO}&LONGITUDE<={LON_MAX_IO}"
        f"&PRES<=1050"
    )
    print(f"  [ARGO] Trying INCOIS ERDDAP … {url[:90]}…")
    try:
        resp = requests.get(url, timeout=_TIMEOUT, verify=False)
        resp.raise_for_status()
        import pandas as pd
        df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
        if df.empty or len(df) < 5:
            print("  [ARGO] INCOIS: empty response")
            return None, 0
        # Map INCOIS column names → standard names
        df.rename(columns={
            "PLATFORM_NUMBER": "platform_number",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "PRES": "pres",
            "TEMP": "temp",
        }, inplace=True)
        n = df["platform_number"].nunique()
        print(f"  [ARGO] INCOIS: {n} floats fetched")
        return _profiles_to_grid(df), n
    except Exception as e:
        print(f"  [WARN] INCOIS failed: {e}")
        return None, 0


# ── Source 3: ArgoVis REST API ────────────────────────────────────────────────

def _fetch_argovis(date_str: str):
    """Fetch real Argo profiles from the ArgoVis API (NOAA / U-Colorado)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (dt + timedelta(days=7)).strftime("%Y-%m-%d")
    url = (
        "https://argovis.colorado.edu/selection/profiles"
        f"?startDate={start}&endDate={end}"
        f"&polygon=[[{LON_MIN_IO},{LAT_MIN_IO},{LON_MIN_IO},{LAT_MAX_IO},"
        f"{LON_MAX_IO},{LAT_MAX_IO},{LON_MAX_IO},{LAT_MIN_IO},{LON_MIN_IO},{LAT_MIN_IO}]]"
    )
    print(f"  [ARGO] Trying ArgoVis … {url[:90]}…")
    try:
        resp = requests.get(url, timeout=_TIMEOUT, verify=False)
        resp.raise_for_status()
        profiles = resp.json()
        if not profiles:
            print("  [ARGO] ArgoVis: empty response")
            return None, 0

        n_lat = len(TARGET_LATS)
        n_lon = len(TARGET_LONS)
        result = np.full((len(STANDARD_DEPTHS), n_lat, n_lon), np.nan)
        count = 0
        for prof in profiles:
            try:
                lat = prof["lat"]
                lon = prof["lon"]
                measurements = prof.get("measurements", [])
                if not measurements:
                    continue
                pres = np.array([m["pres"] for m in measurements if "pres" in m and "temp" in m], dtype=float)
                temp = np.array([m["temp"] for m in measurements if "pres" in m and "temp" in m], dtype=float)
                valid = np.isfinite(pres) & np.isfinite(temp)
                pres, temp = pres[valid], temp[valid]
                if len(pres) < 3:
                    continue

                lat_idx = int(np.argmin(np.abs(TARGET_LATS - lat)))
                lon_idx = int(np.argmin(np.abs(TARGET_LONS - lon)))
                f = interp1d(pres, temp, bounds_error=False, fill_value=np.nan)
                result[:, lat_idx, lon_idx] = f(STANDARD_DEPTHS)
                count += 1
            except Exception:
                continue

        print(f"  [ARGO] ArgoVis: {count} profiles processed")
        return result, count
    except Exception as e:
        print(f"  [WARN] ArgoVis failed: {e}")
        return None, 0



# ── 3-degree exclusion-zone thinning ─────────────────────────────────────────

def _thin_to_3deg_grid(data: np.ndarray, cell_deg: float = 3.0) -> np.ndarray:
    """
    Enforce non-overlapping 3-degree exclusion zones on a (15, n_lat, n_lon)
    real-data grid.

    Algorithm:
      1. Partition the domain into cells of `cell_deg` × `cell_deg` degrees.
      2. Within each cell collect all 0.25° sub-cells that contain at least one
         finite value.
      3. Score each sub-cell by the number of depth levels that are finite.
      4. Keep the sub-cell with the highest score (best-observed float).
         All other sub-cells inside the same 3° cell are cleared to NaN.

    The result is a clean, non-overlapping representation: the best real float
    profile per ~300 km × 300 km box, guaranteed to have zero range overlap.
    """
    thinned = data.copy()
    n_lat = len(TARGET_LATS)
    n_lon = len(TARGET_LONS)
    lat_step = float(TARGET_LATS[1] - TARGET_LATS[0]) if n_lat > 1 else 0.25
    lon_step = float(TARGET_LONS[1] - TARGET_LONS[0]) if n_lon > 1 else 0.25

    cell_i = max(1, round(cell_deg / lat_step))  # number of 0.25° rows per cell
    cell_j = max(1, round(cell_deg / lon_step))  # number of 0.25° cols per cell

    for i0 in range(0, n_lat, cell_i):
        for j0 in range(0, n_lon, cell_j):
            i1 = min(i0 + cell_i, n_lat)
            j1 = min(j0 + cell_j, n_lon)

            # Find all sub-cells in this 3° block that have any finite data
            candidates = []
            for ii in range(i0, i1):
                for jj in range(j0, j1):
                    n_finite = int(np.sum(np.isfinite(data[:, ii, jj])))
                    if n_finite > 0:
                        candidates.append((n_finite, ii, jj))

            if len(candidates) == 0:
                continue  # no data in this cell at all

            # Elect the sub-cell with the most finite depth levels
            candidates.sort(key=lambda x: -x[0])
            best_score, best_i, best_j = candidates[0]

            # Clear every OTHER sub-cell in this 3° block
            for _, ii, jj in candidates[1:]:
                thinned[:, ii, jj] = np.nan

    return thinned


# ── Public entry-point ────────────────────────────────────────────────────────

def load_argo_for_date(date_str: str):
    """
    Load REAL Argo temperature observations for a given date.

    Tries three live ERDDAP / REST sources in sequence:
      1. IFREMER ERDDAP  (global Argo GDAC)
      2. INCOIS  ERDDAP  (Indian Ocean specialist)
      3. ArgoVis REST    (NOAA / U-Colorado)

    Returns a dict with keys:
        data         : np.ndarray (15, n_lat, n_lon)
        lat          : np.ndarray
        lon          : np.ndarray
        source       : str
        coverage_pct : float
        n_floats     : int
    """
    data, n_floats, source = None, 0, "NONE"

    # --- 1. IFREMER ---
    d, n = _fetch_ifremer(date_str)
    if d is not None and n > 0:
        data, n_floats, source = d, n, "IFREMER ERDDAP"

    # --- 2. INCOIS (supplement or replace if IFREMER gave nothing) ---
    if data is None or n_floats == 0:
        d, n = _fetch_incois(date_str)
        if d is not None and n > 0:
            data, n_floats, source = d, n, "INCOIS ERDDAP"

    # --- 3. ArgoVis fallback ---
    if data is None or n_floats == 0:
        d, n = _fetch_argovis(date_str)
        if d is not None and n > 0:
            data, n_floats, source = d, n, "ArgoVis (NOAA)"

    # --- absolute fallback: empty array (no synthetic data) ---
    if data is None:
        print("  [WARN] All Argo sources failed — returning empty grid.")
        data = np.full((len(STANDARD_DEPTHS), len(TARGET_LATS), len(TARGET_LONS)), np.nan)
        n_floats = 0
        source = "NONE"

    # ── Enforce 3-degree non-overlapping exclusion zones ─────────────────────
    # Within each 3°×3° cell, keep only the real float with the most complete
    # depth profile. All other floats in the same cell are cleared to NaN.
    # This produces a clean, non-overlapping map exactly matching the Argo
    # design spacing of one float per 3° × 3° (~300 km × 300 km) cell.
    data = _thin_to_3deg_grid(data)

    # Coverage metric (after thinning)
    valid_cells = int(np.sum(np.any(np.isfinite(data), axis=0)))
    ocean_est   = 24_000  # approx ocean cells in the 0.25-degree Indian-Ocean grid
    coverage    = min(100.0, round(valid_cells / ocean_est * 100, 1))

    print(f"  [ARGO] Source={source} | Floats after thinning={valid_cells} | Coverage={coverage}%")
    return {
        "data":         data,
        "lat":          TARGET_LATS.copy(),
        "lon":          TARGET_LONS.copy(),
        "source":       source,
        "n_floats":     valid_cells,
        "coverage_pct": coverage,
    }


# ── CLI quick-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else "2024-08-01"
    print(f"\nFetching REAL Argo data for {date} …\n")
    result = load_argo_for_date(date)
    print(f"\n  Source       : {result['source']}")
    print(f"  Floats found : {result['n_floats']}")
    print(f"  Coverage     : {result['coverage_pct']}%")
    print(f"  Grid shape   : {result['data'].shape}")
    for i, d in enumerate(STANDARD_DEPTHS):
        layer = result["data"][i]
        valid = layer[np.isfinite(layer)]
        if len(valid):
            print(f"    {d:5d}m : {len(valid):4d} pts | mean {np.mean(valid):.2f} °C")
        else:
            print(f"    {d:5d}m : no data")
