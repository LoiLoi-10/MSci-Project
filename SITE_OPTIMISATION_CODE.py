# ============================================================
# Script overview 
#
# Pipeline:
#   1) Build candidate offshore wind + tidal sites (EEZ/land/dist/depth filters)
#   2) Convert metocean data to hourly capacity-factor time series per candidate
#   3) Solve PyPSA capacity expansion with exact build targets
#   4) Export built sites + produce maps
#   5) Optional OAT sensitivity: rerun optimisation with one parameter changed
#
# Conventions:
#   - PyPSA capital_cost is ANNUALISED (GBP/MW-year)
#   - Backup is excluded from LCOE numerators/denominators (can be included by turning on generator attribute)
# ============================================================

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, asdict
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pypsa
import geopandas as gpd
import shapely


# ============================================================
# 0) CONFIG
# ============================================================

# ---- Force temporary files to a known directory (solver/model can use TMP a lot)
TMP_DIR = Path(r"D:\tmp_linopy")
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(TMP_DIR)
os.environ["TEMP"] = str(TMP_DIR)

# ---- Scenario mode:
#   "wind_only": only wind, exact wind build
#   "tidal_only": only tidal, exact tidal build (+ storage)
#   "joint": wind + tidal, exact total build (+ storage), plus optional diversification bonus
SCENARIO = "joint"

# ---- Exact build targets (MW)
TARGET_BUILD_MW_WIND = 15000.0
TARGET_BUILD_MW_TIDAL = 30000.0
TARGET_BUILD_MW_TOTAL_JOINT = 30000.0

# ---- Input paths
ERA5_WIND_FILE = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\ERA5_DATA\era5_uk_offshore_2023_12.nc")
POWER_CURVE_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\POWER_CURVE.csv")
DEMAND_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\DEMAND_DATA\demanddata_2023.csv")
UK_LAND_SHP = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\SHAPES\uk_land.shp")
UK_EEZ_SHP = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\SHAPES\eez.shp")
TIDAL_CURRENTS_NETCDF = Path(r"D:\cmems_data\cmems_mod_nws_phy-cur_anfc_1.5km-3D_PT1H-i_uo-vo_6.00W-2.00E_48.00N-61.50N_20.00-50.00m_2023-12-01-2023-12-30.nc")

# ---- Bathymetry config (GEBCO extract)
BATHY_NETCDF = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\GEBCO_21_Dec_2025_5f2f49514b76\gebco_2025_n62.0_s48.0_w-8.0_e4.0.nc")
BATHY_VAR = "elevation"
BATHY_LAT_NAME = "lat"
BATHY_LON_NAME = "lon"
BATHY_POSITIVE_DOWN = False  # GEBCO elevation is negative offshore; convert to depth via -elev (change if dataset differs)

# ---- Depth bands (m, positive down)
WIND_MIN_DEPTH_M = 10.0
WIND_MAX_DEPTH_M = 1000.0

TIDAL_MIN_DEPTH_M = 20.0
TIDAL_MAX_DEPTH_M = 50.0

# ---- Existing site exclusion (buffer around known sites)
EXISTING_WIND_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\OPERATIONAL_SITES\uk_offshore_wind_sites.csv")
EXISTING_TIDAL_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\OPERATIONAL_SITES\uk_tidal_sites.csv")
EXCLUDE_KM_WIND = 15.0
EXCLUDE_KM_TIDAL = 2.25

# ---- Max distance from shore (candidate screening)
MAX_DIST_KM_WIND = 200.0
MAX_DIST_KM_TIDAL = 35.0

# ---- Min distance from shore (candidate screening)
MIN_DIST_KM_WIND = 10.0
MIN_DIST_KM_TIDAL = 2.0

# ---- Wind vertical shear (power law) if using 10m winds and 100m wind missing from dataset
ALPHA = 0.14
HUB_HEIGHT_M = 100.0

# ---- Tidal turbine/device CF curve parameters (m/s)
TIDAL_CUT_IN = 0.6
TIDAL_RATED = 2.5
TIDAL_CUT_OUT = 5.0

# -----------------------------
# Capacity limits
# -----------------------------
# Wind: treated as site-level (no clustering), per-site p_nom_max (due to large resolution size)
P_NOM_MAX_PER_SITE_MW_WIND = 862.0

# Tidal: start as cells, later clustered; each cell contributes up to TIDAL_CELL_CAPACITY_MW
TIDAL_CELL_CAPACITY_MW = 51.0
TIDAL_CLUSTER_MAX_CELLS = 20
# -----------------------------

# ---- Distance-based losses and cable capex adders
LOSS_PER_KM = 0.0003
CABLE_COST_GBP_PER_MWKM = 22_500.0

# ---- Storage settings (storage added for tidal_only AND joint later)
ADD_TIDAL_STORAGE = True
STORAGE_MAX_HOURS = 6.0
STORAGE_EFF_STORE = 0.80
STORAGE_EFF_DISPATCH = 0.95
STORAGE_CAPEX_GBP_PER_MW = 250_000.0
STORAGE_MARGINAL_GBP_PER_MWH = 0.0

# ---- Baseline techno-economic assumptions
CAPEX_WIND_GBP_PER_MW = 3_000_000.0
MARGINAL_WIND_GBP_PER_MWH = 49.0

CAPEX_TIDAL_GBP_PER_MW = 3_500_000.0
MARGINAL_TIDAL_GBP_PER_MWH = 179.0

CAPEX_BACKUP_GBP_PER_MW = 700_000.0
MARGINAL_BACKUP_GBP_PER_MWH = 200.0
BACKUP_P_NOM_MAX_MW = 60000.0

# ---- Diversification credit (implemented as a negative-capex "generator" with p_max_pu=0)
DIVERSIFICATION_BONUS_GBP_PER_MW = 000_000.0

# ============================================================
# Annualisation (discount rate affects capital_cost via annuity)
# ============================================================

# Baseline discount rate (OAT will change this)
DISCOUNT_RATE = 0.07

# Lifetimes (years) used in annuity factor
LIFETIME_WIND_Y = 25
LIFETIME_TIDAL_Y = 25
LIFETIME_STORAGE_Y = 20
LIFETIME_BACKUP_Y = 30

def annuity(r: float, n: int) -> float:
    """Fixed-charge rate (FCR) used to annualise CAPEX."""
    return r / (1.0 - (1.0 + r) ** (-n))

def refresh_annualised_costs() -> None:
    """
    Recompute annualised CAPEX globals whenever DISCOUNT_RATE or CAPEX changes.
    PyPSA expects capital_cost in currency per MW-year (annualised).
    """
    global FCR_WIND, FCR_TIDAL, FCR_STORAGE, FCR_BACKUP
    global CAPEX_WIND_GBP_PER_MW_ANNUAL, CAPEX_TIDAL_GBP_PER_MW_ANNUAL
    global STORAGE_CAPEX_GBP_PER_MW_ANNUAL, CAPEX_BACKUP_GBP_PER_MW_ANNUAL

    FCR_WIND    = annuity(float(DISCOUNT_RATE), int(LIFETIME_WIND_Y))
    FCR_TIDAL   = annuity(float(DISCOUNT_RATE), int(LIFETIME_TIDAL_Y))
    FCR_STORAGE = annuity(float(DISCOUNT_RATE), int(LIFETIME_STORAGE_Y))
    FCR_BACKUP  = annuity(float(DISCOUNT_RATE), int(LIFETIME_BACKUP_Y))

    CAPEX_WIND_GBP_PER_MW_ANNUAL   = float(CAPEX_WIND_GBP_PER_MW)   * FCR_WIND
    CAPEX_TIDAL_GBP_PER_MW_ANNUAL  = float(CAPEX_TIDAL_GBP_PER_MW)  * FCR_TIDAL
    STORAGE_CAPEX_GBP_PER_MW_ANNUAL = float(STORAGE_CAPEX_GBP_PER_MW) * FCR_STORAGE
    CAPEX_BACKUP_GBP_PER_MW_ANNUAL = float(CAPEX_BACKUP_GBP_PER_MW) * FCR_BACKUP

# Run once at import time so *ANNUAL globals exist before solving
refresh_annualised_costs()

# ---- Candidate down-selection sizes (after filtering)
TOP_N_WIND = 400
TOP_N_TIDAL = 800

# Wind NOT clustered; tidal clustered radius
CLUSTER_KM_TIDAL = 4.5

# ---- Output paths
OUTDIR = Path("figures_site_selection")
OUTDIR.mkdir(exist_ok=True)

OUT_WIND_SITES = Path("selected_wind_sites.csv")
OUT_TIDAL_SITES = Path("selected_tidal_sites.csv")
OUT_JOINT_WIND = Path("selected_joint_wind_sites.csv")
OUT_JOINT_TIDAL = Path("selected_joint_tidal_sites.csv")


# ============================================================
# 1) HELPERS
# ============================================================

def load_power_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load power curve (v,p) and normalise p to [0,1] for CF interpolation."""
    df = pd.read_csv(csv_path).dropna().sort_values("v")
    if not {"v", "p"}.issubset(df.columns):
        raise ValueError("power_curve.csv must contain columns: v, p")
    v = df["v"].astype(float).to_numpy()
    p = df["p"].astype(float).to_numpy()
    if p.max() > 1.0:
        p = p / p.max()
    return v, np.clip(p, 0.0, 1.0)

def detect_time_lat_lon(ds: xr.Dataset) -> Tuple[str, str, str]:
    """
    Try to infer coordinate names for time/lat/lon across ERA5/CMEMS-style datasets.
    Returns (time_dim, lat_name, lon_name).
    """
    time_dim: Optional[str] = None
    for c in ds.coords:
        if ds[c].ndim == 1 and np.issubdtype(ds[c].dtype, np.datetime64):
            time_dim = str(c)
            break
    if time_dim is None:
        for cand in ("time", "valid_time", "datetime", "date"):
            if cand in ds.coords or cand in ds.variables:
                time_dim = cand
                break
    if time_dim is None:
        raise ValueError("Could not find time coordinate.")

    lat_name: Optional[str] = None
    lon_name: Optional[str] = None
    for nm in list(ds.dims) + list(ds.coords):
        low = str(nm).lower()
        if lat_name is None and "lat" in low:
            lat_name = str(nm)
        if lon_name is None and "lon" in low:
            lon_name = str(nm)

    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not find lat/lon in dataset. dims={list(ds.dims)} coords={list(ds.coords)}")
    return str(time_dim), str(lat_name), str(lon_name)

def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame has CRS and convert to EPSG:4326 (WGS84)."""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")

def dissolve_to_single_geometry(gdf: gpd.GeoDataFrame):
    """Union all geometries into one (faster for vectorised contains/within checks)."""
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    return gdf.geometry.union_all()

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (km) between two WGS84 points/arrays. Model the Earth"""
    R = 6371.0
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def trans_eff_from_distance_km(dist_km: np.ndarray) -> np.ndarray:
    """Transmission efficiency (loss model) as a function of distance to shore."""
    return np.exp(-LOSS_PER_KM * dist_km)

def capex_with_cable(base_capex: float, dist_km: np.ndarray) -> np.ndarray:
    """Add cable capex component proportional to distance (annualised base is passed in)."""
    return base_capex + CABLE_COST_GBP_PER_MWKM * dist_km

def wind_cf_from_u_v(
    u: xr.DataArray,
    v: xr.DataArray,
    v_curve: np.ndarray,
    p_curve: np.ndarray,
    hub_h: float,
    ref_h: float
) -> np.ndarray:
    """
    Convert u/v wind components to CF using:
      - speed magnitude
      - shear adjustment (hub_h/ref_h)^ALPHA
      - interpolation onto turbine power curve
    """
    ws = np.hypot(u.values, v.values)
    factor = (hub_h / ref_h) ** ALPHA
    ws = ws * factor
    cf = np.interp(ws, v_curve, p_curve, left=p_curve[0], right=p_curve[-1])
    return np.clip(cf, 0.0, 1.0)

def tidal_cf_from_speed(speed: np.ndarray) -> np.ndarray:
    """
    Convert tidal speed to CF:
      - 0 below cut-in
      - cubic ramp to rated
      - 1.0 from rated to cut-out
      - 0 above cut-out
    """
    v = np.asarray(speed, dtype=float)
    cf = np.zeros_like(v, dtype=np.float32)
    v0, vr, vc = TIDAL_CUT_IN, TIDAL_RATED, TIDAL_CUT_OUT
    ramp = (v >= v0) & (v < vr)
    cf[ramp] = ((v[ramp] ** 3) - (v0 ** 3)) / ((vr ** 3) - (v0 ** 3))
    rated = (v >= vr) & (v <= vc)
    cf[rated] = 1.0
    cf[v > vc] = 0.0
    return np.clip(cf, 0.0, 1.0)

def exclude_near_existing_sites(
    candidates: pd.DataFrame,
    existing_csv: Path,
    exclude_km: float,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Remove candidate points that fall within exclude_km of known existing sites.
    Uses EPSG:27700 (meters) to buffer accurately.
    """
    if not existing_csv or str(existing_csv).strip() == "" or (not existing_csv.exists()):
        print(f"[exclude] No existing sites file found at: {existing_csv} (skipping exclusion)")
        return candidates

    existing = pd.read_csv(existing_csv)
    if lat_col not in existing.columns or lon_col not in existing.columns:
        raise ValueError(f"Existing sites CSV must contain columns '{lat_col}' and '{lon_col}'.")

    cand_gdf = gpd.GeoDataFrame(
        candidates.copy(),
        geometry=gpd.points_from_xy(candidates["lon"], candidates["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:27700")

    ex_gdf = gpd.GeoDataFrame(
        existing.copy(),
        geometry=gpd.points_from_xy(existing[lon_col], existing[lat_col]),
        crs="EPSG:4326",
    ).to_crs("EPSG:27700")

    ex_union_buffer = ex_gdf.buffer(exclude_km * 1000.0).union_all()
    keep_mask = ~cand_gdf.geometry.within(ex_union_buffer)

    removed = int((~keep_mask).sum())
    kept = int(keep_mask.sum())
    print(f"[exclude] Removed {removed} candidates within {exclude_km:.2f} km. Kept {kept}.")

    return cand_gdf.loc[keep_mask].drop(columns=["geometry"]).copy()

def load_demand_auto(csv_path: Path, snapshots: pd.DatetimeIndex) -> pd.Series:
    """
    Load demand data and align to optimisation snapshots.
    Supports:
      - NGESO settlement format (half-hour) -> resampled to hourly
      - 2-column format: (time, demand_MW)  -> hourly_clean csv's files
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Demand CSV not found: {csv_path}")

    df0 = pd.read_csv(csv_path, nrows=5)
    cols = {c.strip().replace("\ufeff", "") for c in df0.columns}

    if {"SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"}.issubset(cols):
        df = pd.read_csv(csv_path, usecols=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"])
        date = pd.to_datetime(df["SETTLEMENT_DATE"], format="%d-%b-%y", errors="coerce")
        if date.isna().any():
            date2 = pd.to_datetime(df.loc[date.isna(), "SETTLEMENT_DATE"], dayfirst=True, errors="coerce")
            date.loc[date.isna()] = date2
        if date.isna().any():
            bad = df.loc[date.isna(), "SETTLEMENT_DATE"].astype(str).unique()[:10]
            raise ValueError(f"Could not parse some SETTLEMENT_DATE values. Examples: {list(bad)}")

        sp = pd.to_numeric(df["SETTLEMENT_PERIOD"], errors="coerce").astype("Int64")
        if sp.isna().any():
            raise ValueError("Some SETTLEMENT_PERIOD values are not numeric.")
        sp = sp.astype(int)

        nd = pd.to_numeric(df["ND"], errors="coerce")
        if nd.isna().any():
            raise ValueError("Some ND values are not numeric.")

        t_half = date + pd.to_timedelta((sp - 1) * 30, unit="min")
        s_half = pd.Series(nd.to_numpy(float), index=pd.DatetimeIndex(t_half)).sort_index()
        s_hour = s_half.resample("1h").mean()

    elif {"time", "demand_MW"}.issubset(cols):
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

        time_str = (
            df["time"].astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("\u00a0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        t = pd.to_datetime(time_str, errors="coerce", dayfirst=True)
        if t.isna().any():
            t2 = pd.to_datetime(time_str[t.isna()], format="%d/%m/%Y %H:%M", errors="coerce")
            t.loc[t.isna()] = t2
        if t.isna().any():
            t3 = pd.to_datetime(time_str[t.isna()], format="%Y-%m-%d %H:%M:%S", errors="coerce")
            t.loc[t.isna()] = t3
        if t.isna().any():
            bad = time_str[t.isna()].unique()[:10]
            raise ValueError(f"Unparseable timestamps in two-col demand CSV. Examples: {list(bad)}")

        demand = (
            df["demand_MW"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        demand = pd.to_numeric(demand, errors="coerce")

        out = pd.DataFrame({"demand_MW": demand.values}, index=pd.DatetimeIndex(t)).dropna()
        out = out[~out.index.duplicated(keep="first")].sort_index()
        s_hour = out["demand_MW"].resample("1h").mean()
    else:
        raise ValueError(
            "Demand CSV format not recognised.\n"
            "Provide either NGESO (SETTLEMENT_DATE, SETTLEMENT_PERIOD, ND) or (time, demand_MW)."
        )

    # Align to snapshots window, then reindex to snapshots
    start = snapshots[0]
    end = snapshots[-1] + pd.Timedelta(hours=1)
    s_win = s_hour.loc[(s_hour.index >= start) & (s_hour.index <= end)]
    load = s_win.reindex(snapshots)

    # If no overlap, fall back to positional remapping
    if load.isna().all():
        print("[load] Demand does not overlap snapshots (likely year mismatch). Remapping by position.")
        vals = s_hour.dropna().to_numpy()
        if len(vals) < len(snapshots):
            raise ValueError("Not enough demand data to remap by position.")
        load = pd.Series(vals[: len(snapshots)], index=snapshots)

    # Small gap filling for stability is needed
    load = load.interpolate(limit=2).bfill().ffill()
    if load.isna().any():
        raise ValueError("Load still contains NaNs after alignment.")

    print("\nDemand stats over model period:")
    print("  min:", float(load.min()), "max:", float(load.max()), "mean:", float(load.mean()))
    return load

def load_bathymetry_uk(nc_path: Path) -> xr.Dataset:
    """Load bathymetry NetCDF and validate expected variable exists."""
    if not nc_path.exists():
        raise FileNotFoundError(f"Bathymetry file not found: {nc_path}")
    ds = xr.open_dataset(nc_path)
    if BATHY_VAR not in ds.variables and BATHY_VAR not in ds.data_vars:
        raise ValueError(f"Bathymetry variable '{BATHY_VAR}' not found. Available: {list(ds.data_vars)}")
    return ds

def bathy_depth_at_points(bathy: xr.Dataset, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbour bathymetry lookup for candidate points.
    Converts elevation to positive-down depth if required.
    """
    latv = np.asarray(bathy[BATHY_LAT_NAME].values, dtype=float)
    lonv = np.asarray(bathy[BATHY_LON_NAME].values, dtype=float)
    z = bathy[BATHY_VAR]

    # Handle datasets stored in [0,360] longitudes
    lons_use = np.asarray(lons, dtype=float).copy()
    lonv_min = float(np.nanmin(lonv))
    lonv_max = float(np.nanmax(lonv))
    if lonv_min >= 0.0 and lonv_max > 180.0:
        lons_use = np.where(lons_use < 0.0, lons_use + 360.0, lons_use)

    lats = np.asarray(lats, dtype=float)
    ilat = np.argmin(np.abs(latv[:, None] - lats[None, :]), axis=0)
    ilon = np.argmin(np.abs(lonv[:, None] - lons_use[None, :]), axis=0)

    vals = z.values[ilat, ilon].astype(float)

    depth = vals if BATHY_POSITIVE_DOWN else -vals
    depth = np.asarray(depth, dtype=float)
    depth[~np.isfinite(depth)] = np.nan
    depth[depth <= 0.0] = np.nan
    return depth

def fast_candidate_mask(
    lat1d: np.ndarray,
    lon1d: np.ndarray,
    eez_geom_wgs84,
    land_geom_wgs84,
    max_dist_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast vectorised candidate screening:
      - Create all grid points from (lat1d, lon1d)
      - Keep points within EEZ and not on land (vectorised shapely)
      - Compute distance to land (EPSG:27700) and keep <= max_dist_km
    Returns:
      cand_ij: indices into lat2d/lon2d of kept candidates
      lat2d/lon2d: meshgrids
      dist_km: distance-to-land for kept candidates
    """
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    ny, nx = lat2d.shape

    xs = lon2d.ravel()
    ys = lat2d.ravel()

    pts_wgs = shapely.points(xs, ys)
    inside_eez = shapely.within(pts_wgs, eez_geom_wgs84)
    on_land = shapely.within(pts_wgs, land_geom_wgs84)
    keep = inside_eez & (~on_land)

    pts_27700 = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs="EPSG:4326").to_crs("EPSG:27700")
    land_27700 = (
        gpd.GeoSeries([land_geom_wgs84], crs="EPSG:4326")
        .to_crs("EPSG:27700")
        .to_numpy(dtype=object)[0]
    )
    dist_m = pts_27700.distance(land_27700).to_numpy(dtype=float)

    keep = keep & (dist_m <= max_dist_km * 1000.0)

    mask = keep.reshape(ny, nx)
    cand_ij = np.argwhere(mask)

    dist_km = (dist_m.reshape(ny, nx))[cand_ij[:, 0], cand_ij[:, 1]] / 1000.0
    return cand_ij, lat2d, lon2d, dist_km


# ============================================================
# TIDAL clustering (max 20 cells per cluster)
# Greedy: high CF first, assign to nearest eligible cluster if within radius and size < max_cells
# ============================================================

def cluster_tidal_candidates_greedy(
    candidates: pd.DataFrame,
    cf_df: pd.DataFrame,
    cluster_km: float,
    max_cells: int,
    prefix: str = "tidal",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster tidal cells into sites:
      - Sort cells by cf_mean descending
      - For each unassigned cell: attach to nearest cluster within cluster_km if cluster not full
      - Else start a new cluster
    Outputs:
      - candidates dataframe with one row per cluster/site
      - cf dataframe with one CF time series per cluster/site (mean of members)
    """
    if len(candidates) == 0:
        return candidates, cf_df

    cand = candidates.reset_index(drop=True).copy()
    gens = cand["gen"].tolist()

    order = np.argsort(-cand["cf_mean"].to_numpy(dtype=float))
    assigned = np.full(len(cand), -1, dtype=int)

    clusters: List[List[int]] = []
    cluster_centroids: List[tuple[float, float]] = []

    lat = cand["lat"].to_numpy(dtype=float)
    lon = cand["lon"].to_numpy(dtype=float)

    for idx in order:
        if assigned[idx] != -1:
            continue

        best_c = -1
        best_d = 1e9

        for c_id, members in enumerate(clusters):
            if len(members) >= max_cells:
                continue
            clat, clon = cluster_centroids[c_id]
            d = float(haversine_km(lat[idx], lon[idx], clat, clon))
            if d <= cluster_km and d < best_d:
                best_d = d
                best_c = c_id

        if best_c == -1:
            clusters.append([int(idx)])
            assigned[idx] = len(clusters) - 1
            cluster_centroids.append((float(lat[idx]), float(lon[idx])))
        else:
            clusters[best_c].append(int(idx))
            assigned[idx] = best_c
            mem = clusters[best_c]
            cluster_centroids[best_c] = (float(np.mean(lat[mem])), float(np.mean(lon[mem])))

    new_rows = []
    new_cf_cols: Dict[str, pd.Series] = {}

    for k, members in enumerate(clusters):
        member_gens = [gens[i] for i in members]

        lat_c = float(np.mean(lat[members]))
        lon_c = float(np.mean(lon[members]))

        dist_c = float(np.mean(cand.iloc[members]["dist_km"].to_numpy(dtype=float)))
        cf_mean_c = float(np.mean(cand.iloc[members]["cf_mean"].to_numpy(dtype=float)))
        depth_c = float(np.nanmean(cand.iloc[members].get("depth_m", pd.Series(np.nan)).to_numpy(dtype=float)))

        new_gen = f"{prefix}_site_{k:05d}"

        new_cf = cf_df[member_gens].mean(axis=1).clip(0.0, 1.0)
        new_cf_cols[new_gen] = new_cf

        new_rows.append({
            "gen": new_gen,
            "lat": lat_c,
            "lon": lon_c,
            "dist_km": dist_c,
            "cf_mean": cf_mean_c,
            "depth_m": depth_c,
            "n_cells": int(len(members)),
        })

    cand_out = pd.DataFrame(new_rows).sort_values("cf_mean", ascending=False).reset_index(drop=True)
    cf_out = pd.DataFrame(new_cf_cols, index=cf_df.index)
    return cand_out, cf_out

def add_exact_build_constraint(n: pypsa.Network, gens: list[str], total_mw: float, cname: str):
    """
    Add equality constraint sum(p_nom[gens]) == total_mw.
    Uses PyPSA/Linopy model variable "Generator-p_nom".
    """
    m = n.model
    var = m.variables["Generator-p_nom"]
    gen_dim = "name" if "name" in var.dims else var.dims[0]
    available = set(var.indexes[gen_dim])
    gens = [g for g in gens if g in available]
    if not gens:
        raise ValueError(f"No generator names matched in model for constraint '{cname}'.")
    expr = var.sel({gen_dim: gens}).sum(dim=gen_dim)
    m.add_constraints(expr == float(total_mw), name=cname)

def extract_built_table(n: pypsa.Network, prefix: str) -> pd.DataFrame:
    """Return table of built generators (p_nom_opt > 0) matching a name prefix."""
    gens = n.generators.copy()
    gens["p_nom_opt"] = gens.get("p_nom_opt", 0.0)
    gens["p_nom_opt"] = gens["p_nom_opt"].fillna(0.0).clip(lower=0.0)
    built = gens[gens["p_nom_opt"] > 1e-6].copy()
    built = built.reset_index().rename(columns={"name": "gen"})
    return built[built["gen"].str.startswith(prefix)].copy()


# ============================================================
# PLOTTING: marker size by built capacity (p_nom_opt_MW), separate scales, 5 bins
# ============================================================

def _bins_equal_width(vmin: float, vmax: float, n_bins: int = 5) -> np.ndarray:
    """Helper for equal-width bin edges in MW."""
    if vmax <= vmin:
        return np.linspace(vmin, vmin + 1.0, n_bins + 1)
    return np.linspace(vmin, vmax, n_bins + 1)

def size_bins_and_labels(vmin: float, vmax: float, n_bins: int = 5) -> tuple[np.ndarray, List[str]]:
    """Legend bin edges and labels as 'a–b MW'."""
    edges = _bins_equal_width(vmin, vmax, n_bins=n_bins)
    labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f} MW" for i in range(len(edges) - 1)]
    return edges, labels

def marker_sizes_from_capacity(
    p_mw: np.ndarray,
    vmin: float,
    vmax: float,
    s_min: float = 20.0,
    s_max: float = 220.0,
) -> np.ndarray:
    """
    Visual-only size scaling for map markers:
      - normalise p between vmin/vmax
      - sqrt scaling (compress high values)
      - map to [s_min, s_max]
    """
    p = np.asarray(p_mw, dtype=float)
    if len(p) == 0:
        return np.array([])
    denom = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0
    x = (p - vmin) / denom
    x = np.clip(x, 0.0, 1.0)
    return s_min + (s_max - s_min) * np.sqrt(x)

def add_capacity_size_legend(ax, tech: str, vmin: float, vmax: float, color: str, loc: str):
    """Add marker-size legend for one technology."""
    edges, labels = size_bins_and_labels(vmin, vmax, n_bins=5)
    mids = 0.5 * (edges[:-1] + edges[1:])
    handles = []
    for mid, lab in zip(mids, labels):
        s = float(marker_sizes_from_capacity(np.array([mid]), vmin, vmax)[0])
        handles.append(ax.scatter([], [], s=s, c=color, edgecolors="black", linewidths=0.6, label=lab))
    leg = ax.legend(handles=handles, title=f"{tech} site size (built MW)", loc=loc, frameon=True)
    return leg

def plot_joint_map(
    land_gdf,
    eez_gdf,
    wind_df: pd.DataFrame,
    tidal_df: pd.DataFrame,
    outfile: Path,
    title: str,
):
    """Joint plot: wind + tidal with separate size legends."""
    fig, ax = plt.subplots(figsize=(10, 12))
    eez_gdf.boundary.plot(ax=ax, linewidth=1.2, color="black", zorder=1)
    land_gdf.plot(ax=ax, color="#dddddd", edgecolor="black", linewidth=0.5, zorder=2)

    WIND_VMIN, WIND_VMAX = 0.0, float(P_NOM_MAX_PER_SITE_MW_WIND)
    TIDAL_VMIN, TIDAL_VMAX = 0.0, float(TIDAL_CELL_CAPACITY_MW * TIDAL_CLUSTER_MAX_CELLS)

    if isinstance(wind_df, pd.DataFrame) and len(wind_df) > 0:
        gw = gpd.GeoDataFrame(
            wind_df.copy(),
            geometry=gpd.points_from_xy(wind_df["lon"], wind_df["lat"]),
            crs="EPSG:4326",
        )
        s_w = marker_sizes_from_capacity(wind_df["p_nom_opt_MW"].to_numpy(float), WIND_VMIN, WIND_VMAX)
        gw.plot(ax=ax, marker="o", markersize=s_w, color="green", edgecolor="black",
                linewidth=0.5, alpha=0.9, zorder=10)

    if isinstance(tidal_df, pd.DataFrame) and len(tidal_df) > 0:
        gt = gpd.GeoDataFrame(
            tidal_df.copy(),
            geometry=gpd.points_from_xy(tidal_df["lon"], tidal_df["lat"]),
            crs="EPSG:4326",
        )
        s_t = marker_sizes_from_capacity(tidal_df["p_nom_opt_MW"].to_numpy(float), TIDAL_VMIN, TIDAL_VMAX)
        gt.plot(ax=ax, marker="o", markersize=s_t, color="red", edgecolor="black",
                linewidth=0.5, alpha=0.9, zorder=11)

    tech_handles = [
        Line2D([0], [0], marker="o", color="w", label="Wind", markerfacecolor="green", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="Tidal", markerfacecolor="red", markeredgecolor="black", markersize=10),
    ]
    leg1 = ax.legend(handles=tech_handles, title="Technology", loc="upper left", frameon=True)
    ax.add_artist(leg1)

    # Size legends separated to avoid overlap
    leg_w = add_capacity_size_legend(ax, "Wind", WIND_VMIN, WIND_VMAX, color="green", loc="upper right")
    ax.add_artist(leg_w)
    leg_t = add_capacity_size_legend(ax, "Tidal", TIDAL_VMIN, TIDAL_VMAX, color="red", loc="lower right")
    ax.add_artist(leg_t)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-8, 7)
    ax.set_ylim(49, 62)
    ax.grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.show()
    plt.close(fig)

def plot_single_tech_map(
    land_gdf,
    eez_gdf,
    df: pd.DataFrame,
    outfile: Path,
    title: str,
    tech: str,
    color: str,
    vmin: float,
    vmax: float,
):
    """Single-tech plot with size legend."""
    fig, ax = plt.subplots(figsize=(10, 12))
    eez_gdf.boundary.plot(ax=ax, linewidth=1.2, color="black", zorder=1)
    land_gdf.plot(ax=ax, color="#dddddd", edgecolor="black", linewidth=0.5, zorder=2)

    if isinstance(df, pd.DataFrame) and len(df) > 0:
        g = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
        s = marker_sizes_from_capacity(df["p_nom_opt_MW"].to_numpy(float), vmin=vmin, vmax=vmax)
        g.plot(ax=ax, marker="o", markersize=s, color=color, edgecolor="black", linewidth=0.5, alpha=0.9, zorder=10)

        leg1 = ax.legend(
            handles=[Line2D([0], [0], marker="o", color="w", label=tech,
                            markerfacecolor=color, markeredgecolor="black", markersize=10)],
            title="Technology",
            loc="upper left",
            frameon=True,
        )
        ax.add_artist(leg1)

        _ = add_capacity_size_legend(ax, tech, vmin, vmax, color=color, loc="upper right")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-8, 7)
    ax.set_ylim(49, 62)
    ax.grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.show()
    plt.close(fig)


# ============================================================
# 2) LOAD SHAPEFILES + BATHYMETRY
# ============================================================

# Ensure required shapefiles exist before loading
for shp in (UK_LAND_SHP, UK_EEZ_SHP):
    if not shp.exists():
        raise FileNotFoundError(f"Missing shapefile: {shp}")

# Load land + EEZ polygons (WGS84), dissolve for faster spatial checks
land = ensure_wgs84(gpd.read_file(UK_LAND_SHP))
eez = ensure_wgs84(gpd.read_file(UK_EEZ_SHP))
land_geom = dissolve_to_single_geometry(land)
eez_geom = dissolve_to_single_geometry(eez)

# Load bathymetry for depth screening
bathy = load_bathymetry_uk(BATHY_NETCDF)

# Scenario flags
RUN_WIND = SCENARIO in ("wind_only", "joint")
RUN_TIDAL = SCENARIO in ("tidal_only", "joint")


# ============================================================
# 3) PREPARE WIND CANDIDATES  (NO CLUSTERING)
# ============================================================

wind_candidates = None
wind_cf_df = None
wind_snapshots = None

if RUN_WIND:
    # Validate required inputs
    for p in (ERA5_WIND_FILE, POWER_CURVE_CSV, DEMAND_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # Load ERA5 wind dataset
    ds = xr.open_dataset(ERA5_WIND_FILE)

    # Normalise variable names across ERA5 exports
    rename_map = {}
    if "10m_u_component_of_wind" in ds: rename_map["10m_u_component_of_wind"] = "u10"
    if "10m_v_component_of_wind" in ds: rename_map["10m_v_component_of_wind"] = "v10"
    if "100m_u_component_of_wind" in ds: rename_map["100m_u_component_of_wind"] = "u100"
    if "100m_v_component_of_wind" in ds: rename_map["100m_v_component_of_wind"] = "v100"
    if rename_map:
        ds = ds.rename(rename_map)

    # Identify time/lat/lon coordinate names
    time_dim, lat_name, lon_name = detect_time_lat_lon(ds)
    wind_snapshots = pd.DatetimeIndex(ds[time_dim].values)

    # Prefer 100m winds if present, else use 10m and shear to hub height
    if "u100" in ds and "v100" in ds:
        u = ds["u100"]; v = ds["v100"]; ref_h = 100.0
    elif "u10" in ds and "v10" in ds:
        u = ds["u10"]; v = ds["v10"]; ref_h = 10.0
    else:
        raise ValueError("ERA5 dataset missing u/v wind components.")

    # Ensure consistent dimension order: (time, lat, lon)
    u = u.transpose(time_dim, lat_name, lon_name)
    v = v.transpose(time_dim, lat_name, lon_name)

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    # Candidate screening (EEZ, not land, within max distance to shore)
    cand_ij, lat2d, lon2d, dist_km = fast_candidate_mask(lat, lon, eez_geom, land_geom, MAX_DIST_KM_WIND)
    print(f"[wind] candidates after EEZ/land/dist filtering: {cand_ij.shape[0]} cells")

    # Transmission efficiency per candidate (distance-based)
    eta = trans_eff_from_distance_km(dist_km)

    # Compute wind CF grid from metocean u/v and turbine power curve
    v_curve, p_curve = load_power_curve(POWER_CURVE_CSV)
    cf_grid = wind_cf_from_u_v(u, v, v_curve, p_curve, HUB_HEIGHT_M, ref_h)

    # Build CF time series matrix for candidates (apply transmission losses)
    T = len(wind_snapshots)
    N = cand_ij.shape[0]
    cf_arr = np.empty((T, N), dtype=np.float32)
    for k, (iy, ix) in enumerate(cand_ij):
        cf_arr[:, k] = cf_grid[:, iy, ix] * eta[k]

    gen_names = [f"wind_site_{k:05d}" for k in range(N)]
    wind_cf_df = pd.DataFrame(cf_arr, index=wind_snapshots, columns=gen_names).clip(0.0, 1.0)

    # Candidate metadata table (lat/lon, distance, mean CF)
    wind_candidates = pd.DataFrame({
        "gen": gen_names,
        "lat": lat2d[cand_ij[:, 0], cand_ij[:, 1]],
        "lon": lon2d[cand_ij[:, 0], cand_ij[:, 1]],
        "dist_km": dist_km,
        "cf_mean": wind_cf_df.mean(axis=0).to_numpy(),
    })

    # Depth lookup + depth-band filter + min distance from shore
    depth_m = bathy_depth_at_points(bathy, wind_candidates["lat"].to_numpy(), wind_candidates["lon"].to_numpy())
    wind_candidates["depth_m"] = depth_m

    keep = np.isfinite(depth_m) & (depth_m >= WIND_MIN_DEPTH_M) & (depth_m <= WIND_MAX_DEPTH_M)
    keep = keep & (wind_candidates["dist_km"].to_numpy(dtype=float) >= MIN_DIST_KM_WIND)

    wind_candidates = wind_candidates.loc[keep].reset_index(drop=True)
    wind_cf_df = wind_cf_df[wind_candidates["gen"].tolist()].copy()
    print(f"[wind] kept after depth+min-dist: {len(wind_candidates)}")

    # Exclude near existing wind sites
    wind_candidates = exclude_near_existing_sites(wind_candidates, EXISTING_WIND_CSV, EXCLUDE_KM_WIND)
    wind_cf_df = wind_cf_df[wind_candidates["gen"].tolist()].copy()

    # Keep top-N by mean CF
    wind_candidates = wind_candidates.sort_values("cf_mean", ascending=False).head(TOP_N_WIND).reset_index(drop=True)
    wind_cf_df = wind_cf_df[wind_candidates["gen"].tolist()].copy()
    print(f"[wind] reduced to TOP_N_WIND={len(wind_candidates)}")


# ============================================================
# 4) PREPARE TIDAL CANDIDATES  (CLUSTERING ON, <= 20 cells)
# ============================================================

tidal_candidates = None
tidal_cf_df = None
tidal_snapshots = None

if RUN_TIDAL:
    # Validate required inputs
    for p in (TIDAL_CURRENTS_NETCDF, DEMAND_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # Load CMEMS tidal currents
    ds = xr.open_dataset(TIDAL_CURRENTS_NETCDF)
    if "uo" not in ds.variables or "vo" not in ds.variables:
        raise ValueError(f"CMEMS file must contain 'uo' and 'vo'. Found: {list(ds.variables)[:30]}")

    # Identify time/lat/lon coordinate names
    time_dim, lat_name, lon_name = detect_time_lat_lon(ds)
    tidal_snapshots = pd.DatetimeIndex(ds[time_dim].values)

    uo = ds["uo"]
    vo = ds["vo"]

    # If there is a depth dimension, select the first depth level (surface/upper layer)
    depth_dim = None
    for d in uo.dims:
        dlow = str(d).lower()
        if ("depth" in dlow) or ("lev" in dlow) or (dlow == "z"):
            depth_dim = str(d)
            break
    if depth_dim is not None:
        uo = uo.isel({depth_dim: 0})
        vo = vo.isel({depth_dim: 0})

    # Ensure consistent dimension order: (time, lat, lon)
    uo = uo.transpose(time_dim, lat_name, lon_name)
    vo = vo.transpose(time_dim, lat_name, lon_name)

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    # Candidate screening (EEZ, not land, within max distance to shore)
    cand_ij, lat2d, lon2d, dist_km = fast_candidate_mask(lat, lon, eez_geom, land_geom, MAX_DIST_KM_TIDAL)
    print(f"[tidal] candidates after EEZ/land/dist filtering: {cand_ij.shape[0]} cells")

    # Transmission efficiency per candidate (distance-based)
    eta = trans_eff_from_distance_km(dist_km)

    # Compute speed and CF grid
    speed = np.hypot(uo.values, vo.values)
    cf_grid = tidal_cf_from_speed(speed)

    # Build CF time series matrix for candidates (apply transmission losses)
    T = len(tidal_snapshots)
    N = cand_ij.shape[0]
    cf_arr = np.empty((T, N), dtype=np.float32)
    for k, (iy, ix) in enumerate(cand_ij):
        cf_arr[:, k] = cf_grid[:, iy, ix] * eta[k]

    gen_names = [f"tidal_cell_{k:05d}" for k in range(N)]
    tidal_cf_df = pd.DataFrame(cf_arr, index=tidal_snapshots, columns=gen_names).clip(0.0, 1.0)

    # Candidate metadata table (lat/lon, distance, mean CF)
    tidal_candidates = pd.DataFrame({
        "gen": gen_names,
        "lat": lat2d[cand_ij[:, 0], cand_ij[:, 1]],
        "lon": lon2d[cand_ij[:, 0], cand_ij[:, 1]],
        "dist_km": dist_km,
        "cf_mean": tidal_cf_df.mean(axis=0).to_numpy(),
    })

    # Depth lookup + depth-band filter + min distance from shore
    depth_m = bathy_depth_at_points(bathy, tidal_candidates["lat"].to_numpy(), tidal_candidates["lon"].to_numpy())
    tidal_candidates["depth_m"] = depth_m

    keep = np.isfinite(depth_m) & (depth_m >= TIDAL_MIN_DEPTH_M) & (depth_m <= TIDAL_MAX_DEPTH_M)
    keep = keep & (tidal_candidates["dist_km"].to_numpy(dtype=float) >= MIN_DIST_KM_TIDAL)

    tidal_candidates = tidal_candidates.loc[keep].reset_index(drop=True)
    tidal_cf_df = tidal_cf_df[tidal_candidates["gen"].tolist()].copy()
    print(f"[tidal] kept after depth+min-dist: {len(tidal_candidates)}")

    # Exclude near existing tidal sites
    tidal_candidates = exclude_near_existing_sites(tidal_candidates, EXISTING_TIDAL_CSV, EXCLUDE_KM_TIDAL)
    tidal_cf_df = tidal_cf_df[tidal_candidates["gen"].tolist()].copy()

    # Keep top-N by mean CF
    tidal_candidates = tidal_candidates.sort_values("cf_mean", ascending=False).head(TOP_N_TIDAL).reset_index(drop=True)
    tidal_cf_df = tidal_cf_df[tidal_candidates["gen"].tolist()].copy()
    print(f"[tidal] reduced to TOP_N_TIDAL={len(tidal_candidates)}")

    # Cluster top candidates into sites (cap cluster size)
    tidal_candidates, tidal_cf_df = cluster_tidal_candidates_greedy(
        tidal_candidates,
        tidal_cf_df,
        cluster_km=float(CLUSTER_KM_TIDAL),
        max_cells=int(TIDAL_CLUSTER_MAX_CELLS),
        prefix="tidal",
    )
    print(f"[tidal] clustered into {len(tidal_candidates)} sites (cluster_km={CLUSTER_KM_TIDAL}, max_cells={TIDAL_CLUSTER_MAX_CELLS})")


# ============================================================
# 5) JOINT TIME ALIGNMENT
# ============================================================

def get_solve_snapshots() -> pd.DatetimeIndex:
    """
    Determine the snapshot index used for optimisation:
      - wind_only: wind timestamps
      - tidal_only: tidal timestamps
      - joint: intersection of wind and tidal timestamps
    """
    if SCENARIO == "wind_only":
        if wind_snapshots is None:
            raise RuntimeError("Wind snapshots missing.")
        return wind_snapshots
    if SCENARIO == "tidal_only":
        if tidal_snapshots is None:
            raise RuntimeError("Tidal snapshots missing.")
        return tidal_snapshots

    if wind_snapshots is None or tidal_snapshots is None:
        raise RuntimeError("Joint requires both wind and tidal snapshots.")
    idx = wind_snapshots.intersection(tidal_snapshots)
    if len(idx) == 0:
        raise RuntimeError("No overlapping timestamps between wind and tidal datasets for joint run.")
    return idx

# Solve snapshots and aligned demand
snapshots = get_solve_snapshots()
load = load_demand_auto(DEMAND_CSV, snapshots)

# Align CF time series to solve snapshots and fill small gaps (interpolation)
if wind_cf_df is not None:
    wind_cf_df = wind_cf_df.reindex(snapshots).interpolate(limit=2).bfill().ffill().clip(0.0, 1.0)
if tidal_cf_df is not None:
    tidal_cf_df = tidal_cf_df.reindex(snapshots).interpolate(limit=2).bfill().ffill().clip(0.0, 1.0)


# ============================================================
# 6) SOLVE NETWORK
# ============================================================

def solve_network(
    snapshots: pd.DatetimeIndex,
    load: pd.Series,
    wind_cand: Optional[pd.DataFrame],
    wind_cf: Optional[pd.DataFrame],
    tidal_cand: Optional[pd.DataFrame],
    tidal_cf: Optional[pd.DataFrame],
) -> pypsa.Network:
    """
    Build and solve a single-bus PyPSA network:
      - Load is fixed (p_set)
      - Wind/tidal generators are extendable with p_max_pu = CF time series
      - Backup generator is extendable with high marginal cost (ignored for optimisation)
      - Optional diversification "bonus" is a generator with negative capital_cost and p_max_pu=0
      - Storage is added for tidal_only and joint if enabled
      - Exact build constraints are enforced via extra_functionality (linopy constraints)
    """
    n = pypsa.Network()
    n.set_snapshots(list(snapshots))

    # ------------------------
    # Carriers
    # ------------------------
    for c in ["electricity", "wind", "tidal", "backup", "storage", "diversification"]:
        if c not in n.carriers.index:
            n.add("Carrier", c)

    # ------------------------
    # Bus + Load
    # ------------------------
    n.add("Bus", "bus", carrier="electricity")
    n.add("Load", "load", bus="bus", p_set=load)

    # ------------------------
    # Backup (dispatchable)
    # ------------------------
    n.add(
        "Generator", "backup",
        bus="bus",
        carrier="backup",
        p_nom_extendable=True,
        p_nom_max=float(BACKUP_P_NOM_MAX_MW),
        marginal_cost=float(MARGINAL_BACKUP_GBP_PER_MWH),
        capital_cost=float(CAPEX_BACKUP_GBP_PER_MW_ANNUAL),
    )

    wind_names: list[str] = []
    tidal_names: list[str] = []

    # ------------------------
    # Wind generators (candidate sites)
    # ------------------------
    if wind_cand is not None and wind_cf is not None and len(wind_cand) > 0:
        wind_names = wind_cand["gen"].tolist()
        wind_capex = capex_with_cable(
            CAPEX_WIND_GBP_PER_MW_ANNUAL,
            wind_cand["dist_km"].to_numpy(dtype=float),
        )
        for g, cap in zip(wind_names, wind_capex):
            n.add(
                "Generator", g,
                bus="bus",
                carrier="wind",
                p_nom_extendable=True,
                p_nom_max=float(P_NOM_MAX_PER_SITE_MW_WIND),
                marginal_cost=float(MARGINAL_WIND_GBP_PER_MWH),
                capital_cost=float(cap),
            )

    # ------------------------
    # Tidal generators (clustered sites)
    # ------------------------
    if tidal_cand is not None and tidal_cf is not None and len(tidal_cand) > 0:
        tidal_names = tidal_cand["gen"].tolist()
        tidal_capex = capex_with_cable(
            CAPEX_TIDAL_GBP_PER_MW_ANNUAL,
            tidal_cand["dist_km"].to_numpy(dtype=float),
        )

        # Per-site p_nom_max is proportional to number of cells in cluster
        n_cells = tidal_cand["n_cells"].to_numpy(dtype=float)
        tidal_pmax = TIDAL_CELL_CAPACITY_MW * n_cells

        for g, cap, pmax_site in zip(tidal_names, tidal_capex, tidal_pmax):
            n.add(
                "Generator", g,
                bus="bus",
                carrier="tidal",
                p_nom_extendable=True,
                p_nom_max=float(pmax_site),
                marginal_cost=float(MARGINAL_TIDAL_GBP_PER_MWH),
                capital_cost=float(cap),
            )

    # ------------------------
    # Diversification "discount" (capacity-only, no energy)
    # Implemented as a generator with:
    #   - negative capital_cost
    #   - p_max_pu forced to 0.0 (cannot generate)
    #   - constraints later cap it by wind and tidal build
    #   - Give 'direct' incentive structure for mixed renewables
    # ------------------------
    if SCENARIO == "joint" and (wind_names and tidal_names):
        n.add(
            "Generator", "diversification_bonus",
            bus="bus",
            carrier="diversification",
            p_nom=0.0,
            p_nom_extendable=True,
            p_nom_max=float(TARGET_BUILD_MW_TOTAL_JOINT),
            capital_cost=-float(DIVERSIFICATION_BONUS_GBP_PER_MW),
            marginal_cost=0.0,
        )

    # ------------------------
    # Storage for tidal_only AND joint
    # ------------------------
    if (SCENARIO in ("tidal_only", "joint")) and ADD_TIDAL_STORAGE and tidal_names:
        n.add(
            "StorageUnit", "tidal_storage",
            bus="bus",
            carrier="storage",
            p_nom_extendable=True,
            max_hours=float(STORAGE_MAX_HOURS),
            efficiency_store=float(STORAGE_EFF_STORE),
            efficiency_dispatch=float(STORAGE_EFF_DISPATCH),
            capital_cost=float(STORAGE_CAPEX_GBP_PER_MW_ANNUAL),
            marginal_cost=float(STORAGE_MARGINAL_GBP_PER_MWH),
        )

    # ============================================================
    # Ensure generators_t.p_max_pu exists and matches snapshots/generators
    # ============================================================
    if (
        getattr(n.generators_t, "p_max_pu", None) is None
        or not isinstance(n.generators_t.p_max_pu, pd.DataFrame)
        or n.generators_t.p_max_pu.empty
    ):
        n.generators_t.p_max_pu = pd.DataFrame(1.0, index=snapshots, columns=n.generators.index)
    else:
        n.generators_t.p_max_pu = n.generators_t.p_max_pu.reindex(
            index=snapshots,
            columns=n.generators.index,
            fill_value=1.0,
        )

    pmax = n.generators_t.p_max_pu
    assert isinstance(pmax, pd.DataFrame)

    # ------------------------
    # Assign CF limits (p_max_pu)
    # ------------------------
    if wind_names:
        assert wind_cf is not None
        pmax.loc[:, wind_names] = wind_cf.loc[:, wind_names].to_numpy()

    if tidal_names:
        assert tidal_cf is not None
        pmax.loc[:, tidal_names] = tidal_cf.loc[:, tidal_names].to_numpy()

    # diversification bonus can NEVER generate energy
    if "diversification_bonus" in n.generators.index:
        pmax.loc[:, "diversification_bonus"] = 0.0

    # ------------------------
    # Extra constraints (linopy)
    # ------------------------
    def extra_fn(n, sns):
        """
        Inject constraints into the optimisation model:
          - exact build targets:
              wind_only: sum(wind p_nom) == TARGET_BUILD_MW_WIND
              tidal_only: sum(tidal p_nom) == TARGET_BUILD_MW_TIDAL
              joint: sum(wind+tidal p_nom) == TARGET_BUILD_MW_TOTAL_JOINT
          - diversification bonus caps:
              bonus <= wind build
              bonus <= tidal build
              bonus >= 0
        """
        # --- build target ---
        if SCENARIO == "wind_only":
            add_exact_build_constraint(n, wind_names, TARGET_BUILD_MW_WIND, "build_exact_wind")
        elif SCENARIO == "tidal_only":
            add_exact_build_constraint(n, tidal_names, TARGET_BUILD_MW_TIDAL, "build_exact_tidal")
        else:
            # IMPORTANT: build target is ONLY wind + tidal (exclude diversification_bonus)
            add_exact_build_constraint(
                n,
                wind_names + tidal_names,
                TARGET_BUILD_MW_TOTAL_JOINT,
                "build_exact_total_joint",
            )

        # --- diversification caps: bonus <= wind_build and bonus <= tidal_build ---
        if "diversification_bonus" in n.generators.index:
            m = n.model
            var = m.variables["Generator-p_nom"]
            gen_dim = "name" if "name" in var.dims else var.dims[0]

            wind_expr  = var.sel({gen_dim: wind_names}).sum(dim=gen_dim)
            tidal_expr = var.sel({gen_dim: tidal_names}).sum(dim=gen_dim)
            bonus_expr = var.sel({gen_dim: ["diversification_bonus"]}).sum(dim=gen_dim)

            m.add_constraints(bonus_expr <= wind_expr,  name="bonus_le_wind_build")
            m.add_constraints(bonus_expr <= tidal_expr, name="bonus_le_tidal_build")
            m.add_constraints(bonus_expr >= 0.0,       name="bonus_ge_0")

    # Solve with HiGHS (linear optimisation)
    n.optimize(solver_name="highs", extra_functionality=extra_fn)
    return n

# Run the solve once for the selected scenario/candidates
n = solve_network(snapshots, load, wind_candidates, wind_cf_df, tidal_candidates, tidal_cf_df)

print("\n[solve] done.")
print("Total built MW:", float(n.generators["p_nom_opt"].fillna(0.0).sum()))


# ============================================================
# 7) EXPORT RESULTS + MAPS
# ============================================================

# Extract built wind and tidal site tables by name prefix
built_wind = extract_built_table(n, "wind_site_")
built_tidal = extract_built_table(n, "tidal_site_")

sel_wind = pd.DataFrame()
sel_tidal = pd.DataFrame()

# Wind export + (if not joint) wind-only map
if len(built_wind) > 0 and wind_candidates is not None:
    built_wind = built_wind.rename(columns={"p_nom_opt": "p_nom_opt_MW"})
    sel_wind = built_wind.merge(
        wind_candidates[["gen", "lat", "lon", "cf_mean", "dist_km", "depth_m"]],
        on="gen", how="left"
    ).sort_values("p_nom_opt_MW", ascending=False)

    out = OUT_WIND_SITES if SCENARIO != "joint" else OUT_JOINT_WIND
    sel_wind.to_csv(out, index=False)
    print(f"[wind] Saved: {out}")

    if SCENARIO != "joint":
        plot_single_tech_map(
            land, eez,
            sel_wind,
            OUTDIR / f"map_wind_{SCENARIO}.png",
            title=f"Optimised Wind Sites ({SCENARIO})",
            tech="Wind",
            color="green",
            vmin=0.0,
            vmax=float(P_NOM_MAX_PER_SITE_MW_WIND),
        )

# Tidal export + (if not joint) tidal-only map
if len(built_tidal) > 0 and tidal_candidates is not None:
    built_tidal = built_tidal.rename(columns={"p_nom_opt": "p_nom_opt_MW"})
    sel_tidal = built_tidal.merge(
        tidal_candidates[["gen", "lat", "lon", "cf_mean", "dist_km", "depth_m", "n_cells"]],
        on="gen", how="left"
    ).sort_values("p_nom_opt_MW", ascending=False)

    out = OUT_TIDAL_SITES if SCENARIO != "joint" else OUT_JOINT_TIDAL
    sel_tidal.to_csv(out, index=False)
    print(f"[tidal] Saved: {out}")

    if SCENARIO != "joint":
        plot_single_tech_map(
            land, eez,
            sel_tidal,
            OUTDIR / f"map_tidal_{SCENARIO}.png",
            title=f"Optimised Tidal Sites ({SCENARIO})",
            tech="Tidal",
            color="red",
            vmin=0.0,
            vmax=float(TIDAL_CELL_CAPACITY_MW * TIDAL_CLUSTER_MAX_CELLS),
        )

# Joint combined plot + summary diagnostics
if SCENARIO == "joint":
    plot_joint_map(
        land, eez,
        sel_wind,
        sel_tidal,
        OUTDIR / "map_joint_wind_tidal.png",
        title="Optimised Wind and Tidal Sites (joint)",
    )
    print("[joint] Saved combined map:", OUTDIR / "map_joint_wind_tidal.png")

    print("\nBuilt capacity by carrier (MW):")
    print(n.generators["p_nom_opt"].fillna(0).groupby(n.generators["carrier"]).sum().sort_values(ascending=False))

    print("\nBuilt capacity by carrier (excluding backup) (MW):")
    mask = n.generators["carrier"].isin(["wind", "tidal"])
    print(n.generators.loc[mask, "p_nom_opt"].fillna(0).groupby(n.generators.loc[mask, "carrier"]).sum())

    wind_built = n.generators.query("carrier=='wind'")["p_nom_opt"].fillna(0).sum()
    tidal_built = n.generators.query("carrier=='tidal'")["p_nom_opt"].fillna(0).sum()
    print("\nWind built (MW):", float(wind_built))
    print("Tidal built (MW):", float(tidal_built))
    print("Wind+Tidal (MW):", float(wind_built + tidal_built), " target:", TARGET_BUILD_MW_TOTAL_JOINT)


# ============================================================
# 8) OAT SENSITIVITY ANALYSIS (OPTIONAL)
# ============================================================
# OAT re-runs ONLY the optimisation stage with one parameter changed at a time.
# Candidate sets (wind/tidal sites) are NOT rebuilt, isolating techno-economic sensitivity.
# 
#For larger target loads, consider running OAT separately to avoid re-running the main solve. If ran here model can take hours to complete.
#
# Accounting convention used here:
#   - PyPSA capital_cost is already ANNUALISED (GBP/MW-year)
#   - LCOE CAPEX term = sum(p_nom_opt * capital_cost)  [no annuity again]
#
# LCOE reporting convention:
#   - BACKUP excluded from ALL LCOE numerators/denominators
#   - tidal_only: report tidal LCOE (tidal + storage costs / tidal energy)
#   - joint: report system LCOE excl backup, plus wind-only and tidal-only LCOE
# ============================================================

RUN_OAT = True

OAT_OUT_CSV = Path("oat_summary.csv")

OAT_FIG_DIR = OUTDIR / "oat_figures"
OAT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# OAT sweeps (thesis factors + optional extras)
OAT_SWEEPS = [
    ("CAPEX_MULT",    [0.8, 1.0, 1.2]),                 # ±20%
    ("CF_MULT",       [0.9, 1.0, 1.1]),                 # ±10%
    ("DISCOUNT_RATE", [0.02, 0.05, 0.07, 0.09]),        # 2–9%
    ("MARGINAL_WIND_GBP_PER_MWH",  [30.0, 49.0, 70.0]), # OPEX Changes
    ("MARGINAL_TIDAL_GBP_PER_MWH", [120.0, 179.0, 250.0]), # OPEX Changes
]

def _safe_objective_value(n: pypsa.Network) -> float:
    """Best-effort objective extraction across PyPSA versions."""
    for attr in ("objective", "objective_value"):
        if hasattr(n, attr):
            try:
                v = getattr(n, attr)
                return float(v) if v is not None else float("nan")
            except Exception:
                pass
    if hasattr(n, "model") and n.model is not None:
        for attr in ("objective", "objective_value"):
            if hasattr(n.model, attr):
                try:
                    v = getattr(n.model, attr)
                    return float(v) if v is not None else float("nan")
                except Exception:
                    pass
    return float("nan")

def _snapshot_weights_hours(n: pypsa.Network) -> pd.Series:
    """
    Snapshot weights in hours. If missing, assume 1 hour per snapshot.
    Works across PyPSA versions (Series or DataFrame).
    """
    sw = getattr(n, "snapshot_weightings", None)
    if sw is None:
        return pd.Series(1.0, index=n.snapshots)

    if isinstance(sw, pd.DataFrame):
        for col in ("objective", "generators", "stores"):
            if col in sw.columns:
                return pd.Series(sw[col].values, index=n.snapshots)
        return pd.Series(sw.iloc[:, 0].values, index=n.snapshots)

    if isinstance(sw, pd.Series):
        return pd.Series(sw.values, index=n.snapshots)

    return pd.Series(1.0, index=n.snapshots)

def _total_hours(n: pypsa.Network) -> float:
    """Total weighted hours covered by the optimisation horizon."""
    return float(_snapshot_weights_hours(n).sum())

def _annualise_factor(n: pypsa.Network) -> float:
    """
    Scale model-horizon energy to annual energy.
    If horizon covers H weighted hours, annual factor = 8760 / H.
    """
    H = _total_hours(n)
    if not np.isfinite(H) or H <= 0:
        return float("nan")
    return 8760.0 / H

def _energy_by_generator_mwh(n: pypsa.Network) -> pd.Series:
    """Generator energy (MWh) over the model horizon."""
    if not hasattr(n, "generators_t") or not hasattr(n.generators_t, "p"):
        return pd.Series(dtype=float)
    p = n.generators_t.p.copy()  # MW
    w = _snapshot_weights_hours(n).reindex(p.index).fillna(1.0)  # hours
    e = p.mul(w, axis=0).sum(axis=0)  # MWh
    return e.astype(float)

def _energy_by_storage_unit_discharge_mwh(n: pypsa.Network) -> pd.Series:
    """StorageUnit DISCHARGE energy (MWh) over the horizon (positive p only)."""
    if not hasattr(n, "storage_units_t") or not hasattr(n.storage_units_t, "p"):
        return pd.Series(dtype=float)
    p = n.storage_units_t.p.copy()  # MW (positive discharge, negative charge)
    p = p.clip(lower=0.0)
    w = _snapshot_weights_hours(n).reindex(p.index).fillna(1.0)
    e = p.mul(w, axis=0).sum(axis=0)
    return e.astype(float)

def _served_load_mwh(n: pypsa.Network) -> float:
    """Served electrical load (MWh) over the horizon."""
    if not hasattr(n, "loads_t") or not hasattr(n.loads_t, "p"):
        return float("nan")
    pL = n.loads_t.p.copy()  # MW
    w = _snapshot_weights_hours(n).reindex(pL.index).fillna(1.0)
    return float(pL.mul(w, axis=0).sum().sum())

def _built_capacity_by_carrier(n: pypsa.Network) -> pd.Series:
    """Sum built p_nom_opt by generator carrier."""
    s = n.generators.get("p_nom_opt", pd.Series(0.0, index=n.generators.index)).fillna(0.0).astype(float)
    return s.groupby(n.generators["carrier"].astype(str)).sum()

# -----------------------------
# CAPEX + OPEX accounting (capital_cost already annualised)
# -----------------------------

def _annual_capex_gbp_per_year_excl_backup(n: pypsa.Network) -> float:
    """Annualised CAPEX (GBP/year) for generators + storage, excluding backup."""
    total = 0.0

    # Generators
    if hasattr(n, "generators") and len(n.generators) > 0:
        g = n.generators
        p = g.get("p_nom_opt", pd.Series(0.0, index=g.index)).fillna(0.0).astype(float)
        cap = g.get("capital_cost", pd.Series(0.0, index=g.index)).fillna(0.0).astype(float)
        carrier = g.get("carrier", pd.Series("", index=g.index)).astype(str)

        mask = carrier != "backup"
        total += float((p[mask] * cap[mask]).sum())

    # StorageUnits
    if hasattr(n, "storage_units") and len(n.storage_units) > 0:
        su = n.storage_units
        p = su.get("p_nom_opt", pd.Series(0.0, index=su.index)).fillna(0.0).astype(float)
        cap = su.get("capital_cost", pd.Series(0.0, index=su.index)).fillna(0.0).astype(float)
        total += float((p * cap).sum())

    return float(total)

def _annual_opex_gbp_per_year_excl_backup(n: pypsa.Network) -> float:
    """
    Annualised OPEX proxy (GBP/year) from marginal costs:
      marginal_cost (GBP/MWh) * energy (MWh over horizon) * annualise_factor
    Excludes backup.
    """
    e_gen = _energy_by_generator_mwh(n)
    if e_gen.empty:
        return float("nan")

    annual_factor = _annualise_factor(n)
    if not np.isfinite(annual_factor):
        return float("nan")

    g = n.generators
    mc = g.get("marginal_cost", pd.Series(0.0, index=g.index)).fillna(0.0).astype(float)
    carrier = g.get("carrier", pd.Series("", index=g.index)).astype(str)

    cost_horizon = 0.0
    for name, e_mwh in e_gen.items():
        if name not in g.index:
            continue
        if carrier.loc[name] == "backup":
            continue
        cost_horizon += float(e_mwh * mc.loc[name])

    # Storage marginal cost (optional)
    if hasattr(n, "storage_units_t") and hasattr(n.storage_units_t, "p") and hasattr(n, "storage_units"):
        e_dis = _energy_by_storage_unit_discharge_mwh(n)
        su = n.storage_units
        mc_su = su.get("marginal_cost", pd.Series(0.0, index=su.index)).fillna(0.0).astype(float)
        for name, e_mwh in e_dis.items():
            if name in su.index:
                cost_horizon += float(e_mwh * mc_su.loc[name])

    return float(cost_horizon * annual_factor)

def _annual_energy_mwh_excl_backup(n: pypsa.Network) -> float:
    """
    Annualised served energy excluding backup contribution:
      annual_nonbackup = (served_load - backup_energy) * (8760/H)
    """
    served = _served_load_mwh(n)
    e_gen = _energy_by_generator_mwh(n)
    annual_factor = _annualise_factor(n)

    if not np.isfinite(served) or not np.isfinite(annual_factor):
        return float("nan")

    backup_e = 0.0
    if not e_gen.empty and "carrier" in n.generators.columns:
        carriers = n.generators["carrier"].astype(str)
        backup_gens = carriers[carriers == "backup"].index
        backup_e = float(e_gen.reindex(backup_gens).fillna(0.0).sum())

    nonbackup_served = served - backup_e
    if nonbackup_served <= 0:
        return float("nan")

    return float(nonbackup_served * annual_factor)

def _lcoe_system_excl_backup(n: pypsa.Network) -> float:
    """
    System LCOE excluding backup:
      (annual capex excl backup + annual opex excl backup) / annual energy excl backup
    """
    denom = _annual_energy_mwh_excl_backup(n)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")

    capex_y = _annual_capex_gbp_per_year_excl_backup(n)
    opex_y = _annual_opex_gbp_per_year_excl_backup(n)
    if not np.isfinite(opex_y):
        return float("nan")

    return float((capex_y + opex_y) / denom)

def _lcoe_for_carrier(n: pypsa.Network, target: str) -> float:
    """
    Tech LCOE:
      (annual capex for that tech + annual opex for that tech) / annual energy from that tech

    Notes:
    - wind: wind generators only
    - tidal: tidal generators + tidal_storage annual costs
    - Denominator uses generator energy only (not storage discharge) to avoid double-counting.
    """
    target = str(target)
    annual_factor = _annualise_factor(n)
    if not np.isfinite(annual_factor):
        return float("nan")

    e_gen = _energy_by_generator_mwh(n)
    if e_gen.empty:
        return float("nan")

    carriers = n.generators["carrier"].astype(str).reindex(e_gen.index)
    e_target_h = float(e_gen[carriers == target].sum())
    e_target_y = e_target_h * annual_factor
    if e_target_y <= 0:
        return float("nan")

    capex_y = 0.0
    g = n.generators
    p = g.get("p_nom_opt", pd.Series(0.0, index=g.index)).fillna(0.0).astype(float)
    cap = g.get("capital_cost", pd.Series(0.0, index=g.index)).fillna(0.0).astype(float)

    for name in g.index:
        if str(g.at[name, "carrier"]) != target:
            continue
        capex_y += float(p.loc[name] * cap.loc[name])

    # Include storage annualised costs in tidal LCOE interpretation
    if target == "tidal" and hasattr(n, "storage_units") and len(n.storage_units) > 0:
        su = n.storage_units
        p_su = su.get("p_nom_opt", pd.Series(0.0, index=su.index)).fillna(0.0).astype(float)
        cap_su = su.get("capital_cost", pd.Series(0.0, index=su.index)).fillna(0.0).astype(float)
        capex_y += float((p_su * cap_su).sum())

    mc = g.get("marginal_cost", pd.Series(0.0, index=g.index)).fillna(0.0).astype(float)
    opex_h = 0.0
    for name, e_mwh in e_gen.items():
        if name in g.index and str(g.at[name, "carrier"]) == target:
            opex_h += float(e_mwh * mc.loc[name])
    opex_y = opex_h * annual_factor

    # Storage marginal costs included in tidal LCOE
    if target == "tidal" and hasattr(n, "storage_units_t") and hasattr(n.storage_units_t, "p") and hasattr(n, "storage_units"):
        e_dis = _energy_by_storage_unit_discharge_mwh(n)
        su = n.storage_units
        mc_su = su.get("marginal_cost", pd.Series(0.0, index=su.index)).fillna(0.0).astype(float)
        for name, e_mwh in e_dis.items():
            if name in su.index:
                opex_y += float(e_mwh * mc_su.loc[name] * annual_factor)

    return float((capex_y + opex_y) / e_target_y)

def _plot_oat_figures(df: pd.DataFrame, outdir: Path):
    """Basic OAT plots: (1) sorted system LCOE, (2) LCOE vs each swept parameter."""
    d = df.copy().reset_index(drop=True)

    if "lcoe_system_excl_backup_gbp_per_mwh" in d.columns:
        dd = d.dropna(subset=["lcoe_system_excl_backup_gbp_per_mwh"]).copy()
        if len(dd) > 1:
            dd = dd.sort_values("lcoe_system_excl_backup_gbp_per_mwh", ascending=True).reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(np.arange(len(dd)), dd["lcoe_system_excl_backup_gbp_per_mwh"].to_numpy(float), marker="o", linestyle="-")
            ax.set_xlabel("OAT case (sorted)")
            ax.set_ylabel("LCOE excl backup (GBP/MWh)")
            ax.set_title("OAT: System LCOE (excl backup), sorted")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / "oat_lcoe_system_excl_backup_sorted.png", dpi=300)
            plt.close(fig)

    for pname in sorted(set(d["param"].dropna().astype(str))):
        if pname.strip() == "":
            continue
        sub = d[d["param"] == pname].copy()
        sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.dropna(subset=["value_num"]).sort_values("value_num")

        ycol = "lcoe_system_excl_backup_gbp_per_mwh"
        if ycol not in sub.columns or sub[ycol].isna().all():
            continue

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sub["value_num"].to_numpy(float), sub[ycol].to_numpy(float), marker="o", linestyle="-")
        ax.set_xlabel(pname)
        ax.set_ylabel("LCOE excl backup (GBP/MWh)")
        ax.set_title(f"OAT: LCOE vs {pname}")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"oat_lcoe_vs_{pname}.png", dpi=300)
        plt.close(fig)

def run_oat():
    """
    Run baseline optimisation + OAT sweeps.
    Implementation notes:
      - CF_MULT perturbs CF time series without rebuilding candidates.
      - CAPEX_MULT and DISCOUNT_RATE modify globals then call refresh_annualised_costs().
      - extra_overrides temporarily override other globals (e.g., marginal costs).
    """
    global_vars = globals()

    needed_globals = ["CAPEX_WIND_GBP_PER_MW", "CAPEX_TIDAL_GBP_PER_MW", "DISCOUNT_RATE"]
    for gname in needed_globals:
        if gname not in global_vars:
            raise KeyError(f"Missing required global '{gname}'. Add it in Section 0 config.")

    rows = []

    def _solve_case(
        capex_mult: float,
        cf_mult: float,
        discount_rate: float,
        extra_overrides: Optional[Dict[str, float]] = None,
    ) -> pypsa.Network:
        """
        Solve one OAT case:
          - Apply CF multiplier to CF time series (wind_cf_df/tidal_cf_df).
          - Apply CAPEX_MULT and DISCOUNT_RATE to globals and refresh annualised costs.
          - Optionally override other globals for the duration of the solve.
        """
        wind_cf_use = wind_cf_df
        tidal_cf_use = tidal_cf_df
        if wind_cf_use is not None:
            wind_cf_use = (wind_cf_use * float(cf_mult)).clip(0.0, 1.0)
        if tidal_cf_use is not None:
            tidal_cf_use = (tidal_cf_use * float(cf_mult)).clip(0.0, 1.0)

        cap_w0 = float(global_vars["CAPEX_WIND_GBP_PER_MW"])
        cap_t0 = float(global_vars["CAPEX_TIDAL_GBP_PER_MW"])
        dr0    = float(global_vars["DISCOUNT_RATE"])

        overrides0: Dict[str, float] = {}
        if extra_overrides:
            for k, v in extra_overrides.items():
                if k not in global_vars:
                    raise KeyError(f"OAT override '{k}' not found in globals.")
                overrides0[k] = global_vars[k]
                global_vars[k] = v

        global_vars["CAPEX_WIND_GBP_PER_MW"]  = cap_w0 * float(capex_mult)
        global_vars["CAPEX_TIDAL_GBP_PER_MW"] = cap_t0 * float(capex_mult)
        global_vars["DISCOUNT_RATE"]          = float(discount_rate)

        refresh_annualised_costs()

        try:
            n_case = solve_network(
                snapshots,
                load,
                wind_candidates, wind_cf_use,
                tidal_candidates, tidal_cf_use,
            )
            return n_case
        finally:
            global_vars["CAPEX_WIND_GBP_PER_MW"]  = cap_w0
            global_vars["CAPEX_TIDAL_GBP_PER_MW"] = cap_t0
            global_vars["DISCOUNT_RATE"]          = dr0

            if extra_overrides:
                for k, v0 in overrides0.items():
                    global_vars[k] = v0

            refresh_annualised_costs()

    def _record(ncase: pypsa.Network, label: str, pname: str, pval, case_discount_rate: float) -> Dict:
        """Summarise a solved case into one output row."""
        car = _built_capacity_by_carrier(ncase)

        out = {
            "case": label,
            "param": pname,
            "value": pval,
            "discount_rate_used": float(case_discount_rate),

            "wind_MW": float(car.get("wind", 0.0)),
            "tidal_MW": float(car.get("tidal", 0.0)),
            "backup_MW": float(car.get("backup", 0.0)),
            "diversification_MW": float(car.get("diversification", 0.0)),

            "objective": _safe_objective_value(ncase),
            "served_load_MWh_horizon": float(_served_load_mwh(ncase)),
            "annualise_factor": float(_annualise_factor(ncase)),
        }

        out["lcoe_system_excl_backup_gbp_per_mwh"] = float(_lcoe_system_excl_backup(ncase))

        if SCENARIO == "tidal_only":
            out["lcoe_tidal_gbp_per_mwh"] = float(_lcoe_for_carrier(ncase, "tidal"))
        else:
            out["lcoe_wind_gbp_per_mwh"]  = float(_lcoe_for_carrier(ncase, "wind"))
            out["lcoe_tidal_gbp_per_mwh"] = float(_lcoe_for_carrier(ncase, "tidal"))

        out["annual_capex_excl_backup_gbp_per_year"] = float(_annual_capex_gbp_per_year_excl_backup(ncase))
        out["annual_opex_excl_backup_gbp_per_year"]  = float(_annual_opex_gbp_per_year_excl_backup(ncase))

        return out

    # Baseline
    print("\n[OAT] Running baseline optimisation...")
    baseline_dr = float(DISCOUNT_RATE)
    n0 = _solve_case(capex_mult=1.0, cf_mult=1.0, discount_rate=baseline_dr)
    rows.append(_record(n0, "baseline", "", "", baseline_dr))

    # OAT variables
    for pname, values in OAT_SWEEPS:
        for v in values:
            print(f"[OAT] {pname} = {v}")

            capex_mult = 1.0
            cf_mult = 1.0
            disc = float(DISCOUNT_RATE)

            if pname == "CAPEX_MULT":
                capex_mult = float(v)
                n_i = _solve_case(capex_mult=capex_mult, cf_mult=cf_mult, discount_rate=disc)
                rows.append(_record(n_i, f"{pname}={v}", pname, v, disc))

            elif pname == "CF_MULT":
                cf_mult = float(v)
                n_i = _solve_case(capex_mult=capex_mult, cf_mult=cf_mult, discount_rate=disc)
                rows.append(_record(n_i, f"{pname}={v}", pname, v, disc))

            elif pname == "DISCOUNT_RATE":
                disc = float(v)
                n_i = _solve_case(capex_mult=capex_mult, cf_mult=cf_mult, discount_rate=disc)
                rows.append(_record(n_i, f"{pname}={v}", pname, v, disc))

            else:
                n_i = _solve_case(
                    capex_mult=capex_mult,
                    cf_mult=cf_mult,
                    discount_rate=disc,
                    extra_overrides={pname: float(v)},
                )
                rows.append(_record(n_i, f"{pname}={v}", pname, v, disc))

    df = pd.DataFrame(rows)
    df.to_csv(OAT_OUT_CSV, index=False)
    print(f"\n[OAT] Saved summary to: {OAT_OUT_CSV.resolve()}")
    print(df.head(12))

    _plot_oat_figures(df, OAT_FIG_DIR)
    print(f"[OAT] Saved figures to: {OAT_FIG_DIR.resolve()}")

# Run OAT if toggled on
if RUN_OAT:
    run_oat()
