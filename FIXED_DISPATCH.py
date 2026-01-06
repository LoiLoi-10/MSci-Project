# FIXED_DISPATCH.py
#
# Fixed UK offshore wind + tidal sites with real capacities.
# - Wind CF from ERA5 + power curve (per site)
# - Tidal CF per site from CMEMS CSV (already CF, 0..1)
# - Demand from NGESO ND
# - p_nom for each site fixed to installed MW from CSV
# - PyPSA optimises dispatch only (no capacity expansion)
# - Saves hourly + daily CF and power time series

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pypsa

# ============================================================
# 0) CONFIG: FILE PATHS
# ============================================================

# Input data
ERA5_WIND_FILE = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\ERA5_DATA\era5_uk_offshore_2023_12.nc")
POWER_CURVE_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\POWER_CURVE.csv")
DEMAND_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\DEMAND_DATA\demanddata_2023.csv")

# Site metadata (locations + installed capacities)
WIND_SITES_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\OPERATIONAL_SITES\uk_offshore_wind_sites.csv")
TIDAL_SITES_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\OPERATIONAL_SITES\uk_tidal_sites.csv")

# Multi-site tidal CF file (hourly CF columns, indexed by time)
TIDAL_CF_MULTISITE_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\CMEMS_DATA\tidal_cf_2023_DEC_per_site.csv")

# Output CSVs (capacity factors + power time series)
OUT_CF_WIND_HOURLY   = Path("wind_cf_hourly_per_site.csv")
OUT_CF_TIDAL_HOURLY  = Path("tidal_cf_hourly_per_site.csv")
OUT_CF_WIND_DAILY    = Path("wind_cf_daily_per_site.csv")
OUT_CF_TIDAL_DAILY   = Path("tidal_cf_daily_per_site.csv")
OUT_P_WIND_HOURLY    = Path("wind_power_hourly_per_site.csv")
OUT_P_TIDAL_HOURLY   = Path("tidal_power_hourly_per_site.csv")
OUT_P_WIND_DAILY     = Path("wind_power_daily_per_site.csv")
OUT_P_TIDAL_DAILY    = Path("tidal_power_daily_per_site.csv")

# Quick sanity checks before running
print("ERA5 exists:", ERA5_WIND_FILE.exists())
print("Demand exists:", DEMAND_CSV.exists())
print("Wind sites CSV exists:", WIND_SITES_CSV.exists())
print("Tidal sites CSV exists:", TIDAL_SITES_CSV.exists())
print("Tidal CF CSV exists:", TIDAL_CF_MULTISITE_CSV.exists())

# ============================================================
# 0b) MODEL SETTINGS & COSTS
# ============================================================

# Wind shear exponent for scaling ref-height wind speeds to hub height (1/7th power law)
ALPHA = 0.14

# Simplified costs (used only for dispatch economics; no wind/tidal CAPEX here), make model scalable if needed
MARG_WIND  = 49
MARG_TIDAL = 179


SCENARIO_LABEL = "fixed_wind_tidal"

# Column mapping for the site CSVs
WIND_COLS = {
    "id":         "site_id",
    "name":       "name",
    "lat":        "lat",
    "lon":        "lon",
    "p_nom":      "p_nom_max_MW",
    "hub_height": "hub_height_m",
}
TIDAL_COLS = {
    "id":    "site_id",
    "name":  "name",
    "lat":   "lat",
    "lon":   "lon",
    "p_nom": "p_nom_max_MW",
}

# ============================================================
# 1) HELPERS
# ============================================================

def find_lat_lon_dims(dset: xr.Dataset) -> tuple[str, str]:
    """Find the dataset's latitude and longitude dimension/coord names."""
    lat = None
    lon = None

    for d in dset.dims:
        nm = str(d).lower()
        if "lat" in nm:
            lat = d
        if "lon" in nm:
            lon = d

    if lat is None:
        for c in dset.coords:
            nm = str(c).lower()
            if "lat" in nm:
                lat = c
                break

    if lon is None:
        for c in dset.coords:
            nm = str(c).lower()
            if "lon" in nm:
                lon = c
                break

    if lat is None or lon is None:
        raise ValueError(f"Could not detect lat/lon. Dims={list(dset.dims)}, Coords={list(dset.coords)}")
    return str(lat), str(lon)


def detect_or_build_time(dset: xr.Dataset) -> tuple[str, pd.DatetimeIndex]:
    """Detect the dataset's time coordinate and return (name, DatetimeIndex)."""
    for c in dset.coords:
        arr = dset[c]
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.datetime64):
            return str(c), pd.DatetimeIndex(arr.values)

    for nm in ("time", "valid_time", "forecast_time", "date", "datetime"):
        if nm in dset.variables:
            arr = dset[nm]
            try:
                return nm, pd.DatetimeIndex(arr.values)
            except Exception:
                pass

    raise ValueError("Could not determine time coordinate.")


def load_power_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load turbine power curve and return (wind_speeds, capacity_factor_curve)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing power curve: {csv_path}")
    df = pd.read_csv(csv_path).dropna().sort_values("v")
    if not {"v", "p"}.issubset(df.columns) or len(df) < 2:
        raise ValueError("power_curve.csv must contain columns v and p with at least 2 rows.")

    v_curve = df["v"].to_numpy(float)
    p_raw = df["p"].to_numpy(float)

    # Normalise to 0..1 if input is in kW/MW
    if p_raw.max() > 1.0:
        p_curve = p_raw / p_raw.max()
    else:
        p_curve = p_raw.copy()

    # Ensure CF bounds
    p_curve = np.clip(p_curve, 0.0, 1.0)
    return v_curve, p_curve


def nearest_grid_index(lat_array: np.ndarray, lon_array: np.ndarray, lat: float, lon: float) -> tuple[int, int]:
    """Find nearest ERA5 grid cell index for a given (lat, lon)."""
    lat_array = np.asarray(lat_array)
    lon_array = np.asarray(lon_array)
    ilat = int(np.argmin(np.abs(lat_array - lat)))
    ilon = int(np.argmin(np.abs(lon_array - lon)))
    return ilat, ilon


def speed_to_cf_from_curve(speed_series: pd.Series, v_curve: np.ndarray, p_curve: np.ndarray,
                           snapshots: pd.DatetimeIndex) -> pd.Series:
    """Map wind speed time series -> CF using linear interpolation of the power curve, aligned to snapshots."""
    vals = np.asarray(speed_series.values, dtype=float)
    cf_arr = np.interp(vals, v_curve, p_curve, left=p_curve[0], right=p_curve[-1])
    cf_arr = np.clip(cf_arr, 0.0, 1.0)

    s = pd.Series(cf_arr, index=pd.DatetimeIndex(speed_series.index))
    # Align to network snapshots (fill small gaps)
    s = s.sort_index().reindex(snapshots).interpolate(limit=3).bfill().ffill()
    return s


def build_load_from_ngeso_nd(csv_path: Path, snapshots: pd.DatetimeIndex) -> pd.Series:
    """Load NGESO ND half-hourly demand, resample to hourly, align to snapshots."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Demand CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in demand CSV. Columns: {df.columns.tolist()}")

    date = pd.to_datetime(df["SETTLEMENT_DATE"], dayfirst=True, errors="coerce")
    half_hours = df["SETTLEMENT_PERIOD"].astype(int) - 1
    t_half = date + pd.to_timedelta(half_hours * 30, unit="min")

    s_half = pd.Series(df["ND"].astype(float).to_numpy(), index=t_half).sort_index()

    # Trim to model window (snapshots are hourly)
    start = snapshots[0]
    end = snapshots[-1] + pd.Timedelta(hours=1)
    s_half = s_half.loc[(s_half.index >= start) & (s_half.index <= end)]

    # Half-hourly -> hourly and align
    s_hour = s_half.resample("1h").mean()
    load = s_hour.reindex(snapshots).interpolate(limit=2).bfill().ffill()

    print("\nDemand ND stats over model period:")
    print("  min:", float(load.min()), "max:", float(load.max()), "mean:", float(load.mean()))
    return load


def load_wind_sites(csv_path: Path, cols: dict) -> pd.DataFrame:
    """Read wind site metadata (id, name, lat/lon, installed MW, hub height)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Wind sites CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = [cols["id"], cols["name"], cols["lat"], cols["lon"], cols["p_nom"], cols["hub_height"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Wind site CSV missing columns: {missing}. Found: {df.columns.tolist()}")

    return pd.DataFrame({
        "site_id":      df[cols["id"]].astype(str),
        "site_name":    df[cols["name"]].astype(str),
        "lat":          df[cols["lat"]].astype(float),
        "lon":          df[cols["lon"]].astype(float),
        "p_nom_mw":     df[cols["p_nom"]].astype(float),
        "hub_height_m": df[cols["hub_height"]].astype(float),
    })


def load_tidal_sites(csv_path: Path, cols: dict) -> pd.DataFrame:
    """Read tidal site metadata (id, name, lat/lon, installed MW)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Tidal sites CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = [cols["id"], cols["name"], cols["lat"], cols["lon"], cols["p_nom"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tidal site CSV missing columns: {missing}. Found: {df.columns.tolist()}")

    return pd.DataFrame({
        "site_id":   df[cols["id"]].astype(str),
        "site_name": df[cols["name"]].astype(str),
        "lat":       df[cols["lat"]].astype(float),
        "lon":       df[cols["lon"]].astype(float),
        "p_nom_mw":  df[cols["p_nom"]].astype(float),
    })


def load_tidal_cf_multisite(csv_path: Path, snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load multi-site tidal CF CSV (columns are site_id, values are CF 0..1).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Tidal CF CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain a 'time' column; columns are {df.columns.tolist()}")

    # Parse timestamps
    time_raw = df["time"].astype(str).str.strip()
    t = pd.to_datetime(time_raw, errors="coerce", dayfirst=True)
    bad = t.isna()
    if bad.any():
        ex = time_raw[bad].unique()[:5]
        raise ValueError(f"Unparseable tidal times (examples): {list(ex)}")

    df = df.drop(columns=["time"])
    df.index = pd.DatetimeIndex(t).sort_values()
    df = df.sort_index()

    # Coerce all CF columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Align to snapshots
    overlap = df.index.intersection(snapshots)
    if len(overlap) == 0:
        if len(df) == len(snapshots):
            print(
                "\n[tidal] No timestamp overlap with ERA5 snapshots, "
                "but lengths match; remapping by position."    
            )
            df = df.copy()
            df.index = snapshots
        else:
            raise ValueError(
                f"Tidal CF period does not overlap snapshots AND lengths differ "
                f"(tidal rows={len(df)} vs snapshots={len(snapshots)}). "
                f"Either regenerate tidal CF for July-2023 or resample/trim to match."
            )
    else:
        df = df.reindex(snapshots)

    # Fill short gaps, confine to [0,1]
    df = df.interpolate(limit=3).bfill().ffill()
    df = df.clip(lower=0.0, upper=1.0).fillna(0.0)

    print("\nTidal CF diagnostics after alignment:")
    print("  index:", df.index.min(), "→", df.index.max(), "n=", len(df))
    print("  column means:", df.mean().to_dict())

    return df


# ============================================================
# 2) LOAD ERA5 & COMPUTE WIND SPEED FIELD
# ============================================================

assert ERA5_WIND_FILE.exists(), f"Missing ERA5 wind file: {ERA5_WIND_FILE}"
ds = xr.open_dataset(ERA5_WIND_FILE)

# Rename common ERA5 variable names to simpler u10/v10/u100/v100 if needed
rename_map = {}
if "10m_u_component_of_wind" in ds:
    rename_map["10m_u_component_of_wind"] = "u10"
if "10m_v_component_of_wind" in ds:
    rename_map["10m_v_component_of_wind"] = "v10"
if "100m_u_component_of_wind" in ds:
    rename_map["100m_u_component_of_wind"] = "u100"
if "100m_v_component_of_wind" in ds:
    rename_map["100m_v_component_of_wind"] = "v100"
if rename_map:
    ds = ds.rename(rename_map)

lat_dim, lon_dim = find_lat_lon_dims(ds)
time_dim, time_index = detect_or_build_time(ds)

# Ensure time,lat,lon ordering (best effort)
try:
    ds = ds.transpose(time_dim, lat_dim, lon_dim)
except Exception:
    pass

snapshots = pd.DatetimeIndex(time_index)
print(f"\nUsing '{time_dim}' as time coord: {snapshots[0]} → {snapshots[-1]} (n={len(snapshots)})")

# Wind speed magnitude at reference height (prefer 100m if available)
if "u100" in ds and "v100" in ds:
    ws_ref = xr.apply_ufunc(np.hypot, ds["u100"], ds["v100"])
    REF_HEIGHT = 100.0
else:
    ws_ref = xr.apply_ufunc(np.hypot, ds["u10"], ds["v10"])
    REF_HEIGHT = 10.0

ws_ref = ws_ref.assign_coords({time_dim: xr.DataArray(np.asarray(time_index), dims=(time_dim,))})

lat_vals = ds[lat_dim].values
lon_vals = ds[lon_dim].values

# ============================================================
# 3) BUILD NETWORK & LOAD
# ============================================================

n = pypsa.Network()

# Carriers are just labels; helps with filtering and bookkeeping
for c in ["electricity", "wind", "tidal", "gas"]:
    if c not in n.carriers.index:
        n.add("Carrier", c)

n.set_snapshots(list(snapshots))
n.add("Bus", "uk_bus", v_nom=400.0, carrier="electricity")

# Add demand as a fixed load time series
load_series = build_load_from_ngeso_nd(DEMAND_CSV, snapshots)
n.add("Load", "uk_load", bus="uk_bus", p_set=load_series)

# ============================================================
# 4) LOAD SITES & POWER CURVE
# ============================================================

wind_sites = load_wind_sites(WIND_SITES_CSV, WIND_COLS)
tidal_sites = load_tidal_sites(TIDAL_SITES_CSV, TIDAL_COLS)

print(f"\nLoaded {len(wind_sites)} offshore wind sites.")
print(f"Loaded {len(tidal_sites)} tidal sites.")

v_curve, p_curve = load_power_curve(POWER_CURVE_CSV)

# ============================================================
# 5) ADD WIND GENERATORS
# ============================================================

wind_cf_dict = {}
wind_p_dict  = {}

for _, row in wind_sites.iterrows():
    site_id      = row["site_id"]
    lat          = float(row["lat"])
    lon          = float(row["lon"])
    p_nom_mw     = float(row["p_nom_mw"])
    hub_height_m = float(row["hub_height_m"])

    # Extract nearest ERA5 grid-cell wind speed and scale to hub height
    ilat, ilon = nearest_grid_index(lat_vals, lon_vals, lat, lon)
    da_cell_ref = ws_ref.isel({lat_dim: ilat, lon_dim: ilon})
    factor = (hub_height_m / REF_HEIGHT) ** ALPHA
    speed_site = da_cell_ref * factor

    # Convert to pandas and map to capacity factor via power curve
    speed_series = pd.Series(
        np.asarray(speed_site.values, dtype=float),
        index=pd.DatetimeIndex(np.asarray(speed_site[time_dim].values)),
    )
    cf = speed_to_cf_from_curve(speed_series, v_curve, p_curve, snapshots)

    gen_name = f"wind_{site_id}"
    wind_cf_dict[gen_name] = cf
    wind_p_dict[gen_name]  = cf * p_nom_mw

    # Add fixed-capacity wind generator; time-varying availability goes in p_max_pu
    n.add(
        "Generator", gen_name,
        bus="uk_bus", carrier="wind",
        p_nom=p_nom_mw,
        p_nom_extendable=False,
        marginal_cost=MARG_WIND,
        capital_cost=0.0,
    )
    n.generators_t.p_max_pu.loc[:, gen_name] = cf.values

print(f"Added {len(wind_sites)} fixed-capacity wind generators.")

# ============================================================
# 6) ADD TIDAL GENERATORS (PER-SITE CF, OPTION A DATE ALIGN)
# ============================================================

tidal_cf_df = load_tidal_cf_multisite(TIDAL_CF_MULTISITE_CSV, snapshots)
tidal_p_dict = {}
missing_cols = []

for _, row in tidal_sites.iterrows():
    site_id  = row["site_id"]
    p_nom_mw = float(row["p_nom_mw"])
    gen_name = f"tidal_{site_id}"

    # Each tidal site must have a matching CF column named by site_id
    if site_id not in tidal_cf_df.columns:
        missing_cols.append(site_id)
        continue

    cf_site = tidal_cf_df[site_id]
    tidal_p_dict[gen_name] = cf_site * p_nom_mw

    n.add(
        "Generator", gen_name,
        bus="uk_bus", carrier="tidal",
        p_nom=p_nom_mw,
        p_nom_extendable=False,
        marginal_cost=MARG_TIDAL,
        capital_cost=0.0,
    )
    n.generators_t.p_max_pu.loc[:, gen_name] = cf_site.values

if missing_cols:
    print("\nWARNING: These tidal site_ids were not found as columns in the tidal CF CSV:")
    print(" ", missing_cols)

print(f"Added {len(tidal_p_dict)} tidal generators with site-specific CF.")

# ============================================================
# 7) SAVE CF & POWER TIME SERIES
# ============================================================

# Save per-site CFs and per-site power (CF * installed capacity)
wind_cf_df = pd.DataFrame(wind_cf_dict, index=snapshots)
wind_cf_df.to_csv(OUT_CF_WIND_HOURLY)
wind_cf_df.resample("D").mean().to_csv(OUT_CF_WIND_DAILY)
print(f"\nSaved wind CF hourly to {OUT_CF_WIND_HOURLY}")
print(f"Saved wind CF daily means to {OUT_CF_WIND_DAILY}")

tidal_cf_gen = pd.DataFrame(index=snapshots)
for sid in tidal_sites["site_id"].astype(str):
    if sid in tidal_cf_df.columns:
        tidal_cf_gen[f"tidal_{sid}"] = tidal_cf_df[sid]
tidal_cf_gen.to_csv(OUT_CF_TIDAL_HOURLY)
tidal_cf_gen.resample("D").mean().to_csv(OUT_CF_TIDAL_DAILY)
print(f"Saved tidal CF hourly to {OUT_CF_TIDAL_HOURLY}")
print(f"Saved tidal CF daily means to {OUT_CF_TIDAL_DAILY}")

wind_p_df = pd.DataFrame(wind_p_dict, index=snapshots)
wind_p_df.to_csv(OUT_P_WIND_HOURLY)
wind_p_df.resample("D").mean().to_csv(OUT_P_WIND_DAILY)
print(f"Saved wind power hourly to {OUT_P_WIND_HOURLY}")
print(f"Saved wind power daily means to {OUT_P_WIND_DAILY}")

tidal_p_df = pd.DataFrame(tidal_p_dict, index=snapshots)
tidal_p_df.to_csv(OUT_P_TIDAL_HOURLY)
tidal_p_df.resample("D").mean().to_csv(OUT_P_TIDAL_DAILY)
print(f"Saved tidal power hourly to {OUT_P_TIDAL_HOURLY}")
print(f"Saved tidal power daily means to {OUT_P_TIDAL_DAILY}")

# ============================================================
# 8) SOLVE OPTIMISATION
# ============================================================

# Dispatch optimisation: meet demand at least cost given fixed wind/tidal availability
print("\nRunning optimisation (fixed capacities, dispatch only)…")
n.optimize(solver_name="highs")
print("…done.")

# ============================================================
# 9A) RESULTS & PLOT — Dual-axis (clean, with total renewables)
# ============================================================

print("\n=== RESULTS (wind + tidal only) ===")

# Generator dispatch (MW) and load (MW)
gen_p = n.generators_t.p.copy()
load_h = n.loads_t.p_set.sum(axis=1)

wind_cols  = n.generators.query("carrier == 'wind'").index
tidal_cols = n.generators.query("carrier == 'tidal'").index

# Total wind and total tidal generation (hourly)
wind_h  = gen_p[wind_cols].sum(axis=1)
tidal_h = gen_p[tidal_cols].sum(axis=1)

# Daily mean series for clearer plots
wind_d  = wind_h.resample("D").mean()
tidal_d = tidal_h.resample("D").mean()
total_d = wind_d + tidal_d
load_d  = load_h.resample("D").mean()

# Plot daily mean wind + total renewables + load (left axis) and tidal (right axis)
fig, ax1 = plt.subplots(figsize=(13, 6))

ax1.fill_between(
    wind_d.index,
    wind_d.values,
    color="orange",
    alpha=0.35,
    label="Wind (daily mean)",
)

ax1.plot(
    total_d.index,
    total_d.values,
    color="tab:green",
    linewidth=2.5,
    label="Wind + tidal (daily mean)",
)

ax1.plot(
    load_d.index,
    load_d.values,
    color="black",
    linestyle="--",
    linewidth=2.5,
    label="Load (daily mean)",
)

ax1.set_ylabel("Wind & demand (MW)")
ax1.set_xlabel("Date")
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
ax2.plot(
    tidal_d.index,
    tidal_d.values,
    color="tab:blue",
    marker="o",
    linewidth=2.5,
    label="Tidal (daily mean)",
)
ax2.set_ylabel("Tidal generation (MW)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")
ax2.set_ylim(bottom=0)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title(f"Daily mean wind, tidal and demand ({SCENARIO_LABEL})")

plt.tight_layout()
plt.savefig(
    f"daily_dual_axis_wind_tidal_{SCENARIO_LABEL}.png",
    dpi=220,
    bbox_inches="tight",
)
plt.show()
plt.close()

# ============================================================
# 9B) RESULTS & PLOT — Two-panel (recommended for thesis)
# ============================================================

print("\n=== RESULTS (wind + tidal only, two-panel plot) ===")

# Recompute (different plot type)
gen_p = n.generators_t.p.copy()
load_h = n.loads_t.p_set.sum(axis=1)

wind_cols  = n.generators.query("carrier == 'wind'").index
tidal_cols = n.generators.query("carrier == 'tidal'").index

wind_h  = gen_p[wind_cols].sum(axis=1)
tidal_h = gen_p[tidal_cols].sum(axis=1)

wind_d  = wind_h.resample("D").mean()
tidal_d = tidal_h.resample("D").mean()
load_d  = load_h.resample("D").mean()

# Two-panel layout: wind+load on top, tidal on bottom
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(13, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

ax1.fill_between(
    wind_d.index,
    wind_d.values,
    color="orange",
    alpha=0.4,
    label="Wind (daily mean)",
)
ax1.plot(
    load_d.index,
    load_d.values,
    color="black",
    linestyle="--",
    linewidth=2.5,
    label="Load (daily mean)",
)

ax1.set_ylabel("Power (MW)")
ax1.set_title(f"Daily mean wind generation and demand ({SCENARIO_LABEL})")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(
    tidal_d.index,
    tidal_d.values,
    color="tab:blue",
    marker="o",
    linewidth=2.5,
    label="Tidal (daily mean)",
)

ax2.set_ylabel("Tidal power (MW)")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    f"daily_two_panel_wind_tidal_{SCENARIO_LABEL}.png",
    dpi=220,
    bbox_inches="tight",
)
plt.show()
plt.close()
