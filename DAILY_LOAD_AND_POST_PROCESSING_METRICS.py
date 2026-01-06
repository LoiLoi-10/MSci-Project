# Fixed-capacity dispatch-only PyPSA model.
#
# What this script does:
#   1) Loads ERA5 wind and converts it to hourly wind CF per wind site using a power curve.
#   2) Loads a multi-site tidal CF CSV (already precomputed) and aligns it to ERA5 snapshots.
#   3) Loads a strict two-column hourly demand file (time, demand_MW) and aligns to ERA5.
#   4) Builds a PyPSA network with fixed wind + fixed tidal + fixed large gas backup.
#   5) Solves *dispatch only* (min variable cost) so demand is always met (gas fills gaps).
#   6) Exports per-site CFs and dispatch to feed into postprocess_metrics.py.

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pypsa

# ============================================================
# 0) CONFIG: FILE PATHS (CHANGE PER YEAR/MONTH)
# ============================================================

# ERA5 NetCDF containing wind components for UK offshore region and time window.
ERA5_WIND_FILE = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\ERA5_DATA\era5_uk_offshore_2023_12.nc")
# Turbine power curve CSV with columns: v (m/s), p (power or normalised power).
POWER_CURVE_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\power_curve.csv")
# Demand CSV: strict two-column hourly file.
# Must contain "time" and "demand_MW".
DEMAND_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\DEMAND_DATA\demand_clean_hourly_2023.csv")
# Site tables holding *installed/fixed capacity* for each generator site.
WIND_SITES_CSV = Path( r"C:\Users\one\OneDrive\Desktop\Msci Project\uk_offshore_wind_sites.csv")
TIDAL_SITES_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Project\uk_tidal_sites.csv")
# Precomputed tidal CF (wide format):
#  - one column "time"
#  - one column per tidal site_id
TIDAL_CF_MULTISITE_CSV = Path(r"C:\Users\one\OneDrive\Desktop\Msci Code\CMEMS_DATA\tidal_cf_2023_DEC_per_site.csv")

# ============================================================
# OUTPUT SWITCHES (MINIMAL)
# ============================================================

# These toggles control optional outputs (time series + plots + network export).
SAVE_TOTAL_DISPATCH_CSV = True
SAVE_TWO_PANEL_PLOT = True
EXPORT_NETWORK_RESULTS = True

# Exports used by postprocess_metrics.py
SAVE_CF_HOURLY_PER_SITE = True         # capacity factor time series per generator
SAVE_DISPATCH_PER_SITE = True          # dispatch time series per generator
SAVE_FIXED_CAPACITY_TABLE = True       # installed capacity per generator


SCENARIO_LABEL = "fixed_wind_tidal_2023_12"

# Outputs that postprocess_metrics.py expects (filenames).
OUT_WIND_CF_HOURLY = Path("wind_cf_hourly_per_site.csv")
OUT_TIDAL_CF_HOURLY = Path("tidal_cf_hourly_per_site.csv")

# Optional exports (per-generator dispatch).
OUT_WIND_DISPATCH_HOURLY = Path("wind_dispatch_hourly_per_site_MW.csv")
OUT_TIDAL_DISPATCH_HOURLY = Path("tidal_dispatch_hourly_per_site_MW.csv")

# Useful “installed capacity table” for sanity checks and post-processing.
OUT_FIXED_CAP_TABLE = Path("generators_fixed_capacities_MW.csv")

# ============================================================
# MODEL SETTINGS & COSTS
# ============================================================

# Wind shear exponent to scale reference wind speed (10m or 100m) to site hub height:
#   v(hub) = v(ref) * (hub/ref)^ALPHA
ALPHA = 0.14

# Dispatch costs (this is dispatch-only optimisation):
# - wind/tidal are free to dispatch (0)
# - gas has a positive marginal cost so it is used last
MARG_WIND = 0.0
MARG_TIDAL = 0.0
MARG_GAS = 150.0  # not realy used honestly since gas is just backup


# p_nom column is treated as the FIXED installed capacity in MW.
WIND_COLS = {
    "id": "site_id",
    "name": "name",
    "lat": "lat",
    "lon": "lon",
    "p_nom": "p_nom_max_MW",
    "hub_height": "hub_height_m",
}
TIDAL_COLS = {
    "id": "site_id",
    "name": "name",
    "lat": "lat",
    "lon": "lon",
    "p_nom": "p_nom_max_MW",
}

# ============================================================
# HELPERS
# ============================================================

def find_lat_lon_dims(dset: xr.Dataset) -> tuple[str, str]:
    """
    Detect latitude and longitude dimension/coordinate names in an xarray Dataset.
    Handles common naming variations containing 'lat'/'lon'.
    """
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
            if "lat" in str(c).lower():
                lat = c
                break
    if lon is None:
        for c in dset.coords:
            if "lon" in str(c).lower():
                lon = c
                break
    if lat is None or lon is None:
        raise ValueError(f"Could not detect lat/lon. Dims={list(dset.dims)}, Coords={list(dset.coords)}")
    return str(lat), str(lon)


def detect_or_build_time(dset: xr.Dataset) -> tuple[str, pd.DatetimeIndex]:
    """
    Detect a 1D datetime coordinate to use as the time axis.
    Returns the coordinate name and a DatetimeIndex.
    """
    for c in dset.coords:
        arr = dset[c]
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.datetime64):
            return str(c), pd.DatetimeIndex(arr.values)
    for nm in ("time", "valid_time", "forecast_time", "date", "datetime"):
        if nm in dset.variables:
            arr = dset[nm]
            return nm, pd.DatetimeIndex(arr.values)
    raise ValueError("Could not determine time coordinate.")


def load_power_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a turbine power curve.
    Expected columns: v (wind speed), p (power or already-normalised power).
    Normalises p to [0,1] if max(p) > 1.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing power curve: {csv_path}")
    df = pd.read_csv(csv_path).dropna().sort_values("v")
    if not {"v", "p"}.issubset(df.columns) or len(df) < 2:
        raise ValueError("power_curve.csv must contain columns v and p with at least 2 rows.")
    v_curve = df["v"].to_numpy(float)
    p_raw = df["p"].to_numpy(float)
    p_curve = p_raw / p_raw.max() if p_raw.max() > 1.0 else p_raw.copy()
    p_curve = np.clip(p_curve, 0.0, 1.0)
    return v_curve, p_curve


def nearest_grid_index(lat_array: np.ndarray, lon_array: np.ndarray, lat: float, lon: float) -> tuple[int, int]:
    """
    Nearest-neighbour lookup of lat/lon grid indices for a given site coordinate.
    (No interpolation; the site is assigned to the nearest ERA5 cell.)
    """
    lat_array = np.asarray(lat_array)
    lon_array = np.asarray(lon_array)
    ilat = int(np.argmin(np.abs(lat_array - lat)))
    ilon = int(np.argmin(np.abs(lon_array - lon)))
    return ilat, ilon


def speed_to_cf_from_curve(
    speed_series: pd.Series,
    v_curve: np.ndarray,
    p_curve: np.ndarray,
    snapshots: pd.DatetimeIndex,
) -> pd.Series:
    """
    Convert a wind speed time series to capacity factor using the power curve,
    then reindex strictly onto the target snapshots.
    Fills small gaps via interpolation + edge fill.
    """
    vals = np.asarray(speed_series.values, dtype=float)
    cf_arr = np.interp(vals, v_curve, p_curve, left=p_curve[0], right=p_curve[-1])
    cf_arr = np.clip(cf_arr, 0.0, 1.0)
    s = pd.Series(cf_arr, index=pd.DatetimeIndex(speed_series.index)).sort_index()
    s = s.reindex(snapshots)
    if s.isna().any():
        s = s.interpolate(limit=3).bfill().ffill()
    return s


def load_demand_hourly_two_col_strict(csv_path: Path, snapshots: pd.DatetimeIndex) -> pd.Series:
    """
    Strict demand loader for a 2-column hourly file:
        time, demand_MW (hourly_clean.csv format)

    Behaviour:
      - robust time parsing (tries multiple formats)
      - drops non-numeric demand rows
      - slices to [snapshots[0], snapshots[-1]]
      - reindexes to snapshots and requires full coverage (raises if gaps remain)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Demand CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    if "time" not in df.columns or "demand_MW" not in df.columns:
        raise ValueError(f"Demand CSV must have columns ['time','demand_MW']. Found: {df.columns.tolist()}")

    # Clean up weird whitespace/BOM/nbsp patterns before parsing times.
    time_str = (
        df["time"].astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Parse time with fallback formats.
    t = pd.to_datetime(time_str, errors="coerce")
    if t.isna().any():
        t2 = pd.to_datetime(time_str[t.isna()], format="%d/%m/%Y %H:%M", errors="coerce")
        t.loc[t.isna()] = t2
    if t.isna().any():
        t3 = pd.to_datetime(time_str[t.isna()], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        t.loc[t.isna()] = t3

    # Parse numeric demand and drop bad rows.
    demand = (
        df["demand_MW"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    demand = pd.to_numeric(demand, errors="coerce")
    bad_rows = int(demand.isna().sum())
    if bad_rows > 0:
        print(f"[demand] Dropping {bad_rows} rows with missing/non-numeric demand_MW.")

    out = pd.DataFrame({"demand_MW": demand.values}, index=pd.DatetimeIndex(t)).dropna()
    out = out[~out.index.duplicated(keep="first")].sort_index()

    # Restrict demand to the ERA5 time window and enforce full coverage.
    start, end = snapshots[0], snapshots[-1]
    out = out.loc[(out.index >= start) & (out.index <= end)]

    s = out["demand_MW"].reindex(snapshots)
    if s.isna().any():
        s = s.interpolate(limit=2)

    # If any missing remains after small interpolation: fail with diagnostics.
    if s.isna().any():
        missing = s.index[s.isna()][:10]
        raise ValueError(
            "Demand does not fully cover ERA5 snapshots.\n"
            f"ERA5: {start} → {end} (n={len(snapshots)})\n"
            f"Demand range after slicing: {out.index.min()} → {out.index.max()} (n={len(out)})\n"
            f"Missing examples: {missing.tolist()}"
        )

    print("\nDemand stats (MW) over model horizon:")
    print("  min:", float(s.min()), "max:", float(s.max()), "mean:", float(s.mean()))
    return s


def load_sites(csv_path: Path, cols: dict, kind: str) -> pd.DataFrame:
    """
    Load wind/tidal site tables and output a standardised DataFrame with:
        site_id, site_name, lat, lon, p_nom_mw
    plus hub_height_m for wind.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"{kind} sites CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = [cols["id"], cols["name"], cols["lat"], cols["lon"], cols["p_nom"]]
    if kind == "wind":
        required.append(cols["hub_height"])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{kind} site CSV missing columns: {missing}. Found: {df.columns.tolist()}")

    out = pd.DataFrame({
        "site_id": df[cols["id"]].astype(str),
        "site_name": df[cols["name"]].astype(str),
        "lat": df[cols["lat"]].astype(float),
        "lon": df[cols["lon"]].astype(float),
        "p_nom_mw": pd.to_numeric(df[cols["p_nom"]], errors="coerce").astype(float),
    })

    # Wind requires hub height for shear scaling.
    if kind == "wind":
        out["hub_height_m"] = pd.to_numeric(df[cols["hub_height"]], errors="coerce").astype(float)
        if out["hub_height_m"].isna().any():
            bad = out.loc[out["hub_height_m"].isna(), "site_id"].head(10).tolist()
            raise ValueError(f"wind sites file has non-numeric hub_height entries for site_id: {bad}")

    return out


def load_tidal_cf_multisite_strict(csv_path: Path, snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load multi-site tidal CF file and reindex strictly to model snapshots.
    Requirements:
      - has a 'time' column
      - remaining columns are site_id strings matching tidal_sites['site_id']
      - full coverage over snapshots (raises if gaps)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Tidal CF CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError(f"Tidal CF CSV must contain a 'time' column. Found: {df.columns.tolist()}")

    # Parse time, allowing mixed formats with a fallback.
    t = pd.to_datetime(df["time"], errors="coerce", format="mixed", dayfirst=True)
    if t.isna().any():
        t2 = pd.to_datetime(df.loc[t.isna(), "time"], errors="coerce", dayfirst=True)
        t.loc[t.isna()] = t2

    if t.isna().any():
        bad = df.loc[t.isna(), "time"].astype(str).unique()[:10]
        raise ValueError(f"Unparseable timestamps in tidal CF CSV. Examples: {list(bad)}")

    # Move time into index, coerce all CF columns to numeric.
    df = df.drop(columns=["time"]).copy()
    df.index = pd.DatetimeIndex(t)
    df = df.sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Window and strict reindex to snapshots.
    df_win = df.loc[(df.index >= snapshots[0]) & (df.index <= snapshots[-1])]
    out = df_win.reindex(snapshots)

    if out.isna().any().any():
        raise ValueError(
            "Tidal CF CSV does not fully cover model snapshots.\n"
            f"Snapshots: {snapshots[0]} → {snapshots[-1]}\n"
            f"CF range (after window): {df_win.index.min()} → {df_win.index.max()}\n"
        )

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Basic existence checks so missing inputs fail early with clear output.
    print("ERA5 exists:", ERA5_WIND_FILE.exists())
    print("Demand exists:", DEMAND_CSV.exists())
    print("Wind sites CSV exists:", WIND_SITES_CSV.exists())
    print("Tidal sites CSV exists:", TIDAL_SITES_CSV.exists())
    print("Tidal CF CSV exists:", TIDAL_CF_MULTISITE_CSV.exists())
    print()

    # ---- Load ERA5 dataset
    ds = xr.open_dataset(ERA5_WIND_FILE)

    # Normalise variable names across common ERA5 encodings.
    rename_map = {}
    if "10m_u_component_of_wind" in ds: rename_map["10m_u_component_of_wind"] = "u10"
    if "10m_v_component_of_wind" in ds: rename_map["10m_v_component_of_wind"] = "v10"
    if "100m_u_component_of_wind" in ds: rename_map["100m_u_component_of_wind"] = "u100"
    if "100m_v_component_of_wind" in ds: rename_map["100m_v_component_of_wind"] = "v100"
    if rename_map:
        ds = ds.rename(rename_map)

    # Identify grid dims and time index.
    lat_dim, lon_dim = find_lat_lon_dims(ds)
    time_dim, time_index = detect_or_build_time(ds)

    # Try to reorder dims to (time, lat, lon) for consistent indexing.
    try:
        ds = ds.transpose(time_dim, lat_dim, lon_dim)
    except Exception:
        pass

    snapshots = pd.DatetimeIndex(time_index)
    print(f"ERA5 snapshots: {snapshots[0]} → {snapshots[-1]} (n={len(snapshots)})")

    # Compute reference wind speed magnitude at either 100m or 10m depending on available variables.
    if "u100" in ds and "v100" in ds:
        ws_ref = xr.apply_ufunc(np.hypot, ds["u100"], ds["v100"])
        REF_HEIGHT = 100.0
    elif "u10" in ds and "v10" in ds:
        ws_ref = xr.apply_ufunc(np.hypot, ds["u10"], ds["v10"])
        REF_HEIGHT = 10.0
    else:
        raise ValueError("Neither u100/v100 nor u10/v10 present in ERA5 file.")

    lat_vals = ds[lat_dim].values
    lon_vals = ds[lon_dim].values

    # ---- Demand: strict alignment to ERA5 snapshots
    load_series = load_demand_hourly_two_col_strict(DEMAND_CSV, snapshots)

    # ---- Sites + turbine power curve
    wind_sites = load_sites(WIND_SITES_CSV, WIND_COLS, "wind")
    tidal_sites = load_sites(TIDAL_SITES_CSV, TIDAL_COLS, "tidal")
    v_curve, p_curve = load_power_curve(POWER_CURVE_CSV)

    print(f"\nLoaded {len(wind_sites)} offshore wind sites.")
    print(f"Loaded {len(tidal_sites)} tidal sites.")

    # ---- Build PyPSA network (single bus system)   ---> improve for future with multiple buses?
    n = pypsa.Network()
    for c in ["electricity", "wind", "tidal", "gas"]:
        if c not in n.carriers.index:
            n.add("Carrier", c)

    n.set_snapshots(list(snapshots))
    n.add("Bus", "uk_bus", v_nom=400.0, carrier="electricity")
    n.add("Load", "uk_load", bus="uk_bus", p_set=load_series)

    # Pre-create p_max_pu DataFrame to store per-generator dispatch limits.
    n.generators_t.p_max_pu = pd.DataFrame(index=snapshots)

    # ---- Gas generator sized large so optimisation is always feasible.
    peak_load = float(load_series.max())
    gas_p_nom = max(60_000.0, 1.2 * peak_load)
    n.add(
        "Generator", "gas_uk",
        bus="uk_bus", carrier="gas",
        p_nom=gas_p_nom,
        p_nom_extendable=False,
        marginal_cost=MARG_GAS,
    )
    print(f"[gas] peak_load={peak_load:.1f} MW, using fixed p_nom={gas_p_nom:.1f} MW")

    # ---- DataFrames to export per-site CF time series (used by postprocess_metrics.py)
    wind_cf_out = pd.DataFrame(index=snapshots)
    tidal_cf_out = pd.DataFrame(index=snapshots)

    # ---- Add wind generators
    # For each site:
    #   - find nearest ERA5 cell
    #   - scale speed to hub height using shear power law
    #   - map speed to CF via power curve
    #   - add fixed-capacity generator with p_max_pu = CF(t)
    added_wind = 0
    for _, row in wind_sites.iterrows():
        site_id = row["site_id"]
        lat = float(row["lat"])
        lon = float(row["lon"])
        p_nom_mw = float(row["p_nom_mw"])
        hub_height_m = float(row["hub_height_m"])

        ilat, ilon = nearest_grid_index(lat_vals, lon_vals, lat, lon)
        ws_cell_ref = ws_ref.isel({lat_dim: ilat, lon_dim: ilon})

        factor = (hub_height_m / REF_HEIGHT) ** ALPHA
        speed_site = ws_cell_ref * factor

        speed_series = pd.Series(
            np.asarray(speed_site.values, dtype=float),
            index=pd.DatetimeIndex(np.asarray(ds[time_dim].values)),
        )

        cf = speed_to_cf_from_curve(speed_series, v_curve, p_curve, snapshots)

        gen_name = f"wind_{site_id}"
        n.add(
            "Generator", gen_name,
            bus="uk_bus",
            carrier="wind",
            p_nom=p_nom_mw,
            p_nom_extendable=False,
            marginal_cost=MARG_WIND,
        )

        # Set dispatch ceiling for every hour.
        n.generators_t.p_max_pu[gen_name] = cf.values

        # Export CF for post-processing.
        wind_cf_out[gen_name] = cf.values
        added_wind += 1

    print(f"Added {added_wind} FIXED wind generators.")

    # ---- Add FIXED tidal generators from precomputed CF CSV
    # CF CSV columns must match tidal site_id values.
    tidal_cf_df = load_tidal_cf_multisite_strict(TIDAL_CF_MULTISITE_CSV, snapshots)

    missing_cols = []
    added_tidal = 0
    for _, row in tidal_sites.iterrows():
        site_id = row["site_id"]
        p_nom_mw = float(row["p_nom_mw"])
        gen_name = f"tidal_{site_id}"

        # If CF file doesn't include this site_id, skip it and warn.
        if site_id not in tidal_cf_df.columns:
            missing_cols.append(site_id)
            continue

        cf_site = tidal_cf_df[site_id]

        n.add(
            "Generator", gen_name,
            bus="uk_bus",
            carrier="tidal",
            p_nom=p_nom_mw,
            p_nom_extendable=False,
            marginal_cost=MARG_TIDAL,
        )

        # Hourly dispatch ceiling = CF(t)
        n.generators_t.p_max_pu[gen_name] = cf_site.values

        # Export CF.
        tidal_cf_out[gen_name] = cf_site.values
        added_tidal += 1

    if missing_cols:
        print("\nWARNING: These tidal site_ids were not found as columns in the tidal CF CSV:")
        print(" ", missing_cols)

    print(f"Added {added_tidal} FIXED tidal generators.")

    # ---- Export hourly CF per generator for postprocess_metrics.py
    if SAVE_CF_HOURLY_PER_SITE:
        wind_cf_out.to_csv(OUT_WIND_CF_HOURLY)
        tidal_cf_out.to_csv(OUT_TIDAL_CF_HOURLY)
        print(f"Saved: {OUT_WIND_CF_HOURLY}")
        print(f"Saved: {OUT_TIDAL_CF_HOURLY}")

    # ---- Export installed capacities table (quick sanity check / post-processing input)
    if SAVE_FIXED_CAPACITY_TABLE:
        cap_tbl = n.generators[["carrier", "p_nom"]].copy()
        cap_tbl.to_csv(OUT_FIXED_CAP_TABLE)
        print(f"Saved: {OUT_FIXED_CAP_TABLE}")

    # ---- Solve dispatch-only optimisation:
    # Objective is variable cost; since wind/tidal are free, gas is used only when needed.
    print("\nRunning optimisation (dispatch with fixed capacities)…")
    status, condition = n.optimize(solver_name="highs")
    print("…done.")
    if condition != "optimal":
        raise RuntimeError(f"Optimisation failed: termination_condition={condition}")

    # ---- Extract dispatch results
    gen_p = n.generators_t.p.copy()

    # Some PyPSA versions return MultiIndex columns; flatten them if needed.
    if isinstance(gen_p.columns, pd.MultiIndex):
        gen_p.columns = gen_p.columns.get_level_values(-1)

    # Identify generator groups by carrier.
    wind_cols = n.generators.query("carrier=='wind'").index
    tidal_cols = n.generators.query("carrier=='tidal'").index
    gas_cols = n.generators.query("carrier=='gas'").index

    # Aggregate dispatch per tech (MW at each hour).
    wind_h = gen_p.loc[:, pd.Index(gen_p.columns).intersection(wind_cols)].sum(axis=1)
    tidal_h = gen_p.loc[:, pd.Index(gen_p.columns).intersection(tidal_cols)].sum(axis=1)
    gas_h = gen_p.loc[:, pd.Index(gen_p.columns).intersection(gas_cols)].sum(axis=1)
    load_h = n.loads_t.p_set.sum(axis=1)

    # Reliability proxy: hours where dispatch does not meet load.
    # In a feasible linear dispatch with large gas, this should be ~0.
    residual = load_h - (wind_h + tidal_h + gas_h)
    LOLH = int((residual > 1e-3).sum())

    print("\n=== FIXED CAPACITIES (MW) ===")
    print(n.generators[["carrier", "p_nom"]].sort_values("p_nom", ascending=False).head(30))

    # Energy over horizon (MWh) since hourly series: MW summed over hours.
    print("\nDispatch summary (MWh over horizon):")
    print(f"  Wind : {float(wind_h.sum()):,.0f}")
    print(f"  Tidal: {float(tidal_h.sum()):,.0f}")
    print(f"  Gas  : {float(gas_h.sum()):,.0f}")
    print(f"\nLoss-of-load hours (LOLH): {LOLH} h over {len(n.snapshots)} h")

    # ---- Export per-site dispatch 
    if SAVE_DISPATCH_PER_SITE:
        wind_dispatch = gen_p.loc[:, pd.Index(gen_p.columns).intersection(wind_cols)].copy()
        tidal_dispatch = gen_p.loc[:, pd.Index(gen_p.columns).intersection(tidal_cols)].copy()
        wind_dispatch.to_csv(OUT_WIND_DISPATCH_HOURLY)
        tidal_dispatch.to_csv(OUT_TIDAL_DISPATCH_HOURLY)
        print(f"Saved: {OUT_WIND_DISPATCH_HOURLY}")
        print(f"Saved: {OUT_TIDAL_DISPATCH_HOURLY}")

    # ---- Minimal export: total dispatch time series (aggregated)
    if SAVE_TOTAL_DISPATCH_CSV:
        total_dispatch = pd.DataFrame({
            "wind_MW": wind_h,
            "tidal_MW": tidal_h,
            "gas_MW": gas_h,
            "total_MW": wind_h + tidal_h + gas_h,
            "load_MW": load_h,
        })
        total_dispatch.to_csv("total_dispatch_hourly_MW.csv")
        print("Saved: total_dispatch_hourly_MW.csv")

    # ---- Simple daily plot for quick visual checks
    if SAVE_TWO_PANEL_PLOT:
        wind_d = wind_h.resample("D").mean()
        tidal_d = tidal_h.resample("D").mean()
        load_d = load_h.resample("D").mean()

        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=(13, 8),
            sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )

        ax1.fill_between(wind_d.index, wind_d.values, alpha=0.4, label="Wind (Daily Mean)")
        ax1.plot(load_d.index, load_d.values, linestyle="--", linewidth=2.5, label="Load (Daily Mean)")
        ax1.set_ylabel("Power (MW)")
        ax1.set_title(f"Daily mean wind & demand ({SCENARIO_LABEL})")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(tidal_d.index, tidal_d.values, marker="o", linewidth=2.5, label="Tidal (Daily Mean)")
        ax2.set_ylabel("Power (MW)")
        ax2.set_xlabel("Date")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plot_name = f"daily_two_panel_wind_tidal_{SCENARIO_LABEL}.png"
        plt.savefig(plot_name, dpi=220, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"Saved plot: {plot_name}")

    # ---- Export full network tables to CSV folder (so postprocess can import network)
    if EXPORT_NETWORK_RESULTS:
        out_dir = f"results_wind_tidal_{SCENARIO_LABEL}"
        n.export_to_csv_folder(out_dir)
        print(f"Saved results to '{out_dir}'.")


if __name__ == "__main__":
    main()
