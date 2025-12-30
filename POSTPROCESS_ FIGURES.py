# Purpose:
#   Post-process a solved PyPSA network + per-site hourly capacity-factor (CF) time series
#   to compute system-level metrics for wind, tidal, and their combined portfolio.
#
# Outputs (printed):
#   - Temporal variability (CV) for system-wide wind and tidal CF
#   - Complementarity (Pearson correlation between CF series)
#   - System adequacy proxy (coincident low hours)
#   - Economic proxy (LCOE for wind, tidal, and portfolio using CRF annualised CAPEX + OPEX)
#   - Reliability proxy (LOLH based on residual load > 0)

from pathlib import Path
import numpy as np
import pandas as pd
import pypsa

# ----------------------------
# 1) INPUTS
# ----------------------------

# Hourly CF time series per generator/site.
# Expected: index is timestamps; values in [0, 1].
WIND_CF_HOURLY  = Path("wind_cf_hourly_per_site.csv")
TIDAL_CF_HOURLY = Path("tidal_cf_hourly_per_site.csv")

# Folder containing a PyPSA network exported as CSVs (pypsa.Network().export_to_csv_folder(...) format).
NETWORK_RESULTS_FOLDER = Path("results_wind_tidal_fixed_fixed_wind_tidal")

# Overnight CAPEX (£/MW) — i.e., total upfront cost per MW installed (NOT annualised).
CAPEX_WIND_GBP_PER_MW  = 3_000_000.0
CAPEX_TIDAL_GBP_PER_MW = 3_500_000.0

# Fixed OPEX as a fraction of CAPEX per year (dimensionless / year).
# Example: 0.03 means fixed OPEX = 3% of overnight CAPEX each year.
OPEX_FRAC_PER_YEAR_WIND  = 0.03
OPEX_FRAC_PER_YEAR_TIDAL = 0.04

# --- Economic realism variables for annualising CAPEX with a CRF (capital recovery factor)
DISCOUNT_RATE = 0.07
LIFETIME_WIND_Y  = 25
LIFETIME_TIDAL_Y = 25

# Optional variable OPEX (£/MWh) applied to energy produced.
# If you want “CAPEX + fixed OPEX only”, set these to 0.0.
VAR_OPEX_WIND_GBP_PER_MWH  = 2.38
VAR_OPEX_TIDAL_GBP_PER_MWH = 5.00

# ----------------------------
# 2) LOAD DATA
# ----------------------------

# Read CF files.
# - index_col=0 assumes first column is the timestamp column
# - parse_dates=True converts index to DatetimeIndex
wind_cf  = pd.read_csv(WIND_CF_HOURLY, index_col=0, parse_dates=True)
tidal_cf = pd.read_csv(TIDAL_CF_HOURLY, index_col=0, parse_dates=True)

# Align timestamps exactly by intersection so wind and tidal have identical time axis.
idx = wind_cf.index.intersection(tidal_cf.index)
wind_cf  = wind_cf.loc[idx].copy()
tidal_cf = tidal_cf.loc[idx].copy()

# Load the solved (or at least exported) PyPSA network from CSV folder.
n = pypsa.Network()
n.import_from_csv_folder(str(NETWORK_RESULTS_FOLDER))

# Identify wind/tidal generators by carrier in the network.
# (Assumes carrier is set to exactly 'wind' and 'tidal'.)
wind_gens  = n.generators.query("carrier == 'wind'").index
tidal_gens = n.generators.query("carrier == 'tidal'").index

# Ensure generator names exist in CF CSV columns too (name matching is critical).
# After this, wind_gens/tidal_gens are the common set.
wind_gens  = [g for g in wind_gens  if g in wind_cf.columns]
tidal_gens = [g for g in tidal_gens if g in tidal_cf.columns]

# Hard fail early if name matching yields nothing.
if len(wind_gens) == 0:
    raise ValueError("No wind generators found in both network and wind CF CSV.")
if len(tidal_gens) == 0:
    raise ValueError("No tidal generators found in both network and tidal CF CSV.")

# Installed capacities (MW) taken from network generators table.
# NOTE: using "p_nom" here means “fixed installed capacity”.
# If you exported an optimised network and wanted the optimised builds, you might use "p_nom_opt".
p_nom_wind  = n.generators.loc[wind_gens,  "p_nom"]
p_nom_tidal = n.generators.loc[tidal_gens, "p_nom"]

# System-wide CF time series computed as capacity-weighted average CF:
#   CF_sys(t) = sum_i CF_i(t) * p_nom_i / sum_i p_nom_i
CFwind_sys = (wind_cf[wind_gens].multiply(p_nom_wind, axis=1).sum(axis=1) /
              p_nom_wind.sum())

CFtidal_sys = (tidal_cf[tidal_gens].multiply(p_nom_tidal, axis=1).sum(axis=1) /
               p_nom_tidal.sum())

# ----------------------------
# 3) TEMPORAL VARIABILITY (CV)
# ----------------------------
# CV = std / mean of the system-wide CF time series.
def coeff_of_variation(x: pd.Series) -> float:
    mu = float(x.mean())
    sigma = float(x.std(ddof=0))
    return np.nan if abs(mu) < 1e-12 else sigma / mu

CV_wind  = coeff_of_variation(CFwind_sys)
CV_tidal = coeff_of_variation(CFtidal_sys)

# ----------------------------
# 4) COMPLEMENTARITY (Pearson correlation)
# ----------------------------
# Pearson r between wind and tidal system-wide CF series.
# Negative/low correlation suggests complementarity (less co-variation).
rw_t = float(CFwind_sys.corr(CFtidal_sys))

# ----------------------------
# 5) SYSTEM ADEQUACY (Coincident-low-hours ratio)
# ----------------------------
# Define “low” as below the 20th percentile for each technology.
# Then measure fraction of hours when both are simultaneously “low”.
p20_wind  = float(CFwind_sys.quantile(0.20))
p20_tidal = float(CFtidal_sys.quantile(0.20))

coincident_low_mask = (CFwind_sys < p20_wind) & (CFtidal_sys < p20_tidal)
coincident_low_hours = int(coincident_low_mask.sum())
coincident_low_ratio = coincident_low_hours / len(idx)

# ----------------------------
# 6) ECONOMIC (LCOE)  --- FIXED ---
# ----------------------------
# LCOE formulation here:
#   Annualised cost (annualised CAPEX via CRF + annual fixed OPEX + annual variable OPEX)
#   divided by
#   Annualised energy production (computed over model horizon and scaled to 8760 hours).
#
# Important conventions embedded here:
#   - CAPEX inputs are "overnight" £/MW (not annualised)
#   - CRF turns overnight CAPEX into an equivalent annual payment (£/year)
#   - Fixed OPEX is a fraction of overnight CAPEX per year
#   - Variable OPEX is £/MWh applied to annual energy output
def crf(r: float, n_years: int) -> float:
    """Capital recovery factor / annuity factor."""
    if n_years <= 0:
        raise ValueError("n_years must be > 0")
    if abs(r) < 1e-12:
        return 1.0 / n_years
    return (r * (1 + r) ** n_years) / ((1 + r) ** n_years - 1)

# Model horizon duration (assumes hourly sampling; uses number of timestamps).
hours_period = len(idx)

# Annual scaling factor: convert “energy over horizon” to “energy per year equivalent”
# by multiplying by 8760 / hours_period.
hours_year = 8760.0
annualisation_factor = hours_year / hours_period

# Energy produced over the horizon (MWh).
# Since CF is dimensionless and p_nom is MW, CF * p_nom gives MW output each hour.
# Summing over hours gives MW·h = MWh.
Ewind_period_MWh  = float((CFwind_sys  * p_nom_wind.sum()).sum())   # MW * h
Etidal_period_MWh = float((CFtidal_sys * p_nom_tidal.sum()).sum())

# Annual equivalent energy (MWh/year)
Ewind_annual_MWh  = Ewind_period_MWh  * annualisation_factor
Etidal_annual_MWh = Etidal_period_MWh * annualisation_factor

# Total overnight CAPEX (£) for each tech = installed MW * £/MW
Ccapex_wind  = float(p_nom_wind.sum()  * CAPEX_WIND_GBP_PER_MW)
Ccapex_tidal = float(p_nom_tidal.sum() * CAPEX_TIDAL_GBP_PER_MW)

# Annualised CAPEX (£/year) using CRF
Ccapex_wind_annual  = Ccapex_wind  * crf(DISCOUNT_RATE, LIFETIME_WIND_Y)
Ccapex_tidal_annual = Ccapex_tidal * crf(DISCOUNT_RATE, LIFETIME_TIDAL_Y)

# Fixed OPEX (£/year) as % of overnight CAPEX
Opex_wind_annual  = Ccapex_wind  * OPEX_FRAC_PER_YEAR_WIND
Opex_tidal_annual = Ccapex_tidal * OPEX_FRAC_PER_YEAR_TIDAL

# Variable OPEX (£/year) = annual energy * (£/MWh)
VarOpex_wind_annual  = Ewind_annual_MWh  * VAR_OPEX_WIND_GBP_PER_MWH
VarOpex_tidal_annual = Etidal_annual_MWh * VAR_OPEX_TIDAL_GBP_PER_MWH

# LCOE (£/MWh) for each technology
LCOE_wind  = (Ccapex_wind_annual  + Opex_wind_annual  + VarOpex_wind_annual)  / Ewind_annual_MWh
LCOE_tidal = (Ccapex_tidal_annual + Opex_tidal_annual + VarOpex_tidal_annual) / Etidal_annual_MWh

# Portfolio LCOE (wind + tidal combined)
# Computed as (total annual costs) / (total annual energy).
E_total_annual = Ewind_annual_MWh + Etidal_annual_MWh
C_total_annual = (Ccapex_wind_annual + Ccapex_tidal_annual) + (Opex_wind_annual + Opex_tidal_annual) + (VarOpex_wind_annual + VarOpex_tidal_annual)
LCOE_portfolio = C_total_annual / E_total_annual

# ----------------------------
# 7) RELIABILITY (LOLH from optimisation results)
# ----------------------------
# LOLH here is computed from the network dispatch results:
#   residual(t) = load(t) - total_generation(t)
# Count hours where residual > 0.
#
# NOTE: This assumes:
#   - n.generators_t.p exists and is aligned with the time index
#   - the network results include all relevant generation (incl. backup, storage links, etc. if present, not much useful in this code)
load = n.loads_t.p_set.sum(axis=1).reindex(idx)
gen_total = n.generators_t.p.sum(axis=1).reindex(idx)

residual = load - gen_total
LOLH = int((residual > 1e-3).sum())

# ----------------------------
# 8) REPORT
# ----------------------------
# Prints all derived metrics to file.
print("\n=== POST-PROCESSING METRICS (system-wide, capacity-weighted) ===")
print("Temporal variability (CV):")
print(f"  CV_wind  = {CV_wind:.3f}")
print(f"  CV_tidal = {CV_tidal:.3f}")

print("\nComplementarity (Pearson r):")
print(f"  r_w,t = {rw_t:.3f}")

print("\nSystem adequacy (coincident low hours):")
print(f"  wind 20th percentile  = {p20_wind:.3f}")
print(f"  tidal 20th percentile = {p20_tidal:.3f}")
print(f"  coincident_low_hours  = {coincident_low_hours} h / {hours_period} h")
print(f"  coincident_low_ratio  = {coincident_low_ratio:.3%}")

print("\nEconomic (LCOE) — annualised CAPEX (CRF) + fixed+variable OPEX, energy scaled to 8760h:")
print(f"  Assumptions: r={DISCOUNT_RATE:.2%}, wind life={LIFETIME_WIND_Y}y, tidal life={LIFETIME_TIDAL_Y}y")
print(f"  Wind: CAPEX £/MW={CAPEX_WIND_GBP_PER_MW:,.0f}, fixed OPEX={OPEX_FRAC_PER_YEAR_WIND:.1%} of CAPEX/yr, var OPEX £/MWh={VAR_OPEX_WIND_GBP_PER_MWH:.2f}")
print(f"  Tidal: CAPEX £/MW={CAPEX_TIDAL_GBP_PER_MW:,.0f}, fixed OPEX={OPEX_FRAC_PER_YEAR_TIDAL:.1%} of CAPEX/yr, var OPEX £/MWh={VAR_OPEX_TIDAL_GBP_PER_MWH:.2f}")
print(f"  Wind LCOE  = £{LCOE_wind:,.2f} / MWh")
print(f"  Tidal LCOE = £{LCOE_tidal:,.2f} / MWh")
print(f"  Portfolio LCOE (wind+tidal) = £{LCOE_portfolio:,.2f} / MWh")

print("\nReliability (LOLH):")
print(f"  LOLH = {LOLH} h over {hours_period} h")
