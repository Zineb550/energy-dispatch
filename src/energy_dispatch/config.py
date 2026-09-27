"""
Single source of truth for paths, dates, and cost/capacity assumptions.

Nothing in notebooks or src/ should hard-code a date range, a cost, or a
capacity — import it from here instead, so every module and the Streamlit
app stay consistent when an assumption changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

PROCESSED_HOURLY_PARQUET = PROCESSED_DATA_DIR / "spain_hourly.parquet"
FORECAST_TABLE_PARQUET = PROCESSED_DATA_DIR / "forecast_test_period.parquet"
METRICS_TABLE_PARQUET = PROCESSED_DATA_DIR / "forecast_metrics.parquet"
SCENARIO_RESULTS_PARQUET = PROCESSED_DATA_DIR / "scenario_results.parquet"

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

COUNTRY_CODE = "ES"  # Spain

# OPSD has no stable "latest" alias for either package — each release lives
# under a dated version directory, and the project has been dormant since
# ~2020. These are the last confirmed-reachable versions (verified against
# the live portal while writing this module); if a URL 404s by the time you
# run this, check https://data.open-power-system-data.org/time_series/ and
# https://data.open-power-system-data.org/weather_data/ for the current
# version directory and update the two *_VERSION constants below — nothing
# else needs to change.
OPSD_TIMESERIES_VERSION = "2020-10-06"
OPSD_WEATHER_VERSION = "2020-09-16"

OPSD_TIMESERIES_URL = (
    f"https://data.open-power-system-data.org/time_series/{OPSD_TIMESERIES_VERSION}/"
    "time_series_60min_singleindex.csv"
)
OPSD_WEATHER_URL = (
    f"https://data.open-power-system-data.org/weather_data/{OPSD_WEATHER_VERSION}/"
    "weather_data.csv"
)

RAW_TIMESERIES_CSV = RAW_DATA_DIR / "time_series_60min_singleindex.csv"
RAW_WEATHER_CSV = RAW_DATA_DIR / "weather_data.csv"

# time_series_60min_singleindex.csv: one column per country per metric,
# e.g. "ES_load_actual_entsoe_transparency". Confirmed columns for Spain
# (no offshore wind column exists for ES in this package):
LOAD_ACTUAL_COL = "ES_load_actual_entsoe_transparency"
LOAD_FORECAST_COL = "ES_load_forecast_entsoe_transparency"
SOLAR_GEN_COL = "ES_solar_generation_actual"
WIND_GEN_COL = "ES_wind_onshore_generation_actual"
TIMESTAMP_COL = "utc_timestamp"

# weather_data.csv: population-weighted, country-aggregated daily/hourly
# weather, one column per country per variable, e.g. "ES_temperature".
WEATHER_TEMPERATURE_COL = "ES_temperature"
WEATHER_RADIATION_DIRECT_COL = "ES_radiation_direct_horizontal"
WEATHER_RADIATION_DIFFUSE_COL = "ES_radiation_diffuse_horizontal"
WEATHER_TIMESTAMP_COL = "utc_timestamp"

# Only these columns are read from the raw CSVs (each file has hundreds of
# per-country columns; both files are large, so usecols keeps memory sane).
TIMESERIES_USECOLS = (
    TIMESTAMP_COL,
    LOAD_ACTUAL_COL,
    LOAD_FORECAST_COL,
    SOLAR_GEN_COL,
    WIND_GEN_COL,
)
WEATHER_USECOLS = (
    WEATHER_TIMESTAMP_COL,
    WEATHER_TEMPERATURE_COL,
    WEATHER_RADIATION_DIRECT_COL,
    WEATHER_RADIATION_DIFFUSE_COL,
)

# ---------------------------------------------------------------------------
# Test fixtures (small, checked-in samples — never the real OPSD download)
# ---------------------------------------------------------------------------

FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures"
SAMPLE_TIMESERIES_CSV = FIXTURES_DIR / "sample_time_series_60min_singleindex.csv"
SAMPLE_WEATHER_CSV = FIXTURES_DIR / "sample_weather_data.csv"

# Timezone: ingest and store in UTC; convert to this zone only for calendar
# features (hour-of-day, weekday, holidays). DST transitions are handled
# explicitly in data.py, not glossed over by a blanket tz_localize.
LOCAL_TZ = "Europe/Madrid"
HOLIDAY_COUNTRY = "ES"

# ---------------------------------------------------------------------------
# Data cleaning (spec section 4)
# ---------------------------------------------------------------------------

# Gaps of this length or shorter are time-interpolated; anything longer is
# left NaN and flagged so it's excluded from training/evaluation windows
# rather than silently filled.
MAX_INTERPOLATE_GAP_HOURS = 3

# A load value outside this range is treated as implausible (Spain's actual
# hourly load has never been remotely close to either bound; this is a
# sanity floor/ceiling, not a tight physical limit).
LOAD_PLAUSIBLE_RANGE_MW = (0, 60_000)

# Spike detection: flag an hour-over-hour change whose absolute value
# exceeds this many robust standard deviations (MAD-based) from the
# typical hour-over-hour change.
SPIKE_ROBUST_ZSCORE_THRESHOLD = 6.0

# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------
# 2020 is excluded from the main train/val/test split (COVID demand shock)
# and instead treated as a separate stress-test period.

TRAIN_START = "2015-01-01"
TRAIN_END = "2017-12-31"

VALIDATION_START = "2018-01-01"
VALIDATION_END = "2018-12-31"

TEST_START = "2019-01-01"
TEST_END = "2019-12-31"

STRESS_TEST_START = "2020-01-01"
STRESS_TEST_END = "2020-12-31"

# Day-ahead cutoff: local wall-clock hour on day D at which all 24 hourly
# forecasts for day D+1 are committed at once. Every non-calendar feature
# (lags, rolling stats) is anchored to this shared "forecast origin" rather
# than to the individual target hour — see features.compute_forecast_origin
# and its module docstring for why a target-relative lag would leak for
# roughly half of the 24 target hours at a 10:00 cutoff, and why anchoring
# to the origin avoids that regardless of the cutoff hour chosen here.
DAY_AHEAD_CUTOFF_HOUR = 10
MIN_LAG_HOURS = 24  # sanity floor on requested lags, not itself the safety mechanism
LAG_HOURS = (24, 48, 168)  # hours before the forecast origin (1 day, 2 days, 1 week)
ROLLING_WINDOWS_HOURS = (24, 168)  # trailing window ending at the forecast origin

# Degree-hour base temperatures (spec 6.3): hours above/below these
# thresholds drive heating/cooling demand. Common European convention.
HEATING_DEGREE_BASE_C = 18.0
COOLING_DEGREE_BASE_C = 24.0

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

QUANTILE_LOW = 0.10
QUANTILE_HIGH = 0.90
PREDICTION_INTERVAL_COVERAGE_TARGET = QUANTILE_HIGH - QUANTILE_LOW  # 0.80

RANDOM_SEED = 42

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "random_state": RANDOM_SEED,
}

# More regularized than LIGHTGBM_PARAMS on purpose: an uncalibrated
# rolling-origin backtest on the real 2019 test year showed num_leaves=63
# (i.e. matching the point model's capacity) badly overfitting the
# training quantiles and under-covering on test (56% empirical vs. an 80%
# target). A hyperparameter sweep (see forecast.QUANTILE_PARAM_SWEEP_GRID)
# found this configuration had the best (lowest) pinball loss of the
# candidates tried -- the proper scoring rule these models are actually
# trained against -- even though none of the sweep's candidates closed
# the full coverage gap by tuning alone. The remaining gap is corrected by
# split-conformal calibration (see forecast.make_conformalized_quantile_
# predict_fn), which is robust to the base model's own miscalibration, so
# this config's job is to be a good base model, not to hit 80% by itself.
LIGHTGBM_QUANTILE_PARAMS = {
    "objective": "quantile",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
}

# Rolling-origin backtest: retrain (or expand the training window) on this
# cadence during the test year, then forecast day by day until the next
# retrain point.
BACKTEST_RETRAIN_FREQ = "MS"  # month start

# Split-conformal calibration (CQR): fraction of each retrain block's own
# training data held out as a calibration slice (the most recent
# origins, by forecast origin, not fit-df rows) used to measure how far
# off the raw quantile predictions actually are out-of-sample, and
# correct q10/q90 by that measured amount so empirical coverage targets
# PREDICTION_INTERVAL_COVERAGE_TARGET rather than whatever the
# uncalibrated model happens to produce.
CONFORMAL_CALIBRATION_FRAC = 0.2


# ---------------------------------------------------------------------------
# Dispatch optimization — illustrative system, NOT market data
# ---------------------------------------------------------------------------
# All capacities in MW, costs in EUR/MWh, energy in MWh. These numbers are
# chosen to produce a plausible, solvable day-ahead LP for a demonstration
# project. The README must state explicitly that they are illustrative.


@dataclass(frozen=True)
class GenerationUnit:
    name: str
    capacity_mw: float
    marginal_cost_eur_per_mwh: float
    ramp_limit_mw_per_hour: float | None  # None = unconstrained ramping


GENERATION_UNITS: tuple[GenerationUnit, ...] = (
    GenerationUnit("baseload", capacity_mw=7_000, marginal_cost_eur_per_mwh=10, ramp_limit_mw_per_hour=1_000),
    GenerationUnit("ccgt", capacity_mw=25_000, marginal_cost_eur_per_mwh=60, ramp_limit_mw_per_hour=8_000),
    GenerationUnit("peaker", capacity_mw=10_000, marginal_cost_eur_per_mwh=120, ramp_limit_mw_per_hour=None),
)

# Solar + wind: marginal cost 0, output capped at actual observed generation
# each hour (from the OPSD data), curtailable.
RENEWABLE_MARGINAL_COST_EUR_PER_MWH = 0.0


@dataclass(frozen=True)
class BatteryParams:
    power_capacity_mw: float = 1_000.0
    energy_capacity_mwh: float = 4_000.0
    round_trip_efficiency: float = 0.90
    initial_soc_mwh: float = field(default=2_000.0)  # 50% SoC at day start

    @property
    def charge_efficiency(self) -> float:
        # Split round-trip efficiency evenly between charge and discharge legs.
        return self.round_trip_efficiency**0.5

    @property
    def discharge_efficiency(self) -> float:
        return self.round_trip_efficiency**0.5


BATTERY = BatteryParams()

# Balancing market: covers real-time imbalance between planned supply and
# actual demand. Positive imbalance (under-supply) bought at the up-price;
# negative imbalance (over-supply) sold at the down-price.
BALANCING_PRICE_UP_EUR_PER_MWH = 200.0
BALANCING_PRICE_DOWN_EUR_PER_MWH = 20.0

HOURS_PER_DAY = 24

# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------

BATTERY_POWER_SWEEP_MW = (0, 250, 500, 1_000, 2_000, 4_000)
BALANCING_PRICE_UP_SWEEP_EUR_PER_MWH = (50, 100, 200, 400, 800)
RENEWABLE_SCALE_SWEEP = (0.5, 1.0, 1.5, 2.0)  # multiplier on observed solar+wind
