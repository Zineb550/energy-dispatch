"""
Feature engineering for day-ahead demand forecasting (spec section 6.3).

THE LEAKAGE MODEL
------------------
We forecast all 24 (occasionally 23 or 25, on a DST-transition day) local
hours of day D+1 from a single cutoff: config.DAY_AHEAD_CUTOFF_HOUR (10:00)
local time on day D. That single cutoff is the "forecast origin" shared by
every target hour of D+1 — see compute_forecast_origin().

A feature defined *relative to the target hour itself* ("value at t - 24h")
is NOT uniformly safe here: the gap between the origin and the target hour
ranges from ~14h (the first hour of D+1) to ~37h (the last hour of D+1)
depending on which of the 24 hours t is. A naive "t - 24h" lag would reach
*after* the cutoff (i.e. into the future) for every target hour whose local
hour-of-day exceeds the cutoff hour — roughly 13 of the 24 hours at a 10:00
cutoff. Only lags >= ~37h would be uniformly safe if defined relative to t.

Resolution: every lag/rolling feature in this module is anchored to the
shared origin O(t), not to t — "value at O(t) - k hours", not "value at
t - k hours". Since O(t) <= t always (by construction), and every lag/
window subtracts a non-negative number of hours from O(t), every one of
these features is safe for every target hour, regardless of the cutoff
hour, with no per-hour reasoning required. This does mean these features
are constant across all ~24 rows of a given target day (they describe
"what was known when the forecast was made", not "what happened at this
specific hour historically") — the calendar features below (which DO vary
per row, since they're deterministic facts about the target hour) are what
let a downstream model learn the within-day and within-week shape.

Weather features are the one deliberate exception: they use the *observed*
value at the target hour t as a proxy for a day-ahead weather forecast
(OPSD has no such forecast). This is a modeling assumption, disclosed in
the README's limitations — not an accidental leak of the load series'
own future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from holidays import country_holidays

from energy_dispatch import config

# ---------------------------------------------------------------------------
# Forecast origin
# ---------------------------------------------------------------------------

def compute_forecast_origin(target_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """For each UTC target timestamp, return the UTC instant of the shared
    day-ahead forecast origin: config.DAY_AHEAD_CUTOFF_HOUR local time on
    the day *before* the target's local calendar day.

    Every timestamp in the same local calendar day maps to the same
    origin — that's what "day-ahead" (one cutoff, 24 outputs) means.

    Implementation note: the hour-of-day is added to the origin's local
    midnight in *naive* (tz-unaware) time and localized only at the very
    end. Adding a Timedelta directly to a tz-aware timestamp is duration
    (not wall-clock) arithmetic in pandas, which would shift the result by
    an hour on a day that itself contains a DST transition between
    midnight and the cutoff hour. Localizing 10:00 is always unambiguous
    for Europe/Madrid (transitions happen around 02:00-03:00), so no
    ambiguous/nonexistent handling is needed at that final step.
    """
    if target_index.tz is None:
        raise ValueError("target_index must be tz-aware (UTC)")

    local = target_index.tz_convert(config.LOCAL_TZ)
    target_local_date = local.normalize()
    origin_local_date = target_local_date - pd.DateOffset(days=1)

    origin_naive = origin_local_date.tz_localize(None) + pd.Timedelta(
        hours=config.DAY_AHEAD_CUTOFF_HOUR
    )
    origin_local = origin_naive.tz_localize(config.LOCAL_TZ)
    return origin_local.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Calendar features (safe: deterministic facts about the target hour)
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour/weekday/month/day-of-year (sin/cos), weekend flag, Spanish
    national holidays, and bridge days ("puente": a Monday before a Tuesday
    holiday, or a Friday after a Thursday holiday) — all computed from the
    target timestamp's own local calendar date. None of this depends on
    any load/weather observation, so none of it carries leakage risk.
    """
    local = df.index.tz_convert(config.LOCAL_TZ)
    out = df.copy()

    out["hour"] = local.hour
    out["weekday"] = local.weekday  # Monday=0 .. Sunday=6
    out["month"] = local.month
    out["is_weekend"] = local.weekday >= 5

    day_of_year = local.dayofyear.to_numpy(dtype=float)
    days_in_year = np.where(local.is_leap_year, 366.0, 365.0)
    angle = 2 * np.pi * day_of_year / days_in_year
    out["day_of_year_sin"] = np.sin(angle)
    out["day_of_year_cos"] = np.cos(angle)

    years = range(local.min().year, local.max().year + 1)
    es_holidays = country_holidays(config.HOLIDAY_COUNTRY, years=years)
    dates = local.date  # array of python date objects, in local wall-clock terms
    is_holiday = np.array([d in es_holidays for d in dates])
    out["is_holiday"] = is_holiday

    one_day = pd.Timedelta(days=1).to_pytimedelta()
    is_holiday_tomorrow = np.array([(d + one_day) in es_holidays for d in dates])
    is_holiday_yesterday = np.array([(d - one_day) in es_holidays for d in dates])
    weekday = local.weekday.to_numpy()
    is_bridge_monday = (weekday == 0) & is_holiday_tomorrow
    is_bridge_friday = (weekday == 4) & is_holiday_yesterday
    out["is_bridge_day"] = is_bridge_monday | is_bridge_friday

    return out


# ---------------------------------------------------------------------------
# Lag & rolling features (origin-anchored: leak-safe by construction)
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame, target_col: str, lags: tuple[int, ...] = config.LAG_HOURS
) -> pd.DataFrame:
    """Add lagged load features, each anchored to the row's forecast
    origin O(t) rather than to t itself: lag_{k}h = value observed at
    (O(t) - k hours).

    Every lag must be >= config.MIN_LAG_HOURS: not because that threshold
    is itself what makes these safe (origin-anchoring already guarantees
    that for any k >= 0), but as a sanity floor against a caller passing
    a suspiciously small lag by mistake.
    """
    bad = [k for k in lags if k < config.MIN_LAG_HOURS]
    if bad:
        raise ValueError(
            f"lag(s) {bad} are below config.MIN_LAG_HOURS={config.MIN_LAG_HOURS}h"
        )

    origin = compute_forecast_origin(df.index)
    raw = df[target_col]
    out = df.copy()
    for k in lags:
        lookup_times = origin - pd.Timedelta(hours=k)
        out[f"lag_{k}h"] = raw.reindex(lookup_times).to_numpy()
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: tuple[int, ...] = config.ROLLING_WINDOWS_HOURS,
) -> pd.DataFrame:
    """Add rolling mean/std, each a trailing window *ending at the row's
    forecast origin O(t)* (inclusive of the observation at O(t) itself —
    the value observed exactly at the cutoff is "known now", not future).

    Computed once as a rolling calculation over the raw hourly series
    (efficient, and correct regardless of how many target rows share the
    same origin), then looked up at each row's O(t).
    """
    origin = compute_forecast_origin(df.index)
    raw = df[target_col]
    out = df.copy()
    for w in windows:
        window_str = f"{w}h"
        roll_mean = raw.rolling(window_str).mean()
        roll_std = raw.rolling(window_str).std()
        out[f"rolling_mean_{w}h"] = roll_mean.reindex(origin).to_numpy()
        out[f"rolling_std_{w}h"] = roll_std.reindex(origin).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Weather features (target-relative by deliberate design — see module docstring)
# ---------------------------------------------------------------------------

def add_weather_features(
    df: pd.DataFrame,
    heating_base_c: float = config.HEATING_DEGREE_BASE_C,
    cooling_base_c: float = config.COOLING_DEGREE_BASE_C,
) -> pd.DataFrame:
    """Add heating/cooling degree-hours derived from the observed
    temperature at the target hour. Raw temperature and radiation columns
    (config.WEATHER_TEMPERATURE_COL etc.) are already present on df from
    data.merge_weather and are left as-is — this function only adds the
    derived degree-hour columns.

    Uses observed weather as a proxy for a day-ahead weather forecast
    (OPSD has no such forecast) — a stated modeling assumption, not
    accidental leakage; see the module docstring.
    """
    out = df.copy()
    temp = df[config.WEATHER_TEMPERATURE_COL]
    out["heating_degree_hours"] = (heating_base_c - temp).clip(lower=0)
    out["cooling_degree_hours"] = (temp - cooling_base_c).clip(lower=0)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame, target_col: str = config.LOAD_ACTUAL_COL
) -> pd.DataFrame:
    """Run the full feature pipeline (calendar + lag + rolling + weather)
    and attach origin_utc so downstream code (and assert_no_leakage) can
    see, for every row, exactly which forecast origin it was built from.

    This is the single entry point forecast.py should call rather than
    composing the add_*_features functions itself, so the feature set
    used for training and for backtesting can never drift apart.
    """
    out = add_calendar_features(df)
    out = add_lag_features(out, target_col)
    out = add_rolling_features(out, target_col)
    out = add_weather_features(out)
    out["origin_utc"] = compute_forecast_origin(df.index)
    return out


def assert_no_leakage(feature_matrix: pd.DataFrame) -> None:
    """Invariant check used by tests/test_features.py. Verifies, purely
    from the feature_matrix's own bookkeeping (no access to raw data
    needed):

      1. every row has an origin_utc,
      2. origin_utc is never after the row's own target timestamp,
      3. origin_utc falls exactly at config.DAY_AHEAD_CUTOFF_HOUR local
         time (i.e. it really is "the cutoff", not some other instant),
      4. origin_utc's local calendar date is exactly one day before the
         target's local calendar date (catches an off-by-one-day bug),
      5. every target timestamp sharing the same local calendar day has
         the *same* origin_utc (the single-cutoff-per-batch property that
         is the whole point of the origin-anchored design — this is what
         would catch a regression back to a per-hour, target-relative
         definition).

    Raises AssertionError with a message identifying which check failed.
    """
    if "origin_utc" not in feature_matrix.columns:
        raise AssertionError("feature_matrix has no origin_utc column")

    target_idx = feature_matrix.index
    origin = pd.DatetimeIndex(pd.to_datetime(feature_matrix["origin_utc"], utc=True))

    if (origin > target_idx).any():
        raise AssertionError("origin_utc is after the target timestamp for at least one row")

    origin_local = origin.tz_convert(config.LOCAL_TZ)
    if not (origin_local.hour == config.DAY_AHEAD_CUTOFF_HOUR).all():
        raise AssertionError(
            "origin_utc does not fall exactly at config.DAY_AHEAD_CUTOFF_HOUR local time "
            "for every row"
        )

    target_local_date = target_idx.tz_convert(config.LOCAL_TZ).normalize()
    origin_local_date = origin_local.normalize()
    if not (origin_local_date == (target_local_date - pd.DateOffset(days=1))).all():
        raise AssertionError(
            "origin's local calendar date is not exactly one day before the target's "
            "local calendar date for every row"
        )

    grouping = pd.Series(origin_local.to_numpy(), index=target_local_date)
    n_unique_per_day = grouping.groupby(level=0).nunique()
    if (n_unique_per_day > 1).any():
        bad_days = n_unique_per_day[n_unique_per_day > 1].index.tolist()
        raise AssertionError(
            f"multiple distinct origins found within a single target day: {bad_days}"
        )
