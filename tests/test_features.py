"""
Tests for energy_dispatch.features — the leakage invariants this module
exists to guarantee, checked two ways: (1) mechanically, via
assert_no_leakage's bookkeeping checks, and (2) empirically, by poisoning
every value after a known cutoff with an outlandish sentinel and
confirming it never appears in a constructed feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_dispatch import config, features

POISON = 1e9


@pytest.fixture
def two_week_frame() -> pd.DataFrame:
    """14 days of hourly data (enough for a 168h lag/rolling window with
    a few days of margin), local Europe/Madrid time, spanning no DST
    transition — a simple ramp so lag values are trivial to hand-check.
    """
    idx = pd.date_range("2019-06-01", periods=24 * 14, freq="h", tz="Europe/Madrid")
    idx = idx.tz_convert("UTC")
    values = np.arange(len(idx), dtype=float)
    df = pd.DataFrame({config.LOAD_ACTUAL_COL: values}, index=idx)
    df[config.WEATHER_TEMPERATURE_COL] = 20.0
    return df


def _target_day(date_str: str) -> pd.DatetimeIndex:
    local = pd.date_range(f"{date_str} 00:00", periods=24, freq="h", tz="Europe/Madrid")
    return local.tz_convert("UTC")


# ---------------------------------------------------------------------------
# compute_forecast_origin
# ---------------------------------------------------------------------------

def test_origin_is_shared_across_every_hour_of_the_target_day():
    target_day = _target_day("2019-06-05")
    origin = features.compute_forecast_origin(target_day)
    assert origin.nunique() == 1


def test_origin_is_cutoff_hour_on_the_day_before():
    target_day = _target_day("2019-06-05")
    origin = features.compute_forecast_origin(target_day)
    origin_local = origin.tz_convert(config.LOCAL_TZ)[0]
    assert origin_local.hour == config.DAY_AHEAD_CUTOFF_HOUR
    assert origin_local.date() == pd.Timestamp("2019-06-04").date()


def test_origin_handles_spring_forward_day():
    # Spain, 2019-03-31: clocks go 02:00 -> 03:00, a 23-hour local day.
    target_day = pd.date_range(
        "2019-03-31 00:00", "2019-03-31 23:00", freq="h", tz="Europe/Madrid"
    ).tz_convert("UTC")
    assert len(target_day) == 23
    origin = features.compute_forecast_origin(target_day)
    assert origin.nunique() == 1
    origin_local = origin.tz_convert(config.LOCAL_TZ)[0]
    assert origin_local.hour == config.DAY_AHEAD_CUTOFF_HOUR
    assert origin_local.date() == pd.Timestamp("2019-03-30").date()


def test_origin_handles_day_after_fall_back():
    # Spain, 2019-10-27: fall-back day itself; origin for the day after
    # falls ON the transition day and must still resolve to 10:00 local.
    target_day = _target_day("2019-10-28")
    origin = features.compute_forecast_origin(target_day)
    origin_local = origin.tz_convert(config.LOCAL_TZ)[0]
    assert origin_local.hour == config.DAY_AHEAD_CUTOFF_HOUR
    assert origin_local.date() == pd.Timestamp("2019-10-27").date()


def test_origin_rejects_tz_naive_index():
    naive = pd.date_range("2019-06-05", periods=24, freq="h")
    with pytest.raises(ValueError, match="tz-aware"):
        features.compute_forecast_origin(naive)


# ---------------------------------------------------------------------------
# add_lag_features — structural + poison tests
# ---------------------------------------------------------------------------

def test_add_lag_features_rejects_sub_floor_lag(two_week_frame):
    with pytest.raises(ValueError, match="MIN_LAG_HOURS"):
        features.add_lag_features(two_week_frame, config.LOAD_ACTUAL_COL, lags=(1,))


def test_add_lag_features_matches_hand_computed_value(two_week_frame):
    target_day = _target_day("2019-06-10")
    origin = features.compute_forecast_origin(target_day)[0]
    expected_lag24 = two_week_frame.loc[origin - pd.Timedelta(hours=24), config.LOAD_ACTUAL_COL]
    expected_lag48 = two_week_frame.loc[origin - pd.Timedelta(hours=48), config.LOAD_ACTUAL_COL]
    expected_lag168 = two_week_frame.loc[origin - pd.Timedelta(hours=168), config.LOAD_ACTUAL_COL]

    lagged = features.add_lag_features(two_week_frame, config.LOAD_ACTUAL_COL)
    day_rows = lagged.reindex(target_day)

    assert (day_rows["lag_24h"] == expected_lag24).all()
    assert (day_rows["lag_48h"] == expected_lag48).all()
    assert (day_rows["lag_168h"] == expected_lag168).all()


def test_add_lag_features_constant_across_target_day(two_week_frame):
    target_day = _target_day("2019-06-10")
    lagged = features.add_lag_features(two_week_frame, config.LOAD_ACTUAL_COL)
    day_rows = lagged.reindex(target_day)
    for col in ("lag_24h", "lag_48h", "lag_168h"):
        assert day_rows[col].nunique() == 1


def test_add_lag_features_never_leaks_poisoned_future_values(two_week_frame):
    target_day = _target_day("2019-06-10")
    origin = features.compute_forecast_origin(target_day)[0]

    poisoned = two_week_frame.copy()
    poisoned.loc[poisoned.index > origin, config.LOAD_ACTUAL_COL] = POISON

    lagged = features.add_lag_features(poisoned, config.LOAD_ACTUAL_COL)
    day_rows = lagged.reindex(target_day)
    lag_cols = [c for c in day_rows.columns if c.startswith("lag_")]

    assert not (day_rows[lag_cols] >= POISON).any().any()


# ---------------------------------------------------------------------------
# add_rolling_features — structural + poison tests
# ---------------------------------------------------------------------------

def test_add_rolling_features_matches_hand_computed_value(two_week_frame):
    target_day = _target_day("2019-06-10")
    origin = features.compute_forecast_origin(target_day)[0]
    window = two_week_frame.loc[origin - pd.Timedelta(hours=23) : origin, config.LOAD_ACTUAL_COL]
    expected_mean_24h = window.mean()

    rolled = features.add_rolling_features(two_week_frame, config.LOAD_ACTUAL_COL)
    day_rows = rolled.reindex(target_day)

    assert day_rows["rolling_mean_24h"].nunique() == 1
    assert np.isclose(day_rows["rolling_mean_24h"].iloc[0], expected_mean_24h)


def test_add_rolling_features_never_leaks_poisoned_future_values(two_week_frame):
    target_day = _target_day("2019-06-10")
    origin = features.compute_forecast_origin(target_day)[0]

    poisoned = two_week_frame.copy()
    poisoned.loc[poisoned.index > origin, config.LOAD_ACTUAL_COL] = POISON

    rolled = features.add_rolling_features(poisoned, config.LOAD_ACTUAL_COL)
    day_rows = rolled.reindex(target_day)
    roll_cols = [c for c in day_rows.columns if c.startswith("rolling_")]

    assert not (day_rows[roll_cols] >= POISON / 2).any().any()  # poison would dominate any mean/std


# ---------------------------------------------------------------------------
# add_calendar_features
# ---------------------------------------------------------------------------

def test_add_calendar_features_known_holiday(two_week_frame):
    # 2019-06-01 is not a Spanish national holiday; construct a frame
    # covering a known one instead: 2019-08-15 (Assumption of Mary).
    idx = pd.date_range("2019-08-10", periods=24 * 10, freq="h", tz="Europe/Madrid")
    idx = idx.tz_convert("UTC")
    df = pd.DataFrame({config.LOAD_ACTUAL_COL: 0.0}, index=idx)
    out = features.add_calendar_features(df)
    local_dates = out.index.tz_convert(config.LOCAL_TZ).date
    holiday_rows = out[local_dates == pd.Timestamp("2019-08-15").date()]
    assert holiday_rows["is_holiday"].all()


def test_add_calendar_features_hour_and_weekday(two_week_frame):
    out = features.add_calendar_features(two_week_frame)
    local = two_week_frame.index.tz_convert(config.LOCAL_TZ)
    assert (out["hour"] == local.hour).all()
    assert (out["weekday"] == local.weekday).all()
    assert (out["is_weekend"] == (local.weekday >= 5)).all()


# ---------------------------------------------------------------------------
# add_weather_features
# ---------------------------------------------------------------------------

def test_add_weather_features_degree_hours(two_week_frame):
    df = two_week_frame.copy()
    df[config.WEATHER_TEMPERATURE_COL] = 10.0  # below heating base (18), above nothing
    out = features.add_weather_features(df)
    assert (out["heating_degree_hours"] == 8.0).all()
    assert (out["cooling_degree_hours"] == 0.0).all()

    df[config.WEATHER_TEMPERATURE_COL] = 30.0  # above cooling base (24)
    out = features.add_weather_features(df)
    assert (out["heating_degree_hours"] == 0.0).all()
    assert (out["cooling_degree_hours"] == 6.0).all()


# ---------------------------------------------------------------------------
# build_feature_matrix + assert_no_leakage integration
# ---------------------------------------------------------------------------

def test_build_feature_matrix_passes_leakage_check(two_week_frame):
    matrix = features.build_feature_matrix(two_week_frame, target_col=config.LOAD_ACTUAL_COL)
    features.assert_no_leakage(matrix)  # should not raise


def test_assert_no_leakage_catches_origin_after_target(two_week_frame):
    matrix = features.build_feature_matrix(two_week_frame, target_col=config.LOAD_ACTUAL_COL)
    broken = matrix.copy()
    broken.iloc[0, broken.columns.get_loc("origin_utc")] = broken.index[0] + pd.Timedelta(hours=1)
    with pytest.raises(AssertionError, match="after the target timestamp"):
        features.assert_no_leakage(broken)


def test_assert_no_leakage_catches_wrong_cutoff_hour(two_week_frame):
    matrix = features.build_feature_matrix(two_week_frame, target_col=config.LOAD_ACTUAL_COL)
    broken = matrix.copy()
    broken.iloc[0, broken.columns.get_loc("origin_utc")] -= pd.Timedelta(hours=1)
    with pytest.raises(AssertionError, match="DAY_AHEAD_CUTOFF_HOUR"):
        features.assert_no_leakage(broken)


def test_assert_no_leakage_catches_wrong_origin_date(two_week_frame):
    matrix = features.build_feature_matrix(two_week_frame, target_col=config.LOAD_ACTUAL_COL)
    broken = matrix.copy()
    # A full extra day back keeps the cutoff-hour check passing (still
    # exactly 10:00 local) while breaking the "exactly one day before"
    # check. Note: given that check plus the cutoff-hour check, the
    # "same target day -> same origin" check in assert_no_leakage is
    # actually implied/redundant (both checks pin origin_utc to a single
    # instant once a row's target date is fixed) — it's kept as cheap,
    # independent future-proofing (e.g. against a cutoff hour that isn't
    # DST-unambiguous in some other timezone), not because it's reachable
    # in isolation here.
    broken.iloc[0, broken.columns.get_loc("origin_utc")] -= pd.Timedelta(days=1)
    with pytest.raises(AssertionError, match="one day before"):
        features.assert_no_leakage(broken)


def test_assert_no_leakage_requires_origin_column(two_week_frame):
    with pytest.raises(AssertionError, match="origin_utc"):
        features.assert_no_leakage(two_week_frame)


# ---------------------------------------------------------------------------
# select_feature_columns
# ---------------------------------------------------------------------------

def test_select_feature_columns_excludes_target_and_bookkeeping(two_week_frame):
    matrix = features.build_feature_matrix(two_week_frame, target_col=config.LOAD_ACTUAL_COL)
    selected = features.select_feature_columns(matrix)

    assert config.LOAD_ACTUAL_COL not in selected
    assert "origin_utc" not in selected
    # sanity: real engineered features are still present
    assert "hour" in selected
    assert "lag_24h" in selected
    assert "rolling_mean_24h" in selected
    assert "heating_degree_hours" in selected


def test_select_feature_columns_excludes_qc_flags_and_non_target_load_columns():
    idx = pd.date_range("2019-06-01", periods=24 * 14, freq="h", tz="Europe/Madrid").tz_convert(
        "UTC"
    )
    df = pd.DataFrame(
        {
            config.LOAD_ACTUAL_COL: np.arange(len(idx), dtype=float),
            config.LOAD_FORECAST_COL: 0.0,
            config.SOLAR_GEN_COL: 0.0,
            config.WIND_GEN_COL: 0.0,
            config.WEATHER_TEMPERATURE_COL: 20.0,
            "is_duplicate_timestamp": False,
            "is_missing_hour": False,
            "is_implausible_load": False,
            "is_spike": False,
            "exclude_long_gap": False,
        },
        index=idx,
    )
    matrix = features.build_feature_matrix(df, target_col=config.LOAD_ACTUAL_COL)
    selected = features.select_feature_columns(matrix)

    for excluded_col in (
        config.LOAD_FORECAST_COL,
        config.SOLAR_GEN_COL,
        config.WIND_GEN_COL,
        "is_duplicate_timestamp",
        "is_missing_hour",
        "is_implausible_load",
        "is_spike",
        "exclude_long_gap",
    ):
        assert excluded_col not in selected
