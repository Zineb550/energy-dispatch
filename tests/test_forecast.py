"""
Tests for energy_dispatch.forecast — baselines, time-based splits, point
metrics, and the rolling-origin backtest harness. LightGBM-dependent
functions are intentionally not implemented yet and are not tested here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_dispatch import config, features, forecast


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A few months of hourly data with a clear weekly pattern (so
    seasonal-naive is easy to hand-check), spanning parts of 2017/2018/2019
    to exercise time_based_split's boundary handling.
    """
    idx = pd.date_range("2017-12-20", "2019-01-10", freq="h", tz="Europe/Madrid").tz_convert("UTC")
    # Phase is a plain elapsed-hour counter (not local wall-clock
    # hour-of-week): the index spans two DST transitions, and a signal
    # built from *local* hour-of-week would have a 167h or 169h "week"
    # across a transition, breaking exact recovery by t-168h even though
    # nothing is wrong with seasonal_naive_forecast itself. t - 168h
    # measures real elapsed hours (our data is UTC/hourly-regular), so
    # the fixture's periodicity needs to match that basis.
    hour_position = np.arange(len(idx))
    load = 30_000 + 2_000 * np.sin(2 * np.pi * hour_position / 168)
    df = pd.DataFrame(
        {
            config.LOAD_ACTUAL_COL: load,
            config.LOAD_FORECAST_COL: load + 500,  # a deliberately-offset "TSO forecast"
            config.WEATHER_TEMPERATURE_COL: 20.0,
        },
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_seasonal_naive_matches_hand_computed_value(synthetic_df):
    naive = forecast.seasonal_naive_forecast(synthetic_df)
    t = synthetic_df.index[500]
    expected = synthetic_df.loc[t - pd.Timedelta(hours=168), config.LOAD_ACTUAL_COL]
    assert naive.loc[t] == expected


def test_seasonal_naive_recovers_exact_load_on_a_pure_weekly_signal(synthetic_df):
    # The synthetic load is exactly periodic with period 168h, so the
    # seasonal-naive forecast should be a perfect reconstruction wherever
    # a full week of history exists.
    naive = forecast.seasonal_naive_forecast(synthetic_df)
    valid = naive.notna()
    assert valid.sum() > 0
    expected = synthetic_df.loc[valid.index[valid], config.LOAD_ACTUAL_COL]
    np.testing.assert_allclose(naive[valid].to_numpy(), expected.to_numpy())


def test_tso_forecast_returns_the_forecast_column_unchanged(synthetic_df):
    tso = forecast.tso_forecast(synthetic_df)
    assert (tso == synthetic_df[config.LOAD_FORECAST_COL]).all()
    assert tso.name == "y_tso"


# ---------------------------------------------------------------------------
# time_based_split
# ---------------------------------------------------------------------------

def test_time_based_split_boundaries(synthetic_df):
    splits = forecast.time_based_split(synthetic_df)

    train_local_dates = splits["train"].index.tz_convert(config.LOCAL_TZ).normalize().unique()
    val_local_dates = splits["validation"].index.tz_convert(config.LOCAL_TZ).normalize().unique()

    assert train_local_dates.max() <= pd.Timestamp(config.TRAIN_END, tz=config.LOCAL_TZ)
    assert val_local_dates.min() >= pd.Timestamp(config.VALIDATION_START, tz=config.LOCAL_TZ)

    # Dec 31 2017 23:00 UTC is already Jan 1 2018 local (CET, UTC+1) —
    # must land in validation, not train, since the split is by local date.
    boundary_ts = pd.Timestamp("2017-12-31 23:00", tz="UTC")
    assert boundary_ts in splits["validation"].index
    assert boundary_ts not in splits["train"].index


def test_time_based_split_covers_disjoint_periods(synthetic_df):
    splits = forecast.time_based_split(synthetic_df)
    train_idx = set(splits["train"].index)
    val_idx = set(splits["validation"].index)
    assert train_idx.isdisjoint(val_idx)


# ---------------------------------------------------------------------------
# compute_point_metrics
# ---------------------------------------------------------------------------

def test_compute_point_metrics_hand_computed():
    table = pd.DataFrame(
        {
            "y_true": [100.0, 200.0, 300.0, 400.0],
            "y_naive": [110.0, 190.0, 330.0, 360.0],
        }
    )
    metrics = forecast.compute_point_metrics(table, model_cols=("y_naive",))
    errors = table["y_true"] - table["y_naive"]
    assert np.isclose(metrics.loc["y_naive", "MAE"], errors.abs().mean())
    assert np.isclose(metrics.loc["y_naive", "RMSE"], (errors**2).mean() ** 0.5)
    assert np.isclose(metrics.loc["y_naive", "MAPE"], (errors.abs() / table["y_true"]).mean() * 100)
    assert metrics.loc["y_naive", "n"] == 4


def test_compute_point_metrics_drops_nan_rows_per_model():
    table = pd.DataFrame(
        {
            "y_true": [100.0, 200.0, 300.0],
            "y_naive": [110.0, np.nan, 330.0],
            "y_tso": [90.0, 210.0, 290.0],
        }
    )
    metrics = forecast.compute_point_metrics(table, model_cols=("y_naive", "y_tso"))
    assert metrics.loc["y_naive", "n"] == 2
    assert metrics.loc["y_tso", "n"] == 3


# ---------------------------------------------------------------------------
# rolling_origin_backtest
# ---------------------------------------------------------------------------

def test_rolling_origin_backtest_without_predict_fn_returns_baselines_only(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    result = forecast.rolling_origin_backtest(
        matrix,
        config.LOAD_ACTUAL_COL,
        predict_fn=None,
        test_start="2018-06-01",
        test_end="2018-06-30",
    )
    assert list(result.columns) == ["y_true", "y_naive", "y_tso", "y_model"]
    assert result["y_model"].isna().all()
    assert len(result) > 0

    local_dates = result.index.tz_convert(config.LOCAL_TZ).normalize().unique()
    assert local_dates.min() == pd.Timestamp("2018-06-01", tz=config.LOCAL_TZ)
    assert local_dates.max() == pd.Timestamp("2018-06-30", tz=config.LOCAL_TZ)


def test_rolling_origin_backtest_train_df_never_contains_future_rows(synthetic_df):
    """A spy predict_fn that asserts, on every call, that every row handed
    to it as "training data" has an origin at or before the forecast
    block's own origin — i.e. the harness never lets a later block's
    retrain see rows from inside or after the block it's forecasting.
    """
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    calls = []

    def spy_predict_fn(train_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.Series:
        calls.append((train_df.copy(), forecast_df.copy()))
        block_origin = forecast_df["origin_utc"].min()
        assert (train_df["origin_utc"] <= block_origin).all(), "train_df contains a future origin"
        return pd.Series(0.0, index=forecast_df.index)

    result = forecast.rolling_origin_backtest(
        matrix,
        config.LOAD_ACTUAL_COL,
        predict_fn=spy_predict_fn,
        retrain_freq="MS",
        test_start="2018-01-01",
        test_end="2018-03-31",
    )

    assert len(calls) >= 3  # at least one call per calendar month in the range
    assert result["y_model"].notna().any()

    # Every train_df handed to predict_fn must exclude all rows that
    # belong to that same call's forecast block (no peeking at the days
    # being forecast, let alone days after them).
    for train_df, forecast_df in calls:
        assert set(train_df.index).isdisjoint(set(forecast_df.index))


def test_rolling_origin_backtest_covers_every_day_in_range_exactly_once(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    seen_days = []

    def counting_predict_fn(train_df, forecast_df):
        seen_days.extend(forecast_df.index.tz_convert(config.LOCAL_TZ).normalize().unique().tolist())
        return pd.Series(0.0, index=forecast_df.index)

    forecast.rolling_origin_backtest(
        matrix,
        config.LOAD_ACTUAL_COL,
        predict_fn=counting_predict_fn,
        retrain_freq="MS",
        test_start="2018-01-01",
        test_end="2018-02-28",
    )

    assert len(seen_days) == len(set(seen_days)), "a day was forecast in more than one block"
    expected_days = pd.date_range("2018-01-01", "2018-02-28", freq="D", tz=config.LOCAL_TZ)
    assert set(seen_days) == set(expected_days)
