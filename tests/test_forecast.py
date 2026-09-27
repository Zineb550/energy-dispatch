"""
Tests for energy_dispatch.forecast — baselines, time-based splits, point
and interval metrics, the LightGBM point and quantile models, and the
rolling-origin backtest harness.
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
    assert list(result.columns) == ["y_true", "y_naive", "y_tso", "y_lgbm"]
    assert result["y_lgbm"].isna().all()
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
    assert result["y_lgbm"].notna().any()

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


# ---------------------------------------------------------------------------
# prepare_training_frame
# ---------------------------------------------------------------------------

def test_prepare_training_frame_drops_missing_target_rows(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    with_gap = matrix.copy()
    missing_at = with_gap.index[1000]
    with_gap.loc[missing_at, config.LOAD_ACTUAL_COL] = np.nan

    prepared = forecast.prepare_training_frame(with_gap, config.LOAD_ACTUAL_COL)
    assert missing_at not in prepared.index
    assert len(prepared) == len(with_gap) - 1


def test_prepare_training_frame_drops_exclude_long_gap_rows(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    flagged = matrix.copy()
    flagged["exclude_long_gap"] = False
    excluded_at = flagged.index[2000]
    flagged.loc[excluded_at, "exclude_long_gap"] = True

    prepared = forecast.prepare_training_frame(flagged, config.LOAD_ACTUAL_COL)
    assert excluded_at not in prepared.index
    assert len(prepared) == len(flagged) - 1


def test_prepare_training_frame_keeps_rows_with_missing_features(synthetic_df):
    # The first 168h of any series has NaN lag_168h/rolling_*_168h features
    # by construction (no history yet) — prepare_training_frame must keep
    # those rows (LightGBM handles NaN features natively), only dropping
    # rows for a missing *target* or an excluded long gap.
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    first_row = matrix.index[0]
    assert pd.isna(matrix.loc[first_row, "lag_168h"])

    prepared = forecast.prepare_training_frame(matrix, config.LOAD_ACTUAL_COL)
    assert first_row in prepared.index


# ---------------------------------------------------------------------------
# fit_lightgbm_point / make_lightgbm_predict_fn
# ---------------------------------------------------------------------------

def test_fit_lightgbm_point_fits_and_predicts(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    prepared = forecast.prepare_training_frame(matrix, config.LOAD_ACTUAL_COL)
    feature_cols = features.select_feature_columns(prepared)

    fast_params = {"objective": "regression", "n_estimators": 20, "random_state": 42}
    model = forecast.fit_lightgbm_point(
        prepared[feature_cols], prepared[config.LOAD_ACTUAL_COL], params=fast_params
    )
    predictions = model.predict(prepared[feature_cols])
    assert len(predictions) == len(prepared)
    assert np.isfinite(predictions).all()


def test_make_lightgbm_predict_fn_returns_series_aligned_to_forecast_df(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    split_at = matrix.index[len(matrix) // 2]
    train_df = matrix[matrix.index < split_at]
    forecast_df = matrix[
        (matrix.index >= split_at) & (matrix.index < split_at + pd.Timedelta(hours=24))
    ]

    fast_params = {"objective": "regression", "n_estimators": 20, "random_state": 42}
    predict_fn = forecast.make_lightgbm_predict_fn(params=fast_params)
    predictions = predict_fn(train_df, forecast_df)

    assert isinstance(predictions, pd.Series)
    assert predictions.name == "y_lgbm"
    assert list(predictions.index) == list(forecast_df.index)
    assert predictions.notna().all()


def test_rolling_origin_backtest_with_real_lightgbm_beats_dumb_constant(synthetic_df):
    # Integration test: run the actual harness with the real LightGBM
    # predict_fn (not a spy) over a short period of the synthetic weekly
    # signal, and sanity-check it learned *something* — comfortably beats
    # a model that just predicts the training mean for everything.
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    fast_params = {"objective": "regression", "n_estimators": 50, "random_state": 42}

    result = forecast.rolling_origin_backtest(
        matrix,
        config.LOAD_ACTUAL_COL,
        predict_fn=forecast.make_lightgbm_predict_fn(params=fast_params),
        retrain_freq="MS",
        test_start="2018-03-01",
        test_end="2018-03-31",
    )

    assert result["y_lgbm"].notna().all()
    lgbm_mae = (result["y_true"] - result["y_lgbm"]).abs().mean()

    train_mean = matrix.loc[matrix.index < result.index.min(), config.LOAD_ACTUAL_COL].mean()
    dumb_mae = (result["y_true"] - train_mean).abs().mean()

    assert lgbm_mae < dumb_mae


# ---------------------------------------------------------------------------
# rolling_origin_backtest — output_col
# ---------------------------------------------------------------------------

def test_rolling_origin_backtest_output_col_is_overridable(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    result = forecast.rolling_origin_backtest(
        matrix,
        config.LOAD_ACTUAL_COL,
        predict_fn=lambda train_df, forecast_df: pd.Series(0.0, index=forecast_df.index),
        output_col="q10",
        retrain_freq="MS",
        test_start="2018-01-01",
        test_end="2018-01-31",
    )
    assert "q10" in result.columns
    assert "y_lgbm" not in result.columns
    assert (result["q10"] == 0.0).all()


# ---------------------------------------------------------------------------
# fit_lightgbm_quantile / make_lightgbm_quantile_predict_fn
# ---------------------------------------------------------------------------

def test_fit_lightgbm_quantile_fits_and_predicts(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    prepared = forecast.prepare_training_frame(matrix, config.LOAD_ACTUAL_COL)
    feature_cols = features.select_feature_columns(prepared)

    fast_params = {"objective": "quantile", "n_estimators": 20, "random_state": 42}
    model = forecast.fit_lightgbm_quantile(
        prepared[feature_cols], prepared[config.LOAD_ACTUAL_COL], quantile=0.1, params=fast_params
    )
    predictions = model.predict(prepared[feature_cols])
    assert len(predictions) == len(prepared)
    assert np.isfinite(predictions).all()


def test_low_quantile_predicts_below_high_quantile_on_average(synthetic_df):
    # Not a pointwise guarantee (LightGBM's separately-trained quantile
    # models have no built-in monotonicity constraint), but on a smooth
    # synthetic signal with plenty of training data, q90's average
    # prediction should clear q10's average prediction comfortably.
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    split_at = matrix.index[len(matrix) // 2]
    train_df = matrix[matrix.index < split_at]
    forecast_df = matrix[
        (matrix.index >= split_at) & (matrix.index < split_at + pd.Timedelta(hours=48))
    ]

    fast_params = {"objective": "quantile", "n_estimators": 50, "random_state": 42}
    low_fn = forecast.make_lightgbm_quantile_predict_fn(0.10, params=fast_params)
    high_fn = forecast.make_lightgbm_quantile_predict_fn(0.90, params=fast_params)
    low_predictions = low_fn(train_df, forecast_df)
    high_predictions = high_fn(train_df, forecast_df)

    assert low_predictions.name == "q10"
    assert high_predictions.name == "q90"
    assert high_predictions.mean() > low_predictions.mean()


def test_run_quantile_backtests_returns_expected_columns(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    fast_params = {"objective": "quantile", "n_estimators": 20, "random_state": 42}

    result = forecast.run_quantile_backtests(
        matrix,
        config.LOAD_ACTUAL_COL,
        params=fast_params,
        retrain_freq="MS",
        test_start="2018-03-01",
        test_end="2018-03-31",
    )

    assert set(result.columns) == {"y_true", "y_naive", "y_tso", "q10", "q90"}
    assert result["q10"].notna().all()
    assert result["q90"].notna().all()


# ---------------------------------------------------------------------------
# compute_interval_metrics
# ---------------------------------------------------------------------------

def test_compute_interval_metrics_hand_computed():
    table = pd.DataFrame(
        {
            "y_true": [100.0, 200.0, 300.0, 1000.0],  # last row: true value outside the interval
            "q10": [90.0, 180.0, 250.0, 100.0],
            "q90": [110.0, 220.0, 350.0, 200.0],
        }
    )
    metrics = forecast.compute_interval_metrics(table, target_coverage=0.80)

    assert metrics["n"] == 4
    assert metrics["coverage"] == 0.75  # 3 of 4 rows covered
    assert metrics["target_coverage"] == 0.80
    expected_width = ((110 - 90) + (220 - 180) + (350 - 250) + (200 - 100)) / 4
    assert np.isclose(metrics["avg_interval_width"], expected_width)
    assert metrics["pinball_loss_low"] > 0  # q10 misses on the last (uncovered) row
    assert metrics["pinball_loss_high"] > 0


def test_compute_interval_metrics_perfect_coverage_has_zero_pinball_loss_at_the_bounds():
    # If the true value always sits exactly on the bound being scored, the
    # pinball loss for that side is exactly 0 (the loss function's minimum).
    table = pd.DataFrame({"y_true": [100.0, 200.0], "q10": [100.0, 200.0], "q90": [150.0, 250.0]})
    metrics = forecast.compute_interval_metrics(table)
    assert np.isclose(metrics["pinball_loss_low"], 0.0)


def test_compute_interval_metrics_drops_rows_with_missing_values():
    table = pd.DataFrame(
        {
            "y_true": [100.0, np.nan, 300.0],
            "q10": [90.0, 180.0, 250.0],
            "q90": [110.0, 220.0, 350.0],
        }
    )
    metrics = forecast.compute_interval_metrics(table)
    assert metrics["n"] == 2


# ---------------------------------------------------------------------------
# sweep_quantile_params
# ---------------------------------------------------------------------------

def test_sweep_quantile_params_returns_one_row_per_candidate(synthetic_df):
    matrix = features.build_feature_matrix(synthetic_df, target_col=config.LOAD_ACTUAL_COL)
    grid = {
        "few_leaves": {
            "objective": "quantile",
            "n_estimators": 15,
            "num_leaves": 7,
            "random_state": 42,
        },
        "more_leaves": {
            "objective": "quantile",
            "n_estimators": 15,
            "num_leaves": 31,
            "random_state": 42,
        },
    }

    result = forecast.sweep_quantile_params(
        matrix, grid, retrain_freq="MS", test_start="2018-03-01", test_end="2018-03-31"
    )

    assert list(result.index) == ["few_leaves", "more_leaves"]
    assert "coverage" in result.columns
    assert "avg_interval_width" in result.columns
    assert (result["n"] > 0).all()
