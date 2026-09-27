"""
Module 3 (partial) — demand forecasting: baselines, time-based splits,
point metrics, and a model-agnostic rolling-origin backtest harness.

LightGBM (point + quantile), interval metrics, SHAP, and the
Diebold-Mariano test are intentionally NOT implemented yet — there's no
learned model to evaluate until LightGBM lands, and this module is scoped
to what's usable before that: the two parameter-free benchmarks, the
splits, and MAE/RMSE/MAPE. rolling_origin_backtest is written to be
model-agnostic (a pluggable predict_fn) precisely so it doesn't need to
change when LightGBM is added later.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from energy_dispatch import config, features

# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def seasonal_naive_forecast(
    df: pd.DataFrame, target_col: str = config.LOAD_ACTUAL_COL
) -> pd.Series:
    """Floor baseline: for every row, the load 168h (exactly one week)
    before that row's own target timestamp — "same hour, same weekday,
    last week".

    This is target-relative (not origin-anchored, unlike the ML features
    in features.py): 168h clears the worst-case origin-to-target gap
    (~37h at a 10:00 cutoff) by a wide margin regardless of which of the
    24 hours is being forecast, so a single literal "t - 168h" is safe
    for the whole day at once — see features.py's module docstring for
    why the same isn't true of a naive 24h target-relative lag.
    """
    lookup_times = df.index - pd.Timedelta(hours=168)
    values = df[target_col].reindex(lookup_times).to_numpy()
    return pd.Series(values, index=df.index, name="y_naive")


def tso_forecast(df: pd.DataFrame) -> pd.Series:
    """Professional benchmark: ENTSO-E's own published day-ahead forecast,
    as-is. This is an already-published forecast for comparison, not a
    feature fed into our own model, so it carries no leakage concern.
    """
    return df[config.LOAD_FORECAST_COL].rename("y_tso")


# ---------------------------------------------------------------------------
# LightGBM models — deferred
# ---------------------------------------------------------------------------

def fit_lightgbm_point(X_train: pd.DataFrame, y_train: pd.Series):
    """Train the point-forecast LightGBM model using config.LIGHTGBM_PARAMS.

    Deliberately not implemented yet — see module docstring.
    """
    raise NotImplementedError("LightGBM is deferred until the forecasting setup is verified")


def fit_lightgbm_quantile(X_train: pd.DataFrame, y_train: pd.Series, quantile: float):
    """Train a single quantile LightGBM model (config.QUANTILE_LOW / HIGH).

    Deliberately not implemented yet — see module docstring.
    """
    raise NotImplementedError("LightGBM is deferred until the forecasting setup is verified")


# ---------------------------------------------------------------------------
# Time-based splits
# ---------------------------------------------------------------------------

def time_based_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split df into train/validation/test/stress_test by the *local*
    calendar date of each row's own index timestamp, using the date
    ranges in config.py (TRAIN_START/END, VALIDATION_START/END,
    TEST_START/END, STRESS_TEST_START/END).

    Splitting on the local date (not the raw UTC timestamp) matters at
    the boundaries: Dec 31 2017 23:00 UTC is already Jan 1 2018 in
    Europe/Madrid in winter (CET, UTC+1), and getting that boundary
    row's split membership wrong by working in UTC would leak a few
    hours of 2018 into the 2015-2017 training set (or the reverse).
    """
    local_date = df.index.tz_convert(config.LOCAL_TZ).normalize()

    def _between(start: str, end: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start, tz=config.LOCAL_TZ)
        end_ts = pd.Timestamp(end, tz=config.LOCAL_TZ)
        return df[(local_date >= start_ts) & (local_date <= end_ts)]

    return {
        "train": _between(config.TRAIN_START, config.TRAIN_END),
        "validation": _between(config.VALIDATION_START, config.VALIDATION_END),
        "test": _between(config.TEST_START, config.TEST_END),
        "stress_test": _between(config.STRESS_TEST_START, config.STRESS_TEST_END),
    }


# ---------------------------------------------------------------------------
# Rolling-origin backtest (model-agnostic)
# ---------------------------------------------------------------------------

PredictFn = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


def _retrain_block_bounds(
    test_start: str, test_end: str, retrain_freq: str
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Local-date block boundaries [block_start, block_end) covering
    [test_start, test_end], anchored to retrain_freq (e.g. "MS" = month
    start). Each block is forecast using a single train/fit taken at
    block_start — see rolling_origin_backtest.
    """
    tz = config.LOCAL_TZ
    start = pd.Timestamp(test_start, tz=tz)
    end_exclusive = pd.Timestamp(test_end, tz=tz) + pd.Timedelta(days=1)

    retrain_dates = pd.date_range(start, end_exclusive, freq=retrain_freq, tz=tz)
    if len(retrain_dates) == 0 or retrain_dates[0] > start:
        retrain_dates = retrain_dates.insert(0, start)
    if retrain_dates[-1] < end_exclusive:
        retrain_dates = retrain_dates.append(pd.DatetimeIndex([end_exclusive]))

    return list(zip(retrain_dates[:-1], retrain_dates[1:], strict=True))


def rolling_origin_backtest(
    feature_matrix: pd.DataFrame,
    target_col: str,
    predict_fn: PredictFn | None = None,
    retrain_freq: str = config.BACKTEST_RETRAIN_FREQ,
    test_start: str = config.TEST_START,
    test_end: str = config.TEST_END,
) -> pd.DataFrame:
    """Roll through [test_start, test_end] in retrain_freq-sized blocks
    (default: calendar months). For each block [block_start, block_end):

      - train_df = every row of feature_matrix whose forecast origin is at
        or before block_start's own origin (i.e. everything knowable at
        the moment this block starts being forecast) — this is what
        predict_fn would "fit" on, if it fits anything.
      - forecast_df = the rows of feature_matrix whose target local date
        falls in [block_start, block_end).

    If predict_fn is given, it's called once per block as
    predict_fn(train_df, forecast_df) -> pd.Series aligned to
    forecast_df.index, and the concatenated result is returned as the
    'y_model' column. If predict_fn is None (the current state — no
    learned model exists yet), the block iteration is skipped entirely
    and only the two baselines are computed.

    Returns a DataFrame over the full [test_start, test_end] range with
    columns y_true, y_naive, y_tso, y_model — y_model is always present,
    all-NA when predict_fn is None, so the table's shape doesn't change
    once a model is plugged in later.
    """
    local_date = feature_matrix.index.tz_convert(config.LOCAL_TZ).normalize()
    blocks = _retrain_block_bounds(test_start, test_end, retrain_freq)

    test_mask = (local_date >= pd.Timestamp(test_start, tz=config.LOCAL_TZ)) & (
        local_date <= pd.Timestamp(test_end, tz=config.LOCAL_TZ)
    )
    test_index = feature_matrix.index[test_mask]

    result = pd.DataFrame(index=test_index)
    result["y_true"] = feature_matrix.loc[test_index, target_col]
    result["y_naive"] = seasonal_naive_forecast(feature_matrix, target_col).loc[test_index]
    result["y_tso"] = tso_forecast(feature_matrix).loc[test_index]
    result["y_model"] = pd.NA

    if predict_fn is None:
        return result

    predictions = []
    for block_start, block_end in blocks:
        block_origin = features.compute_forecast_origin(pd.DatetimeIndex([block_start]))[0]
        # Strictly before block_origin: block_start's own target day shares
        # that exact origin (every row of one target day shares one
        # origin, by construction — see features.compute_forecast_origin),
        # so a "<=" here would silently include the block's own rows —
        # the very thing being forecast — in its own training set.
        train_df = feature_matrix[feature_matrix["origin_utc"] < block_origin]

        block_mask = (local_date >= block_start) & (local_date < block_end)
        forecast_df = feature_matrix.loc[block_mask]
        if forecast_df.empty:
            continue

        block_predictions = predict_fn(train_df, forecast_df)
        predictions.append(block_predictions)

    if predictions:
        all_predictions = pd.concat(predictions)
        result["y_model"] = all_predictions.reindex(test_index)

    return result


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def compute_point_metrics(
    forecast_table: pd.DataFrame,
    model_cols: tuple[str, ...] = ("y_naive", "y_tso"),
    true_col: str = "y_true",
) -> pd.DataFrame:
    """MAE, RMSE, MAPE for each column in model_cols against true_col.
    Rows where either value is NaN are dropped per model (not
    per-table), and the surviving row count is reported alongside the
    metrics so a model scored on fewer rows (e.g. y_naive missing its
    first week) is visible rather than silently averaged over less data.
    """
    y_true = forecast_table[true_col]
    rows = []
    for col in model_cols:
        y_pred = forecast_table[col]
        valid = y_true.notna() & y_pred.notna()
        n = int(valid.sum())
        if n == 0:
            nan = float("nan")
            rows.append({"model": col, "MAE": nan, "RMSE": nan, "MAPE": nan, "n": 0})
            continue

        err = y_true[valid] - y_pred[valid]
        mae = err.abs().mean()
        rmse = (err**2).mean() ** 0.5
        mape = (err.abs() / y_true[valid].abs()).mean() * 100
        rows.append({"model": col, "MAE": mae, "RMSE": rmse, "MAPE": mape, "n": n})

    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Everything below needs a fitted model — deferred alongside LightGBM
# ---------------------------------------------------------------------------

def compute_interval_metrics(forecast_table: pd.DataFrame) -> dict:
    """Coverage and pinball loss for the quantile models.

    Deliberately not implemented yet — needs fit_lightgbm_quantile.
    """
    raise NotImplementedError("needs the quantile models, deferred alongside LightGBM")


def error_breakdown(forecast_table: pd.DataFrame) -> pd.DataFrame:
    """Error metrics sliced by hour, season, holiday flag, and extreme-
    temperature days.

    Deliberately not implemented yet — revisit once there's a model worth
    slicing (the two baselines can already be sliced with
    compute_point_metrics per-subset if useful sooner).
    """
    raise NotImplementedError("deferred alongside LightGBM")


def shap_feature_importance(model, X: pd.DataFrame):
    """SHAP values for the LightGBM point model.

    Deliberately not implemented yet — needs fit_lightgbm_point.
    """
    raise NotImplementedError("needs the LightGBM point model, deferred alongside it")


def diebold_mariano_test(errors_a: pd.Series, errors_b: pd.Series) -> dict:
    """Diebold-Mariano test for whether two models' errors differ
    significantly.

    Deliberately not implemented yet — the spec uses this to compare
    LightGBM against a benchmark, so it's deferred alongside LightGBM
    rather than wired up against only the two baselines for now.
    """
    raise NotImplementedError("deferred alongside LightGBM")
