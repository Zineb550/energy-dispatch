"""
Module 3 (partial) — demand forecasting: baselines, time-based splits,
point metrics, the LightGBM point model, and a model-agnostic
rolling-origin backtest harness.

The LightGBM point model (fit_lightgbm_point / make_lightgbm_predict_fn) is
now implemented and wired into rolling_origin_backtest as its predict_fn.
LightGBM quantile models, interval metrics, SHAP, and the Diebold-Mariano
test remain intentionally NOT implemented yet — those need the quantile
models specifically, not just a fitted point model, and are deferred to a
later step. rolling_origin_backtest itself needed no changes to accept the
new predict_fn: it was written model-agnostic from the start precisely so
this addition wouldn't require touching the harness.
"""

from __future__ import annotations

import argparse
import logging
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
# LightGBM point model
# ---------------------------------------------------------------------------

def prepare_training_frame(
    feature_matrix: pd.DataFrame, target_col: str = config.LOAD_ACTUAL_COL
) -> pd.DataFrame:
    """Rows of a built feature matrix usable for *fitting* a model: drops
    rows flagged exclude_long_gap (data.py's marker for a gap too long to
    interpolate, so the target itself is unreliable there) and rows with a
    missing target value.

    Deliberately does NOT drop rows for missing *feature* values (e.g. the
    lag_168h/rolling_168h columns are NaN for the first 168h of the whole
    series) — LightGBM's own split logic handles NaN features natively, so
    dropping those rows would only throw away otherwise-usable training
    examples for no benefit.
    """
    usable = pd.Series(True, index=feature_matrix.index)
    if "exclude_long_gap" in feature_matrix.columns:
        usable &= ~feature_matrix["exclude_long_gap"].fillna(False)
    usable &= feature_matrix[target_col].notna()
    return feature_matrix[usable]


def fit_lightgbm_point(X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None):
    """Train the point-forecast LightGBM model using config.LIGHTGBM_PARAMS
    (or an override), returning the fitted lgb.LGBMRegressor.
    """
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**(params if params is not None else config.LIGHTGBM_PARAMS))
    model.fit(X_train, y_train)
    return model


def make_lightgbm_predict_fn(
    target_col: str = config.LOAD_ACTUAL_COL, params: dict | None = None
) -> PredictFn:
    """Build a predict_fn suitable for rolling_origin_backtest's predict_fn
    argument: on every call it re-fits a fresh LightGBM model on train_df
    (via prepare_training_frame + features.select_feature_columns) and
    predicts on forecast_df, so each retrain block gets its own model
    trained only on data knowable as of that block's own forecast origin.
    """

    def _predict(train_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.Series:
        usable_train = prepare_training_frame(train_df, target_col)
        feature_cols = features.select_feature_columns(usable_train)
        model = fit_lightgbm_point(
            usable_train[feature_cols], usable_train[target_col], params=params
        )
        predictions = model.predict(forecast_df[feature_cols])
        return pd.Series(predictions, index=forecast_df.index, name="y_lgbm")

    return _predict


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
    'y_lgbm' column (named for the LightGBM predict_fn this harness is
    normally used with — see make_lightgbm_predict_fn — though predict_fn
    can be any model-agnostic callable). If predict_fn is None (no model
    supplied), the block iteration is skipped entirely and only the two
    baselines are computed.

    Returns a DataFrame over the full [test_start, test_end] range with
    columns y_true, y_naive, y_tso, y_lgbm — y_lgbm is always present,
    all-NA when predict_fn is None, so the table's shape doesn't change
    whether or not a model is plugged in.
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
    result["y_lgbm"] = pd.NA

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
        result["y_lgbm"] = all_predictions.reindex(test_index)

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

    fit_lightgbm_point now exists, but this is still deliberately not
    implemented yet — deferred to a later step, not blocked on anything.
    """
    raise NotImplementedError("deferred to a later step")


def diebold_mariano_test(errors_a: pd.Series, errors_b: pd.Series) -> dict:
    """Diebold-Mariano test for whether two models' errors differ
    significantly.

    Deliberately not implemented yet — deferred to a later step, alongside
    the quantile models.
    """
    raise NotImplementedError("deferred to a later step")


# ---------------------------------------------------------------------------
# CLI: run the real LightGBM backtest against the local processed dataset
# ---------------------------------------------------------------------------

def run_lightgbm_backtest(
    processed_path=config.PROCESSED_HOURLY_PARQUET,
    test_start: str = config.TEST_START,
    test_end: str = config.TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed hourly dataset, build the feature matrix, check
    it for leakage, run the rolling-origin backtest with the LightGBM
    predict_fn over [test_start, test_end], and compute point metrics for
    all three columns (seasonal-naive, TSO, LightGBM).

    Returns (forecast_table, metrics_table); does not write anything to
    disk — see main() for that.
    """
    logging.info("Loading processed dataset from %s", processed_path)
    df = pd.read_parquet(processed_path)

    logging.info("Building feature matrix...")
    matrix = features.build_feature_matrix(df, target_col=config.LOAD_ACTUAL_COL)
    features.assert_no_leakage(matrix)
    logging.info("Leakage check passed.")

    logging.info(
        "Running rolling-origin backtest with LightGBM over %s to %s "
        "(this re-fits a model at every retrain block, so it can take a "
        "while)...",
        test_start,
        test_end,
    )
    forecast_table = rolling_origin_backtest(
        matrix,
        config.LOAD_ACTUAL_COL,
        predict_fn=make_lightgbm_predict_fn(),
        test_start=test_start,
        test_end=test_end,
    )

    metrics_table = compute_point_metrics(
        forecast_table, model_cols=("y_naive", "y_tso", "y_lgbm")
    )
    return forecast_table, metrics_table


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LightGBM point-forecast rolling-origin backtest against "
            "the local processed dataset and write the forecast table and "
            "metrics table to data/processed/."
        )
    )
    parser.add_argument(
        "--test-start",
        default=config.TEST_START,
        help=f"First local date of the backtest period (default: {config.TEST_START}).",
    )
    parser.add_argument(
        "--test-end",
        default=config.TEST_END,
        help=f"Last local date of the backtest period (default: {config.TEST_END}).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args(argv)

    forecast_table, metrics_table = run_lightgbm_backtest(
        test_start=args.test_start, test_end=args.test_end
    )

    pd.set_option("display.float_format", "{:.2f}".format)
    logging.info("\n%s", metrics_table)

    config.FORECAST_TABLE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    forecast_table.to_parquet(config.FORECAST_TABLE_PARQUET)
    metrics_table.to_parquet(config.METRICS_TABLE_PARQUET)
    logging.info("Wrote %s", config.FORECAST_TABLE_PARQUET)
    logging.info("Wrote %s", config.METRICS_TABLE_PARQUET)


if __name__ == "__main__":
    main()
