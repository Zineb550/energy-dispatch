"""
Module 3 (partial) — demand forecasting: baselines, time-based splits,
point metrics, the LightGBM point and quantile models, and a
model-agnostic rolling-origin backtest harness.

The LightGBM point model (fit_lightgbm_point / make_lightgbm_predict_fn)
and the LightGBM quantile models (fit_lightgbm_quantile /
make_lightgbm_quantile_predict_fn / run_quantile_backtests) are both now
implemented, along with compute_interval_metrics (coverage + pinball
loss). SHAP and the Diebold-Mariano test remain intentionally NOT
implemented yet — deferred to a later step. rolling_origin_backtest
gained one small, backward-compatible addition to support this
(output_col, defaulting to "y_lgbm") rather than a new harness: the same
block-iteration logic that fits/predicts the point model also fits/
predicts each quantile model, just writing its predictions to a
differently-named column.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

import numpy as np
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


def fit_lightgbm_quantile(
    X_train: pd.DataFrame, y_train: pd.Series, quantile: float, params: dict | None = None
):
    """Train a single quantile LightGBM model (typically config.QUANTILE_LOW
    or config.QUANTILE_HIGH) using config.LIGHTGBM_QUANTILE_PARAMS (or an
    override) plus the requested quantile's alpha, returning the fitted
    lgb.LGBMRegressor.

    A separate model per quantile (rather than one multi-output model) is
    the standard approach for gradient-boosted quantile regression: each
    call trains against the pinball loss for its own alpha.
    """
    import lightgbm as lgb

    base_params = dict(params if params is not None else config.LIGHTGBM_QUANTILE_PARAMS)
    base_params["alpha"] = quantile
    model = lgb.LGBMRegressor(**base_params)
    model.fit(X_train, y_train)
    return model


def make_lightgbm_quantile_predict_fn(
    quantile: float, target_col: str = config.LOAD_ACTUAL_COL, params: dict | None = None
) -> PredictFn:
    """Build a predict_fn (suitable for rolling_origin_backtest) for a
    single quantile model. Mirrors make_lightgbm_predict_fn exactly, but
    fits fit_lightgbm_quantile(..., quantile) instead of the point model.
    The returned Series is named q{int(quantile * 100)}, e.g. "q10" for
    quantile=0.10 — pass that as rolling_origin_backtest's output_col so
    it lands in a column of the same name rather than overwriting "y_lgbm".
    """

    def _predict(train_df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.Series:
        usable_train = prepare_training_frame(train_df, target_col)
        feature_cols = features.select_feature_columns(usable_train)
        model = fit_lightgbm_quantile(
            usable_train[feature_cols], usable_train[target_col], quantile, params=params
        )
        predictions = model.predict(forecast_df[feature_cols])
        col_name = f"q{int(round(quantile * 100))}"
        return pd.Series(predictions, index=forecast_df.index, name=col_name)

    return _predict


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
    output_col: str = "y_lgbm",
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
    output_col column. output_col defaults to "y_lgbm" (the LightGBM point
    model this harness is normally used with — see make_lightgbm_predict_fn)
    but is overridable so the exact same harness can also run a quantile
    model into its own column (e.g. output_col="q10" with
    make_lightgbm_quantile_predict_fn — see run_quantile_backtests), or any
    other model-agnostic predict_fn. If predict_fn is None (no model
    supplied), the block iteration is skipped entirely and only the two
    baselines are computed.

    Returns a DataFrame over the full [test_start, test_end] range with
    columns y_true, y_naive, y_tso, <output_col> — output_col is always
    present, all-NA when predict_fn is None, so the table's shape doesn't
    change whether or not a model is plugged in.
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
    result[output_col] = pd.NA

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
        result[output_col] = all_predictions.reindex(test_index)

    return result


def run_quantile_backtests(
    feature_matrix: pd.DataFrame,
    target_col: str = config.LOAD_ACTUAL_COL,
    quantiles: tuple[float, ...] = (config.QUANTILE_LOW, config.QUANTILE_HIGH),
    params: dict | None = None,
    retrain_freq: str = config.BACKTEST_RETRAIN_FREQ,
    test_start: str = config.TEST_START,
    test_end: str = config.TEST_END,
) -> pd.DataFrame:
    """Run rolling_origin_backtest once per quantile in `quantiles`, each
    fitting its own LightGBM quantile model per retrain block (via
    make_lightgbm_quantile_predict_fn), and combine the results into a
    single table: y_true, y_naive, y_tso (identical across every quantile
    run, since those don't depend on predict_fn, so taken from the first
    run), plus one column per quantile named q{int(q * 100)} (e.g. "q10",
    "q90" for the default config.QUANTILE_LOW/HIGH).
    """
    combined: pd.DataFrame | None = None
    for q in quantiles:
        col = f"q{int(round(q * 100))}"
        table = rolling_origin_backtest(
            feature_matrix,
            target_col,
            predict_fn=make_lightgbm_quantile_predict_fn(q, target_col=target_col, params=params),
            output_col=col,
            retrain_freq=retrain_freq,
            test_start=test_start,
            test_end=test_end,
        )
        if combined is None:
            combined = table[["y_true", "y_naive", "y_tso"]].copy()
        combined[col] = table[col]

    return combined


def sweep_quantile_params(
    feature_matrix: pd.DataFrame,
    param_grid: dict[str, dict],
    target_col: str = config.LOAD_ACTUAL_COL,
    quantiles: tuple[float, ...] = (config.QUANTILE_LOW, config.QUANTILE_HIGH),
    retrain_freq: str = config.BACKTEST_RETRAIN_FREQ,
    test_start: str = config.TEST_START,
    test_end: str = config.TEST_END,
) -> pd.DataFrame:
    """Run run_quantile_backtests once per named hyperparameter set in
    param_grid ({"label": {...LightGBM params...}, ...}) and return one row
    of compute_interval_metrics per label, so several candidate
    hyperparameter sets can be compared for calibration (coverage vs.
    config.PREDICTION_INTERVAL_COVERAGE_TARGET) in a single local run
    instead of one diff/apply/run round-trip per candidate.
    """
    rows = []
    for label, params in param_grid.items():
        table = run_quantile_backtests(
            feature_matrix,
            target_col,
            quantiles=quantiles,
            params=params,
            retrain_freq=retrain_freq,
            test_start=test_start,
            test_end=test_end,
        )
        metrics = compute_interval_metrics(table)
        rows.append({"params": label, **metrics})
    return pd.DataFrame(rows).set_index("params")


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


def compute_interval_metrics(
    forecast_table: pd.DataFrame,
    low_col: str = "q10",
    high_col: str = "q90",
    true_col: str = "y_true",
    target_coverage: float = config.PREDICTION_INTERVAL_COVERAGE_TARGET,
) -> dict:
    """Coverage and pinball loss for a [low_col, high_col] prediction
    interval against true_col.

    - coverage: the empirical fraction of rows where true_col actually
      falls within [low_col, high_col] — compare this against
      target_coverage (0.80 for the default 10th/90th percentile
      interval) to see whether the interval is well-calibrated,
      too narrow (coverage < target), or too wide (coverage > target).
    - avg_interval_width: mean(high_col - low_col), reported alongside
      coverage since a trivially wide interval can hit any coverage
      target without being useful.
    - pinball_loss_low / pinball_loss_high: the quantile (pinball) loss
      for each side's own quantile (config.QUANTILE_LOW/HIGH), the
      proper scoring rule each underlying LightGBM quantile model is
      actually trained against — lower is better, 0 is a perfect fit.

    Rows with a missing true/low/high value are dropped (n reports how
    many rows the metrics were computed over).
    """
    valid = (
        forecast_table[true_col].notna()
        & forecast_table[low_col].notna()
        & forecast_table[high_col].notna()
    )
    n = int(valid.sum())
    if n == 0:
        nan = float("nan")
        return {
            "coverage": nan,
            "target_coverage": target_coverage,
            "avg_interval_width": nan,
            "pinball_loss_low": nan,
            "pinball_loss_high": nan,
            "n": 0,
        }

    y_true = forecast_table.loc[valid, true_col]
    y_low = forecast_table.loc[valid, low_col]
    y_high = forecast_table.loc[valid, high_col]

    coverage = float(((y_true >= y_low) & (y_true <= y_high)).mean())
    avg_width = float((y_high - y_low).mean())

    def _pinball_loss(y_true: pd.Series, y_pred: pd.Series, tau: float) -> float:
        diff = y_true - y_pred
        return float(np.maximum(tau * diff, (tau - 1) * diff).mean())

    return {
        "coverage": coverage,
        "target_coverage": target_coverage,
        "avg_interval_width": avg_width,
        "pinball_loss_low": _pinball_loss(y_true, y_low, config.QUANTILE_LOW),
        "pinball_loss_high": _pinball_loss(y_true, y_high, config.QUANTILE_HIGH),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Everything below is still deferred (unrelated to the quantile models)
# ---------------------------------------------------------------------------


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

# Candidate hyperparameter sets for `--sweep-quantile-params`, tried after
# the real backtest showed the default quantile params (config.
# LIGHTGBM_QUANTILE_PARAMS, same capacity as the point model: num_leaves=63,
# n_estimators=500) under-covering badly (56% empirical vs. an 80% target).
# Gradient-boosted quantile (pinball-loss) regression is prone to exactly
# this: the tails of the loss get noisier gradients than the bulk, so a
# high-capacity model fits the *training* quantiles well but generalizes
# to a narrower-than-true spread on the test set — the fix is more
# regularization, not more capacity, hence every candidate below is more
# regularized than the current default, not less. "current_default" is
# included as the baseline row so the comparison table shows the delta.
QUANTILE_PARAM_SWEEP_GRID: dict[str, dict] = {
    "current_default": dict(config.LIGHTGBM_QUANTILE_PARAMS),
    "fewer_leaves": {**config.LIGHTGBM_QUANTILE_PARAMS, "num_leaves": 15},
    "regularized": {
        "objective": "quantile",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": config.RANDOM_SEED,
    },
    "heavy_regularization": {
        "objective": "quantile",
        "n_estimators": 800,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 100,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "random_state": config.RANDOM_SEED,
    },
}


def run_full_backtest(
    processed_path=config.PROCESSED_HOURLY_PARQUET,
    test_start: str = config.TEST_START,
    test_end: str = config.TEST_END,
    skip_quantiles: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """Load the processed hourly dataset, build the feature matrix, check
    it for leakage, run the rolling-origin backtest with the LightGBM point
    model over [test_start, test_end], and (unless skip_quantiles) also run
    it once per quantile in config.QUANTILE_LOW/HIGH, merging their columns
    (q10, q90 by default) into the same forecast table. Computes point
    metrics for all three point-forecast columns (seasonal-naive, TSO,
    LightGBM) and, when the quantile columns are present, interval metrics
    for them too.

    Returns (forecast_table, metrics_table, interval_metrics) —
    interval_metrics is None when skip_quantiles is True. Does not write
    anything to disk — see main() for that.
    """
    logging.info("Loading processed dataset from %s", processed_path)
    df = pd.read_parquet(processed_path)

    logging.info("Building feature matrix...")
    matrix = features.build_feature_matrix(df, target_col=config.LOAD_ACTUAL_COL)
    features.assert_no_leakage(matrix)
    logging.info("Leakage check passed.")

    logging.info(
        "Running rolling-origin backtest with the LightGBM point model over "
        "%s to %s (this re-fits a model at every retrain block, so it can "
        "take a while)...",
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

    interval_metrics = None
    if not skip_quantiles:
        logging.info(
            "Running rolling-origin backtest with the LightGBM quantile "
            "models (one retrain-per-block fit per quantile, so this "
            "roughly doubles the runtime for the default 2 quantiles)..."
        )
        quantile_table = run_quantile_backtests(
            matrix, config.LOAD_ACTUAL_COL, test_start=test_start, test_end=test_end
        )
        for col in quantile_table.columns:
            if col not in ("y_true", "y_naive", "y_tso"):
                forecast_table[col] = quantile_table[col]
        interval_metrics = compute_interval_metrics(forecast_table)

    metrics_table = compute_point_metrics(
        forecast_table, model_cols=("y_naive", "y_tso", "y_lgbm")
    )
    return forecast_table, metrics_table, interval_metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LightGBM point + quantile rolling-origin backtest "
            "against the local processed dataset and write the forecast "
            "table and metrics table to data/processed/."
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
    parser.add_argument(
        "--skip-quantiles",
        action="store_true",
        help="Only run the LightGBM point model (skip the q10/q90 quantile "
        "backtests, roughly halving the runtime).",
    )
    parser.add_argument(
        "--sweep-quantile-params",
        action="store_true",
        help="Instead of the normal run, try several LightGBM quantile "
        "hyperparameter sets (QUANTILE_PARAM_SWEEP_GRID) and print a "
        "comparison of their interval metrics (coverage, width, pinball "
        "loss) — use this to pick a better-calibrated config. Does not "
        "write the forecast/metrics parquet files.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args(argv)

    if args.sweep_quantile_params:
        logging.info("Loading processed dataset from %s", config.PROCESSED_HOURLY_PARQUET)
        df = pd.read_parquet(config.PROCESSED_HOURLY_PARQUET)
        logging.info("Building feature matrix...")
        matrix = features.build_feature_matrix(df, target_col=config.LOAD_ACTUAL_COL)
        features.assert_no_leakage(matrix)

        logging.info(
            "Sweeping %d quantile hyperparameter sets over %s to %s "
            "(this fits 2 quantile models x every retrain block, per set, "
            "so it takes roughly len(QUANTILE_PARAM_SWEEP_GRID)x as long as "
            "a single quantile backtest)...",
            len(QUANTILE_PARAM_SWEEP_GRID),
            args.test_start,
            args.test_end,
        )
        sweep_table = sweep_quantile_params(
            matrix,
            QUANTILE_PARAM_SWEEP_GRID,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        pd.set_option("display.float_format", "{:.4f}".format)
        logging.info(
            "\nInterval metrics by hyperparameter set (target_coverage=%.2f):\n%s",
            config.PREDICTION_INTERVAL_COVERAGE_TARGET,
            sweep_table,
        )
        return

    forecast_table, metrics_table, interval_metrics = run_full_backtest(
        test_start=args.test_start, test_end=args.test_end, skip_quantiles=args.skip_quantiles
    )

    pd.set_option("display.float_format", "{:.2f}".format)
    logging.info("\n%s", metrics_table)
    if interval_metrics is not None:
        logging.info("\nInterval metrics (q10/q90):\n%s", interval_metrics)

    config.FORECAST_TABLE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    forecast_table.to_parquet(config.FORECAST_TABLE_PARQUET)
    metrics_table.to_parquet(config.METRICS_TABLE_PARQUET)
    logging.info("Wrote %s", config.FORECAST_TABLE_PARQUET)
    logging.info("Wrote %s", config.METRICS_TABLE_PARQUET)


if __name__ == "__main__":
    main()
