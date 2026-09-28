"""
Forecast and decision diagnostics: are the differences we report real,
and what drives the model?

    python -m energy_dispatch.diagnostics

1. Diebold-Mariano tests (forecast.dm_test_on_losses) on
   - forecast errors (squared and absolute) for each pair of point forecasts;
   - hourly realized cost from the 2019 evaluation (battery off), which asks
     whether the euro differences between forecasts are more than noise;
   - hourly realized cost of the tau* quantile plan vs the point-forecast
     plan (planning.py), re-evaluated here because planning.py only keeps
     annual totals.
2. Error breakdown of each point forecast by local hour, day type and month.
3. SHAP feature importance of the LightGBM point model (the model the
   backtest fits for its first test block), explained on 2019 features.

Everything reads existing outputs; only the SHAP step fits one LightGBM
model, and the planning test re-solves two day-ahead plans per day.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from energy_dispatch import config, evaluate, features, forecast, planning

logger = logging.getLogger(__name__)

COST_PAIRS: tuple[tuple[str, str], ...] = (
    ("lightgbm", "seasonal_naive"),
    ("tso", "lightgbm"),
    ("tso", "seasonal_naive"),
)


def cost_dm_table(
    hourly_results: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = COST_PAIRS,
    battery: str = "off",
    max_lag: int = 168,
) -> pd.DataFrame:
    """Diebold-Mariano tests on hourly realized cost between forecast
    scenarios of evaluate.evaluate_year's output (same hours for all)."""
    rows = hourly_results[hourly_results["battery"] == battery]
    cost = rows.pivot_table(
        index=rows.index, columns="forecast_method", values="realized_cost_eur", aggfunc="sum"
    )
    out = []
    for a, b in pairs:
        if a in cost.columns and b in cost.columns:
            result = forecast.dm_test_on_losses(cost[a], cost[b], max_lag=max_lag)
            out.append({"test": "realized_cost", "model_a": a, "model_b": b,
                        "loss": "hourly realized cost (EUR)", **result})  # fmt: skip
    return pd.DataFrame(out)


def planning_dm_table(
    forecast_table: pd.DataFrame,
    renewable_available: pd.Series,
    target_quantile: float | None = None,
    max_lag: int = 168,
    max_days: int | None = None,
) -> pd.DataFrame:
    """Diebold-Mariano test on hourly realized cost: the tau* quantile plan
    vs the point-forecast plan (battery off, baseline day sample)."""
    tau = target_quantile if target_quantile is not None else planning.theoretical_plan_quantile()
    table = forecast_table.copy()
    point_name, tau_name = planning.plan_name(0.5), planning.plan_name(tau)
    table[point_name] = planning.quantile_plan(forecast_table, 0.5)
    table[tau_name] = planning.quantile_plan(forecast_table, tau)
    hourly, _ = evaluate.evaluate_year(
        table,
        renewable_available,
        scenarios={tau_name: tau_name, point_name: point_name},
        batteries={"off": evaluate.NO_BATTERY},
        max_days=max_days,
        availability_forecast_cols=(
            *evaluate.FORECAST_SCENARIOS.values(),
            planning.LOW_COL,
            planning.HIGH_COL,
        ),
    )
    table = cost_dm_table(hourly, pairs=((tau_name, point_name),), max_lag=max_lag)
    table["total_saving_eur"] = -table["mean_loss_difference"] * table["n"]
    return table


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--skip-shap", action="store_true", help="Skip the SHAP step.")
    parser.add_argument(
        "--skip-planning-test", action="store_true", help="Skip the tau* vs point-plan test."
    )
    parser.add_argument("--max-days", type=int, default=None, help="Limit the planning test.")
    args = parser.parse_args(argv)

    forecast_table = pd.read_parquet(config.FORECAST_TABLE_PARQUET)
    tests = [forecast.forecast_dm_table(forecast_table)]
    if config.EVALUATION_HOURLY_PARQUET.exists():
        tests.append(cost_dm_table(pd.read_parquet(config.EVALUATION_HOURLY_PARQUET)))
    else:
        logger.info("No evaluation results; skipping realized-cost tests.")
    if not args.skip_planning_test:
        logger.info("Re-planning with the point forecast and tau* for the planning test...")
        _, renewable_available = evaluate.load_evaluation_inputs()
        tests.append(planning_dm_table(forecast_table, renewable_available, max_days=args.max_days))
    dm = pd.concat(tests, ignore_index=True)
    dm.to_parquet(config.DM_TESTS_PARQUET)

    breakdown = forecast.error_breakdown(forecast_table)
    breakdown.to_parquet(config.ERROR_BREAKDOWN_PARQUET)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:,.4g}".format)
    logger.info("\nDiebold-Mariano tests (mean_loss_difference < 0: model_a better):\n%s",
                dm.to_string(index=False))  # fmt: skip
    for slice_name in ("day_type", "month"):
        view = breakdown[breakdown["slice"] == slice_name].pivot(
            index="group", columns="model", values="mae"
        )
        logger.info("\nMAE by %s (MW):\n%s", slice_name, view)

    if not args.skip_shap:
        logger.info("\nFitting the LightGBM point model for SHAP (about a minute)...")
        matrix = features.build_feature_matrix(
            pd.read_parquet(config.PROCESSED_HOURLY_PARQUET), target_col=config.LOAD_ACTUAL_COL
        )
        model, X_test = forecast.fit_model_for_explanation(matrix)
        importance = forecast.shap_feature_importance(model, X_test)
        importance.to_parquet(config.SHAP_IMPORTANCE_PARQUET)
        logger.info("\nSHAP feature importance (top 15):\n%s", importance.head(15).to_string())
        logger.info("Wrote %s", config.SHAP_IMPORTANCE_PARQUET)
    logger.info("Wrote %s and %s", config.DM_TESTS_PARQUET, config.ERROR_BREAKDOWN_PARQUET)


if __name__ == "__main__":
    main()
