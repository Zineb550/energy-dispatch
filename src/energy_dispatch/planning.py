"""
Uncertainty-aware day-ahead planning (spec section 8, "value of the
prediction intervals").

Question: can the calibrated forecast intervals lower realized cost, not
just describe uncertainty?

The balancing market is asymmetric. With the plan fixed, one extra MWh of
demand that was not planned for is bought at price_up instead of being
generated at the marginal unit's cost; one MWh planned but not needed was
generated at marginal cost and sold back at price_down. So, per MWh:

    cost of under-planning  c_u = price_up - marginal_cost
    cost of over-planning   c_o = marginal_cost - price_down

This is the newsvendor problem: the cost-minimizing plan is not the median
forecast but the quantile

    tau* = c_u / (c_u + c_o)

of the demand distribution. With the illustrative prices (200 / 20 EUR/MWh)
and CCGT as the marginal unit (60 EUR/MWh), tau* = 140 / 180 ~ 0.78: plan
roughly at the 78th percentile. When the peaker (120 EUR/MWh) is
marginal, tau* ~ 0.44 instead, so tau* is a rule of thumb for the typical
hour, not a law.

This module plans each day on a target quantile of the LightGBM forecast
and evaluates it with the unchanged evaluate.evaluate_year machinery. The
target quantile's value is interpolated from the three forecasts
forecast.py already produces (point, conformal-calibrated q10 and q90),
assuming the forecast error is Gaussian-shaped between them, with
separate widths above and below the point forecast:

    plan_tau = point + z(tau) / z(0.90) * (q90 - point)    for tau >= 0.5
    plan_tau = point + z(tau) / z(0.10) * (q10 - point)    for tau <  0.5

so tau = 0.5 is the point forecast and tau = 0.9 is exactly q90. All three
inputs are known at the 10:00 day-ahead cutoff, so every plan uses only
planning-time information.

Reading the results honestly: tau* is chosen from the prices alone, before
looking at any outcome, so its row is a genuine out-of-sample test. The
full sweep over tau is evaluated on the 2019 test year, so picking
whichever tau happens to score best would be selecting on the test set.
The sweep shows the shape of the trade-off; the pre-registered policy is
tau*.
"""

from __future__ import annotations

import argparse
import logging
from statistics import NormalDist

import pandas as pd

from energy_dispatch import config, evaluate

logger = logging.getLogger(__name__)

POINT_COL = "y_lgbm"
LOW_COL = f"q{int(round(config.QUANTILE_LOW * 100))}"  # "q10"
HIGH_COL = f"q{int(round(config.QUANTILE_HIGH * 100))}"  # "q90"

# Quantiles swept in addition to the theoretical tau* (see newsvendor_quantile).
PLAN_QUANTILES: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Reference scenarios evaluated on the same days, from evaluate.FORECAST_SCENARIOS.
REFERENCE_SCENARIOS: tuple[str, ...] = ("perfect_foresight", "tso")


def newsvendor_quantile(price_up: float, price_down: float, marginal_cost: float) -> float:
    """Cost-minimizing plan quantile tau* = c_u / (c_u + c_o) for a fixed plan
    settled at asymmetric balancing prices (see module docstring)."""
    under = price_up - marginal_cost
    over = marginal_cost - price_down
    if under <= 0 or over <= 0:
        raise ValueError(
            "the newsvendor quantile needs price_down < marginal_cost < price_up; "
            f"got {price_down=}, {marginal_cost=}, {price_up=}"
        )
    return under / (under + over)


def theoretical_plan_quantile(
    generation_units: tuple = config.GENERATION_UNITS,
    price_up: float = config.BALANCING_PRICE_UP_EUR_PER_MWH,
    price_down: float = config.BALANCING_PRICE_DOWN_EUR_PER_MWH,
    marginal_unit: str = "ccgt",
) -> float:
    """tau* for the unit that is usually marginal (CCGT in the baseline fleet)."""
    units = {unit.name: unit for unit in generation_units}
    return newsvendor_quantile(price_up, price_down, units[marginal_unit].marginal_cost_eur_per_mwh)


def quantile_plan(
    forecast_table: pd.DataFrame,
    target_quantile: float,
    point_col: str = POINT_COL,
    low_col: str = LOW_COL,
    high_col: str = HIGH_COL,
) -> pd.Series:
    """Planning load at `target_quantile`, interpolated from the point
    forecast and the q10/q90 interval (Gaussian-shaped scaling, separate
    upper and lower widths). Uses only forecast columns, never y_true.
    """
    if not 0 < target_quantile < 1:
        raise ValueError(f"target_quantile must be in (0, 1), got {target_quantile}")
    z = NormalDist().inv_cdf
    point = forecast_table[point_col]
    if target_quantile >= 0.5:
        scale = z(target_quantile) / z(config.QUANTILE_HIGH)
        plan = point + scale * (forecast_table[high_col] - point)
    else:
        scale = z(target_quantile) / z(config.QUANTILE_LOW)
        plan = point + scale * (forecast_table[low_col] - point)
    return plan.rename(plan_name(target_quantile))


def plan_name(target_quantile: float) -> str:
    return f"lightgbm_q{target_quantile * 100:.1f}".replace(".0", "")


def uncertainty_aware_plan(
    forecast_table: pd.DataFrame,
    renewable_available: pd.Series,
    quantiles: tuple[float, ...] | None = None,
    battery: config.BatteryParams = evaluate.NO_BATTERY,
    generation_units: tuple = config.GENERATION_UNITS,
    price_up: float = config.BALANCING_PRICE_UP_EUR_PER_MWH,
    price_down: float = config.BALANCING_PRICE_DOWN_EUR_PER_MWH,
    max_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate day-ahead plans built on quantiles of the LightGBM forecast.

    quantiles defaults to PLAN_QUANTILES plus the theoretical tau*. Perfect
    foresight and TSO are evaluated on the same days as references. The
    battery is off by default, matching the headline forecast comparison.

    Returns (summary, skipped_days). summary has one row per plan with
    evaluate.aggregate_results' columns plus target_quantile,
    mean_uplift_mw (plan - point forecast), is_theoretical_optimum, and the
    realized-cost saving against the point-forecast plan (tau = 0.5).
    """
    tau_star = theoretical_plan_quantile(generation_units, price_up, price_down)
    if quantiles is None:
        quantiles = tuple(sorted(set(PLAN_QUANTILES) | {round(tau_star, 3)}))
    if 0.5 not in quantiles:
        quantiles = tuple(sorted(set(quantiles) | {0.5}))  # the point-forecast plan

    table = forecast_table.copy()
    scenarios = {}
    for tau in quantiles:
        column = plan_name(tau)
        table[column] = quantile_plan(forecast_table, tau)
        scenarios[column] = column
    for name in REFERENCE_SCENARIOS:
        scenarios[name] = evaluate.FORECAST_SCENARIOS[name]

    # Same day sample as the baseline evaluation, plus the interval columns.
    availability = (*evaluate.FORECAST_SCENARIOS.values(), LOW_COL, HIGH_COL)
    hourly, skipped = evaluate.evaluate_year(
        table,
        renewable_available,
        scenarios=scenarios,
        batteries={"off" if battery == evaluate.NO_BATTERY else "on": battery},
        generation_units=generation_units,
        price_up=price_up,
        price_down=price_down,
        max_days=max_days,
        availability_forecast_cols=availability,
    )
    if hourly.empty:
        return pd.DataFrame(), skipped

    summary = evaluate.aggregate_results(hourly).reset_index()
    by_plan = {plan_name(tau): tau for tau in quantiles}
    summary["target_quantile"] = summary["forecast_method"].map(by_plan)
    summary["is_theoretical_optimum"] = summary["target_quantile"].sub(tau_star).abs() < 5e-4
    uplift = {
        plan_name(tau): (table[plan_name(tau)] - table[POINT_COL])
        .reindex(hourly.index.unique())
        .mean()
        for tau in quantiles
    }
    summary["mean_uplift_mw"] = summary["forecast_method"].map(uplift)

    point_cost = summary.loc[
        summary["forecast_method"] == plan_name(0.5), "realized_cost_eur"
    ].iloc[0]
    summary["saving_vs_point_plan_eur"] = point_cost - summary["realized_cost_eur"]
    summary["saving_vs_point_plan_pct"] = summary["saving_vs_point_plan_eur"] / point_cost * 100
    summary.attrs["theoretical_quantile"] = tau_star
    return summary, skipped


# ---------------------------------------------------------------------------
# Real-data entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan day-ahead dispatch on quantiles of the calibrated LightGBM forecast "
            "and compare realized cost with the point-forecast plan."
        )
    )
    parser.add_argument(
        "--battery",
        choices=["off", "on"],
        default="off",
        help="Battery available to the planner (default: off, as in the headline comparison).",
    )
    parser.add_argument(
        "--max-days", type=int, default=None, help="Only use the first N evaluable days."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args(argv)

    forecast_table, renewable_available = evaluate.load_evaluation_inputs()
    missing = [c for c in (POINT_COL, LOW_COL, HIGH_COL) if c not in forecast_table.columns]
    if missing:
        raise KeyError(
            f"forecast table has no {missing}; rerun `python -m energy_dispatch.forecast` "
            "without --skip-quantiles"
        )
    battery = config.BATTERY if args.battery == "on" else evaluate.NO_BATTERY
    summary, skipped = uncertainty_aware_plan(
        forecast_table, renewable_available, battery=battery, max_days=args.max_days
    )
    summary.to_parquet(config.PLANNING_RESULTS_PARQUET)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:,.2f}".format)
    columns = [
        "forecast_method", "target_quantile", "is_theoretical_optimum", "mean_uplift_mw",
        "planned_cost_eur", "imbalance_cost_eur", "realized_cost_eur",
        "saving_vs_point_plan_eur", "saving_vs_point_plan_pct",
        "upward_imbalance_mwh", "downward_imbalance_mwh", "n_days",
    ]  # fmt: skip
    if not skipped.empty:
        logger.info("\nSkipped %d day(s):\n%s", len(skipped), skipped.to_string(index=False))
    logger.info(
        "\nTheoretical plan quantile tau* = %.3f (CCGT marginal, prices %.0f/%.0f EUR/MWh)",
        summary.attrs["theoretical_quantile"],
        config.BALANCING_PRICE_UP_EUR_PER_MWH,
        config.BALANCING_PRICE_DOWN_EUR_PER_MWH,
    )
    logger.info(
        "\nRealized cost by planning quantile (battery %s):\n%s",
        args.battery,
        summary[columns].to_string(index=False),
    )
    logger.info("Wrote %s", config.PLANNING_RESULTS_PARQUET)


if __name__ == "__main__":
    main()
