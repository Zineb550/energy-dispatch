"""
Module 4 (cont.) — realized-cost evaluation (spec section 7.3-7.4).

Answers the project's central question: when a forecast is used to make a
day-ahead dispatch decision, how much does forecast quality affect the
cost of operating the simulated system once actual demand is revealed?

Operational timeline, per local (Europe/Madrid) day D+1:

    10:00 on day D   forecast for every hour of D+1 is fixed
                     (forecast.py's rolling-origin backtest)
          |
          v          plan dispatch with optimize.plan_day_ahead_dispatch,
                     using ONLY the forecast + renewable availability
          |
          v          D+1 happens; actual demand becomes known
          |
          v          imbalance = actual - planned supply, priced at the
                     balancing market -> realized cost

Actual demand enters only in the second half (evaluate_day's "reveal"
step), after the plan exists. The plan is never revised afterwards: there
is no real-time re-dispatch, so every MWh of forecast error is settled on
the balancing market.

Definitions (hourly; MW over one hour = MWh):

    planned_supply = sum(generation) + renewable_used + discharge - charge
        the exact left-hand side of the LP's demand-balance row, so it
        equals forecast_load by construction (charging draws power from
        the grid, hence the minus sign)
    imbalance      = actual_load - planned_supply
        > 0: under-supply, upward energy bought at the up-price
        < 0: over-supply, surplus sold at the down-price
    imbalance_cost = price_up * max(imbalance, 0)
                     - price_down * max(-imbalance, 0)
        surplus is a credit (config.py: "sold at the down-price") because
        its generation cost is already inside planned_cost
    planned_cost   = sum over units of marginal_cost * generation (the LP
                     objective, i.e. the cost of the plan on the forecast)
    realized_cost  = planned_cost + imbalance_cost

Consequence of the fixed plan: planned_supply == forecast_load, so
imbalance == actual_load - forecast_load whether or not the battery is
available. The battery changes planned cost only (moving energy toward
cheaper hours of the *forecast* profile); it cannot absorb forecast
errors, because it does not react to actual demand.

Assumptions carried over from optimize.py: renewable availability is the
observed solar + wind output (perfect renewable foresight), so this
evaluation isolates demand forecast error, not renewable forecast error.
"perfect_foresight" plans on the actual load itself — an idealized
reference / lower bound, not an achievable forecasting method.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd

from energy_dispatch import config, optimize

logger = logging.getLogger(__name__)

# Planning-time forecast inputs, mapped to their column in the forecast
# table written by forecast.py (config.FORECAST_TABLE_PARQUET).
FORECAST_SCENARIOS: dict[str, str] = {
    "perfect_foresight": "y_true",
    "seasonal_naive": "y_naive",
    "tso": "y_tso",
    "lightgbm": "y_lgbm",
}
ACTUAL_COL = "y_true"

# Battery "off": the same LP with a zero-capacity battery, rather than a
# second copy of the optimization logic. Defined here, not in config.py,
# since it is an evaluation scenario, not a system assumption.
NO_BATTERY = config.BatteryParams(
    power_capacity_mw=0.0,
    energy_capacity_mwh=0.0,
    round_trip_efficiency=config.BATTERY.round_trip_efficiency,
    initial_soc_mwh=0.0,
)
BATTERY_SCENARIOS: dict[str, config.BatteryParams] = {"off": NO_BATTERY, "on": config.BATTERY}


# ---------------------------------------------------------------------------
# Hourly building blocks
# ---------------------------------------------------------------------------


def planned_net_supply(
    dispatch: pd.DataFrame, generation_units: tuple = config.GENERATION_UNITS
) -> pd.Series:
    """Planned net supply to the load, per hour, from an optimize.py
    dispatch table: generation + renewable_used + discharge - charge.
    """
    gen_cols = [f"gen_{unit.name}" for unit in generation_units]
    supply = (
        dispatch[gen_cols].sum(axis=1)
        + dispatch["renewable_used"]
        + dispatch["discharge"]
        - dispatch["charge"]
    )
    return supply.rename("planned_supply")


def compute_imbalance(actual_load: pd.Series, planned_supply: pd.Series) -> pd.Series:
    """imbalance_t = actual_load_t - planned_supply_t (positive = under-supply)."""
    if not actual_load.index.equals(planned_supply.index):
        raise ValueError("actual_load and planned_supply must share the same index")
    return (actual_load - planned_supply).rename("imbalance")


def price_imbalance(
    imbalance: pd.Series,
    price_up: float = config.BALANCING_PRICE_UP_EUR_PER_MWH,
    price_down: float = config.BALANCING_PRICE_DOWN_EUR_PER_MWH,
) -> pd.Series:
    """Hourly imbalance cost in EUR: positive imbalance (under-supply) is
    bought at price_up; negative imbalance (over-supply) is sold at
    price_down, i.e. a credit (negative cost).
    """
    upward = imbalance.clip(lower=0)
    downward = (-imbalance).clip(lower=0)
    return (price_up * upward - price_down * downward).rename("imbalance_cost_eur")


def evaluate_day(
    forecast_load: pd.Series,
    actual_load: pd.Series,
    renewable_available: pd.Series,
    battery: config.BatteryParams = config.BATTERY,
    generation_units: tuple = config.GENERATION_UNITS,
    price_up: float = config.BALANCING_PRICE_UP_EUR_PER_MWH,
    price_down: float = config.BALANCING_PRICE_DOWN_EUR_PER_MWH,
) -> pd.DataFrame:
    """Plan one day's dispatch against forecast_load, then settle it
    against actual_load. Returns one row per hour with the planned
    dispatch, imbalance, and planned / imbalance / realized cost.

    Raises RuntimeError (from optimize.plan_day_ahead_dispatch) if the
    day-ahead LP is not solvable to optimality.
    """
    if not (
        forecast_load.index.equals(actual_load.index)
        and forecast_load.index.equals(renewable_available.index)
    ):
        raise ValueError("forecast_load, actual_load and renewable_available must share an index")

    # --- Step 1: plan. Only the forecast and renewables are visible here. ---
    dispatch = optimize.plan_day_ahead_dispatch(
        forecast_load, renewable_available, generation_units=generation_units, battery=battery
    )
    planned_supply = planned_net_supply(dispatch, generation_units)
    if not optimize.validate_demand_balance(dispatch, forecast_load, generation_units):
        raise RuntimeError("planned supply does not match the forecast the plan was built for")

    # --- Step 2: actual demand is revealed; the plan is not revised. ---
    imbalance = compute_imbalance(actual_load, planned_supply)
    imbalance_cost = price_imbalance(imbalance, price_up=price_up, price_down=price_down)

    result = dispatch.drop(columns=["forecast_load", "planned_cost_eur"]).copy()
    result.insert(0, "forecast_load", forecast_load)
    result.insert(1, "actual_load", actual_load)
    result["planned_supply"] = planned_supply
    result["imbalance"] = imbalance
    result["planned_cost_eur"] = dispatch["planned_cost_eur"]
    result["imbalance_cost_eur"] = imbalance_cost
    result["realized_cost_eur"] = result["planned_cost_eur"] + result["imbalance_cost_eur"]
    result.index.name = forecast_load.index.name
    return result


# ---------------------------------------------------------------------------
# Choosing which days can be evaluated
# ---------------------------------------------------------------------------


def expected_local_hours(local_date: dt.date, tz: str = config.LOCAL_TZ) -> int:
    """Number of hours in a local calendar day: 24, or 23/25 on DST changes."""
    start = pd.Timestamp(local_date, tz=tz)
    end = pd.Timestamp(local_date + dt.timedelta(days=1), tz=tz)
    return len(pd.date_range(start, end, freq="h", inclusive="left"))


def _local_dates(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(index.tz_convert(config.LOCAL_TZ).date, name="local_date")


def select_evaluable_days(
    forecast_table: pd.DataFrame,
    renewable_available: pd.Series,
    forecast_cols: tuple[str, ...] = tuple(FORECAST_SCENARIOS.values()),
    start: str | None = None,
    end: str | None = None,
) -> tuple[list[dt.date], pd.DataFrame]:
    """Return (evaluable local dates, skipped days with a reason).

    A local day is evaluable only if it has every one of its local hours
    (23/24/25 on DST days) and, in every hour, the actual load, every
    forecast column, and a non-negative renewable availability. The same
    day set is then used for every scenario so comparisons are
    like-for-like. start/end (local dates) default to the table's own
    first and last local date; days in that range with no rows at all
    are reported too.
    """
    cols = list(dict.fromkeys([ACTUAL_COL, *forecast_cols]))
    missing_cols = [c for c in cols if c not in forecast_table.columns]
    if missing_cols:
        raise KeyError(f"forecast table is missing columns: {missing_cols}")

    table = forecast_table[cols].copy()
    table["renewable_available"] = renewable_available.reindex(table.index)
    local_dates = _local_dates(table.index)

    first = pd.Timestamp(start).date() if start else min(local_dates)
    last = pd.Timestamp(end).date() if end else max(local_dates)
    grouped = {date: rows for date, rows in table.groupby(local_dates)}

    valid, skipped = [], []
    for date in pd.date_range(first, last, freq="D").date:
        rows = grouped.get(date)
        if rows is None:
            skipped.append((date, "no rows in forecast table"))
            continue
        expected = expected_local_hours(date)
        if len(rows) != expected:
            skipped.append((date, f"incomplete horizon: {len(rows)} of {expected} local hours"))
            continue
        nan_cols = [c for c in rows.columns if rows[c].isna().any()]
        if nan_cols:
            skipped.append((date, f"missing values in: {', '.join(nan_cols)}"))
            continue
        if (rows["renewable_available"] < 0).any():
            skipped.append((date, "negative renewable availability"))
            continue
        valid.append(date)

    skipped_df = pd.DataFrame(skipped, columns=["local_date", "reason"])
    return valid, skipped_df


# ---------------------------------------------------------------------------
# Year-level evaluation
# ---------------------------------------------------------------------------


def evaluate_year(
    forecast_table: pd.DataFrame,
    renewable_available: pd.Series,
    scenarios: dict[str, str] = FORECAST_SCENARIOS,
    batteries: dict[str, config.BatteryParams] = BATTERY_SCENARIOS,
    generation_units: tuple = config.GENERATION_UNITS,
    price_up: float = config.BALANCING_PRICE_UP_EUR_PER_MWH,
    price_down: float = config.BALANCING_PRICE_DOWN_EUR_PER_MWH,
    start: str | None = None,
    end: str | None = None,
    max_days: int | None = None,
    availability_forecast_cols: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every (forecast scenario x battery scenario) pair, one
    day-ahead LP per local day.

    Returns (hourly_results, skipped_days). hourly_results is long-format:
    one row per (scenario, hour), with columns forecast_method, battery,
    local_date, and everything evaluate_day returns. A day whose LP is not
    solvable for any scenario is skipped for all scenarios (and reported)
    so every scenario covers exactly the same hours.

    availability_forecast_cols overrides which columns select_evaluable_days
    checks for completeness. Default (None) checks tuple(scenarios.values())
    as before. Pass the full forecast-scenario column set explicitly when
    evaluating a *subset* of scenarios (e.g. sensitivity.py running one
    forecast at a time) so the day-availability check doesn't loosen just
    because fewer columns are in play — otherwise two runs over different
    scenario subsets could silently end up evaluating different samples.
    """
    valid_days, skipped_df = select_evaluable_days(
        forecast_table,
        renewable_available,
        availability_forecast_cols
        if availability_forecast_cols is not None
        else tuple(scenarios.values()),
        start=start,
        end=end,
    )
    if max_days is not None:
        valid_days = valid_days[:max_days]

    table = forecast_table.copy()
    table["renewable_available"] = renewable_available.reindex(table.index)
    local_dates = _local_dates(table.index)
    grouped = {date: rows.sort_index() for date, rows in table.groupby(local_dates)}

    results, extra_skips = [], []
    for i, date in enumerate(valid_days, start=1):
        rows = grouped[date]
        actual = rows[ACTUAL_COL]
        renewable = rows["renewable_available"]
        day_results = []
        try:
            for forecast_name, col in scenarios.items():
                for battery_name, battery in batteries.items():
                    hourly = evaluate_day(
                        rows[col],
                        actual,
                        renewable,
                        battery=battery,
                        generation_units=generation_units,
                        price_up=price_up,
                        price_down=price_down,
                    )
                    hourly.insert(0, "forecast_method", forecast_name)
                    hourly.insert(1, "battery", battery_name)
                    hourly.insert(2, "local_date", date)
                    day_results.append(hourly)
        except RuntimeError as exc:
            reason = f"dispatch not solved ({forecast_name}, battery {battery_name}): {exc}"
            extra_skips.append((date, reason))
            continue
        results.extend(day_results)
        if i % 30 == 0:
            logger.info("Evaluated %d / %d days", i, len(valid_days))

    if extra_skips:
        skipped_df = pd.concat(
            [skipped_df, pd.DataFrame(extra_skips, columns=["local_date", "reason"])],
            ignore_index=True,
        ).sort_values("local_date", ignore_index=True)

    hourly_results = pd.concat(results) if results else pd.DataFrame()
    return hourly_results, skipped_df


def aggregate_results(hourly_results: pd.DataFrame) -> pd.DataFrame:
    """Annual summary per (forecast_method, battery) scenario.

    Imbalance sign convention as elsewhere in this module: positive =
    actual above plan (upward energy bought). forecast_bias_mw uses the
    same sign (actual - forecast), so positive bias = under-forecasting,
    which is the expensive direction at the configured prices.
    """
    df = hourly_results.assign(
        abs_imbalance=hourly_results["imbalance"].abs(),
        upward_imbalance=hourly_results["imbalance"].clip(lower=0),
        downward_imbalance=(-hourly_results["imbalance"]).clip(lower=0),
        forecast_error=hourly_results["actual_load"] - hourly_results["forecast_load"],
    )
    df["abs_forecast_error"] = df["forecast_error"].abs()

    grouped = df.groupby(["forecast_method", "battery"], sort=False)
    summary = pd.DataFrame(
        {
            "n_days": grouped["local_date"].nunique(),
            "n_hours": grouped.size(),
            "planned_cost_eur": grouped["planned_cost_eur"].sum(),
            "imbalance_cost_eur": grouped["imbalance_cost_eur"].sum(),
            "realized_cost_eur": grouped["realized_cost_eur"].sum(),
            "mean_hourly_realized_cost_eur": grouped["realized_cost_eur"].mean(),
            "total_abs_imbalance_mwh": grouped["abs_imbalance"].sum(),
            "mean_abs_imbalance_mw": grouped["abs_imbalance"].mean(),
            "upward_imbalance_mwh": grouped["upward_imbalance"].sum(),
            "downward_imbalance_mwh": grouped["downward_imbalance"].sum(),
            "forecast_mae_mw": grouped["abs_forecast_error"].mean(),
            "forecast_bias_mw": grouped["forecast_error"].mean(),
        }
    )
    return summary


def compare_scenarios(summary: pd.DataFrame) -> pd.DataFrame:
    """Pairwise realized-cost comparisons from an aggregate_results table.

    saving_eur = reference realized cost - scenario realized cost, so a
    positive saving means the scenario is cheaper than its reference.
    saving_pct = saving_eur / reference realized cost * 100 (the
    denominator is always the reference's realized cost).

    Rows:
      - vs_perfect_foresight / vs_seasonal_naive / vs_tso: each forecast
        against that reference, at the same battery setting
      - battery_value: battery "on" against "off", for each forecast
        (compare these rows across forecasts for the forecasting x
        storage interaction)
    """
    realized = summary["realized_cost_eur"]
    forecasts = list(dict.fromkeys(summary.index.get_level_values("forecast_method")))
    batteries = list(dict.fromkeys(summary.index.get_level_values("battery")))

    rows = []

    def _add(comparison: str, scenario: tuple[str, str], reference: tuple[str, str]) -> None:
        if scenario not in realized.index or reference not in realized.index:
            return
        ref_cost, cost = realized[reference], realized[scenario]
        saving = ref_cost - cost
        rows.append(
            {
                "comparison": comparison,
                "scenario": f"{scenario[0]} / battery {scenario[1]}",
                "reference": f"{reference[0]} / battery {reference[1]}",
                "scenario_realized_cost_eur": cost,
                "reference_realized_cost_eur": ref_cost,
                "saving_eur": saving,
                "saving_pct": saving / ref_cost * 100 if ref_cost else float("nan"),
            }
        )

    for reference in ("perfect_foresight", "seasonal_naive", "tso"):
        for battery in batteries:
            for forecast in forecasts:
                if forecast != reference:
                    _add(f"vs_{reference}", (forecast, battery), (reference, battery))
    for forecast in forecasts:
        _add("battery_value", (forecast, "on"), (forecast, "off"))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Real-data entry point
# ---------------------------------------------------------------------------


def load_evaluation_inputs(
    forecast_path=config.FORECAST_TABLE_PARQUET,
    processed_path=config.PROCESSED_HOURLY_PARQUET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the forecast table written by forecast.py and build the hourly
    renewable availability (solar + wind) from the processed dataset.
    Each file is read once; nothing is retrained.
    """
    forecast_table = pd.read_parquet(forecast_path)
    generation = pd.read_parquet(
        processed_path, columns=[config.SOLAR_GEN_COL, config.WIND_GEN_COL]
    )
    renewable_available = (
        generation[config.SOLAR_GEN_COL] + generation[config.WIND_GEN_COL]
    ).rename("renewable_available")
    return forecast_table, renewable_available


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan day-ahead dispatch against each forecast in the test-period "
            "forecast table, settle it against actual demand, and write the "
            "hourly results and scenario summary to data/processed/."
        )
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Only evaluate the first N evaluable days (quick smoke run).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args(argv)

    forecast_table, renewable_available = load_evaluation_inputs()
    logger.info(
        "Evaluating %d scenarios (forecast x battery), one LP per day and scenario...",
        len(FORECAST_SCENARIOS) * len(BATTERY_SCENARIOS),
    )
    hourly_results, skipped = evaluate_year(
        forecast_table, renewable_available, max_days=args.max_days
    )
    if hourly_results.empty:
        raise RuntimeError("no day could be evaluated; see the skipped-day reasons above")

    summary = aggregate_results(hourly_results)
    comparisons = compare_scenarios(summary)

    hourly_results.to_parquet(config.EVALUATION_HOURLY_PARQUET)
    summary.to_parquet(config.SCENARIO_RESULTS_PARQUET)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:,.2f}".format)
    if not skipped.empty:
        logger.info("\nSkipped %d day(s):\n%s", len(skipped), skipped.to_string(index=False))
    logger.info("\nScenario summary:\n%s", summary)
    logger.info(
        "\nComparisons (saving = reference - scenario):\n%s", comparisons.to_string(index=False)
    )
    logger.info("Wrote %s", config.EVALUATION_HOURLY_PARQUET)
    logger.info("Wrote %s", config.SCENARIO_RESULTS_PARQUET)


if __name__ == "__main__":
    main()
