"""
Module 5 — sensitivity analysis (spec section 8).

Question: under what system conditions does battery storage become
economically valuable?

The real 2019 evaluation (evaluate.py) found the battery saving only
~0.04% of realized cost under the baseline system in config.py. That
result is specific to this illustrative system (its generation mix, ramp
limits and prices), and those generation and balancing prices are
illustrative assumptions, not Spanish market data. This module varies one
system parameter at a time around the baseline, holding everything else
fixed, and re-measures

    battery_savings = realized_cost(battery off) - realized_cost(battery on)
    battery_savings_pct = battery_savings / realized_cost(battery off) * 100

split into its planned-cost part (intertemporal arbitrage inside the
day-ahead plan) and its imbalance-cost part. The model keeps its fixed
day-ahead plan (no real-time re-dispatch), so imbalance = actual -
forecast whatever the battery does, and the imbalance part is ~0 by
construction. It is still reported, so that stays visible rather than
assumed.

Nothing here changes optimize.py or evaluate.py semantics. Every variant
is evaluated with evaluate.evaluate_year on the same day sample as the
baseline evaluation (days complete for all four forecast columns, not
just the one being run), so results are comparable with it.

To avoid redundant LP solves, results are cached per (forecast, generation
fleet, battery): the battery-off run is shared by every battery-size
variant, and the baseline is solved once rather than once per dimension.
Balancing prices never enter the LP, so price variants re-price the
cached imbalances instead of re-solving.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging

import pandas as pd

from energy_dispatch import config, evaluate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System parameterization
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SystemParams:
    """Everything a sensitivity variant can change, as one immutable value.
    Variants are built with dataclasses.replace, so the module-level config
    objects are never mutated.
    """

    generation_units: tuple[config.GenerationUnit, ...] = config.GENERATION_UNITS
    battery: config.BatteryParams = config.BATTERY
    price_up: float = config.BALANCING_PRICE_UP_EUR_PER_MWH
    price_down: float = config.BALANCING_PRICE_DOWN_EUR_PER_MWH


BASELINE_SYSTEM = SystemParams()

SENSITIVITY_DIMENSIONS: dict[str, tuple] = {
    "battery_power_mw": config.BATTERY_POWER_SWEEP_MW,
    "battery_energy_mwh": config.BATTERY_ENERGY_SWEEP_MWH,
    "ccgt_ramp_mw_per_hour": config.CCGT_RAMP_SWEEP_MW_PER_HOUR,
    "ccgt_capacity_mw": config.CCGT_CAPACITY_SWEEP_MW,
    "peaker_cost_eur_per_mwh": config.PEAKER_COST_SWEEP_EUR_PER_MWH,
    "balancing_prices_eur_per_mwh": config.BALANCING_PRICE_SWEEP_EUR_PER_MWH,
}


def with_battery(
    system: SystemParams, power_mw: float | None = None, energy_mwh: float | None = None
) -> SystemParams:
    """Change battery power and/or energy capacity. When the energy
    capacity changes, the start-of-day SoC keeps the same *fraction* of
    capacity as before (50% in the baseline); otherwise a 1,000 MWh battery
    would start the day above its own capacity.
    """
    battery = system.battery
    changes: dict[str, float] = {}
    if power_mw is not None:
        changes["power_capacity_mw"] = float(power_mw)
    if energy_mwh is not None:
        soc_fraction = (
            battery.initial_soc_mwh / battery.energy_capacity_mwh
            if battery.energy_capacity_mwh
            else 0.0
        )
        changes["energy_capacity_mwh"] = float(energy_mwh)
        changes["initial_soc_mwh"] = soc_fraction * float(energy_mwh)
    return dataclasses.replace(system, battery=dataclasses.replace(battery, **changes))


def with_unit(system: SystemParams, unit_name: str, **changes) -> SystemParams:
    """Change fields of one generation unit (by name), leaving the others alone."""
    if unit_name not in {unit.name for unit in system.generation_units}:
        raise KeyError(f"no generation unit named {unit_name!r}")
    units = tuple(
        dataclasses.replace(unit, **changes) if unit.name == unit_name else unit
        for unit in system.generation_units
    )
    return dataclasses.replace(system, generation_units=units)


def with_balancing_prices(system: SystemParams, price_up: float, price_down: float) -> SystemParams:
    return dataclasses.replace(system, price_up=float(price_up), price_down=float(price_down))


def build_variant(parameter: str, value, base: SystemParams = BASELINE_SYSTEM) -> SystemParams:
    """The system with one parameter set to `value`, everything else at `base`."""
    if parameter == "battery_power_mw":
        return with_battery(base, power_mw=value)
    if parameter == "battery_energy_mwh":
        return with_battery(base, energy_mwh=value)
    if parameter == "ccgt_ramp_mw_per_hour":
        return with_unit(base, "ccgt", ramp_limit_mw_per_hour=value)
    if parameter == "ccgt_capacity_mw":
        return with_unit(base, "ccgt", capacity_mw=value)
    if parameter == "peaker_cost_eur_per_mwh":
        return with_unit(base, "peaker", marginal_cost_eur_per_mwh=value)
    if parameter == "balancing_prices_eur_per_mwh":
        price_up, price_down = value
        return with_balancing_prices(base, price_up, price_down)
    expected = list(SENSITIVITY_DIMENSIONS)
    raise KeyError(f"unknown sensitivity parameter {parameter!r}; expected one of {expected}")


def describe_system(system: SystemParams) -> dict:
    """Flat description of a system, so every result row is self-describing."""
    units = {unit.name: unit for unit in system.generation_units}
    row = {
        "battery_power_mw": system.battery.power_capacity_mw,
        "battery_energy_mwh": system.battery.energy_capacity_mwh,
        "price_up_eur_per_mwh": system.price_up,
        "price_down_eur_per_mwh": system.price_down,
    }
    if "ccgt" in units:
        row["ccgt_capacity_mw"] = units["ccgt"].capacity_mw
        row["ccgt_ramp_mw_per_hour"] = units["ccgt"].ramp_limit_mw_per_hour
    if "peaker" in units:
        row["peaker_cost_eur_per_mwh"] = units["peaker"].marginal_cost_eur_per_mwh
    return row


# ---------------------------------------------------------------------------
# Costs and savings
# ---------------------------------------------------------------------------


def reprice(hourly: pd.DataFrame, price_up: float, price_down: float) -> pd.DataFrame:
    """Recompute imbalance and realized cost at different balancing prices.
    Planned dispatch and imbalance volumes are unchanged, since prices do
    not enter the day-ahead LP.
    """
    out = hourly.copy()
    out["imbalance_cost_eur"] = evaluate.price_imbalance(out["imbalance"], price_up, price_down)
    out["realized_cost_eur"] = out["planned_cost_eur"] + out["imbalance_cost_eur"]
    return out


def summarize_hourly(hourly: pd.DataFrame) -> dict:
    """Totals for one (variant, forecast, battery) run."""
    if hourly.empty:  # every day infeasible for this variant
        nan = float("nan")
        return {
            "evaluated_days": 0, "evaluated_hours": 0, "total_planned_cost_eur": nan,
            "total_imbalance_cost_eur": nan, "total_realized_cost_eur": nan,
            "peaker_energy_mwh": nan, "battery_discharge_mwh": nan,
        }  # fmt: skip
    return {
        "evaluated_days": int(hourly["local_date"].nunique()) if len(hourly) else 0,
        "evaluated_hours": len(hourly),
        "total_planned_cost_eur": float(hourly["planned_cost_eur"].sum()),
        "total_imbalance_cost_eur": float(hourly["imbalance_cost_eur"].sum()),
        "total_realized_cost_eur": float(hourly["realized_cost_eur"].sum()),
        "peaker_energy_mwh": float(hourly["gen_peaker"].sum())
        if "gen_peaker" in hourly
        else float("nan"),
        "battery_discharge_mwh": float(hourly["discharge"].sum()) if len(hourly) else 0.0,
    }


def battery_savings(off: dict, on: dict) -> dict:
    """Savings from enabling the battery, and their split by cost component.

    battery_savings_eur = realized(off) - realized(on); positive means the
    battery lowers cost. battery_savings_pct uses realized(off) as the
    denominator. planned + imbalance savings add up to the total.
    """
    saving = off["total_realized_cost_eur"] - on["total_realized_cost_eur"]
    denominator = off["total_realized_cost_eur"]
    return {
        "battery_savings_eur": saving,
        "battery_savings_pct": saving / denominator * 100 if denominator else float("nan"),
        "planned_cost_savings_eur": off["total_planned_cost_eur"] - on["total_planned_cost_eur"],
        "imbalance_cost_savings_eur": off["total_imbalance_cost_eur"]
        - on["total_imbalance_cost_eur"],
    }


# ---------------------------------------------------------------------------
# Running variants
# ---------------------------------------------------------------------------


class SensitivityRunner:
    """Evaluates system variants on a fixed day sample, caching LP results
    per (forecast method, generation fleet, battery).
    """

    def __init__(
        self,
        forecast_table: pd.DataFrame,
        renewable_available: pd.Series,
        max_days: int | None = None,
    ):
        self.forecast_table = forecast_table
        self.renewable_available = renewable_available
        self.max_days = max_days
        # Same availability rule as the baseline 8-scenario evaluation: a day
        # counts only if all four forecast columns are complete, even when a
        # sensitivity run uses just one of them.
        self.availability_cols = tuple(evaluate.FORECAST_SCENARIOS.values())
        valid_days, _ = evaluate.select_evaluable_days(
            forecast_table, renewable_available, self.availability_cols
        )
        self.sample_days: list[dt.date] = valid_days[:max_days] if max_days else valid_days
        self._cache: dict[tuple, tuple[pd.DataFrame, set]] = {}
        self.lp_runs = 0  # number of evaluate_year calls actually made

    def dispatch(
        self, forecast_method: str, generation_units: tuple, battery: config.BatteryParams
    ) -> tuple[pd.DataFrame, set]:
        """Hourly results (at baseline prices) and infeasible days for one
        forecast/fleet/battery combination, solved at most once.
        """
        key = (forecast_method, generation_units, battery)
        if key not in self._cache:
            self.lp_runs += 1
            hourly, skipped = evaluate.evaluate_year(
                self.forecast_table,
                self.renewable_available,
                scenarios={forecast_method: evaluate.FORECAST_SCENARIOS[forecast_method]},
                batteries={"run": battery},
                generation_units=generation_units,
                max_days=self.max_days,
                availability_forecast_cols=self.availability_cols,
            )
            not_solved = skipped["reason"].str.startswith("dispatch not solved")
            infeasible = set(skipped.loc[not_solved, "local_date"])
            self._cache[key] = (hourly, infeasible)
        return self._cache[key]

    def evaluate_system(
        self, system: SystemParams, forecast_method: str
    ) -> tuple[dict[str, pd.DataFrame], set]:
        """Battery-off and battery-on hourly results for one variant, priced
        at the variant's balancing prices and restricted to the days both
        could be planned for (so off and on always cover the same hours).
        """
        off, off_infeasible = self.dispatch(
            forecast_method, system.generation_units, evaluate.NO_BATTERY
        )
        on, on_infeasible = self.dispatch(forecast_method, system.generation_units, system.battery)
        infeasible = off_infeasible | on_infeasible
        results = {}
        for name, hourly in (("off", off), ("on", on)):
            if not hourly.empty:
                hourly = hourly[~hourly["local_date"].isin(infeasible)]
            if not hourly.empty:
                hourly = reprice(hourly, system.price_up, system.price_down)
            results[name] = hourly
        return results, infeasible


def run_sensitivity(
    forecast_table: pd.DataFrame,
    renewable_available: pd.Series,
    dimensions: dict[str, tuple] | None = None,
    forecast_methods: tuple[str, ...] = ("lightgbm",),
    base: SystemParams = BASELINE_SYSTEM,
    max_days: int | None = None,
    runner: SensitivityRunner | None = None,
) -> pd.DataFrame:
    """One-factor-at-a-time sensitivity of battery value.

    dimensions maps a parameter name (see SENSITIVITY_DIMENSIONS) to the
    values to try; default is every dimension with its config grid.

    Returns one row per (parameter, value, forecast_method, battery
    enabled/disabled) with the variant's settings, the day/hour counts,
    planned / imbalance / realized cost, peaker energy and battery
    discharge. The battery_savings* columns are filled on the
    battery-enabled row of each pair (NaN on the disabled row).
    """
    dimensions = SENSITIVITY_DIMENSIONS if dimensions is None else dimensions
    runner = runner or SensitivityRunner(forecast_table, renewable_available, max_days=max_days)
    n_sample_days = len(runner.sample_days)

    rows = []
    for parameter, values in dimensions.items():
        for value in values:
            system = build_variant(parameter, value, base)
            is_price_pair = isinstance(value, tuple)
            for method in forecast_methods:
                logger.info("Sensitivity: %s = %s (%s)", parameter, value, method)
                results, infeasible = runner.evaluate_system(system, method)
                summaries = {name: summarize_hourly(h) for name, h in results.items()}
                savings = battery_savings(summaries["off"], summaries["on"])
                for name, enabled in (("off", False), ("on", True)):
                    summary = summaries[name]
                    rows.append(
                        {
                            "parameter": parameter,
                            "value": float(value[0] if is_price_pair else value),
                            "value_label": "/".join(str(v) for v in value)
                            if is_price_pair
                            else str(value),
                            "is_baseline": system == base,
                            "forecast_method": method,
                            "battery_enabled": enabled,
                            **describe_system(system),
                            "infeasible_days": len(infeasible),
                            "comparable_to_baseline_sample": summary["evaluated_days"]
                            == n_sample_days,
                            **summary,
                            **(savings if enabled else dict.fromkeys(savings, float("nan"))),
                        }
                    )
    logger.info("Sensitivity done: %d distinct dispatch runs.", runner.lp_runs)
    return pd.DataFrame(rows)


def battery_value_table(results: pd.DataFrame) -> pd.DataFrame:
    """Compact view of run_sensitivity: one row per (parameter, value,
    forecast), with the battery-off and battery-on costs side by side and
    the savings. Suited to plotting one sensitivity curve per parameter.
    """
    keys = ["parameter", "value", "value_label", "forecast_method"]
    off = results[~results["battery_enabled"]].set_index(keys)
    on = results[results["battery_enabled"]].set_index(keys)
    table = pd.DataFrame(
        {
            "is_baseline": on["is_baseline"],
            "evaluated_days": on["evaluated_days"],
            "infeasible_days": on["infeasible_days"],
            "realized_cost_off_eur": off["total_realized_cost_eur"],
            "realized_cost_on_eur": on["total_realized_cost_eur"],
            "battery_savings_eur": on["battery_savings_eur"],
            "battery_savings_pct": on["battery_savings_pct"],
            "planned_cost_savings_eur": on["planned_cost_savings_eur"],
            "imbalance_cost_savings_eur": on["imbalance_cost_savings_eur"],
            "peaker_energy_off_mwh": off["peaker_energy_mwh"],
            "peaker_energy_on_mwh": on["peaker_energy_mwh"],
        }
    )
    return table.reset_index()


# ---------------------------------------------------------------------------
# Not yet implemented (later spec section 8 items)
# ---------------------------------------------------------------------------


def sweep_renewable_share(
    df: pd.DataFrame,
    forecast_table: pd.DataFrame,
    scales: tuple = config.RENEWABLE_SCALE_SWEEP,
) -> pd.DataFrame:
    """Scale observed solar+wind output by each factor in `scales` and
    re-run compare_scenarios to show the value of storage in a
    higher-renewable grid.
    """
    raise NotImplementedError


def uncertainty_aware_plan(forecast_table: pd.DataFrame) -> pd.DataFrame:
    """Plan dispatch against the q90 forecast (or point forecast + a
    reserve sized from the [q10, q90] interval width) instead of the
    point forecast, and compare realized cost against the point-forecast
    plan. Demonstrates the practical value of the prediction intervals
    from forecast.py's quantile models.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Real-data entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-factor-at-a-time sensitivity of battery value around the "
            "baseline system, on the same days as the baseline evaluation."
        )
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=list(SENSITIVITY_DIMENSIONS),
        default=list(SENSITIVITY_DIMENSIONS),
        help="Which parameters to vary (default: all).",
    )
    parser.add_argument(
        "--forecast-methods",
        nargs="+",
        choices=list(evaluate.FORECAST_SCENARIOS),
        default=["lightgbm"],
        help="Forecast(s) to plan on (default: lightgbm).",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Only use the first N evaluable days (quick smoke run).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args(argv)

    forecast_table, renewable_available = evaluate.load_evaluation_inputs()
    dimensions = {name: SENSITIVITY_DIMENSIONS[name] for name in args.dimensions}
    results = run_sensitivity(
        forecast_table,
        renewable_available,
        dimensions=dimensions,
        forecast_methods=tuple(args.forecast_methods),
        max_days=args.max_days,
    )
    results.to_parquet(config.SENSITIVITY_RESULTS_PARQUET)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:,.2f}".format)
    table = battery_value_table(results)
    logger.info("\nBattery value by system variant:\n%s", table.to_string(index=False))
    not_comparable = results[~results["comparable_to_baseline_sample"]]
    if not not_comparable.empty:
        logger.info(
            "\nNOTE: some variants could not be planned on every day (e.g. "
            "demand above total capacity). Their totals cover fewer days than "
            "the baseline sample, so compare their battery_savings_pct, not "
            "their absolute costs, with the baseline:\n%s",
            not_comparable[["parameter", "value_label", "battery_enabled", "infeasible_days"]]
            .drop_duplicates()
            .to_string(index=False),
        )
    logger.info("Wrote %s", config.SENSITIVITY_RESULTS_PARQUET)


if __name__ == "__main__":
    main()
