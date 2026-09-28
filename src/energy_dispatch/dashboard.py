"""
Data layer for the Streamlit dashboard (app/streamlit_app.py).

The app is a presentation layer only. Everything it shows comes from the
result files the pipeline already writes (forecast table, forecast
metrics, hourly evaluation results, scenario summary, sensitivity
results). The only live computation is the "what-if" page, which re-plans
a single day through evaluate.evaluate_day with a system built by
sensitivity.build_variant, so no forecasting, optimization, evaluation or
sensitivity logic is duplicated here.

This module has no Streamlit import, so it can be tested directly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

import pandas as pd

from energy_dispatch import config, evaluate, forecast, sensitivity

# Display names, in the order used everywhere in the app.
FORECAST_LABELS: dict[str, str] = {
    "perfect_foresight": "Perfect foresight (reference)",
    "seasonal_naive": "Seasonal naive",
    "tso": "TSO (ENTSO-E)",
    "lightgbm": "LightGBM",
}
FORECAST_COLUMN_LABELS: dict[str, str] = {
    "y_naive": "Seasonal naive",
    "y_tso": "TSO (ENTSO-E)",
    "y_lgbm": "LightGBM",
}
SENSITIVITY_LABELS: dict[str, str] = {
    "ccgt_ramp_mw_per_hour": "CCGT ramp limit (MW/h)",
    "ccgt_capacity_mw": "CCGT capacity (MW)",
    "peaker_cost_eur_per_mwh": "Peaker marginal cost (€/MWh)",
    "battery_power_mw": "Battery power (MW)",
    "battery_energy_mwh": "Battery energy (MWh)",
    "balancing_prices_eur_per_mwh": "Balancing prices up/down (€/MWh)",
}

RESULT_FILES: dict[str, tuple[Path, str]] = {
    "forecast_table": (config.FORECAST_TABLE_PARQUET, "python -m energy_dispatch.forecast"),
    "forecast_metrics": (config.METRICS_TABLE_PARQUET, "python -m energy_dispatch.forecast"),
    "evaluation_hourly": (config.EVALUATION_HOURLY_PARQUET, "python -m energy_dispatch.evaluate"),
    "scenario_summary": (config.SCENARIO_RESULTS_PARQUET, "python -m energy_dispatch.evaluate"),
    "sensitivity": (config.SENSITIVITY_RESULTS_PARQUET, "python -m energy_dispatch.sensitivity"),
    "planning": (config.PLANNING_RESULTS_PARQUET, "python -m energy_dispatch.planning"),
    "dm_tests": (config.DM_TESTS_PARQUET, "python -m energy_dispatch.diagnostics"),
    "error_breakdown": (config.ERROR_BREAKDOWN_PARQUET, "python -m energy_dispatch.diagnostics"),
    "shap_importance": (config.SHAP_IMPORTANCE_PARQUET, "python -m energy_dispatch.diagnostics"),
}
# Extensions whose page section is simply hidden when the file is absent.
OPTIONAL_RESULTS: frozenset[str] = frozenset(
    {"planning", "dm_tests", "error_breakdown", "shap_importance"}
)


# Copies of the result files committed with the app, so a hosted deployment
# (which has no data/processed/) can show the real 2019 results. Written by
# export_demo_results(); data/processed/ always takes precedence.
DEMO_RESULTS_DIR: Path = config.ROOT_DIR / "app" / "demo_results"


def resolve_result_path(name: str) -> Path | None:
    """Where one of RESULT_FILES can be read from: data/processed/ first,
    then the bundled copy in DEMO_RESULTS_DIR; None if neither exists."""
    path = Path(RESULT_FILES[name][0])
    if path.exists():
        return path
    bundled = Path(DEMO_RESULTS_DIR) / path.name
    return bundled if bundled.exists() else None


def is_bundled(name: str) -> bool:
    path = resolve_result_path(name)
    return path is not None and path.parent == Path(DEMO_RESULTS_DIR)


def load_result(name: str) -> pd.DataFrame | None:
    """Read one of RESULT_FILES, or return None if it has not been generated."""
    path = resolve_result_path(name)
    return pd.read_parquet(path) if path is not None else None


def missing_file_message(name: str) -> str:
    path, command = RESULT_FILES[name]
    return f"`{path.name}` not found in `data/processed/`. Generate it with `{command}`."


def export_demo_results(dest: Path | None = None) -> list[Path]:
    """Copy the result files from data/processed/ into DEMO_RESULTS_DIR (or
    dest) so they can be committed with the app for a hosted deployment.
    Only the dashboard result files are copied — never raw data. Optional
    extension results are skipped if they have not been generated."""
    dest = Path(dest or DEMO_RESULTS_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name, (path, command) in RESULT_FILES.items():
        path = Path(path)
        if not path.exists():
            if name in OPTIONAL_RESULTS:
                continue
            raise FileNotFoundError(f"{path} is missing; run `{command}` first")
        target = dest / path.name
        shutil.copy2(path, target)
        written.append(target)
    return written


def main() -> None:
    """python -m energy_dispatch.dashboard --export-demo-results"""
    parser = argparse.ArgumentParser(description="Dashboard data utilities.")
    parser.add_argument(
        "--export-demo-results",
        action="store_true",
        help=f"Copy the dashboard result files into {DEMO_RESULTS_DIR} for deployment.",
    )
    args = parser.parse_args()
    if not args.export_demo_results:
        parser.print_help()
        return
    for target in export_demo_results():
        print(f"Wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def headline_metrics(summary: pd.DataFrame) -> dict[str, dict]:
    """The three headline comparisons, computed from the scenario summary
    with evaluate.compare_scenarios (battery off for the forecast
    comparisons, LightGBM for the battery)."""
    comparisons = evaluate.compare_scenarios(summary)

    def _pick(comparison: str, scenario: str) -> dict:
        row = comparisons[
            (comparisons["comparison"] == comparison) & (comparisons["scenario"] == scenario)
        ]
        if row.empty:
            return {"saving_eur": float("nan"), "saving_pct": float("nan")}
        return {"saving_eur": row["saving_eur"].iloc[0], "saving_pct": row["saving_pct"].iloc[0]}

    tso_vs_lgbm = _pick("vs_tso", "lightgbm / battery off")
    return {
        "lightgbm_vs_naive": _pick("vs_seasonal_naive", "lightgbm / battery off"),
        # TSO's advantage over LightGBM = -(LightGBM's saving against TSO)
        "tso_vs_lightgbm": {k: -v for k, v in tso_vs_lgbm.items()},
        "battery_lightgbm": _pick("battery_value", "lightgbm / battery on"),
    }


# ---------------------------------------------------------------------------
# Forecast page
# ---------------------------------------------------------------------------


def local_dates(index: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(index.tz_convert(config.LOCAL_TZ).date)


def available_days(table: pd.DataFrame) -> list[dt.date]:
    return sorted(set(local_dates(table.index)))


def forecast_day(forecast_table: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    """One local day of the forecast table, indexed by local time."""
    rows = forecast_table[local_dates(forecast_table.index) == day].sort_index()
    rows = rows.copy()
    rows.index = rows.index.tz_convert(config.LOCAL_TZ)
    return rows


def forecast_metrics_for(table: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE/MAPE of each available point forecast (forecast.compute_point_metrics)."""
    cols = tuple(c for c in FORECAST_COLUMN_LABELS if c in table.columns)
    metrics = forecast.compute_point_metrics(table, model_cols=cols)
    return metrics.rename(index=FORECAST_COLUMN_LABELS)


def interval_metrics_for(table: pd.DataFrame) -> dict | None:
    """Coverage/width of the calibrated q10-q90 interval, if present."""
    low, high = quantile_columns()
    if low not in table.columns or high not in table.columns:
        return None
    valid = table[[low, high, "y_true"]].dropna()
    if valid.empty:
        return None
    return forecast.compute_interval_metrics(valid, low_col=low, high_col=high)


def quantile_columns() -> tuple[str, str]:
    """Column names forecast.py uses for the interval (q10/q90 by default)."""
    return (
        f"q{int(round(config.QUANTILE_LOW * 100))}",
        f"q{int(round(config.QUANTILE_HIGH * 100))}",
    )


# ---------------------------------------------------------------------------
# Dispatch page
# ---------------------------------------------------------------------------


def dispatch_day(
    hourly: pd.DataFrame, forecast_method: str, battery: str, day: dt.date
) -> pd.DataFrame:
    """One scenario's planned dispatch for one local day, indexed by local time,
    with renewable availability (used + curtailed) added."""
    rows = hourly[
        (hourly["forecast_method"] == forecast_method)
        & (hourly["battery"] == battery)
        & (hourly["local_date"] == day)
    ].sort_index()
    rows = rows.copy()
    rows.index = pd.DatetimeIndex(rows.index).tz_convert(config.LOCAL_TZ)
    rows["renewable_available"] = rows["renewable_used"] + rows["curtailment"]
    return rows


def dispatch_totals(day: pd.DataFrame) -> dict[str, float]:
    """Daily totals for one scenario-day (MWh and €)."""
    return {
        "planned_cost_eur": day["planned_cost_eur"].sum(),
        "imbalance_cost_eur": day["imbalance_cost_eur"].sum(),
        "realized_cost_eur": day["realized_cost_eur"].sum(),
        "renewable_available_mwh": day["renewable_available"].sum(),
        "renewable_used_mwh": day["renewable_used"].sum(),
        "curtailment_mwh": day["curtailment"].sum(),
        "battery_charge_mwh": day["charge"].sum(),
        "battery_discharge_mwh": day["discharge"].sum(),
        "peaker_mwh": day["gen_peaker"].sum() if "gen_peaker" in day else float("nan"),
    }


# ---------------------------------------------------------------------------
# Cost & forecast value page
# ---------------------------------------------------------------------------


def scenario_table(summary: pd.DataFrame) -> pd.DataFrame:
    """The eight core scenarios with readable labels, in a fixed order."""
    table = summary.reset_index()
    table["forecast"] = table["forecast_method"].map(FORECAST_LABELS)
    order = {name: i for i, name in enumerate(FORECAST_LABELS)}
    table = table.sort_values(
        ["forecast_method", "battery"],
        key=lambda s: s.map(order) if s.name == "forecast_method" else s,
    )
    return table[
        [
            "forecast",
            "battery",
            "planned_cost_eur",
            "imbalance_cost_eur",
            "realized_cost_eur",
            "total_abs_imbalance_mwh",
            "forecast_mae_mw",
            "n_days",
            "n_hours",
        ]
    ].reset_index(drop=True)


def cost_above_reference(summary: pd.DataFrame, battery: str = "off") -> pd.DataFrame:
    """Realized cost of each forecast relative to perfect foresight, and its
    split into planned-cost and imbalance-cost differences."""
    rows = summary.xs(battery, level="battery")
    reference = rows.loc["perfect_foresight"]
    out = pd.DataFrame(
        {
            "realized_above_reference_eur": rows["realized_cost_eur"]
            - reference["realized_cost_eur"],
            "planned_above_reference_eur": rows["planned_cost_eur"] - reference["planned_cost_eur"],
            "imbalance_cost_eur": rows["imbalance_cost_eur"],
            "forecast_mae_mw": rows["forecast_mae_mw"],
        }
    )
    out = out.drop(index="perfect_foresight", errors="ignore")
    out.index = out.index.map(FORECAST_LABELS)
    return out


def planning_view(planning: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split planning.py's summary into the quantile-plan rows (sorted by
    target quantile) and the reference rows (perfect foresight, TSO)."""
    plans = planning[planning["target_quantile"].notna()].sort_values("target_quantile")
    references = planning[planning["target_quantile"].isna()].copy()
    references["label"] = references["forecast_method"].map(FORECAST_LABELS)
    return plans.reset_index(drop=True), references.reset_index(drop=True)


def _dm_label(name: str) -> str:
    if name in FORECAST_COLUMN_LABELS:
        return FORECAST_COLUMN_LABELS[name]
    if name in FORECAST_LABELS:
        return FORECAST_LABELS[name]
    if name.startswith("lightgbm_q"):
        return "LightGBM point plan" if name == "lightgbm_q50" else f"LightGBM {name[10:]}% plan"
    return name


def dm_view(dm: pd.DataFrame, test: str) -> pd.DataFrame:
    """Readable Diebold-Mariano results for one test family
    ("forecast_error" or "realized_cost")."""
    rows = dm[dm["test"] == test]
    better = [
        _dm_label(a if lower == "a" else b)
        for a, b, lower in zip(rows["model_a"], rows["model_b"], rows["lower_loss"], strict=True)
    ]
    return pd.DataFrame(
        {
            "Comparison": [
                f"{_dm_label(a)} vs {_dm_label(b)}"
                for a, b in zip(rows["model_a"], rows["model_b"], strict=True)
            ],
            "Loss": rows["loss"].to_numpy(),
            "Lower loss": better,
            "Mean difference (a − b)": rows["mean_loss_difference"].to_numpy(),
            "DM statistic": rows["dm_statistic"].to_numpy(),
            "p-value": rows["p_value"].to_numpy(),
            "Significant at 5%": (rows["p_value"] < 0.05).to_numpy(),
        }
    )


def battery_value_by_forecast(summary: pd.DataFrame) -> pd.DataFrame:
    """Battery savings (off - on) for each forecast, split by cost component."""
    off = summary.xs("off", level="battery")
    on = summary.xs("on", level="battery")
    out = pd.DataFrame(
        {
            "battery_savings_eur": off["realized_cost_eur"] - on["realized_cost_eur"],
            "planned_cost_savings_eur": off["planned_cost_eur"] - on["planned_cost_eur"],
            "imbalance_cost_savings_eur": off["imbalance_cost_eur"] - on["imbalance_cost_eur"],
        }
    )
    out["battery_savings_pct"] = out["battery_savings_eur"] / off["realized_cost_eur"] * 100
    out.index = out.index.map(FORECAST_LABELS)
    return out


# ---------------------------------------------------------------------------
# Storage sensitivity page
# ---------------------------------------------------------------------------


def sensitivity_view(
    results: pd.DataFrame, parameter: str, forecast_method: str = "lightgbm"
) -> pd.DataFrame:
    """One sensitivity curve: sensitivity.battery_value_table filtered to a
    parameter and forecast, sorted by value."""
    table = sensitivity.battery_value_table(results)
    view = table[(table["parameter"] == parameter) & (table["forecast_method"] == forecast_method)]
    view = view.sort_values("value").reset_index(drop=True)
    view["sample_days"] = results["evaluated_days"].max()
    view["full_sample"] = view["evaluated_days"] == view["sample_days"]
    return view


# ---------------------------------------------------------------------------
# What-if page
# ---------------------------------------------------------------------------


def system_from_overrides(
    overrides: dict, base: sensitivity.SystemParams = sensitivity.BASELINE_SYSTEM
):
    """Apply several sensitivity parameters at once, e.g.
    {"ccgt_ramp_mw_per_hour": 2000, "battery_power_mw": 2000}."""
    system = base
    for parameter, value in overrides.items():
        system = sensitivity.build_variant(parameter, value, system)
    return system


def what_if_day(
    hourly: pd.DataFrame,
    day: dt.date,
    forecast_method: str,
    system: sensitivity.SystemParams,
) -> dict[str, dict]:
    """Re-plan one day for `system`, battery off and on, using the day's
    forecast, actual load and renewable availability from the stored
    evaluation results. Returns daily totals for each battery state.

    Raises RuntimeError (from optimize) if the system cannot meet the
    day's forecast demand.
    """
    rows = hourly[
        (hourly["forecast_method"] == forecast_method)
        & (hourly["battery"] == "off")
        & (hourly["local_date"] == day)
    ].sort_index()
    if rows.empty:
        raise KeyError(f"no stored results for {forecast_method} on {day}")
    forecast_load = rows["forecast_load"]
    actual_load = rows["actual_load"]
    renewable = rows["renewable_used"] + rows["curtailment"]

    out = {}
    for name, battery in (("off", evaluate.NO_BATTERY), ("on", system.battery)):
        result = evaluate.evaluate_day(
            forecast_load,
            actual_load,
            renewable,
            battery=battery,
            generation_units=system.generation_units,
            price_up=system.price_up,
            price_down=system.price_down,
        )
        out[name] = {
            "planned_cost_eur": result["planned_cost_eur"].sum(),
            "imbalance_cost_eur": result["imbalance_cost_eur"].sum(),
            "realized_cost_eur": result["realized_cost_eur"].sum(),
            "peaker_mwh": result["gen_peaker"].sum() if "gen_peaker" in result else float("nan"),
        }
    out["battery_savings_eur"] = out["off"]["realized_cost_eur"] - out["on"]["realized_cost_eur"]
    return out


if __name__ == "__main__":
    main()
