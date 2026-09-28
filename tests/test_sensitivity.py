"""
Tests for energy_dispatch.sensitivity — all synthetic, no OPSD data.

Covers variant construction (each parameter lands in the right place, the
baseline is reproduced, config is never mutated), savings arithmetic, the
planned/imbalance split, caching, determinism, the shared day sample, and
a toy system where the battery's value is known by hand.
"""

import copy
import datetime as dt

import pandas as pd
import pytest

from energy_dispatch import config, evaluate, sensitivity

BASE = sensitivity.BASELINE_SYSTEM


def local_day_index(date: str) -> pd.DatetimeIndex:
    day = pd.Timestamp(date).date()
    start = pd.Timestamp(day, tz=config.LOCAL_TZ)
    end = pd.Timestamp(day + dt.timedelta(days=1), tz=config.LOCAL_TZ)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def make_inputs(dates, load_by_local_hour, forecast_offset=0.0, renewable=3_000.0):
    """Forecast table in forecast.py's shape + renewable series."""
    index = pd.DatetimeIndex([ts for d in dates for ts in local_day_index(d)], name="utc_timestamp")
    hours = index.tz_convert(config.LOCAL_TZ).hour
    actual = pd.Series([float(load_by_local_hour(h)) for h in hours], index=index)
    forecast = actual - forecast_offset
    table = pd.DataFrame(
        {"y_true": actual, "y_naive": forecast, "y_tso": forecast, "y_lgbm": forecast}
    )
    return table, pd.Series(renewable, index=index, name="renewable_available")


def realistic_day(hour: int) -> float:
    """A Spanish-looking daily shape within the baseline fleet's capacity."""
    return 26_000 + 9_000 * (8 <= hour <= 21) + 3_000 * (19 <= hour <= 21)


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------


def test_baseline_values_reproduce_the_baseline_system():
    baseline_values = {
        "battery_power_mw": 1_000,
        "battery_energy_mwh": 4_000,
        "ccgt_ramp_mw_per_hour": 8_000,
        "ccgt_capacity_mw": 25_000,
        "peaker_cost_eur_per_mwh": 120,
        "balancing_prices_eur_per_mwh": (200, 20),
    }
    for parameter, value in baseline_values.items():
        assert value in sensitivity.SENSITIVITY_DIMENSIONS[parameter]  # grid includes baseline
        assert sensitivity.build_variant(parameter, value) == BASE, parameter
    assert BASE.generation_units == config.GENERATION_UNITS
    assert BASE.battery == config.BATTERY
    assert (BASE.price_up, BASE.price_down) == (200.0, 20.0)


def test_battery_power_variant():
    variant = sensitivity.build_variant("battery_power_mw", 4_000)
    assert variant.battery.power_capacity_mw == 4_000
    assert variant.battery.energy_capacity_mwh == config.BATTERY.energy_capacity_mwh
    assert variant.generation_units == BASE.generation_units


def test_battery_energy_variant_keeps_the_start_of_day_soc_fraction():
    variant = sensitivity.build_variant("battery_energy_mwh", 1_000)
    assert variant.battery.energy_capacity_mwh == 1_000
    assert variant.battery.initial_soc_mwh == 500  # 50%, as in the baseline (2,000 of 4,000)
    assert variant.battery.power_capacity_mw == config.BATTERY.power_capacity_mw


def _units(system):
    return {unit.name: unit for unit in system.generation_units}


def test_ccgt_ramp_variant_changes_only_the_ccgt_ramp():
    units = _units(sensitivity.build_variant("ccgt_ramp_mw_per_hour", 2_000))
    base_units = _units(BASE)
    assert units["ccgt"].ramp_limit_mw_per_hour == 2_000
    assert units["ccgt"].capacity_mw == base_units["ccgt"].capacity_mw
    assert units["baseload"] == base_units["baseload"]
    assert units["peaker"] == base_units["peaker"]


def test_ccgt_capacity_variant_changes_only_the_ccgt_capacity():
    units = _units(sensitivity.build_variant("ccgt_capacity_mw", 15_000))
    assert units["ccgt"].capacity_mw == 15_000
    assert units["ccgt"].ramp_limit_mw_per_hour == 8_000
    assert units["peaker"] == _units(BASE)["peaker"]


def test_peaker_cost_variant_changes_only_the_peaker_cost():
    units = _units(sensitivity.build_variant("peaker_cost_eur_per_mwh", 250))
    assert units["peaker"].marginal_cost_eur_per_mwh == 250
    assert units["peaker"].capacity_mw == 10_000
    assert units["ccgt"] == _units(BASE)["ccgt"]


def test_balancing_price_variant():
    variant = sensitivity.build_variant("balancing_prices_eur_per_mwh", (300, 10))
    assert (variant.price_up, variant.price_down) == (300.0, 10.0)
    assert variant.generation_units == BASE.generation_units
    assert variant.battery == BASE.battery


def test_unknown_parameter_is_rejected():
    with pytest.raises(KeyError):
        sensitivity.build_variant("nuclear_capacity_mw", 1)


# ---------------------------------------------------------------------------
# Savings arithmetic
# ---------------------------------------------------------------------------


def test_battery_savings_and_percentage():
    off = {"total_realized_cost_eur": 1_000.0, "total_planned_cost_eur": 800.0,
           "total_imbalance_cost_eur": 200.0}  # fmt: skip
    on = {"total_realized_cost_eur": 950.0, "total_planned_cost_eur": 750.0,
          "total_imbalance_cost_eur": 200.0}  # fmt: skip
    savings = sensitivity.battery_savings(off, on)
    assert savings["battery_savings_eur"] == pytest.approx(50.0)  # off - on
    assert savings["battery_savings_pct"] == pytest.approx(5.0)  # 50 / 1000 * 100
    assert savings["planned_cost_savings_eur"] == pytest.approx(50.0)
    assert savings["imbalance_cost_savings_eur"] == pytest.approx(0.0)


def test_reprice_changes_imbalance_cost_only():
    hourly = pd.DataFrame(
        {"imbalance": [10.0, -10.0], "planned_cost_eur": [100.0, 100.0],
         "imbalance_cost_eur": [0.0, 0.0], "realized_cost_eur": [0.0, 0.0]}
    )  # fmt: skip
    out = sensitivity.reprice(hourly, price_up=300, price_down=10)
    assert out["imbalance_cost_eur"].tolist() == [3_000.0, -100.0]
    assert out["realized_cost_eur"].tolist() == [3_100.0, 0.0]
    assert out["planned_cost_eur"].tolist() == [100.0, 100.0]


# ---------------------------------------------------------------------------
# Running variants on synthetic days
# ---------------------------------------------------------------------------


def test_baseline_row_matches_a_direct_evaluate_year_run():
    table, renewable = make_inputs(["2019-06-12"], realistic_day, forecast_offset=300)
    results = sensitivity.run_sensitivity(
        table, renewable, dimensions={"battery_power_mw": (1_000,)}
    )
    direct, _ = evaluate.evaluate_year(
        table, renewable, scenarios={"lightgbm": "y_lgbm"}, batteries=evaluate.BATTERY_SCENARIOS
    )
    summary = evaluate.aggregate_results(direct)
    for enabled, battery in ((False, "off"), (True, "on")):
        row = results[results["battery_enabled"] == enabled].iloc[0]
        assert row["is_baseline"]
        for col in ("planned_cost_eur", "imbalance_cost_eur", "realized_cost_eur"):
            assert row[f"total_{col}"] == pytest.approx(summary.loc[("lightgbm", battery), col])


def test_planned_and_imbalance_components_stay_distinguishable():
    table, renewable = make_inputs(["2019-06-12"], realistic_day, forecast_offset=500)
    results = sensitivity.run_sensitivity(
        table, renewable, dimensions={"battery_power_mw": (1_000, 4_000)}
    )
    on_rows = results[results["battery_enabled"]]
    off_rows = results[~results["battery_enabled"]]
    # imbalance cost is identical with and without the battery (fixed plan)...
    assert (on_rows["imbalance_cost_savings_eur"].abs() < 1.0).all()
    assert (off_rows["total_imbalance_cost_eur"] > 0).all()  # ...but not zero
    # ...so all of the battery's value shows up as planned-cost savings
    pd.testing.assert_series_equal(
        on_rows["battery_savings_eur"],
        on_rows["planned_cost_savings_eur"] + on_rows["imbalance_cost_savings_eur"],
        check_names=False,
    )
    assert off_rows["battery_savings_eur"].isna().all()


def test_balancing_prices_are_propagated_without_re_solving(monkeypatch):
    table, renewable = make_inputs(["2019-06-12"], realistic_day, forecast_offset=500)
    runner = sensitivity.SensitivityRunner(table, renewable)
    results = sensitivity.run_sensitivity(
        table,
        renewable,
        dimensions={"balancing_prices_eur_per_mwh": ((150, 50), (200, 20), (300, 10))},
        runner=runner,
    )
    assert runner.lp_runs == 2  # one battery-off + one battery-on solve, shared by all prices
    off = results[~results["battery_enabled"]].set_index("value_label")
    # forecast is 500 MW below actual every hour -> 24 h x 500 MWh upward imbalance
    upward_mwh = 24 * 500
    for label, price_up in (("150/50", 150), ("200/20", 200), ("300/10", 300)):
        assert off.loc[label, "total_imbalance_cost_eur"] == pytest.approx(
            upward_mwh * price_up, rel=1e-6
        )
        assert off.loc[label, "price_up_eur_per_mwh"] == price_up
    assert off["total_planned_cost_eur"].nunique() == 1  # prices don't touch the plan


def test_dispatch_results_are_cached_across_variants():
    table, renewable = make_inputs(["2019-06-12"], realistic_day)
    runner = sensitivity.SensitivityRunner(table, renewable)
    sensitivity.run_sensitivity(
        table,
        renewable,
        dimensions={
            "battery_power_mw": (500, 1_000, 2_000, 4_000),  # 1 shared "off" + 4 "on"
            "peaker_cost_eur_per_mwh": (120, 250),  # baseline cached; 250 needs off + on
        },
        runner=runner,
    )
    assert runner.lp_runs == 5 + 2


def test_run_does_not_mutate_the_baseline_configuration():
    before = (
        copy.deepcopy(config.GENERATION_UNITS),
        copy.deepcopy(config.BATTERY),
        config.BALANCING_PRICE_UP_EUR_PER_MWH,
        config.BALANCING_PRICE_DOWN_EUR_PER_MWH,
        copy.deepcopy(sensitivity.BASELINE_SYSTEM),
    )
    table, renewable = make_inputs(["2019-06-12"], realistic_day, forecast_offset=200)
    sensitivity.run_sensitivity(
        table,
        renewable,
        dimensions={
            "battery_energy_mwh": (1_000,),
            "ccgt_ramp_mw_per_hour": (1_000,),
            "balancing_prices_eur_per_mwh": ((300, 10),),
        },
    )
    after = (
        config.GENERATION_UNITS,
        config.BATTERY,
        config.BALANCING_PRICE_UP_EUR_PER_MWH,
        config.BALANCING_PRICE_DOWN_EUR_PER_MWH,
        sensitivity.BASELINE_SYSTEM,
    )
    assert after == before


def test_results_are_deterministic():
    table, renewable = make_inputs(["2019-06-12"], realistic_day, forecast_offset=300)
    dims = {"ccgt_ramp_mw_per_hour": (1_000, 8_000)}
    first = sensitivity.run_sensitivity(table, renewable, dimensions=dims)
    second = sensitivity.run_sensitivity(table, renewable, dimensions=dims)
    pd.testing.assert_frame_equal(first, second)


def test_uses_the_same_day_sample_as_the_baseline_evaluation():
    table, renewable = make_inputs(["2019-06-11", "2019-06-12"], realistic_day)
    day_11 = table.index.tz_convert(config.LOCAL_TZ).date == dt.date(2019, 6, 11)
    table.loc[table.index[day_11][4], "y_naive"] = float("nan")  # naive-only gap

    results = sensitivity.run_sensitivity(
        table, renewable, dimensions={"battery_power_mw": (1_000,)}
    )  # LightGBM only, but the naive gap must still exclude that day
    assert (results["evaluated_days"] == 1).all()
    assert results["comparable_to_baseline_sample"].all()


def test_infeasible_variant_days_are_dropped_for_both_battery_states_and_reported():
    # Peak 38,000 MW: fine at the baseline 25,000 MW CCGT, infeasible at 15,000
    # (7,000 + 15,000 + 10,000 + 3,000 renewables + 1,000 battery < 38,000).
    table, renewable = make_inputs(
        ["2019-06-11", "2019-06-12"],
        lambda h: 38_000 if h == 20 else 28_000,
    )
    day_12 = table.index.tz_convert(config.LOCAL_TZ).date == dt.date(2019, 6, 12)
    for col in ("y_true", "y_naive", "y_tso", "y_lgbm"):
        table.loc[day_12, col] = 28_000.0  # the second day stays feasible

    results = sensitivity.run_sensitivity(
        table, renewable, dimensions={"ccgt_capacity_mw": (15_000, 25_000)}
    )
    small = results[results["value"] == 15_000]
    assert (small["infeasible_days"] == 1).all()
    assert (small["evaluated_days"] == 1).all()  # same single day for off and on
    assert not small["comparable_to_baseline_sample"].any()
    baseline = results[results["value"] == 25_000]
    assert (baseline["evaluated_days"] == 2).all()
    assert baseline["comparable_to_baseline_sample"].all()


# ---------------------------------------------------------------------------
# Toy system where the battery's value is known by hand
# ---------------------------------------------------------------------------

# CCGT 100 MW at 60 EUR/MWh, peaker at 120 EUR/MWh. Demand is 50 MW for the
# first 12 local hours and 150 MW for the last 12, so without storage the
# peaker supplies 50 MW x 12 h = 600 MWh. A 50 MW battery (lossless, to keep
# the arithmetic exact) charges 600 MWh from spare CCGT early in the day and
# discharges it at the peak, replacing all peaker output:
#     savings = 600 MWh x (120 - 60) EUR/MWh = 36,000 EUR.
TOY_SYSTEM = sensitivity.SystemParams(
    generation_units=(
        config.GenerationUnit("ccgt", 100, 60, None),
        config.GenerationUnit("peaker", 1_000, 120, None),
    ),
    battery=config.BatteryParams(
        power_capacity_mw=50.0,
        energy_capacity_mwh=2_000.0,
        round_trip_efficiency=1.0,
        initial_soc_mwh=1_000.0,
    ),
)


def toy_inputs():
    return make_inputs(["2019-06-12"], lambda h: 50 if h < 12 else 150, renewable=0.0)


def _savings_by_value(results):
    on = results[results["battery_enabled"]]
    return dict(zip(on["value"], on["battery_savings_eur"], strict=True))


def test_toy_battery_value_when_peaker_use_is_forced():
    table, renewable = toy_inputs()
    results = sensitivity.run_sensitivity(
        table, renewable, dimensions={"peaker_cost_eur_per_mwh": (120,)}, base=TOY_SYSTEM
    )
    on = results[results["battery_enabled"]].iloc[0]
    off = results[~results["battery_enabled"]].iloc[0]
    assert off["peaker_energy_mwh"] == pytest.approx(600.0, abs=1e-3)
    assert on["peaker_energy_mwh"] == pytest.approx(0.0, abs=1e-3)
    assert on["battery_savings_eur"] == pytest.approx(36_000.0, abs=0.1)
    assert on["imbalance_cost_savings_eur"] == pytest.approx(0.0, abs=1e-6)


def test_toy_battery_value_responds_to_the_system_parameters():
    table, renewable = toy_inputs()
    results = sensitivity.run_sensitivity(
        table,
        renewable,
        dimensions={
            # dearer peaker -> each shifted MWh is worth (cost - 60)
            "peaker_cost_eur_per_mwh": (120, 250),
            # enough CCGT to cover the peak -> nothing for the battery to avoid
            "ccgt_capacity_mw": (100, 200),
            # half the battery power -> only 25 MW x 12 h = 300 MWh shifted
            "battery_power_mw": (25, 50),
        },
        base=TOY_SYSTEM,
    )
    by_param = {p: _savings_by_value(g) for p, g in results.groupby("parameter")}
    assert by_param["peaker_cost_eur_per_mwh"][120] == pytest.approx(600 * 60, abs=0.1)
    assert by_param["peaker_cost_eur_per_mwh"][250] == pytest.approx(600 * 190, abs=0.1)
    assert by_param["ccgt_capacity_mw"][200] == pytest.approx(0.0, abs=0.1)
    assert by_param["battery_power_mw"][25] == pytest.approx(300 * 60, abs=0.1)


def test_battery_value_table_is_one_row_per_variant():
    table, renewable = toy_inputs()
    results = sensitivity.run_sensitivity(
        table, renewable, dimensions={"peaker_cost_eur_per_mwh": (120, 250)}, base=TOY_SYSTEM
    )
    compact = sensitivity.battery_value_table(results)
    assert len(compact) == 2
    row = compact.set_index("value").loc[250.0]
    assert row["battery_savings_eur"] == pytest.approx(
        row["realized_cost_off_eur"] - row["realized_cost_on_eur"]
    )
    assert row["peaker_energy_off_mwh"] == pytest.approx(600.0, abs=1e-3)
