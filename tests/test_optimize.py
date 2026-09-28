"""
Invariant tests for energy_dispatch.optimize.

Core rules under test (spec 7.2, and section 12's engineering standards):
  - demand balance holds exactly in every hour of every solved LP
  - battery state of charge stays within [0, energy_capacity_mwh]
  - battery ends the day at the same SoC it started (cycle constraint)
  - generation never exceeds unit capacity, cheapest units are used first
  - ramp limits are respected hour to hour
  - battery charge/discharge never exceeds its power limit
  - the SoC recursion applies charge/discharge efficiency correctly
  - the battery never charges and discharges in the same hour
  - excess renewable output is curtailed rather than forced onto the grid
  - infeasible demand (exceeding total available capacity) raises loudly
  - the reported planned cost matches a hand-computed objective value

All scenarios here are small, synthetic, and hand-verifiable — none of
this touches the real OPSD dataset.
"""

import pandas as pd
import pytest

from energy_dispatch import config, optimize


@pytest.fixture
def toy_day():
    hours = pd.RangeIndex(config.HOURS_PER_DAY)
    # Ranges ~28,000-37,900 MW: comfortably within the ~42,000 MW of firm
    # generation capacity in config.GENERATION_UNITS (7,000 + 25,000 +
    # 10,000), even before counting renewables or the battery, while still
    # swinging enough hour to hour to exercise ramp and curtailment
    # behavior. (An earlier version of this fixture peaked at 85,000 MW,
    # which exceeds total available capacity and would have made the LP
    # infeasible for several hours a day — not what these happy-path tests
    # are meant to check.)
    forecast_load = pd.Series([28_000 + 900 * (h % 12) for h in hours], index=hours)
    renewable_available = pd.Series([2_000 for _ in hours], index=hours)
    return forecast_load, renewable_available


@pytest.fixture
def zero_battery():
    """A battery with no power/energy capacity, so it never participates
    in dispatch — useful for isolating generation-only behavior.
    """
    return config.BatteryParams(
        power_capacity_mw=0.0,
        energy_capacity_mwh=0.0,
        round_trip_efficiency=0.9,
        initial_soc_mwh=0.0,
    )


@pytest.fixture
def two_unit_system():
    """A small two-unit generation fleet, cheap enough to hand-verify:
    a 100 MW unit at 10 EUR/MWh, and a 50 MW unit at 100 EUR/MWh.
    """
    return (
        config.GenerationUnit(
            "baseload", capacity_mw=100, marginal_cost_eur_per_mwh=10, ramp_limit_mw_per_hour=None
        ),
        config.GenerationUnit(
            "peaker", capacity_mw=50, marginal_cost_eur_per_mwh=100, ramp_limit_mw_per_hour=None
        ),
    )


def test_demand_balance_holds_every_hour(toy_day):
    forecast_load, renewable_available = toy_day
    solution = optimize.plan_day_ahead_dispatch(forecast_load, renewable_available)
    assert optimize.validate_demand_balance(solution, forecast_load)


def test_battery_soc_within_bounds(toy_day):
    forecast_load, renewable_available = toy_day
    solution = optimize.plan_day_ahead_dispatch(forecast_load, renewable_available)
    soc = solution["soc"]
    assert ((soc >= -1e-6) & (soc <= config.BATTERY.energy_capacity_mwh + 1e-6)).all()


def test_battery_cycle_constraint(toy_day):
    forecast_load, renewable_available = toy_day
    solution = optimize.plan_day_ahead_dispatch(forecast_load, renewable_available)
    assert optimize.validate_battery_soc(solution, config.BATTERY)
    assert solution["soc"].iloc[-1] == pytest.approx(config.BATTERY.initial_soc_mwh, abs=1e-4)


def test_generator_capacity_limits_and_merit_order(zero_battery, two_unit_system):
    # Demand (130) exceeds the cheap unit's capacity (100), so the cheap
    # unit should be maxed out and the expensive unit should cover the
    # remaining 30 MW — never the other way around, and never above cap.
    hours = pd.RangeIndex(3)
    forecast_load = pd.Series([130.0] * 3, index=hours)
    renewable_available = pd.Series([0.0] * 3, index=hours)
    solution = optimize.plan_day_ahead_dispatch(
        forecast_load, renewable_available, generation_units=two_unit_system, battery=zero_battery
    )
    assert (solution["gen_baseload"] <= 100 + 1e-6).all()
    assert (solution["gen_peaker"] <= 50 + 1e-6).all()
    assert solution["gen_baseload"].apply(lambda v: v == pytest.approx(100.0, abs=1e-4)).all()
    assert solution["gen_peaker"].apply(lambda v: v == pytest.approx(30.0, abs=1e-4)).all()


def test_ramp_limit_is_enforced(zero_battery):
    units = (
        config.GenerationUnit(
            "slow", capacity_mw=200, marginal_cost_eur_per_mwh=10, ramp_limit_mw_per_hour=20
        ),
        config.GenerationUnit(
            "fast", capacity_mw=200, marginal_cost_eur_per_mwh=100, ramp_limit_mw_per_hour=None
        ),
    )
    hours = pd.RangeIndex(3)
    forecast_load = pd.Series([50.0, 150.0, 150.0], index=hours)
    renewable_available = pd.Series([0.0] * 3, index=hours)
    solution = optimize.plan_day_ahead_dispatch(
        forecast_load, renewable_available, generation_units=units, battery=zero_battery
    )
    ramp_hour_to_hour = solution["gen_slow"].diff().dropna()
    assert (ramp_hour_to_hour.abs() <= 20 + 1e-6).all()
    # the cheap "slow" unit hits its ramp limit exactly on the demand jump...
    assert ramp_hour_to_hour.iloc[0] == pytest.approx(20.0, abs=1e-4)
    # ...and the expensive "fast" unit covers the rest of the jump
    assert solution["gen_fast"].iloc[1] == pytest.approx(80.0, abs=1e-4)


def test_battery_power_limit_is_enforced():
    units = (
        config.GenerationUnit(
            "expensive",
            capacity_mw=1_000,
            marginal_cost_eur_per_mwh=100,
            ramp_limit_mw_per_hour=None,
        ),
    )
    battery = config.BatteryParams(
        power_capacity_mw=10.0,
        energy_capacity_mwh=1_000.0,
        round_trip_efficiency=1.0,
        initial_soc_mwh=0.0,
    )
    hours = pd.RangeIndex(2)
    forecast_load = pd.Series([0.0, 50.0], index=hours)
    renewable_available = pd.Series([100.0, 0.0], index=hours)
    solution = optimize.plan_day_ahead_dispatch(
        forecast_load, renewable_available, generation_units=units, battery=battery
    )
    assert solution["charge"].max() <= battery.power_capacity_mw + 1e-6
    assert solution["discharge"].max() <= battery.power_capacity_mw + 1e-6
    # free renewable at hour 0 charges the battery at its power limit...
    assert solution["charge"].iloc[0] == pytest.approx(10.0, abs=1e-4)
    # ...and it discharges at the same power limit at hour 1 to offset
    # otherwise-expensive generation
    assert solution["discharge"].iloc[1] == pytest.approx(10.0, abs=1e-4)
    assert solution["gen_expensive"].iloc[1] == pytest.approx(40.0, abs=1e-4)


def test_battery_soc_recursion_matches_efficiency_formula(toy_day):
    forecast_load, renewable_available = toy_day
    # round_trip_efficiency=0.81 -> charge/discharge efficiency = sqrt(0.81)
    # = 0.9 exactly, so the recursion is easy to hand-check.
    battery = config.BatteryParams(
        power_capacity_mw=config.BATTERY.power_capacity_mw,
        energy_capacity_mwh=config.BATTERY.energy_capacity_mwh,
        round_trip_efficiency=0.81,
        initial_soc_mwh=config.BATTERY.initial_soc_mwh,
    )
    assert battery.charge_efficiency == pytest.approx(0.9)
    assert battery.discharge_efficiency == pytest.approx(0.9)

    solution = optimize.plan_day_ahead_dispatch(forecast_load, renewable_available, battery=battery)

    expected_soc = battery.initial_soc_mwh
    for _, row in solution.iterrows():
        charge_in = battery.charge_efficiency * row["charge"]
        discharge_out = row["discharge"] / battery.discharge_efficiency
        expected_soc = expected_soc + charge_in - discharge_out
        assert row["soc"] == pytest.approx(expected_soc, abs=1e-4)


def test_battery_never_charges_and_discharges_simultaneously(toy_day):
    forecast_load, renewable_available = toy_day
    solution = optimize.plan_day_ahead_dispatch(forecast_load, renewable_available)
    simultaneous = (solution["charge"] > 1e-6) & (solution["discharge"] > 1e-6)
    assert not simultaneous.any()


def test_renewable_curtailment_when_generation_exceeds_demand(zero_battery):
    units = (
        config.GenerationUnit(
            "baseload", capacity_mw=100, marginal_cost_eur_per_mwh=10, ramp_limit_mw_per_hour=None
        ),
    )
    hours = pd.RangeIndex(1)
    forecast_load = pd.Series([50.0], index=hours)
    renewable_available = pd.Series([80.0], index=hours)
    solution = optimize.plan_day_ahead_dispatch(
        forecast_load, renewable_available, generation_units=units, battery=zero_battery
    )
    # renewable output (80) exceeds demand (50) and generation is free to
    # sit at zero, so exactly 30 MW of renewable must be curtailed.
    assert solution["curtailment"].iloc[0] == pytest.approx(30.0, abs=1e-4)
    assert solution["gen_baseload"].iloc[0] == pytest.approx(0.0, abs=1e-4)
    assert solution["renewable_used"].iloc[0] == pytest.approx(50.0, abs=1e-4)


def test_infeasible_demand_raises_runtime_error(zero_battery):
    units = (
        config.GenerationUnit(
            "only", capacity_mw=100, marginal_cost_eur_per_mwh=10, ramp_limit_mw_per_hour=None
        ),
    )
    hours = pd.RangeIndex(1)
    forecast_load = pd.Series([1_000_000.0], index=hours)  # far beyond any available capacity
    renewable_available = pd.Series([0.0], index=hours)
    with pytest.raises(RuntimeError):
        optimize.plan_day_ahead_dispatch(
            forecast_load, renewable_available, generation_units=units, battery=zero_battery
        )


def test_solve_dispatch_reports_infeasible_status_without_raising(zero_battery):
    units = (
        config.GenerationUnit(
            "only", capacity_mw=100, marginal_cost_eur_per_mwh=10, ramp_limit_mw_per_hour=None
        ),
    )
    hours = pd.RangeIndex(1)
    forecast_load = pd.Series([1_000_000.0], index=hours)
    renewable_available = pd.Series([0.0], index=hours)
    problem, _ = optimize.build_dispatch_lp(
        forecast_load, renewable_available, generation_units=units, battery=zero_battery
    )
    result = optimize.solve_dispatch(problem)
    assert result["status"] != "Optimal"
    assert result["objective_value"] is None


def test_planned_cost_matches_hand_calculated_objective(zero_battery, two_unit_system):
    hours = pd.RangeIndex(1)
    forecast_load = pd.Series([130.0], index=hours)
    renewable_available = pd.Series([0.0], index=hours)
    solution = optimize.plan_day_ahead_dispatch(
        forecast_load, renewable_available, generation_units=two_unit_system, battery=zero_battery
    )
    # cheapest dispatch: baseload maxed at 100 MW (10 EUR/MWh), peaker
    # covers the remaining 30 MW (100 EUR/MWh).
    expected_cost = 100 * 10 + 30 * 100
    assert solution["planned_cost_eur"].sum() == pytest.approx(expected_cost, abs=1e-4)
