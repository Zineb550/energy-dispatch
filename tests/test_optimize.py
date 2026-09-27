"""
Invariant tests for energy_dispatch.optimize.

Core rules under test (spec 7.2, and section 12's engineering standards):
  - demand balance holds exactly in every hour of every solved LP
  - battery state of charge stays within [0, energy_capacity_mwh]
  - battery ends the day at the same SoC it started (cycle constraint)
"""

import pandas as pd
import pytest

from energy_dispatch import config, optimize


@pytest.fixture
def toy_day():
    hours = pd.RangeIndex(config.HOURS_PER_DAY)
    forecast_load = pd.Series([30_000 + 5_000 * (h % 12) for h in hours], index=hours)
    renewable_available = pd.Series([2_000 for _ in hours], index=hours)
    return forecast_load, renewable_available


def test_demand_balance_holds_every_hour(toy_day):
    forecast_load, renewable_available = toy_day
    pytest.skip("implement once optimize.plan_day_ahead_dispatch is implemented")


def test_battery_soc_within_bounds(toy_day):
    pytest.skip("implement once optimize.plan_day_ahead_dispatch is implemented")


def test_battery_cycle_constraint(toy_day):
    pytest.skip("implement once optimize.plan_day_ahead_dispatch is implemented")
