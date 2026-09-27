"""
Module 4 (cont.) — realized-cost evaluation (spec section 7.3-7.4).

Plans dispatch against a forecast, reveals the actual load, prices the
resulting imbalance at the balancing market rates, and compares scenarios
across the (forecast x battery) grid.
"""

from __future__ import annotations

import pandas as pd

from energy_dispatch import config


def compute_imbalance(actual_load: pd.Series, planned_supply: pd.Series) -> pd.Series:
    """Delta_t = L_t - planned_supply_t, per hour."""
    raise NotImplementedError


def price_imbalance(
    imbalance: pd.Series,
    price_up: float = config.BALANCING_PRICE_UP_EUR_PER_MWH,
    price_down: float = config.BALANCING_PRICE_DOWN_EUR_PER_MWH,
) -> pd.Series:
    """Positive imbalance (under-supply) priced at price_up; negative
    imbalance (over-supply) priced as a credit at price_down.
    """
    raise NotImplementedError


def realized_cost_for_day(
    forecast_load: pd.Series,
    actual_load: pd.Series,
    renewable_available: pd.Series,
    battery: config.BatteryParams,
) -> dict:
    """Plan dispatch against forecast_load, then evaluate against
    actual_load. Returns {'planned_cost': ..., 'imbalance_cost': ...,
    'realized_cost': ...} for one day.
    """
    raise NotImplementedError


def run_scenario(
    forecast_table: pd.DataFrame,
    forecast_col: str,
    use_battery: bool,
) -> pd.DataFrame:
    """Run realized_cost_for_day across every day of the test year for one
    (forecast_col, use_battery) combination. use_battery=False solves the
    LP with battery power/energy capacity fixed at 0.

    Returns a daily DataFrame of planned/imbalance/realized cost, to be
    aggregated into the scenario comparison table (spec 7.4).
    """
    raise NotImplementedError


def compare_scenarios(forecast_table: pd.DataFrame) -> pd.DataFrame:
    """Run run_scenario for every cell of the (forecast x battery) grid:
    {perfect foresight, seasonal naive, TSO forecast, LightGBM} x
    {no battery, optimized battery}.

    Returns the annual €-and-% savings table that is the project's
    headline result (spec 7.4), ready to feed the deck and dashboard.
    """
    raise NotImplementedError
