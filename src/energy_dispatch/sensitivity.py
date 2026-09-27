"""
Module 5 — sensitivity & uncertainty sweeps (spec section 8).
"""

from __future__ import annotations

import pandas as pd

from energy_dispatch import config


def sweep_battery_size(
    forecast_table: pd.DataFrame,
    sizes_mw: tuple = config.BATTERY_POWER_SWEEP_MW,
) -> pd.DataFrame:
    """Savings vs. battery power capacity (0 -> 4,000 MW), holding energy
    capacity's MW:MWh ratio fixed at the config.BATTERY default. Produces
    the diminishing-returns curve referenced in spec 8.
    """
    raise NotImplementedError


def sweep_balancing_price(
    forecast_table: pd.DataFrame,
    prices_up: tuple = config.BALANCING_PRICE_UP_SWEEP_EUR_PER_MWH,
) -> pd.DataFrame:
    """How much forecast accuracy is worth as the up-imbalance penalty
    rises, holding the down-price fixed.
    """
    raise NotImplementedError


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
