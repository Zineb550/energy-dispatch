"""
Module 4 — day-ahead dispatch optimization (spec section 7).

Builds and solves the linear program described in the spec: given a
forecast load profile for one day (24 hourly values) and observed
renewable output, choose generation dispatch, battery charge/discharge,
and curtailment to minimize total generation cost subject to demand
balance, capacity, ramping, and battery state-of-charge constraints.

Uses PuLP with the CBC solver (config has no solver-specific settings
since CBC is the PuLP default).
"""

from __future__ import annotations

import pandas as pd

from energy_dispatch import config


def build_dispatch_lp(
    forecast_load: pd.Series,
    renewable_available: pd.Series,
    generation_units: tuple = config.GENERATION_UNITS,
    battery: config.BatteryParams = config.BATTERY,
):
    """Build the PuLP LP for one day (24 hours).

    Parameters
    ----------
    forecast_load : hourly forecast load (MWh/h), length 24, this is
        L-hat_t in the spec — the value the dispatch is planned against.
    renewable_available : hourly observed solar+wind output (MWh/h),
        length 24 — this is R_t, the curtailable renewable ceiling.

    Returns the PuLP problem plus a dict of the decision variables
    (generation per unit per hour, charge/discharge, state of charge,
    curtailment) so the caller can extract a solution after solving.

    Constraints implemented (spec 7.2):
      - demand balance per hour
      - per-unit capacity
      - per-unit ramping (where ramp_limit_mw_per_hour is not None)
      - battery charge/discharge power limits
      - battery energy balance with charge/discharge efficiency
      - battery state of charge bounds [0, energy_capacity_mwh]
      - battery cycle constraint: end-of-day SoC == start-of-day SoC
      - curtailment <= renewable_available
    """
    raise NotImplementedError


def solve_dispatch(problem) -> dict:
    """Solve a built LP (CBC) and return solver status + objective value."""
    raise NotImplementedError


def extract_dispatch_solution(variables: dict) -> pd.DataFrame:
    """Turn solved PuLP variables into an hourly DataFrame: generation by
    unit, battery charge/discharge/SoC, curtailment, and total planned
    cost per hour.
    """
    raise NotImplementedError


def plan_day_ahead_dispatch(
    forecast_load: pd.Series,
    renewable_available: pd.Series,
    battery: config.BatteryParams = config.BATTERY,
) -> pd.DataFrame:
    """Convenience wrapper: build -> solve -> extract for a single day.

    This is the function evaluate.py calls once per day of the test year
    for each (forecast, battery) scenario in spec section 7.4's table.
    """
    raise NotImplementedError
