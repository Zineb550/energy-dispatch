"""
Module 4 — day-ahead dispatch optimization (spec section 7).

Builds and solves the linear program described in the spec: given a
forecast load profile for one day (24 hourly values) and observed
renewable output, choose generation dispatch, battery charge/discharge,
and curtailment to minimize total generation cost subject to demand
balance, capacity, ramping, and battery state-of-charge constraints.

Uses PuLP with the CBC solver (config has no solver-specific settings
since CBC is the PuLP default).

Design notes (confirmed with the project owner before implementation):
  - Solar + wind are combined into a single renewable stream
    (`renewable_available`); the caller is responsible for summing the
    OPSD solar and wind columns before calling in.
  - `renewable_available` is treated as known with perfect foresight at
    planning time (there is no day-ahead renewable forecast in OPSD) —
    the same simplification already documented for weather features in
    forecast.py, and it should be called out again wherever dispatch
    results are reported.
  - Each day's LP is fully self-contained: the end-of-day cycle
    constraint (`soc[24] == soc[0] == battery.initial_soc_mwh`) means
    there is no SoC state to carry across days, and ramp constraints do
    not carry across days either (hour 0 of a day is never compared
    against the previous day's hour 23).
  - The battery has zero operating cost; only generation has a marginal
    cost in the objective.
  - `solve_dispatch` never raises — it reports the solver status
    honestly. `plan_day_ahead_dispatch` is the one place that turns a
    non-optimal status into a loud `RuntimeError`, since a day whose
    forecast demand exceeds total available capacity has no dispatch
    plan to return.
  - Realized cost and balancing-market imbalance (using *actual* demand
    once it is known) are computed in evaluate.py, not here — this
    module only produces the planned, forecast-based dispatch.
"""

from __future__ import annotations

import pandas as pd
import pulp

from energy_dispatch import config


def build_dispatch_lp(
    forecast_load: pd.Series,
    renewable_available: pd.Series,
    generation_units: tuple = config.GENERATION_UNITS,
    battery: config.BatteryParams = config.BATTERY,
) -> tuple[pulp.LpProblem, dict]:
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
    hours = list(forecast_load.index)
    if list(renewable_available.index) != hours:
        raise ValueError("forecast_load and renewable_available must share the same index")
    n_hours = len(hours)

    problem = pulp.LpProblem("day_ahead_dispatch", pulp.LpMinimize)

    # Variables are created via problem.add_variable(name, lowBound, upBound)
    # rather than the pulp.LpVariable(...) constructor directly: some
    # installed PuLP builds have already dropped lowBound/upBound from
    # LpVariable.__init__ itself (PuLP's own deprecation notice says so),
    # with add_variable as the forward-compatible replacement that exists
    # across versions.
    generation = {
        unit.name: {
            t: problem.add_variable(f"gen_{unit.name}_{i}", 0, unit.capacity_mw)
            for i, t in enumerate(hours)
        }
        for unit in generation_units
    }
    curtailment = {
        t: problem.add_variable(f"curtail_{i}", 0, float(renewable_available.iloc[i]))
        for i, t in enumerate(hours)
    }
    charge = {
        t: problem.add_variable(f"charge_{i}", 0, battery.power_capacity_mw)
        for i, t in enumerate(hours)
    }
    discharge = {
        t: problem.add_variable(f"discharge_{i}", 0, battery.power_capacity_mw)
        for i, t in enumerate(hours)
    }
    # soc[i] is the state of charge at the boundary before hour i's dispatch;
    # soc[n_hours] is the state of charge at the end of the day.
    soc = {
        i: problem.add_variable(f"soc_{i}", 0, battery.energy_capacity_mwh)
        for i in range(n_hours + 1)
    }

    # Objective: minimize total generation cost. Renewables (curtailed or
    # not) and the battery have zero marginal cost by assumption, so only
    # dispatchable generation appears here.
    problem += (
        pulp.lpSum(
            unit.marginal_cost_eur_per_mwh * generation[unit.name][t]
            for unit in generation_units
            for t in hours
        ),
        "total_generation_cost",
    )

    # Demand balance, every hour: dispatchable generation + renewable used
    # (available minus curtailed) + battery discharge - battery charge ==
    # the forecast load the day-ahead plan is committed against.
    for t in hours:
        renewable_used = float(renewable_available.loc[t]) - curtailment[t]
        problem += (
            (
                pulp.lpSum(generation[unit.name][t] for unit in generation_units)
                + renewable_used
                + discharge[t]
                - charge[t]
                == float(forecast_load.loc[t])
            ),
            f"demand_balance_{t}",
        )

    # Ramp constraints: no cross-day coupling, so hour 0 of the day is left
    # unconstrained (there is nothing before it in this LP).
    for unit in generation_units:
        if unit.ramp_limit_mw_per_hour is None:
            continue
        for i in range(1, n_hours):
            t_prev, t_curr = hours[i - 1], hours[i]
            gen_prev, gen_curr = generation[unit.name][t_prev], generation[unit.name][t_curr]
            problem += (
                (gen_curr - gen_prev <= unit.ramp_limit_mw_per_hour),
                f"ramp_up_{unit.name}_{i}",
            )
            problem += (
                (gen_prev - gen_curr <= unit.ramp_limit_mw_per_hour),
                f"ramp_down_{unit.name}_{i}",
            )

    # Battery: start the day at initial_soc_mwh, evolve by the efficiency-
    # weighted charge/discharge recursion, and return to initial_soc_mwh by
    # end of day (the cycle constraint — makes each day self-contained).
    problem += soc[0] == battery.initial_soc_mwh, "battery_initial_soc"
    for i, t in enumerate(hours):
        soc_next = (
            soc[i]
            + battery.charge_efficiency * charge[t]
            - discharge[t] / battery.discharge_efficiency
        )
        problem += (soc[i + 1] == soc_next), f"battery_soc_balance_{i}"
    problem += soc[n_hours] == battery.initial_soc_mwh, "battery_end_of_day_soc"

    variables = {
        "hours": hours,
        "generation_units": generation_units,
        "battery": battery,
        "forecast_load": forecast_load,
        "renewable_available": renewable_available,
        "generation": generation,
        "curtailment": curtailment,
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
    }
    return problem, variables


def solve_dispatch(problem: pulp.LpProblem) -> dict:
    """Solve a built LP (CBC) and return solver status + objective value.

    Never raises on an infeasible or otherwise non-optimal problem — it
    reports the solver's own status honestly so the caller decides what
    to do. `plan_day_ahead_dispatch` is where a non-optimal status becomes
    a `RuntimeError`.
    """
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[problem.status]
    objective_value = pulp.value(problem.objective) if status == "Optimal" else None
    return {"status": status, "objective_value": objective_value}


def extract_dispatch_solution(variables: dict) -> pd.DataFrame:
    """Turn solved PuLP variables into an hourly DataFrame: generation by
    unit, battery charge/discharge/SoC, curtailment, and total planned
    cost per hour.
    """
    hours = variables["hours"]
    generation_units = variables["generation_units"]
    generation = variables["generation"]
    curtailment = variables["curtailment"]
    charge = variables["charge"]
    discharge = variables["discharge"]
    soc = variables["soc"]
    forecast_load = variables["forecast_load"]
    renewable_available = variables["renewable_available"]

    rows = []
    for i, t in enumerate(hours):
        row = {"hour": t}
        planned_cost = 0.0
        for unit in generation_units:
            value = pulp.value(generation[unit.name][t]) or 0.0
            row[f"gen_{unit.name}"] = value
            planned_cost += unit.marginal_cost_eur_per_mwh * value
        row["curtailment"] = pulp.value(curtailment[t]) or 0.0
        row["renewable_used"] = float(renewable_available.loc[t]) - row["curtailment"]
        row["charge"] = pulp.value(charge[t]) or 0.0
        row["discharge"] = pulp.value(discharge[t]) or 0.0
        row["soc"] = pulp.value(soc[i + 1]) or 0.0
        row["forecast_load"] = float(forecast_load.loc[t])
        row["planned_cost_eur"] = planned_cost
        rows.append(row)

    return pd.DataFrame(rows).set_index("hour")


def plan_day_ahead_dispatch(
    forecast_load: pd.Series,
    renewable_available: pd.Series,
    generation_units: tuple = config.GENERATION_UNITS,
    battery: config.BatteryParams = config.BATTERY,
) -> pd.DataFrame:
    """Convenience wrapper: build -> solve -> extract for a single day.

    This is the function evaluate.py calls once per day of the test year
    for each (forecast, battery) scenario in spec section 7.4's table.

    Raises `RuntimeError` if the day-ahead LP does not solve to
    optimality — in practice this means the forecast demand for that day
    exceeds total available capacity (generation + renewables +
    battery discharge), which is a modeling/scenario error worth
    surfacing loudly rather than returning a partial or garbage plan.
    """
    problem, variables = build_dispatch_lp(
        forecast_load, renewable_available, generation_units=generation_units, battery=battery
    )
    result = solve_dispatch(problem)
    if result["status"] != "Optimal":
        raise RuntimeError(
            "Day-ahead dispatch optimization did not solve to optimality "
            f"(solver status: {result['status']!r}). This usually means the "
            "forecast demand for this day exceeds total available capacity "
            "(generation + renewables + battery discharge)."
        )
    return extract_dispatch_solution(variables)


def validate_demand_balance(
    solution: pd.DataFrame,
    forecast_load: pd.Series,
    generation_units: tuple = config.GENERATION_UNITS,
    atol: float = 1e-3,
) -> bool:
    """Check that generation + renewable_used + discharge - charge equals
    the forecast load in every hour of a dispatch solution.

    `atol` defaults to 1e-3 MW rather than machine precision: CBC (like
    any LP solver) reports a solution within its own numerical tolerance
    of exactly feasible, not bit-exact, so a very tight atol produces
    false positives on an otherwise-correct solve.
    """
    gen_cols = [f"gen_{unit.name}" for unit in generation_units]
    total_supply = (
        solution[gen_cols].sum(axis=1)
        + solution["renewable_used"]
        + solution["discharge"]
        - solution["charge"]
    )
    aligned_load = forecast_load.reindex(solution.index)
    return bool((total_supply.sub(aligned_load).abs() <= atol).all())


def validate_battery_soc(
    solution: pd.DataFrame,
    battery: config.BatteryParams = config.BATTERY,
    atol: float = 1e-3,
) -> bool:
    """Check that SoC stays within [0, energy_capacity_mwh] throughout the
    day and that the day ends at the same SoC it started (cycle
    constraint).
    """
    soc = solution["soc"]
    within_bounds = bool(((soc >= -atol) & (soc <= battery.energy_capacity_mwh + atol)).all())
    ends_at_initial = abs(float(soc.iloc[-1]) - battery.initial_soc_mwh) <= atol
    return within_bounds and ends_at_initial
