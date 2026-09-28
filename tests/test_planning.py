"""
Tests for energy_dispatch.planning (uncertainty-aware planning) — all
synthetic, no OPSD data.
"""

import datetime as dt

import pandas as pd
import pytest

from energy_dispatch import config, planning


def _local_day_index(date: str) -> pd.DatetimeIndex:
    day = pd.Timestamp(date).date()
    start = pd.Timestamp(day, tz=config.LOCAL_TZ)
    end = pd.Timestamp(day + dt.timedelta(days=1), tz=config.LOCAL_TZ)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def flat_day(point=30_000.0, actual_offset=300.0, upper=600.0, lower=600.0):
    """One day with a flat point forecast; actual demand, q10 and q90 at
    fixed offsets. Net of 3,000 MW renewables, the baseline fleet runs
    baseload flat out with CCGT on the margin in every hour."""
    index = pd.DatetimeIndex(_local_day_index("2019-06-12"), name="utc_timestamp")
    table = pd.DataFrame(
        {
            "y_true": point + actual_offset,
            "y_naive": point,
            "y_tso": point + actual_offset / 2,
            "y_lgbm": point,
            "q10": point - lower,
            "q90": point + upper,
        },
        index=index,
    )
    renewable = pd.Series(3_000.0, index=index, name="renewable_available")
    return table, renewable


# ---------------------------------------------------------------------------
# Newsvendor quantile
# ---------------------------------------------------------------------------


def test_newsvendor_quantile():
    # CCGT marginal: under-planning costs 200 - 60 = 140, over-planning 60 - 20 = 40
    assert planning.newsvendor_quantile(200, 20, 60) == pytest.approx(140 / 180)
    # peaker marginal: 80 vs 100 -> plan below the median
    assert planning.newsvendor_quantile(200, 20, 120) == pytest.approx(80 / 180)
    assert planning.theoretical_plan_quantile() == pytest.approx(0.7778, abs=1e-4)
    with pytest.raises(ValueError):
        planning.newsvendor_quantile(200, 20, 10)  # surplus would be profitable


# ---------------------------------------------------------------------------
# Quantile plans
# ---------------------------------------------------------------------------


def test_quantile_plan_reproduces_the_forecast_columns_it_is_built_from():
    table, _ = flat_day(upper=600, lower=300)
    pd.testing.assert_series_equal(
        planning.quantile_plan(table, 0.5), table["y_lgbm"], check_names=False
    )
    pd.testing.assert_series_equal(
        planning.quantile_plan(table, 0.9), table["q90"], check_names=False
    )
    pd.testing.assert_series_equal(
        planning.quantile_plan(table, 0.1), table["q10"], check_names=False
    )


def test_quantile_plan_is_monotone_and_uses_the_matching_side_of_the_interval():
    table, _ = flat_day(upper=600, lower=300)
    plans = [planning.quantile_plan(table, tau).iloc[0] for tau in (0.2, 0.3, 0.5, 0.7, 0.8)]
    assert plans == sorted(plans)
    # the upper side is twice as wide, so tau = 0.7 moves twice as far as tau = 0.3
    up = planning.quantile_plan(table, 0.7).iloc[0] - 30_000
    down = 30_000 - planning.quantile_plan(table, 0.3).iloc[0]
    assert up == pytest.approx(2 * down)


def test_quantile_plan_never_uses_actual_demand():
    table, _ = flat_day()
    without_actuals = table.drop(columns="y_true")
    pd.testing.assert_series_equal(
        planning.quantile_plan(table, 0.8), planning.quantile_plan(without_actuals, 0.8)
    )


# ---------------------------------------------------------------------------
# Realized cost of quantile plans
# ---------------------------------------------------------------------------


def _row(summary, tau):
    return summary.loc[summary["forecast_method"] == planning.plan_name(tau)].iloc[0]


def test_planning_on_q90_saves_the_hand_computed_amount():
    # Actual demand is 300 MW above the point forecast every hour; q90 is 600 MW above.
    # Point plan: 300 MWh short each hour, bought at 200     -> 60,000 EUR/h
    # q90 plan:   600 MWh more CCGT at 60, 300 MWh sold at 20 -> 36,000 - 6,000 = 30,000 EUR/h
    table, renewable = flat_day()
    summary, skipped = planning.uncertainty_aware_plan(table, renewable, quantiles=(0.5, 0.9))
    assert skipped.empty
    point, q90 = _row(summary, 0.5), _row(summary, 0.9)
    assert point["imbalance_cost_eur"] == pytest.approx(24 * 300 * 200, abs=1)
    assert q90["imbalance_cost_eur"] == pytest.approx(-24 * 300 * 20, abs=1)
    assert q90["planned_cost_eur"] - point["planned_cost_eur"] == pytest.approx(
        24 * 600 * 60, abs=1
    )
    assert q90["saving_vs_point_plan_eur"] == pytest.approx(24 * 30_000, abs=1)
    assert point["saving_vs_point_plan_eur"] == 0
    assert q90["mean_uplift_mw"] == pytest.approx(600)


def test_default_run_includes_tau_star_references_and_one_day_sample():
    table, renewable = flat_day()
    summary, _ = planning.uncertainty_aware_plan(table, renewable)
    methods = set(summary["forecast_method"])
    assert {"perfect_foresight", "tso", planning.plan_name(0.5)} <= methods
    assert summary["is_theoretical_optimum"].sum() == 1
    assert summary.attrs["theoretical_quantile"] == pytest.approx(140 / 180)
    assert summary["n_days"].nunique() == 1
    reference = summary.set_index("forecast_method").loc["perfect_foresight"]
    assert abs(reference["imbalance_cost_eur"]) < 1.0


def test_plans_do_not_change_when_actual_demand_changes():
    table, renewable = flat_day()
    shocked = table.copy()
    shocked["y_true"] += 1_000
    a, _ = planning.uncertainty_aware_plan(table, renewable, quantiles=(0.5, 0.8))
    b, _ = planning.uncertainty_aware_plan(shocked, renewable, quantiles=(0.5, 0.8))
    for tau in (0.5, 0.8):
        assert _row(a, tau)["planned_cost_eur"] == pytest.approx(_row(b, tau)["planned_cost_eur"])
        assert _row(a, tau)["imbalance_cost_eur"] != pytest.approx(
            _row(b, tau)["imbalance_cost_eur"]
        )
