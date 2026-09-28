"""
Tests for energy_dispatch.evaluate — all synthetic, no OPSD data.

Covers the imbalance/price arithmetic, the planned-supply definition, the
plan-then-reveal separation (actual demand must never reach the
optimizer), battery on/off, aggregation, and explicit handling of
incomplete, DST, and infeasible days.
"""

import datetime as dt

import pandas as pd
import pytest

from energy_dispatch import config, evaluate, optimize

UP = config.BALANCING_PRICE_UP_EUR_PER_MWH  # 200
DOWN = config.BALANCING_PRICE_DOWN_EUR_PER_MWH  # 20


def local_day_index(date: str) -> pd.DatetimeIndex:
    """UTC hourly index covering one Europe/Madrid calendar day (23/24/25 h)."""
    day = pd.Timestamp(date).date()
    start = pd.Timestamp(day, tz=config.LOCAL_TZ)
    end = pd.Timestamp(day + dt.timedelta(days=1), tz=config.LOCAL_TZ)
    local = pd.date_range(start, end, freq="h", inclusive="left")
    return local.tz_convert("UTC")


def make_forecast_table(dates: list[str], base_load: float = 30_000.0) -> pd.DataFrame:
    """Forecast table in forecast.py's shape, with deliberately different
    forecasts so each scenario plans on its own input."""
    index = pd.DatetimeIndex(
        [ts for date in dates for ts in local_day_index(date)], name="utc_timestamp"
    )
    hour = pd.Series(index.tz_convert(config.LOCAL_TZ).hour, index=index).astype(float)
    actual = base_load + 300 * hour
    return pd.DataFrame(
        {
            "y_true": actual,
            "y_naive": actual - 900,  # under-forecast
            "y_tso": actual + 150,  # slight over-forecast
            "y_lgbm": actual - 400,
        },
        index=index,
    )


def renewable_for(table: pd.DataFrame, value: float = 3_000.0) -> pd.Series:
    return pd.Series(value, index=table.index, name="renewable_available")


@pytest.fixture
def zero_battery():
    return evaluate.NO_BATTERY


# ---------------------------------------------------------------------------
# Imbalance and pricing
# ---------------------------------------------------------------------------


def test_zero_imbalance_has_zero_cost():
    load = pd.Series([100.0, 250.0, 80.0])
    imbalance = evaluate.compute_imbalance(load, load.copy())
    assert (imbalance == 0).all()
    assert (evaluate.price_imbalance(imbalance) == 0).all()


def test_positive_imbalance_is_bought_at_the_up_price():
    imbalance = evaluate.compute_imbalance(pd.Series([110.0]), pd.Series([100.0]))
    assert imbalance.iloc[0] == 10.0  # actual above plan -> positive
    assert evaluate.price_imbalance(imbalance).iloc[0] == pytest.approx(10 * UP)


def test_negative_imbalance_is_sold_at_the_down_price_as_a_credit():
    imbalance = evaluate.compute_imbalance(pd.Series([90.0]), pd.Series([100.0]))
    assert imbalance.iloc[0] == -10.0  # actual below plan -> negative
    assert evaluate.price_imbalance(imbalance).iloc[0] == pytest.approx(-10 * DOWN)


def test_up_and_down_prices_are_not_swapped():
    imbalance = pd.Series([1.0, -1.0])
    assert evaluate.price_imbalance(imbalance).tolist() == pytest.approx([200.0, -20.0])
    # distinct custom prices make any argument mix-up visible
    custom = evaluate.price_imbalance(imbalance, price_up=7.0, price_down=3.0)
    assert custom.tolist() == pytest.approx([7.0, -3.0])


def test_compute_imbalance_rejects_misaligned_series():
    with pytest.raises(ValueError):
        evaluate.compute_imbalance(pd.Series([1.0], index=[0]), pd.Series([1.0], index=[1]))


# ---------------------------------------------------------------------------
# Planned supply matches the LP's demand-balance row
# ---------------------------------------------------------------------------


def test_planned_net_supply_counts_charging_as_negative():
    units = (config.GenerationUnit("base", 1_000, 10, None),)
    dispatch = pd.DataFrame(
        {"gen_base": [100.0], "renewable_used": [50.0], "discharge": [5.0], "charge": [30.0]}
    )
    assert evaluate.planned_net_supply(dispatch, units).iloc[0] == pytest.approx(125.0)


def test_planned_supply_equals_forecast_and_realized_is_planned_plus_imbalance():
    table = make_forecast_table(["2019-06-12"])
    result = evaluate.evaluate_day(table["y_lgbm"], table["y_true"], renewable_for(table))
    # the plan's net supply is exactly what the LP balanced against the forecast
    assert (result["planned_supply"] - result["forecast_load"]).abs().max() < 1e-3
    # so, with no re-dispatch, imbalance is the forecast error
    expected_imbalance = table["y_true"] - table["y_lgbm"]
    assert (result["imbalance"] - expected_imbalance).abs().max() < 1e-3
    assert result["realized_cost_eur"].sum() == pytest.approx(
        result["planned_cost_eur"].sum() + result["imbalance_cost_eur"].sum()
    )


def test_perfect_foresight_has_zero_imbalance():
    table = make_forecast_table(["2019-06-12"])
    result = evaluate.evaluate_day(table["y_true"], table["y_true"], renewable_for(table))
    assert result["imbalance"].abs().max() < 1e-3
    assert result["imbalance_cost_eur"].abs().sum() < 1.0  # < 1 EUR over the day


# ---------------------------------------------------------------------------
# Leakage: actual demand never reaches the optimizer
# ---------------------------------------------------------------------------


def test_plan_does_not_depend_on_actual_demand():
    table = make_forecast_table(["2019-06-12"])
    renewable = renewable_for(table)
    a = evaluate.evaluate_day(table["y_naive"], table["y_true"], renewable)
    b = evaluate.evaluate_day(table["y_naive"], table["y_true"] + 5_000, renewable)

    plan_cols = ["gen_baseload", "gen_ccgt", "gen_peaker", "charge", "discharge", "soc",
                 "curtailment", "planned_supply", "planned_cost_eur"]  # fmt: skip
    pd.testing.assert_frame_equal(a[plan_cols], b[plan_cols])
    assert not a["imbalance"].equals(b["imbalance"])


def test_optimizer_only_ever_sees_the_forecast(monkeypatch):
    table = make_forecast_table(["2019-06-12"])
    sentinel_actual = pd.Series(987_654.0, index=table.index)  # never a plausible load
    seen = []
    real_plan = optimize.plan_day_ahead_dispatch

    def spy(forecast_load, renewable_available, **kwargs):
        seen.append((forecast_load.copy(), renewable_available.copy()))
        return real_plan(forecast_load, renewable_available, **kwargs)

    monkeypatch.setattr(optimize, "plan_day_ahead_dispatch", spy)
    evaluate.evaluate_day(table["y_tso"], sentinel_actual, renewable_for(table))

    assert len(seen) == 1
    planning_load, planning_renewable = seen[0]
    pd.testing.assert_series_equal(planning_load, table["y_tso"])
    assert (planning_load != 987_654.0).all()
    assert (planning_renewable != 987_654.0).all()


def test_changing_actual_demand_leaves_forecast_scenarios_plans_unchanged():
    table = make_forecast_table(["2019-06-12"])
    shocked = table.copy()
    shocked["y_true"] += 2_000  # actual demand revealed differently

    kwargs = {"batteries": {"on": config.BATTERY}}
    base, _ = evaluate.evaluate_year(table, renewable_for(table), **kwargs)
    alt, _ = evaluate.evaluate_year(shocked, renewable_for(shocked), **kwargs)

    for method in ("seasonal_naive", "tso", "lightgbm"):
        planned_a = base.loc[base["forecast_method"] == method, "planned_cost_eur"]
        planned_b = alt.loc[alt["forecast_method"] == method, "planned_cost_eur"]
        pd.testing.assert_series_equal(planned_a, planned_b)
    # perfect foresight plans on actual demand by definition, so it moves
    pf_a = base.loc[base["forecast_method"] == "perfect_foresight", "planned_cost_eur"].sum()
    pf_b = alt.loc[alt["forecast_method"] == "perfect_foresight", "planned_cost_eur"].sum()
    assert pf_b > pf_a


# ---------------------------------------------------------------------------
# Battery on vs off
# ---------------------------------------------------------------------------


def test_battery_changes_planned_cost_but_not_imbalance(zero_battery):
    # Free renewable at hour 0 can be stored and used at hour 1 instead of
    # the 100 EUR/MWh unit (same toy system as test_optimize's power-limit test).
    units = (config.GenerationUnit("expensive", 1_000, 100, None),)
    battery = config.BatteryParams(
        power_capacity_mw=10.0, energy_capacity_mwh=1_000.0,
        round_trip_efficiency=1.0, initial_soc_mwh=0.0,
    )  # fmt: skip
    index = pd.RangeIndex(2)
    forecast = pd.Series([0.0, 50.0], index=index)
    actual = pd.Series([0.0, 60.0], index=index)  # 10 MWh under-forecast in hour 1
    renewable = pd.Series([100.0, 0.0], index=index)

    off = evaluate.evaluate_day(forecast, actual, renewable, zero_battery, units)
    on = evaluate.evaluate_day(forecast, actual, renewable, battery, units)

    assert off["planned_cost_eur"].sum() == pytest.approx(5_000.0)  # 50 MWh x 100
    assert on["planned_cost_eur"].sum() == pytest.approx(4_000.0)  # 40 MWh x 100
    pd.testing.assert_series_equal(off["imbalance"], on["imbalance"], atol=1e-4)
    assert off["imbalance_cost_eur"].sum() == pytest.approx(10 * UP)
    assert off["realized_cost_eur"].sum() == pytest.approx(7_000.0)
    assert on["realized_cost_eur"].sum() == pytest.approx(6_000.0)


# ---------------------------------------------------------------------------
# Day selection: incomplete, missing, DST, infeasible
# ---------------------------------------------------------------------------


def test_expected_local_hours_on_dst_days():
    assert evaluate.expected_local_hours(dt.date(2019, 3, 31)) == 23
    assert evaluate.expected_local_hours(dt.date(2019, 6, 12)) == 24
    assert evaluate.expected_local_hours(dt.date(2019, 10, 27)) == 25


def test_days_with_missing_or_incomplete_data_are_skipped_for_every_scenario():
    table = make_forecast_table(["2019-06-10", "2019-06-11", "2019-06-12", "2019-06-14"])
    day_11 = pd.DatetimeIndex(table.index).tz_convert(config.LOCAL_TZ).date == dt.date(2019, 6, 11)
    table.loc[table.index[day_11][5], "y_naive"] = float("nan")  # one missing forecast hour
    table = table.drop(local_day_index("2019-06-12")[3])  # one missing hour

    valid, skipped = evaluate.select_evaluable_days(table, renewable_for(table))
    assert valid == [dt.date(2019, 6, 10), dt.date(2019, 6, 14)]
    reasons = dict(zip(skipped["local_date"], skipped["reason"], strict=True))
    assert "y_naive" in reasons[dt.date(2019, 6, 11)]
    assert "incomplete horizon: 23 of 24" in reasons[dt.date(2019, 6, 12)]
    assert reasons[dt.date(2019, 6, 13)] == "no rows in forecast table"

    # the day missing only a naive value is excluded from the LightGBM
    # scenario too, so every scenario covers the same hours
    hourly, _ = evaluate.evaluate_year(
        table, renewable_for(table), batteries={"off": evaluate.NO_BATTERY}
    )
    days_per_method = hourly.groupby("forecast_method")["local_date"].unique()
    for days in days_per_method:
        assert sorted(days) == valid


def test_missing_or_negative_renewables_skip_the_day():
    table = make_forecast_table(["2019-06-10", "2019-06-11"])
    renewable = renewable_for(table)
    renewable.iloc[2] = float("nan")
    renewable.iloc[30] = -5.0
    valid, skipped = evaluate.select_evaluable_days(table, renewable)
    assert valid == []
    assert "renewable_available" in skipped["reason"].iloc[0]
    assert skipped["reason"].iloc[1] == "negative renewable availability"


def test_dst_days_are_evaluated_at_their_true_length():
    table = make_forecast_table(["2019-03-31", "2019-10-27"])
    hourly, skipped = evaluate.evaluate_year(
        table,
        renewable_for(table),
        scenarios={"lightgbm": "y_lgbm"},
        batteries={"on": config.BATTERY},
    )
    dst_days = [dt.date(2019, 3, 31), dt.date(2019, 10, 27)]
    assert not skipped["local_date"].isin(dst_days).any()  # the gap between them is reported
    hours_per_day = hourly.groupby("local_date").size()
    assert hours_per_day[dt.date(2019, 3, 31)] == 23
    assert hours_per_day[dt.date(2019, 10, 27)] == 25


def test_infeasible_day_is_reported_and_skipped_for_all_scenarios():
    units = (config.GenerationUnit("small", 40_000, 50, None),)
    table = make_forecast_table(["2019-06-10", "2019-06-11"])
    day_11 = pd.DatetimeIndex(table.index).tz_convert(config.LOCAL_TZ).date == dt.date(2019, 6, 11)
    table.loc[day_11, "y_tso"] = 1_000_000.0  # impossible to plan for

    hourly, skipped = evaluate.evaluate_year(
        table, renewable_for(table), batteries={"off": evaluate.NO_BATTERY}, generation_units=units
    )
    assert set(hourly["local_date"]) == {dt.date(2019, 6, 10)}
    assert hourly["forecast_method"].nunique() == 4
    assert len(skipped) == 1
    assert "dispatch not solved (tso" in skipped["reason"].iloc[0]


# ---------------------------------------------------------------------------
# Aggregation and comparisons
# ---------------------------------------------------------------------------


def _hourly_row(method, battery, date, forecast, actual, planned_cost):
    imbalance = actual - forecast
    imbalance_cost = UP * max(imbalance, 0) - DOWN * max(-imbalance, 0)
    return {
        "forecast_method": method, "battery": battery, "local_date": date,
        "forecast_load": forecast, "actual_load": actual, "imbalance": imbalance,
        "planned_cost_eur": planned_cost, "imbalance_cost_eur": imbalance_cost,
        "realized_cost_eur": planned_cost + imbalance_cost,
    }  # fmt: skip


def test_aggregate_results_sums_hourly_rows_into_annual_totals():
    d1, d2 = dt.date(2019, 1, 1), dt.date(2019, 1, 2)
    hourly = pd.DataFrame(
        [
            _hourly_row("naive", "off", d1, 100.0, 110.0, 1_000.0),  # +10 -> +2000
            _hourly_row("naive", "off", d1, 100.0, 95.0, 1_000.0),  # -5  -> -100
            _hourly_row("naive", "off", d2, 100.0, 100.0, 1_000.0),  # 0
            _hourly_row("pf", "off", d1, 110.0, 110.0, 1_100.0),
        ]
    )
    summary = evaluate.aggregate_results(hourly)
    naive = summary.loc[("naive", "off")]

    assert naive["n_days"] == 2
    assert naive["n_hours"] == 3
    assert naive["planned_cost_eur"] == pytest.approx(3_000.0)
    assert naive["imbalance_cost_eur"] == pytest.approx(1_900.0)
    assert naive["realized_cost_eur"] == pytest.approx(4_900.0)
    assert naive["mean_hourly_realized_cost_eur"] == pytest.approx(4_900.0 / 3)
    assert naive["upward_imbalance_mwh"] == pytest.approx(10.0)
    assert naive["downward_imbalance_mwh"] == pytest.approx(5.0)
    assert naive["total_abs_imbalance_mwh"] == pytest.approx(15.0)
    assert naive["mean_abs_imbalance_mw"] == pytest.approx(5.0)
    assert naive["forecast_mae_mw"] == pytest.approx(5.0)
    assert naive["forecast_bias_mw"] == pytest.approx(5.0 / 3)  # (10 - 5 + 0) / 3
    assert summary.loc[("pf", "off"), "imbalance_cost_eur"] == 0


def test_compare_scenarios_savings_and_percentages():
    index = pd.MultiIndex.from_tuples(
        [
            ("perfect_foresight", "off"), ("perfect_foresight", "on"),
            ("seasonal_naive", "off"), ("seasonal_naive", "on"),
            ("tso", "off"), ("tso", "on"),
            ("lightgbm", "off"), ("lightgbm", "on"),
        ],
        names=["forecast_method", "battery"],
    )  # fmt: skip
    summary = pd.DataFrame(
        {"realized_cost_eur": [900.0, 850.0, 1_200.0, 1_140.0, 950.0, 900.0, 1_000.0, 960.0]},
        index=index,
    )
    comparisons = evaluate.compare_scenarios(summary).set_index(["comparison", "scenario"])

    naive_vs_lgbm = comparisons.loc[("vs_seasonal_naive", "lightgbm / battery off")]
    assert naive_vs_lgbm["saving_eur"] == pytest.approx(200.0)  # 1200 - 1000
    assert naive_vs_lgbm["saving_pct"] == pytest.approx(200.0 / 1_200.0 * 100)

    lgbm_vs_pf = comparisons.loc[("vs_perfect_foresight", "lightgbm / battery on")]
    assert lgbm_vs_pf["saving_eur"] == pytest.approx(-110.0)  # 110 EUR above the lower bound

    battery = comparisons.loc["battery_value"]
    assert battery.loc["lightgbm / battery on", "saving_eur"] == pytest.approx(40.0)
    assert battery.loc["seasonal_naive / battery on", "saving_eur"] == pytest.approx(60.0)
    assert len(battery) == 4
    # no scenario is compared against itself
    assert not (comparisons["reference"] == comparisons.index.get_level_values("scenario")).any()
