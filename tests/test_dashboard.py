"""
Tests for the dashboard data layer (energy_dispatch.dashboard) and a smoke
test that every page of app/streamlit_app.py renders without errors.

All result files are generated here from synthetic days with the real
evaluate/sensitivity functions, so no OPSD data is involved.
"""

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from energy_dispatch import config, dashboard, evaluate, forecast, sensitivity

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
DAYS = ["2019-01-22", "2019-06-12"]


def _local_day_index(date: str) -> pd.DatetimeIndex:
    day = pd.Timestamp(date).date()
    start = pd.Timestamp(day, tz=config.LOCAL_TZ)
    end = pd.Timestamp(day + dt.timedelta(days=1), tz=config.LOCAL_TZ)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def _synthetic_inputs():
    index = pd.DatetimeIndex([t for d in DAYS for t in _local_day_index(d)], name="utc_timestamp")
    hour = pd.Series(index.tz_convert(config.LOCAL_TZ).hour, index=index)
    actual = 26_000.0 + 9_000 * hour.between(8, 21) + 4_000 * hour.between(19, 21)
    table = pd.DataFrame(
        {
            "y_true": actual,
            "y_naive": actual - 900,
            "y_tso": actual + 200,
            "y_lgbm": actual - 400,
        }
    )
    table["q10"] = table["y_lgbm"] - 800
    table["q90"] = table["y_lgbm"] + 800
    renewable = pd.Series(3_000.0, index=index, name="renewable_available")
    return table, renewable


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Real pipeline outputs for two synthetic days, written like the pipeline does."""
    table, renewable = _synthetic_inputs()
    hourly, _ = evaluate.evaluate_year(table, renewable)
    summary = evaluate.aggregate_results(hourly)
    metrics = forecast.compute_point_metrics(table, model_cols=("y_naive", "y_tso", "y_lgbm"))
    sens = sensitivity.run_sensitivity(
        table,
        renewable,
        dimensions={
            "ccgt_ramp_mw_per_hour": (1_000, 8_000),
            "ccgt_capacity_mw": (15_000, 25_000),
            "balancing_prices_eur_per_mwh": ((150, 50), (200, 20)),
        },
    )
    frames = {
        "forecast_table": table,
        "forecast_metrics": metrics,
        "evaluation_hourly": hourly,
        "scenario_summary": summary,
        "sensitivity": sens,
    }
    folder = tmp_path_factory.mktemp("processed")
    paths = {}
    for name, frame in frames.items():
        paths[name] = folder / f"{name}.parquet"
        frame.to_parquet(paths[name])
    return frames, paths


@pytest.fixture
def point_app_at(monkeypatch):
    """Point dashboard.RESULT_FILES at a folder of result files."""

    def _point(paths: dict[str, Path], demo_dir: Path | None = None):
        for name, (_, command) in list(dashboard.RESULT_FILES.items()):
            monkeypatch.setitem(dashboard.RESULT_FILES, name, (paths[name], command))
        # never pick up a real bundled copy (app/demo_results/) during tests
        monkeypatch.setattr(dashboard, "DEMO_RESULTS_DIR", demo_dir or Path("/nonexistent"))

    return _point


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


def test_missing_result_file_returns_none_with_a_how_to_message(tmp_path, point_app_at):
    point_app_at({name: tmp_path / f"{name}.parquet" for name in dashboard.RESULT_FILES})
    assert dashboard.load_result("scenario_summary") is None
    assert "python -m energy_dispatch.evaluate" in dashboard.missing_file_message(
        "scenario_summary"
    )


def test_bundled_results_are_used_when_data_processed_is_empty(tmp_path, results, point_app_at):
    _, paths = results
    demo = tmp_path / "demo_results"
    # export from the "processed" folder, then point the app at an empty one
    point_app_at(paths)
    written = dashboard.export_demo_results(demo)
    assert sorted(p.name for p in written) == sorted(p.name for p in paths.values())

    empty = {name: tmp_path / "processed" / path.name for name, path in paths.items()}
    point_app_at(empty, demo_dir=demo)
    assert dashboard.is_bundled("scenario_summary")
    pd.testing.assert_frame_equal(
        dashboard.load_result("scenario_summary"), results[0]["scenario_summary"]
    )


def test_headline_metrics_match_compare_scenarios(results):
    frames, _ = results
    summary = frames["scenario_summary"]
    head = dashboard.headline_metrics(summary)
    realized = summary["realized_cost_eur"]
    naive, lgbm = realized[("seasonal_naive", "off")], realized[("lightgbm", "off")]
    tso = realized[("tso", "off")]

    assert head["lightgbm_vs_naive"]["saving_eur"] == pytest.approx(naive - lgbm)
    assert head["lightgbm_vs_naive"]["saving_pct"] == pytest.approx((naive - lgbm) / naive * 100)
    assert head["tso_vs_lightgbm"]["saving_eur"] == pytest.approx(lgbm - tso)
    assert head["battery_lightgbm"]["saving_eur"] == pytest.approx(
        lgbm - realized[("lightgbm", "on")]
    )


def test_dispatch_day_is_one_local_day_with_renewable_availability(results):
    frames, _ = results
    rows = dashboard.dispatch_day(
        frames["evaluation_hourly"], "lightgbm", "on", dt.date(2019, 6, 12)
    )
    assert len(rows) == 24
    assert str(rows.index.tz) == config.LOCAL_TZ
    assert rows.index[0].hour == 0
    assert (rows["renewable_available"] - 3_000.0).abs().max() < 1e-6
    totals = dashboard.dispatch_totals(rows)
    assert totals["realized_cost_eur"] == pytest.approx(
        totals["planned_cost_eur"] + totals["imbalance_cost_eur"]
    )


def test_scenario_table_lists_the_eight_scenarios_in_order(results):
    frames, _ = results
    table = dashboard.scenario_table(frames["scenario_summary"])
    assert len(table) == 8
    assert table["forecast"].iloc[0] == dashboard.FORECAST_LABELS["perfect_foresight"]
    assert list(table["battery"].iloc[:2]) == ["off", "on"]


def test_cost_above_reference_splits_planned_and_imbalance(results):
    frames, _ = results
    summary = frames["scenario_summary"]
    above = dashboard.cost_above_reference(summary)
    assert dashboard.FORECAST_LABELS["perfect_foresight"] not in above.index
    lgbm = above.loc[dashboard.FORECAST_LABELS["lightgbm"]]
    # the imbalance cost of perfect foresight is ~0, so the two parts add up
    assert lgbm["realized_above_reference_eur"] == pytest.approx(
        lgbm["planned_above_reference_eur"] + lgbm["imbalance_cost_eur"], abs=1.0
    )


def test_sensitivity_view_flags_points_on_fewer_days(results):
    frames, _ = results
    capacity = dashboard.sensitivity_view(frames["sensitivity"], "ccgt_capacity_mw")
    assert list(capacity["value"]) == [15_000, 25_000]
    # 7,000 + 15,000 + 10,000 MW + 3,000 renewables + 1,000 battery < 39,000 MW peak
    assert not capacity.loc[0, "full_sample"]
    assert capacity.loc[1, "full_sample"] and capacity.loc[1, "is_baseline"]


def test_what_if_baseline_reproduces_the_stored_evaluation(results):
    frames, _ = results
    hourly = frames["evaluation_hourly"]
    day = dt.date(2019, 6, 12)
    out = dashboard.what_if_day(hourly, day, "lightgbm", sensitivity.BASELINE_SYSTEM)
    for battery in ("off", "on"):
        stored = dashboard.dispatch_totals(dashboard.dispatch_day(hourly, "lightgbm", battery, day))
        assert out[battery]["realized_cost_eur"] == pytest.approx(
            stored["realized_cost_eur"], rel=1e-9
        )


def test_system_from_overrides_combines_several_parameters():
    system = dashboard.system_from_overrides(
        {"battery_power_mw": 2_000, "ccgt_ramp_mw_per_hour": 2_000, "peaker_cost_eur_per_mwh": 250}
    )
    units = {u.name: u for u in system.generation_units}
    assert system.battery.power_capacity_mw == 2_000
    assert units["ccgt"].ramp_limit_mw_per_hour == 2_000
    assert units["peaker"].marginal_cost_eur_per_mwh == 250
    assert sensitivity.BASELINE_SYSTEM.battery.power_capacity_mw == 1_000  # untouched


# ---------------------------------------------------------------------------
# App smoke test
# ---------------------------------------------------------------------------

PAGES = ["Overview", "Forecast", "Dispatch", "Cost & forecast value", "Storage sensitivity",
         "What-if"]  # fmt: skip


def _app():
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    streamlit.cache_data.clear()
    return AppTest.from_file(str(APP), default_timeout=120)


def test_every_page_renders_with_results(results, point_app_at):
    _, paths = results
    point_app_at(paths)
    app = _app()
    app.run()
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page, app.exception)
        assert not app.warning or page == "Storage sensitivity", (page, app.warning)

    # What-if: submitting the form solves the LP and shows the comparison
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3


def test_every_page_explains_missing_results(tmp_path, point_app_at):
    point_app_at({name: tmp_path / f"{name}.parquet" for name in dashboard.RESULT_FILES})
    app = _app()
    app.run()
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, (page, app.exception)
        if page != "Overview":
            assert any("python -m energy_dispatch" in w.value for w in app.warning), page
