"""
Tests for the forecast diagnostics: the Diebold-Mariano test, the error
breakdown, SHAP importance, the model fitted for explanation, and the
realized-cost tests in energy_dispatch.diagnostics. All synthetic.
"""

import datetime as dt
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from energy_dispatch import config, diagnostics, features, forecast, planning

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


def test_dm_without_lags_is_the_classic_t_ratio():
    d = pd.Series(RNG.normal(0.3, 1.0, 500))
    result = forecast.dm_test_on_losses(d, pd.Series(0.0, index=d.index), max_lag=0)
    expected = d.mean() / np.sqrt(d.var(ddof=0) / len(d))
    assert result["dm_statistic"] == pytest.approx(expected)
    assert result["p_value"] == pytest.approx(2 * (1 - NormalDist().cdf(abs(expected))))


def test_dm_detects_a_clearly_better_forecast_and_reports_the_winner():
    errors_a = pd.Series(RNG.normal(0, 100, 2_000))
    errors_b = 2 * errors_a + pd.Series(RNG.normal(0, 50, 2_000))
    for loss in ("squared", "absolute"):
        result = forecast.diebold_mariano_test(errors_a, errors_b, loss=loss)
        assert result["lower_loss"] == "a"
        assert result["mean_loss_difference"] < 0
        assert result["p_value"] < 1e-6


def test_dm_identical_and_constant_differences():
    errors = pd.Series(RNG.normal(0, 1, 100))
    same = forecast.diebold_mariano_test(errors, errors)
    assert same["dm_statistic"] == 0 and same["p_value"] == 1
    shifted = forecast.dm_test_on_losses(errors * 0 + 1.0, errors * 0 + 2.0)
    assert shifted["p_value"] == 0 and shifted["lower_loss"] == "a"


def test_dm_hac_variance_widens_the_test_for_autocorrelated_losses():
    # AR(1) loss differential with a small positive mean
    d = np.zeros(3_000)
    for t in range(1, d.size):
        d[t] = 0.9 * d[t - 1] + RNG.normal()
    d = pd.Series(d + 0.3)
    zero = pd.Series(0.0, index=d.index)
    naive = forecast.dm_test_on_losses(d, zero, max_lag=0)
    robust = forecast.dm_test_on_losses(d, zero, max_lag=168)
    assert abs(robust["dm_statistic"]) < abs(naive["dm_statistic"])
    assert robust["p_value"] > naive["p_value"]


def test_dm_rejects_unknown_loss():
    with pytest.raises(ValueError):
        forecast.diebold_mariano_test(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), loss="huber")


# ---------------------------------------------------------------------------
# Error breakdown
# ---------------------------------------------------------------------------


def _local_days(*dates):
    out = []
    for date in dates:
        day = pd.Timestamp(date).date()
        start = pd.Timestamp(day, tz=config.LOCAL_TZ)
        end = pd.Timestamp(day + dt.timedelta(days=1), tz=config.LOCAL_TZ)
        out += list(pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC"))
    return pd.DatetimeIndex(out, name="utc_timestamp")


def test_error_breakdown_by_hour_and_day_type():
    # 2019-01-01 holiday (New Year), 2019-01-05 Saturday, 2019-01-07 Monday
    index = _local_days("2019-01-01", "2019-01-05", "2019-01-07")
    local_hour = index.tz_convert(config.LOCAL_TZ).hour
    actual = pd.Series(30_000.0, index=index)
    table = pd.DataFrame(
        {
            "y_true": actual,
            "y_naive": actual - 500,  # always 500 MW too low
            "y_tso": actual,
            "y_lgbm": actual - 100 * (local_hour == 18),  # wrong only at 18:00
        },
        index=index,
    )
    out = forecast.error_breakdown(table)
    lgbm_hour = out[(out["slice"] == "hour") & (out["model"] == "y_lgbm")].set_index("group")
    assert lgbm_hour.loc["18", "mae"] == pytest.approx(100)
    assert lgbm_hour.drop(index="18")["mae"].abs().max() == 0
    naive_days = out[(out["slice"] == "day_type") & (out["model"] == "y_naive")].set_index("group")
    assert set(naive_days.index) == {"holiday", "weekend", "weekday"}
    assert (naive_days["n"] == 24).all()
    assert (naive_days["bias"] == 500).all()  # positive = under-forecast


# ---------------------------------------------------------------------------
# SHAP importance and the explained model
# ---------------------------------------------------------------------------


def test_shap_importance_ranks_the_driving_feature_first():
    lgb = pytest.importorskip("lightgbm")
    X = pd.DataFrame({"signal": RNG.normal(size=2_000), "noise": RNG.normal(size=2_000)})
    y = 3 * X["signal"] + RNG.normal(scale=0.1, size=2_000)
    model = lgb.LGBMRegressor(n_estimators=50, verbose=-1).fit(X, y)
    table = forecast.shap_feature_importance(model, X, max_rows=500)
    assert table["feature"].iloc[0] == "signal"
    assert table["share_pct"].sum() == pytest.approx(100)
    assert table["mean_abs_shap_mw"].iloc[0] > 10 * table["mean_abs_shap_mw"].iloc[1]


def test_explanation_model_never_sees_test_period_targets():
    pytest.importorskip("lightgbm")
    index = pd.date_range("2018-12-01", "2019-01-10 23:00", freq="h", tz="UTC")
    hour = index.tz_convert(config.LOCAL_TZ).hour
    matrix = pd.DataFrame(
        {
            config.LOAD_ACTUAL_COL: 30_000 + 100 * hour,
            "hour_feature": hour,
            "origin_utc": features.compute_forecast_origin(index),
        },
        index=index,
    )
    local_date = index.tz_convert(config.LOCAL_TZ).normalize()
    in_test = local_date >= pd.Timestamp("2019-01-01", tz=config.LOCAL_TZ)
    matrix.loc[in_test, config.LOAD_ACTUAL_COL] = 1e9  # poison every test-period target

    model, X_test = forecast.fit_model_for_explanation(
        matrix, test_start="2019-01-01", test_end="2019-01-10"
    )
    assert list(X_test.columns) == ["hour_feature"]
    assert (X_test.index.tz_convert(config.LOCAL_TZ).normalize() >= local_date[in_test][0]).all()
    assert model.predict(X_test).max() < 40_000  # the poisoned targets were never trained on


# ---------------------------------------------------------------------------
# Realized-cost tests
# ---------------------------------------------------------------------------


def test_cost_dm_table_compares_hourly_realized_cost():
    index = pd.date_range("2019-01-01", periods=500, freq="h", tz="UTC")
    base = pd.Series(RNG.normal(1_000_000, 50_000, 500), index=index)
    rows = []
    for method, cost in (("lightgbm", base - 10_000), ("seasonal_naive", base)):
        rows.append(pd.DataFrame({"forecast_method": method, "battery": "off",
                                  "realized_cost_eur": cost}, index=index))  # fmt: skip
    table = diagnostics.cost_dm_table(pd.concat(rows), pairs=(("lightgbm", "seasonal_naive"),))
    row = table.iloc[0]
    assert row["mean_loss_difference"] == pytest.approx(-10_000)
    assert row["lower_loss"] == "a" and row["p_value"] == 0


def test_planning_dm_table_matches_the_hand_computed_saving():
    # flat day: point forecast 30,000, actual +300 MW, q90 +600 MW, CCGT marginal
    index = _local_days("2019-06-12")
    point = pd.Series(30_000.0, index=index)
    table = pd.DataFrame(
        {"y_true": point + 300, "y_naive": point, "y_tso": point, "y_lgbm": point,
         "q10": point - 600, "q90": point + 600}
    )  # fmt: skip
    renewable = pd.Series(3_000.0, index=index)
    out = diagnostics.planning_dm_table(table, renewable)

    tau = planning.theoretical_plan_quantile()
    z = NormalDist().inv_cdf
    uplift = z(tau) / z(0.9) * 600
    tau_cost = uplift * 60 - (uplift - 300) * 20  # extra CCGT, surplus sold back
    point_cost = 300 * 200  # shortfall bought
    assert out["total_saving_eur"].iloc[0] == pytest.approx(24 * (point_cost - tau_cost), abs=1)
    assert out["model_a"].iloc[0] == planning.plan_name(tau)
