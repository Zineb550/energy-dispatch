"""
Streamlit dashboard (spec section 9).

Presentation layer only. Every number comes from the result files the
pipeline writes to data/processed/ (read through energy_dispatch.dashboard);
the only live computation is the What-if page, which re-plans one day
through evaluate.evaluate_day. No forecasting, optimization, evaluation or
sensitivity logic lives in this file.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import datetime as dt
import inspect
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import energy_dispatch  # noqa: F401  (installed with `pip install -e .`)
except ModuleNotFoundError:  # e.g. a hosted deployment that only installs requirements.txt
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from energy_dispatch import config, dashboard, sensitivity  # noqa: E402

st.set_page_config(page_title="Energy dispatch: forecast value & storage", layout="wide")

# ---------------------------------------------------------------------------
# Fixed colors per entity (reference categorical palette; an entity keeps its
# color on every page).
# ---------------------------------------------------------------------------

NEUTRAL = "#8a8984"  # actual demand: neutral ink, not a series hue
FORECAST_COLORS = {
    "y_lgbm": "#2a78d6",  # blue
    "y_naive": "#eb6834",  # orange
    "y_tso": "#1baf7a",  # aqua
}
METHOD_COLORS = {
    "lightgbm": "#2a78d6",
    "seasonal_naive": "#eb6834",
    "tso": "#1baf7a",
    "perfect_foresight": "#4a3aa7",  # violet
}
SUPPLY_COLORS = {
    "gen_baseload": "#4a3aa7",  # violet
    "gen_ccgt": "#2a78d6",  # blue
    "gen_peaker": "#e34948",  # red
    "renewable_used": "#008300",  # green
    "discharge": "#eda100",  # yellow
    "charge": "#e87ba4",  # magenta
}
SUPPLY_LABELS = {
    "gen_baseload": "Baseload",
    "gen_ccgt": "CCGT",
    "gen_peaker": "Peaker",
    "renewable_used": "Solar + wind used",
    "discharge": "Battery discharge",
    "charge": "Battery charge",
}
BATTERY_COLORS = {"off": NEUTRAL, "on": "#eda100"}


def _full_width(element) -> dict:
    """Full-width keyword for a chart/table call: `width="stretch"` on recent
    Streamlit (where use_container_width is deprecated), the older
    `use_container_width=True` otherwise."""
    param = inspect.signature(element).parameters.get("width")
    if param is not None and isinstance(param.default, str):
        return {"width": "stretch"}
    return {"use_container_width": True}


CHART_WIDTH = _full_width(st.plotly_chart)
TABLE_WIDTH = _full_width(st.dataframe)

ILLUSTRATIVE_NOTE = (
    "Simulated 2019 system. Capacities, marginal costs, ramp limits, battery "
    "parameters and balancing prices are illustrative assumptions, not a "
    "calibrated model of the Spanish electricity market."
)


# ---------------------------------------------------------------------------
# Formatting and data loading
# ---------------------------------------------------------------------------


def eur(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    sign = "−" if value < 0 else ""
    value = abs(value)
    if value >= 1e9:
        return f"{sign}€{value / 1e9:,.3f}B"
    if value >= 1e6:
        return f"{sign}€{value / 1e6:,.1f}M"
    if value >= 1e3:
        return f"{sign}€{value / 1e3:,.1f}k"
    return f"{sign}€{value:,.0f}"


def mwh(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 1e6:
        return f"{value / 1e6:,.2f} TWh"
    if abs(value) >= 1e3:
        return f"{value / 1e3:,.1f} GWh"
    return f"{value:,.0f} MWh"


def _mtime(name: str) -> float | None:
    path = dashboard.resolve_result_path(name)
    return path.stat().st_mtime if path is not None else None


@st.cache_data(show_spinner=False)
def _load(name: str, mtime: float | None) -> pd.DataFrame | None:
    # mtime is part of the cache key, so regenerated files are picked up.
    return dashboard.load_result(name)


def load(name: str) -> pd.DataFrame | None:
    return _load(name, _mtime(name))


def require(name: str) -> pd.DataFrame | None:
    """Load a result file, or show how to generate it and return None."""
    frame = load(name)
    if frame is None:
        st.warning(dashboard.missing_file_message(name))
    return frame


def show(fig: go.Figure, title: str | None = None, hovermode: str = "x unified") -> None:
    """Render a figure with its title as text above it, so a horizontal
    legend row never collides with the title."""
    if title:
        st.markdown(f"**{title}**")
    fig.update_layout(
        title=None,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "traceorder": "normal"},
        hovermode=hovermode,
    )
    st.plotly_chart(fig, **CHART_WIDTH)


def delta_text(difference: float) -> str | None:
    """Delta for st.metric, or None (no arrow) when nothing changed."""
    return None if abs(difference) < 0.5 else eur(difference) + " vs baseline"


def pick_day(label: str, days: list[dt.date], key: str, default: dt.date | None = None):
    default = default if default in days else days[len(days) // 2]
    day = st.date_input(label, value=default, min_value=days[0], max_value=days[-1], key=key)
    if day not in set(days):
        st.warning(f"No results for {day}; showing {default} instead.")
        return default
    return day


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    st.title("Day-ahead demand forecasting & dispatch optimization")
    st.markdown(
        "A grid operator must commit tomorrow's hourly supply today. This project "
        "forecasts Spain's hourly electricity demand (OPSD data), plans the next "
        "day's generation and battery dispatch with a linear program built on that "
        "forecast, and then settles the plan against the demand that actually "
        "occurs. It asks two questions: **does a better forecast lower the cost of "
        "operating the system**, and **how does the value of battery storage depend "
        "on the flexibility of the generation fleet?**"
    )
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR; bgcolor="transparent";
            node [shape=box, style="rounded", fontname="sans-serif", fontsize=11];
            edge [color="#8a8984"];
            f [label="Forecast for D+1\\n(made 10:00 on day D)"];
            p [label="Day-ahead dispatch plan\\n(LP: min. planned cost)"];
            a [label="Actual demand\\nrevealed on D+1"];
            i [label="Imbalance =\\nactual − planned supply"];
            r [label="Realized cost =\\nplanned + imbalance cost"];
            f -> p -> a -> i -> r;
        }
        """
    )

    summary = require("scenario_summary")
    if summary is not None:
        head = dashboard.headline_metrics(summary)
        st.subheader("Headline results (simulated 2019 system)")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "LightGBM vs seasonal naive", eur(head["lightgbm_vs_naive"]["saving_eur"]) + "/yr"
        )
        c1.caption(
            f"{head['lightgbm_vs_naive']['saving_pct']:.2f}% lower realized cost "
            "(relative to the naive scenario)"
        )
        c2.metric("TSO vs LightGBM", eur(head["tso_vs_lightgbm"]["saving_eur"]) + "/yr")
        c2.caption(
            f"{head['tso_vs_lightgbm']['saving_pct']:.2f}% lower realized cost for the TSO "
            "forecast (relative to the TSO scenario)"
        )
        c3.metric(
            "Battery, baseline fleet (LightGBM plan)",
            eur(head["battery_lightgbm"]["saving_eur"]) + "/yr",
        )
        c3.caption(
            f"{head['battery_lightgbm']['saving_pct']:.2f}% of realized cost without the battery"
        )
        st.caption(ILLUSTRATIVE_NOTE)

    st.subheader("How to read these results")
    st.markdown(
        "- **Forecast accuracy is not economic value.** A forecast is worth what it "
        "changes: the dispatch plan and the imbalance settled afterwards.\n"
        "- **The plan is fixed.** There is no real-time re-dispatch, so imbalance = "
        "actual − forecast whether or not the battery exists. The battery cannot "
        "absorb forecast error here; its value comes from reshaping the planned "
        "dispatch.\n"
        "- **Storage value depends on the fleet.** See *Storage sensitivity*.\n"
        "- **Perfect foresight** plans on the actual load: a theoretical reference, "
        "not an achievable forecast.\n"
        "- **Simplifications:** the forecasting model uses observed target-hour "
        "weather as a proxy for weather forecasts, and dispatch uses observed "
        "solar and wind output (perfect renewable foresight), which isolates the "
        "demand-forecast question. 2019-12-24 is excluded (missing solar/wind data)."
    )


def page_forecast() -> None:
    st.header("Forecast")
    table = require("forecast_table")
    if table is None:
        return
    days = dashboard.available_days(table)
    day = pick_day("Day", days, key="forecast_day", default=dt.date(2019, 6, 12))
    rows = dashboard.forecast_day(table, day)

    low, high = dashboard.quantile_columns()
    fig = go.Figure()
    if low in rows and high in rows:
        fig.add_trace(
            go.Scatter(x=rows.index, y=rows[high], line={"width": 0}, showlegend=False,
                       hoverinfo="skip")
        )  # fmt: skip
        fig.add_trace(
            go.Scatter(x=rows.index, y=rows[low], fill="tonexty", line={"width": 0},
                       fillcolor="rgba(42,120,214,0.15)",
                       name="LightGBM 80% interval (q10–q90, conformal-calibrated)",
                       hoverinfo="skip")
        )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=rows.index, y=rows["y_true"], name="Actual demand",
                   line={"color": NEUTRAL, "width": 3})
    )  # fmt: skip
    for col, label in dashboard.FORECAST_COLUMN_LABELS.items():
        if col in rows:
            fig.add_trace(
                go.Scatter(x=rows.index, y=rows[col], name=label,
                           line={"color": FORECAST_COLORS[col], "width": 2})
            )  # fmt: skip
    fig.update_layout(yaxis_title="MW")
    show(fig, f"Hourly demand, {day} (local time)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Point-forecast error on {day}**")
        st.dataframe(dashboard.forecast_metrics_for(rows).round(2), **TABLE_WIDTH)
    with c2:
        st.markdown("**Full test period (2019)**")
        metrics = load("forecast_metrics")
        if metrics is not None:
            metrics = metrics.rename(index=dashboard.FORECAST_COLUMN_LABELS)
        else:
            metrics = dashboard.forecast_metrics_for(table)
        st.dataframe(metrics.round(2), **TABLE_WIDTH)

    interval = dashboard.interval_metrics_for(table)
    if interval is not None:
        st.markdown(
            f"**Prediction interval, full period:** empirical coverage "
            f"{interval['coverage']:.1%} against an {interval['target_coverage']:.0%} "
            f"target, average width {interval['avg_interval_width']:,.0f} MW. The raw "
            f"quantile models are corrected with split-conformal calibration."
        )
    st.caption(
        "Seasonal naive = same hour one week earlier. TSO = ENTSO-E day-ahead load "
        "forecast. LightGBM uses observed target-hour weather as a proxy for a "
        "weather forecast, which flatters its accuracy somewhat."
    )


def page_dispatch() -> None:
    st.header("Dispatch")
    hourly = require("evaluation_hourly")
    if hourly is None:
        return
    methods = [m for m in dashboard.FORECAST_LABELS if m in set(hourly["forecast_method"])]
    days = sorted(set(hourly["local_date"]))
    c1, c2, c3 = st.columns([2, 1, 2])
    default = methods.index("lightgbm") if "lightgbm" in methods else 0
    method = c1.selectbox(
        "Planning forecast", methods, index=default, format_func=dashboard.FORECAST_LABELS.get
    )
    battery = c2.radio("Battery", ["on", "off"], horizontal=True)
    with c3:
        day = pick_day("Day", days, key="dispatch_day", default=dt.date(2019, 6, 12))

    rows = dashboard.dispatch_day(hourly, method, battery, day)
    totals = dashboard.dispatch_totals(rows)

    m = st.columns(4)
    m[0].metric("Planned generation cost", eur(totals["planned_cost_eur"]))
    m[1].metric("Imbalance cost (after actuals)", eur(totals["imbalance_cost_eur"]))
    m[2].metric("Realized cost", eur(totals["realized_cost_eur"]))
    m[3].metric("Peaker generation", mwh(totals["peaker_mwh"]))
    m = st.columns(4)
    m[0].metric("Solar + wind available", mwh(totals["renewable_available_mwh"]))
    m[1].metric("Curtailed", mwh(totals["curtailment_mwh"]))
    m[2].metric("Battery charge", mwh(totals["battery_charge_mwh"]))
    m[3].metric("Battery discharge", mwh(totals["battery_discharge_mwh"]))

    fig = go.Figure()
    for col in ("gen_baseload", "gen_ccgt", "gen_peaker", "renewable_used", "discharge"):
        fig.add_trace(
            go.Bar(x=rows.index, y=rows[col], name=SUPPLY_LABELS[col],
                   marker_color=SUPPLY_COLORS[col])
        )  # fmt: skip
    fig.add_trace(
        go.Bar(x=rows.index, y=-rows["charge"], name=SUPPLY_LABELS["charge"],
               marker_color=SUPPLY_COLORS["charge"])
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=rows.index, y=rows["forecast_load"], name="Forecast demand (planned for)",
                   line={"color": "#52514e", "width": 2})
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=rows.index, y=rows["actual_load"], name="Actual demand (revealed later)",
                   line={"color": NEUTRAL, "width": 2, "dash": "dash"})
    )  # fmt: skip
    fig.update_layout(barmode="relative", bargap=0.15, yaxis_title="MW (charging below zero)")
    show(
        fig,
        f"Planned hourly supply, {day}: plan built on {dashboard.FORECAST_LABELS[method]}, "
        f"battery {battery}",
    )
    st.caption(
        "The stacked bars meet the solid forecast line in every hour: the plan is "
        "built only from the forecast. The gap to the dashed actual line is the "
        "imbalance, bought at €200/MWh when actual demand is higher and sold at "
        "€20/MWh when it is lower."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=rows.index, y=rows["renewable_used"], name="Used",
                   marker_color=SUPPLY_COLORS["renewable_used"])
        )  # fmt: skip
        fig.add_trace(
            go.Bar(x=rows.index, y=rows["curtailment"], name="Curtailed", marker_color=NEUTRAL)
        )
        fig.update_layout(barmode="stack", bargap=0.15, yaxis_title="MW")
        show(fig, "Solar + wind: used vs curtailed")
    with c2:
        if battery == "on":
            fig = go.Figure(
                go.Scatter(x=rows.index, y=rows["soc"], name="State of charge",
                           line={"color": SUPPLY_COLORS["discharge"], "width": 2})
            )  # fmt: skip
            fig.update_layout(
                yaxis_title="MWh", yaxis_range=[0, config.BATTERY.energy_capacity_mwh * 1.05]
            )
            show(fig, "Battery state of charge (end of hour)")
        else:
            st.info("Battery is off in this scenario, so there is no state of charge to show.")


def page_cost() -> None:
    st.header("Cost & forecast value")
    summary = require("scenario_summary")
    if summary is None:
        return
    st.caption(ILLUSTRATIVE_NOTE)

    table = dashboard.scenario_table(summary)
    st.markdown("**The eight scenarios (whole evaluation period)**")
    st.dataframe(
        table.rename(
            columns={
                "forecast": "Planning forecast",
                "battery": "Battery",
                "planned_cost_eur": "Planned cost (€)",
                "imbalance_cost_eur": "Imbalance cost (€)",
                "realized_cost_eur": "Realized cost (€)",
                "total_abs_imbalance_mwh": "Imbalance energy (MWh)",
                "forecast_mae_mw": "Forecast MAE (MW)",
                "n_days": "Days",
                "n_hours": "Hours",
            }  # fmt: skip
        ).style.format(
            {
                "Planned cost (€)": "{:,.0f}",
                "Imbalance cost (€)": "{:,.0f}",
                "Realized cost (€)": "{:,.0f}",
                "Imbalance energy (MWh)": "{:,.0f}",
                "Forecast MAE (MW)": "{:,.1f}",
            }  # fmt: skip
        ),
        **TABLE_WIDTH,
        hide_index=True,
    )

    above = dashboard.cost_above_reference(summary, battery="off")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=above.index, y=above["planned_above_reference_eur"] / 1e6,
                   name="Planned cost above perfect foresight", marker_color=NEUTRAL)
        )  # fmt: skip
        fig.add_trace(
            go.Bar(x=above.index, y=above["imbalance_cost_eur"] / 1e6, name="Imbalance cost",
                   marker_color="#e34948")
        )  # fmt: skip
        fig.update_layout(barmode="relative", bargap=0.35, yaxis_title="€ million / year")
        show(fig, "Realized cost above the perfect-foresight reference (battery off)",
             hovermode="closest")  # fmt: skip
    with c2:
        fig = go.Figure()
        for label, row in above.iterrows():
            method = next(k for k, v in dashboard.FORECAST_LABELS.items() if v == label)
            fig.add_trace(
                go.Scatter(x=[row["forecast_mae_mw"]],
                           y=[row["realized_above_reference_eur"] / 1e6],
                           mode="markers+text", text=[label], textposition="top center",
                           marker={"size": 12, "color": METHOD_COLORS[method]}, name=label)
            )  # fmt: skip
        fig.update_layout(
            xaxis_title="Forecast MAE (MW)", yaxis_title="€ million / year", showlegend=False
        )
        fig.update_xaxes(range=[0, above["forecast_mae_mw"].max() * 1.3])
        fig.update_yaxes(range=[0, above["realized_above_reference_eur"].max() / 1e6 * 1.2])
        show(fig, "Forecast error vs realized cost above the reference", hovermode="closest")
    st.markdown(
        "Most of the difference between forecasts is **imbalance cost** paid after actual "
        "demand is revealed, not planned generation cost. A forecast that is slightly too "
        "high can even have a higher planned cost while being cheaper overall."
    )

    head = dashboard.headline_metrics(summary)
    c1, c2 = st.columns(2)
    c1.metric(
        "Replacing seasonal naive with LightGBM",
        eur(head["lightgbm_vs_naive"]["saving_eur"]) + "/yr",
    )
    c1.caption(f"{head['lightgbm_vs_naive']['saving_pct']:.2f}% lower realized cost")
    c2.metric(
        "TSO forecast instead of LightGBM", eur(head["tso_vs_lightgbm"]["saving_eur"]) + "/yr"
    )
    c2.caption(f"{head['tso_vs_lightgbm']['saving_pct']:.2f}% lower realized cost")

    st.markdown("**Battery value under each planning forecast**")
    battery = dashboard.battery_value_by_forecast(summary)
    st.dataframe(
        battery.rename(
            columns={
                "battery_savings_eur": "Battery savings (€)",
                "planned_cost_savings_eur": "…from planned cost (€)",
                "imbalance_cost_savings_eur": "…from imbalance cost (€)",
                "battery_savings_pct": "Savings (% of cost without battery)",
            }
        ).style.format("{:,.2f}"),
        **TABLE_WIDTH,
    )
    st.caption(
        "The battery saves about the same with every forecast, and none of it comes from "
        "imbalance cost: with a fixed day-ahead plan, imbalance equals the forecast error "
        "whether or not the battery exists."
    )
    section_planning()


def section_planning() -> None:
    """Planning on a quantile of the calibrated LightGBM forecast (planning.py)."""
    results = load("planning")
    if results is None:
        return  # optional extension: hidden until `python -m energy_dispatch.planning` has run
    plans, references = dashboard.planning_view(results)
    tau_star = results.attrs.get("theoretical_quantile") or float(
        plans.loc[plans["is_theoretical_optimum"], "target_quantile"].iloc[0]
    )

    st.subheader("Planning on the forecast interval")
    st.markdown(
        "A shortfall is bought at €200/MWh while a surplus is sold at €20/MWh, so under-"
        "planning costs more than over-planning. When CCGT is the marginal unit, the "
        f"cost-minimizing plan is the **{tau_star:.0%} quantile** of the demand forecast "
        "(newsvendor rule: (200 − 60) / ((200 − 60) + (60 − 20))), not the point forecast. "
        "Each bar plans on a quantile of the LightGBM forecast, interpolated from its "
        "conformal-calibrated q10–q90 interval, and shows the realized-cost saving "
        "against planning on the point forecast (battery off)."
    )
    labels = [
        f"τ* = {t:.3f}" if star else f"τ = {t:.2f}"
        for t, star in zip(plans["target_quantile"], plans["is_theoretical_optimum"], strict=True)
    ]
    colors = ["#2a78d6" if star else "#9bc0ea" for star in plans["is_theoretical_optimum"]]
    fig = go.Figure(
        go.Bar(x=labels, y=plans["saving_vs_point_plan_eur"] / 1e6, marker_color=colors,
               text=[f"{v / 1e6:,.1f}" for v in plans["saving_vs_point_plan_eur"]],
               textposition="outside", name="Saving vs point-forecast plan")
    )  # fmt: skip
    tso = references[references["forecast_method"] == "tso"]
    if not tso.empty:
        fig.add_hline(
            y=tso["saving_vs_point_plan_eur"].iloc[0] / 1e6, line_dash="dash",
            line_color=METHOD_COLORS["tso"],
            annotation_text="TSO point forecast", annotation_position="top left",
        )  # fmt: skip
    fig.update_layout(
        yaxis_title="€ million / year", xaxis_title="Planning quantile",
        xaxis_type="category", showlegend=False,
    )  # fmt: skip
    show(fig, "Realized-cost saving by planning quantile", hovermode="closest")

    star = plans[plans["is_theoretical_optimum"]]
    if not star.empty:
        row = star.iloc[0]
        st.markdown(
            f"Planning on τ* raises the plan by {row['mean_uplift_mw']:,.0f} MW on average and "
            f"changes realized cost by **{eur(row['saving_vs_point_plan_eur'])}/yr "
            f"({row['saving_vs_point_plan_pct']:.2f}%)** against the point-forecast plan, "
            "with the same forecast model: only the decision rule changes."
        )
    st.caption(
        "τ* is fixed from the prices before looking at any result, so its bar is an honest "
        "out-of-sample test. The other bars are evaluated on the same 2019 test year; picking "
        "the best of them afterwards would be selecting on the test set."
    )
    with st.expander("Table view"):
        st.dataframe(results, **TABLE_WIDTH, hide_index=True)


def page_sensitivity() -> None:
    st.header("Storage sensitivity")
    results = require("sensitivity")
    if results is None:
        return
    st.markdown(
        "Storage value depends strongly on the flexibility of the generation fleet. In "
        "the baseline system, flexible CCGT generation limits the amount of expensive "
        "peaker generation that storage can displace. Tightening the CCGT ramp "
        "constraint creates a substantially larger opportunity for storage."
    )
    params = [p for p in dashboard.SENSITIVITY_LABELS if p in set(results["parameter"])]
    methods = sorted(set(results["forecast_method"]))
    c1, c2 = st.columns([3, 1])
    parameter = c1.selectbox("Parameter varied (all others at baseline)", params,
                             format_func=dashboard.SENSITIVITY_LABELS.get)  # fmt: skip
    method = c2.selectbox("Planning forecast", methods, format_func=dashboard.FORECAST_LABELS.get)
    view = dashboard.sensitivity_view(results, parameter, method)

    x = view["value_label"]
    colors = ["#2a78d6" if b else "#9bc0ea" for b in view["is_baseline"]]
    patterns = ["" if full else "/" for full in view["full_sample"]]
    c1, c2 = st.columns(2)
    for column, title, scale, axis, container in (
        ("battery_savings_eur", "Battery savings", 1e6, "€ million / year", c1),
        ("battery_savings_pct", "Battery savings, % of realized cost without battery", 1, "%", c2),
    ):
        fig = go.Figure(
            go.Bar(x=x, y=view[column] / scale, marker_color=colors,
                   marker_pattern_shape=patterns, name=title,
                   text=[f"{v / scale:,.2f}" for v in view[column]], textposition="outside")
        )  # fmt: skip
        fig.update_layout(
            title=title, xaxis_title=dashboard.SENSITIVITY_LABELS[parameter], yaxis_title=axis,
            xaxis_type="category", showlegend=False, hovermode="closest",
        )  # fmt: skip
        with container:
            st.plotly_chart(fig, **CHART_WIDTH)
    st.caption("Darker bar = baseline value. Hatched bar = evaluated on fewer days (see below).")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=view["peaker_energy_off_mwh"] / 1e3, name="Battery off",
                         marker_color=NEUTRAL))  # fmt: skip
    fig.add_trace(go.Bar(x=x, y=view["peaker_energy_on_mwh"] / 1e3, name="Battery on",
                         marker_color=BATTERY_COLORS["on"]))  # fmt: skip
    fig.update_layout(
        barmode="group", bargap=0.25, xaxis_title=dashboard.SENSITIVITY_LABELS[parameter],
        yaxis_title="GWh / year", xaxis_type="category",
    )  # fmt: skip
    show(fig, "Peaker generation, with and without the battery", hovermode="closest")

    partial = view[~view["full_sample"]]
    for _, row in partial.iterrows():
        st.warning(
            f"{dashboard.SENSITIVITY_LABELS[parameter]} = {row['value_label']}: the system "
            f"cannot meet demand on {int(row['infeasible_days'])} days (the LP has no "
            f"load-shedding option), so this point covers {int(row['evaluated_days'])} of "
            f"{int(row['sample_days'])} days and is not directly comparable to the others."
        )
    if parameter == "ccgt_ramp_mw_per_hour" and len(view) > 1:
        base, tight = view[view["is_baseline"]], view.iloc[0]
        if not base.empty:
            base = base.iloc[0]
            st.info(
                f"From the baseline {base['value_label']} MW/h ramp "
                f"({eur(base['battery_savings_eur'])}, {base['battery_savings_pct']:.2f}%) to "
                f"{tight['value_label']} MW/h ({eur(tight['battery_savings_eur'])}, "
                f"{tight['battery_savings_pct']:.2f}%): peaker output without a battery rises "
                f"to {mwh(tight['peaker_energy_off_mwh'])}, and the battery cuts it to "
                f"{mwh(tight['peaker_energy_on_mwh'])}."
            )
    if parameter == "balancing_prices_eur_per_mwh":
        st.info(
            "Battery savings do not move with balancing prices. This is expected: prices "
            "change what imbalance costs, but imbalance does not depend on the battery."
        )
    with st.expander("Table view"):
        st.dataframe(view, **TABLE_WIDTH, hide_index=True)


@st.cache_data(show_spinner=False)
def _what_if(method: str, day: dt.date, overrides: tuple, mtime: float | None) -> dict:
    hourly = load("evaluation_hourly")
    system = dashboard.system_from_overrides(dict(overrides))
    return dashboard.what_if_day(hourly, day, method, system)


def page_what_if() -> None:
    st.header("What-if (one day)")
    hourly = require("evaluation_hourly")
    if hourly is None:
        return
    st.markdown(
        "Re-plan a single day with modified system parameters, using the stored forecast, "
        "actual demand and solar/wind availability for that day. Each run solves the "
        "day-ahead LP with and without the battery, for your settings and for the baseline."
    )
    base = sensitivity.BASELINE_SYSTEM
    units = {u.name: u for u in base.generation_units}
    methods = [m for m in dashboard.FORECAST_LABELS if m in set(hourly["forecast_method"])]
    days = sorted(set(hourly["local_date"]))

    with st.form("what_if"):
        c1, c2 = st.columns(2)
        method = c1.selectbox("Planning forecast", methods, index=methods.index("lightgbm"),
                              format_func=dashboard.FORECAST_LABELS.get)  # fmt: skip
        with c2:
            day = pick_day("Day", days, key="what_if_day", default=dt.date(2019, 1, 22))
        c1, c2, c3 = st.columns(3)
        power = c1.slider("Battery power (MW)", 0, 4_000,
                          int(base.battery.power_capacity_mw), step=250)  # fmt: skip
        energy = c1.slider("Battery energy (MWh)", 500, 16_000,
                           int(base.battery.energy_capacity_mwh), step=500)  # fmt: skip
        ramp = c2.slider("CCGT ramp limit (MW/h)", 500, 8_000,
                         int(units["ccgt"].ramp_limit_mw_per_hour), step=500)  # fmt: skip
        capacity = c2.slider("CCGT capacity (MW)", 10_000, 30_000,
                             int(units["ccgt"].capacity_mw), step=1_000)  # fmt: skip
        peaker = c3.slider("Peaker marginal cost (€/MWh)", 60, 300,
                           int(units["peaker"].marginal_cost_eur_per_mwh), step=10)  # fmt: skip
        c3.caption("Balancing prices stay at €200 / €20 per MWh.")
        submitted = st.form_submit_button("Run")

    if not submitted:
        st.info("Choose settings and press **Run**.")
        return

    overrides = (
        ("battery_power_mw", power),
        ("battery_energy_mwh", energy),
        ("ccgt_ramp_mw_per_hour", ramp),
        ("ccgt_capacity_mw", capacity),
        ("peaker_cost_eur_per_mwh", peaker),
    )
    mtime = _mtime("evaluation_hourly")
    try:
        with st.spinner("Solving the day-ahead LP..."):
            baseline = _what_if(method, day, (), mtime)
            variant = _what_if(method, day, overrides, mtime)
    except RuntimeError:
        st.error(
            "With these settings the system cannot meet the forecast demand on this day "
            "(demand exceeds what the fleet can supply or ramp to). Try more CCGT capacity "
            "or a looser ramp limit."
        )
        return

    def _row(label, result):
        return {
            "System": label,
            "Planned cost, battery on": result["on"]["planned_cost_eur"],
            "Realized cost, battery off": result["off"]["realized_cost_eur"],
            "Realized cost, battery on": result["on"]["realized_cost_eur"],
            "Battery savings": result["battery_savings_eur"],
            "Peaker, battery off (MWh)": result["off"]["peaker_mwh"],
            "Peaker, battery on (MWh)": result["on"]["peaker_mwh"],
        }

    table = pd.DataFrame([_row("Baseline", baseline), _row("Your settings", variant)])
    m = st.columns(3)
    m[0].metric(
        "Planned cost (battery on)", eur(variant["on"]["planned_cost_eur"]),
        delta_text(variant["on"]["planned_cost_eur"] - baseline["on"]["planned_cost_eur"]),
        delta_color="inverse",
    )  # fmt: skip
    m[1].metric(
        "Realized cost (battery on)", eur(variant["on"]["realized_cost_eur"]),
        delta_text(variant["on"]["realized_cost_eur"] - baseline["on"]["realized_cost_eur"]),
        delta_color="inverse",
    )  # fmt: skip
    m[2].metric(
        "Battery savings this day", eur(variant["battery_savings_eur"]),
        delta_text(variant["battery_savings_eur"] - baseline["battery_savings_eur"]),
    )  # fmt: skip
    st.dataframe(
        table.style.format({c: "{:,.0f}" for c in table.columns if c != "System"}),
        **TABLE_WIDTH,
        hide_index=True,
    )
    st.caption(
        f"{day}, planned on {dashboard.FORECAST_LABELS[method]}. One day only: annual "
        "effects are on the Storage sensitivity page. " + ILLUSTRATIVE_NOTE
    )


PAGES = {
    "Overview": page_overview,
    "Forecast": page_forecast,
    "Dispatch": page_dispatch,
    "Cost & forecast value": page_cost,
    "Storage sensitivity": page_sensitivity,
    "What-if": page_what_if,
}

choice = st.sidebar.radio("Section", list(PAGES))
st.sidebar.caption(ILLUSTRATIVE_NOTE)
if any(dashboard.is_bundled(name) for name in dashboard.RESULT_FILES):
    st.sidebar.caption("Showing the bundled results of the real 2019 run (app/demo_results/).")
PAGES[choice]()
