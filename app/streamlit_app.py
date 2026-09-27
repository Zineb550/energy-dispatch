"""
Streamlit dashboard (spec section 9). Loads precomputed forecasts and
scenario results from data/processed/; only the day-ahead LP re-runs live
(the "what-if" page).
"""

from __future__ import annotations

import streamlit as st

from energy_dispatch import config

st.set_page_config(page_title="Energy Dispatch — Forecast & Savings", layout="wide")

st.title("Day-ahead demand forecasting & dispatch optimization")

page = st.sidebar.radio("Page", ["Forecast", "Dispatch", "Savings", "What-if"])

if page == "Forecast":
    st.header("Forecast vs. actual")
    st.info(
        "Pick a date -> actual vs. forecasts (naive, TSO, LightGBM) with the "
        "80% band; metrics for that day and the full test period. "
        "Wire this up once forecast.rolling_origin_backtest has produced "
        f"{config.FORECAST_TABLE_PARQUET.name}."
    )

elif page == "Dispatch":
    st.header("Hourly dispatch")
    st.info(
        "Stacked area chart of hourly supply by unit, plus battery "
        "charge/discharge and state of charge for a selected day."
    )

elif page == "Savings":
    st.header("Scenario comparison")
    st.info(
        "Scenario comparison table and bar chart, once "
        f"evaluate.compare_scenarios has produced {config.SCENARIO_RESULTS_PARQUET.name}."
    )

elif page == "What-if":
    st.header("What-if: battery size & balancing price")
    battery_mw = st.slider("Battery power (MW)", 0, 4_000, int(config.BATTERY.power_capacity_mw), step=250)
    price_up = st.slider(
        "Balancing up-price (EUR/MWh)", 0, 800, int(config.BALANCING_PRICE_UP_EUR_PER_MWH), step=50
    )
    st.info(
        "Re-run optimize.plan_day_ahead_dispatch live for the selected day "
        "with these overrides once that module is implemented."
    )
