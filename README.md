# Energy Demand Forecasting & Dispatch Optimization

**Business question:** a grid operator must plan tomorrow's hourly supply.
How much can it save by (a) forecasting demand more accurately and
(b) operating battery storage optimally?

**Headline result:** _TBD — filled in once the full pipeline has run
(spec section 7.4: savings from better forecasting, savings from the
battery, and the combined effect, in €/year and %)._

## Architecture

```
raw data (OPSD) -> cleaning -> EDA -> demand forecast (point + intervals)
    -> day-ahead dispatch optimization (LP) -> realized-cost evaluation
    -> sensitivity analysis -> dashboard + slide deck
```

## Repository structure

```
energy-dispatch/
├── data/
│   ├── raw/                # downloaded CSVs (gitignored)
│   └── processed/          # cleaned parquet files
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_forecasting.ipynb
│   └── 03_optimization.ipynb
├── src/energy_dispatch/
│   ├── config.py           # paths, dates, cost assumptions (single source of truth)
│   ├── data.py              # download, load, clean, merge
│   ├── features.py          # calendar, lag, rolling, weather features
│   ├── forecast.py          # baselines, LightGBM, quantile models, backtesting
│   ├── optimize.py          # LP model build + solve (PuLP)
│   ├── evaluate.py          # realized cost, imbalance, scenario comparison
│   ├── sensitivity.py       # parameter sweeps
│   └── pipeline.py          # single entry point: python -m energy_dispatch.pipeline
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── fixtures/            # small sample CSVs, NOT the real OPSD download
│   ├── test_data.py
│   ├── test_features.py
│   └── test_optimize.py
├── reports/
│   ├── figures/
│   └── deck.pdf
├── pyproject.toml
└── README.md
```

Notebooks explore and narrate; all reusable logic lives in `src/` and is
imported by the notebooks and the app.

## Data

| Source | Content | Use |
|---|---|---|
| Open Power System Data (OPSD), *Time series* package, 60-min single-index CSV | Hourly load (actual), TSO day-ahead load forecast, solar & wind generation, for European countries | Target variable, benchmark forecast, renewable supply |
| OPSD *Weather data* package | Country-level hourly temperature and solar radiation | Forecasting features |

**Getting the data.** Neither raw dataset is checked into this repo
(`data/raw/` and `data/processed/` are gitignored). Run:

```bash
python -m energy_dispatch.data
```

This downloads both CSVs into `data/raw/` (skipping any that already
exist — pass `--force-download` to re-fetch) and writes the cleaned,
merged hourly table to `data/processed/spain_hourly.parquet`.

OPSD has no "latest" URL alias — both download URLs point at dated
release directories pinned in `config.py` (`OPSD_TIMESERIES_VERSION`,
`OPSD_WEATHER_VERSION`). If a download 404s, check
[data.open-power-system-data.org/time_series](https://data.open-power-system-data.org/time_series/)
or [.../weather_data](https://data.open-power-system-data.org/weather_data/)
for the current version and update those two constants — nothing else
needs to change. Alternatively, fetch the two CSVs yourself and place
them at `data/raw/time_series_60min_singleindex.csv` and
`data/raw/weather_data.csv`, then run with `--skip-download`.

Tests never touch the real download — `tests/fixtures/` holds small,
synthetic CSVs in the same shape as the real files (including a
deliberate short gap, a long gap, an implausible value, a spike, and a
duplicated timestamp) so `data.py`'s cleaning logic can be tested in
under a second.

Country: Spain (`ES_*` columns), chosen as a proxy for Morocco's grid —
linked by interconnector, with a similarly solar-heavy generation profile.
Data spans 2015–2019 for the main train/validation/test split; 2020 is
held out as a separate COVID stress test rather than mixed into the main
split.

## How to run

```bash
pip install -e ".[dev]"

# 1. Fetch + clean data (downloads into data/raw/, writes data/processed/spain_hourly.parquet)
python -m energy_dispatch.data

# 2. Run the full pipeline (forecast backtest -> dispatch -> evaluation)
python -m energy_dispatch.pipeline

# 3. Launch the dashboard
streamlit run app/streamlit_app.py

# Tests
pytest
```

## Forecasting design

Day-ahead forecasting means one commitment, made at a single cutoff
(`config.DAY_AHEAD_CUTOFF_HOUR`, 10:00 local time), covering all ~24
hours of the next local calendar day at once. That single cutoff — the
"forecast origin" — is shared by every one of those ~24 target hours.

A feature defined *relative to the target hour* (`value at t - 24h`) is
**not** uniformly safe here: the gap between the origin and the target
hour ranges from ~14h (the target day's first hour) to ~37h (its last
hour), so a naive 24h lag would reach past the cutoff — into data that
doesn't exist yet — for roughly half of the 24 target hours. `features.py`
resolves this by anchoring every lag and rolling-window feature to the
shared origin instead of to the target hour (`value at O(t) - k hours`),
which is safe for any `k >= 0` regardless of the cutoff hour, by
construction. See `features.py`'s module docstring for the full
reasoning, and `features.assert_no_leakage` / `tests/test_features.py`
for how it's checked (both mechanically, via timestamp bookkeeping, and
empirically, by poisoning future values with an out-of-range sentinel and
confirming they never appear in a constructed feature).

Calendar features (hour, weekday, holidays, ...) are the one exception —
they're deterministic facts about the target hour, known arbitrarily far
in advance, so they vary per-row safely. Weather features are a second,
deliberate exception: they use the *observed* value at the target hour as
a stand-in for a day-ahead weather forecast OPSD doesn't provide (see
Limitations below).

## Results

_TBD — forecast metrics table, scenario comparison table, and key charts
go here once the pipeline has been run._

## Assumptions and limitations

- **Generation costs and capacities are illustrative**, not real market
  data (see `config.py` — baseload/CCGT/peaker capacities and marginal
  costs, battery specs, balancing market prices).
- Day-ahead weather **forecasts** are not available in the OPSD dataset;
  observed weather is used as a proxy feature, which overstates the
  accuracy achievable with a real weather forecast.
- Single-node grid: no transmission constraints or nodal pricing.
- No unit commitment — generation units have no on/off decision, start-up
  cost, or minimum up-time; this is a pure economic-dispatch LP, not a
  MILP.
- 2020 is excluded from the main evaluation due to COVID-driven demand
  anomalies, and is treated as a separate stress test instead.

## Stretch extensions

See spec section 13 — forecasting solar/wind directly, unit commitment
as a MILP, two-stage stochastic/robust optimization, conformal
prediction, public Streamlit deployment, and generalizing to a second
country.
