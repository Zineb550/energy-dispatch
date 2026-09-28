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

# Run the stages individually (until pipeline.py is wired up):
python -m energy_dispatch.forecast   # writes data/processed/forecast_test_period.parquet
python -m energy_dispatch.evaluate   # dispatch + realized cost -> scenario_results.parquet
                                     # (add --max-days 7 for a quick smoke run)

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

Interval forecasts (q10/q90) come from two LightGBM quantile models
(`config.LIGHTGBM_QUANTILE_PARAMS`), one per quantile, trained on the
same origin-anchored features. A real backtest showed these raw quantile
models badly under-covering their target interval (56% empirical coverage
against an 80% target, plateauing around 69% even after a hyperparameter
sweep) — a known failure mode of independently-trained pinball-loss
models, not a bug in the harness. `forecast.py` corrects this with
**split-conformal calibration** (CQR — Romano, Patterson & Candès, 2019,
`make_conformalized_quantile_predict_fn`): each retrain block holds out a
calibration slice of its own training data, measures how far off the raw
q10/q90 predictions actually are on it (out-of-sample), and widens (or
narrows) the interval by that measured amount so empirical coverage
targets `config.PREDICTION_INTERVAL_COVERAGE_TARGET` rather than
whatever the uncalibrated model happens to produce.

## Evaluation design

`evaluate.py` measures what forecast quality is worth *after* a decision
has been made on it. For each local (Europe/Madrid) day of the 2019 test
year:

1. **Plan (10:00 on day D).** `optimize.plan_day_ahead_dispatch` builds
   the day-ahead dispatch for D+1 from one forecast plus the renewable
   availability (solar + wind), and nothing else.
2. **Reveal.** D+1 happens and actual demand becomes known. The plan is
   not revised: there is no real-time re-dispatch.
3. **Settle.** The difference between actual demand and the plan is
   bought or sold on the balancing market.

Definitions, per hour (MW over one hour = MWh):

| Quantity | Definition |
|---|---|
| Planned supply | Σ generation + renewable used + battery discharge − battery charge (the LP's demand-balance row, so it equals the forecast) |
| Imbalance | actual demand − planned supply; **positive** = under-supplied (upward energy bought), **negative** = surplus |
| Imbalance cost | €200/MWh × upward imbalance − €20/MWh × surplus (surplus is *sold*, a credit, since its generation cost is already in the planned cost) |
| Planned cost | the LP objective: Σ marginal cost × generation, on the forecast profile |
| Realized cost | planned cost + imbalance cost |

Eight scenarios are evaluated: {perfect foresight, seasonal naive, TSO,
LightGBM} × {battery off, battery on}. Battery-off is the same LP with a
zero-capacity battery. **Perfect foresight** plans on actual demand and is
an idealized reference (lower bound), not an achievable forecasting
system. Every scenario covers exactly the same set of days: a day is
evaluated only if it has all of its local hours (23/24/25 on DST days) and
no missing values in any forecast, the actual load, or renewables. Skipped
days are reported with a reason.

Two properties of this setup matter when reading the results:

- Because the plan is fixed, planned supply equals the forecast, so
  **imbalance equals the forecast error, with or without the battery**.
  The battery lowers planned cost by shifting energy toward cheaper hours
  of the *forecast* profile. It does not absorb forecast errors, since it
  never reacts to actual demand. Any difference in battery value between
  forecasts comes from the forecast's shape.
- Renewable availability uses observed output (perfect renewable
  foresight), so the evaluation isolates **demand** forecast error, not
  total renewable uncertainty.

## Results

Real 2019 test-year backtest (`python -m energy_dispatch.forecast`), point
metrics:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal-naive | 1254.5 | 2000.2 | 4.40% |
| TSO (ENTSO-E) | 272.7 | 371.4 | 0.95% |
| LightGBM | 500.0 | 747.0 | 1.76% |

The point model clears the naive baseline by a wide margin (60% lower
MAE) but doesn't beat the professional TSO forecast — expected, not a
bug: ENTSO-E's forecast has structural advantages this model doesn't
(intraday updates, grid-operator-side data), and it isn't even helped by
this project's weather-proxy assumption above, which should if anything
flatter our model relative to a real forecast-error scenario.

Interval metrics (q10/q90), same backtest, with split-conformal
calibration applied:

| Metric | Value |
|---|---|
| Coverage (target 80%) | **78.8%** |
| Avg. interval width | 1672 MW |
| Pinball loss (low / high) | 146.6 / 118.1 |

Calibration essentially closed the gap (56% → 78.8%, against an 80%
target) — and did so while *lowering* pinball loss versus every
uncalibrated candidate tried in the hyperparameter sweep (previously as
low as 155.3/121.7), so this isn't a coverage-vs-accuracy trade-off, it's
a straightforward improvement on both axes.

The realized-cost scenario comparison (see "Evaluation design" above) is
still _TBD_. It will be filled in once `python -m energy_dispatch.evaluate`
has been run on the real 2019 test year.

## Assumptions and limitations

- **Generation costs and capacities are illustrative**, not real market
  data (see `config.py` — baseload/CCGT/peaker capacities and marginal
  costs, battery specs, balancing market prices).
- Day-ahead weather **forecasts** are not available in the OPSD dataset;
  observed weather is used as a proxy feature, which overstates the
  accuracy achievable with a real weather forecast.
- Renewable availability in the dispatch LP is the *observed* solar + wind
  output (perfect renewable foresight); only demand is forecast.
- No real-time re-dispatch: the day-ahead plan is fixed, and every MWh
  of forecast error is settled on the balancing market, even when
  curtailed renewables or the battery could have covered it.
- Single-node grid: no transmission constraints or nodal pricing.
- No unit commitment — generation units have no on/off decision, start-up
  cost, or minimum up-time; this is a pure economic-dispatch LP, not a
  MILP.
- 2020 is excluded from the main evaluation due to COVID-driven demand
  anomalies, and is treated as a separate stress test instead.

## Stretch extensions

See spec section 13 — forecasting solar/wind directly, unit commitment
as a MILP, two-stage stochastic/robust optimization, public Streamlit
deployment, and generalizing to a second country. (Conformal prediction
was originally listed here too, but is now implemented — see
"Forecasting design" above.)
