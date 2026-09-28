# Energy Demand Forecasting & Dispatch Optimization

A grid operator must commit tomorrow's hourly supply today, before demand
is known. This project forecasts Spain's hourly electricity demand, plans
the next day's generation and battery dispatch on that forecast with a
linear program, then settles the plan against the demand that actually
occurs. The question is not "how accurate is the forecast?" but **"what is
the forecast worth once a decision is made on it?"**

**Live dashboard:** [energydispatchwithzineb.streamlit.app](https://energydispatchwithzineb.streamlit.app/)

## Key findings

Simulated 2019 system (Spain demand, OPSD data; generation fleet and prices
are illustrative, see [Assumptions](#assumptions-and-limitations)). The
cost differences in findings 1 and 2 are statistically significant
(Diebold-Mariano test on hourly realized cost, autocorrelation-robust,
p < 10⁻⁵).

| # | Finding | Result |
|---|---|---|
| 1 | **Better forecasts lower operating cost.** Planning on the LightGBM forecast instead of a seasonal-naive one | **−€604.3M/year (−6.80%)** realized cost. The TSO's professional forecast is a further €155.2M/year cheaper. |
| 2 | **Deciding with the uncertainty, not just the forecast, saves more.** A shortfall costs more than a surplus, so the cost-minimizing plan is the 78th percentile of the calibrated forecast, not its point value (newsvendor rule, fixed from prices before seeing results) | **−€76.6M/year (−0.92%)** with the same model, about half of LightGBM's gap to the TSO forecast |
| 3 | **Storage value depends on the generation fleet, not the battery.** A 1,000 MW / 4,000 MWh battery saves €3.5M/year (0.04%) in the baseline fleet | **€192.9M/year (2.24%)** once the CCGT ramp limit is tightened from 8,000 to 1,000 MW/h; a bigger battery alone levels off near €4M |

Where the forecast falls short: LightGBM's gap to the TSO is concentrated in
**holidays and holiday periods** (MAE 711 MW on national holidays vs 527 MW on
weekdays; worst months December, January, April and August), which makes
richer holiday features the clearest next improvement.

## Approach

```
OPSD data ─► cleaning ─► leakage-safe features ─► forecast for D+1 (made 10:00 on day D)
                                                    point + conformal q10/q90
                                                           │
                              day-ahead dispatch LP ◄──────┘  (plan on a forecast)
                                       │
               actual demand revealed ─► imbalance ─► realized cost = planned + imbalance
                                       │
            scenario comparison · uncertainty-aware planning · storage sensitivity · diagnostics
```

| Stage | Module | What it does |
|---|---|---|
| Data | `data.py` | Downloads and cleans OPSD load, TSO forecast, solar, wind and weather for Spain (gap interpolation, spike and duplicate handling, DST-safe UTC index) |
| Features | `features.py` | Calendar, lag, rolling and weather features, every lag anchored to the 10:00 forecast origin so none can leak future data (checked mechanically and by poisoning future values) |
| Forecast | `forecast.py` | Seasonal-naive and TSO baselines; LightGBM point and q10/q90 quantile models in a monthly rolling-origin backtest; split-conformal calibration of the interval; Diebold-Mariano tests, SHAP importance, error breakdown |
| Dispatch | `optimize.py` | Day-ahead LP (PuLP/CBC): baseload, CCGT, peaker, curtailable solar + wind, battery with efficiency losses, ramp limits and a state-of-charge cycle constraint |
| Evaluation | `evaluate.py` | Plans each day on a forecast, reveals actual demand, prices the imbalance at the balancing market, for 4 forecasts × battery on/off |
| Planning | `planning.py` | Plans on quantiles of the calibrated forecast and compares realized cost with the point-forecast plan |
| Sensitivity | `sensitivity.py` | One-factor-at-a-time sweeps of battery value over fleet and battery parameters |
| Diagnostics | `diagnostics.py` | Runs the significance tests, SHAP and error breakdown on the stored results |
| Dashboard | `dashboard.py`, `app/` | Streamlit presentation layer over the stored results, with a one-day what-if |
| Pipeline | `pipeline.py` | Runs every stage in order |

## Results in detail

### Forecast accuracy (2019 test year, monthly rolling-origin backtest)

| Model | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|
| Seasonal naive (same hour last week) | 1,254.5 | 2,000.2 | 4.40% |
| TSO (ENTSO-E day-ahead forecast) | 272.7 | 371.4 | 0.95% |
| LightGBM | 500.0 | 747.0 | 1.76% |

LightGBM cuts the naive error by 60% but does not beat the TSO, which has
structural advantages (grid-operator data, intraday updates). Both
differences are significant (Diebold-Mariano, squared and absolute error,
p < 10⁻⁴).

**Prediction interval.** The raw q10/q90 quantile models covered only 56% of
hours against an 80% target, and about 69% after a hyperparameter sweep.
Split-conformal calibration (CQR, Romano et al. 2019), fitted inside each
retrain block on held-out past data, brought coverage to **78.8%** while also
lowering pinball loss (146.6 / 118.1 vs 155.3 / 121.7 at best before).

**Where the model errs.** LightGBM's MAE is 711 MW on national holidays,
527 MW on weekdays and 418 MW on weekends (TSO: 353 / 281 / 247). By month,
its worst are December (871), January (694), April (664, Easter) and August
(620, vacation month), while the TSO stays between 207 and 369 MW all year.

**What drives it (SHAP, LightGBM TreeSHAP on 2019 features).** Hour of day
(40%) and weekday (21%) dominate, then the recent demand level (24 h and
168 h rolling means, 15%) and weather (about 12% combined). `is_holiday`
accounts for only 2%: with few holidays in three training years the model
barely learns them, which matches the error breakdown. `lag_24h` carries
only 1.4% because lags are anchored to the 10:00 cutoff (the value 24 h
before the cutoff, not before the target hour): leakage safety costs some
signal, deliberately.

### Finding 1: economic value of forecasting

364 days / 8,736 hours, battery off. 2019-12-24 is excluded from every
scenario (missing solar/wind data).

| Planning forecast | MAE (MW) | Planned cost | Imbalance cost | **Realized cost** | Imbalance energy |
|---|---|---|---|---|---|
| Perfect foresight (reference) | 0 | €7.917B | €0.0M | **€7.917B** | 0 TWh |
| TSO (ENTSO-E) | 272.5 | €7.921B | €208.5M | **€8.130B** | 2.38 TWh |
| LightGBM | 491.7 | €7.937B | €348.0M | **€8.285B** | 4.30 TWh |
| Seasonal naive | 1,238.2 | €7.923B | €965.8M | **€8.889B** | 10.82 TWh |

- **LightGBM vs seasonal naive:** €604.3M/year lower realized cost (6.80% of
  the naive scenario's cost; p = 1.2 × 10⁻⁸).
- **TSO vs LightGBM:** €155.2M/year lower for the TSO (1.91% of the TSO
  scenario's cost; p < 10⁻¹³), consistent with its lower error.
- **Perfect foresight** plans on actual demand: a theoretical lower bound,
  not an achievable forecast. LightGBM is €367.8M (4.65%) above it.

Accuracy and economic value are related but not the same. Planned cost
alone would rank the forecasts wrongly: LightGBM has the *highest* planned
cost because it over-forecasts slightly on average (−40 MW) and so
schedules more generation. What separates the forecasts is the imbalance
settled after demand is revealed.

### Finding 2: planning on the forecast interval

With the plan fixed, a missing MWh is bought at €200 instead of being
generated at the marginal cost, and a spare MWh is sold back at €20. With
CCGT on the margin (€60/MWh), under-planning costs €140/MWh and
over-planning €40/MWh, so the cost-minimizing plan is the quantile

  τ\* = 140 / (140 + 40) ≈ 0.78

of the demand forecast (the newsvendor problem). `planning.py` plans each day
on quantiles of the LightGBM forecast, interpolated from the point forecast
and the conformal q10/q90.

| Plan (LightGBM) | Mean uplift | Planned cost | Imbalance cost | Realized cost | vs point plan |
|---|---|---|---|---|---|
| Point forecast (τ = 0.5) | 0 MW | €7.937B | €348.0M | €8.285B | — |
| **τ\* = 0.78** | +496 MW | €8.200B | €8.6M | **€8.208B** | **−€76.6M (−0.92%)** |

Raising the plan by about 500 MW costs €263M more in generation but removes
€339M of imbalance cost, mostly expensive shortfalls. The saving is
significant (p = 4.4 × 10⁻⁶) and closes about half of LightGBM's gap to the
TSO forecast by changing only the decision rule.

The full sweep has the expected U-shape (τ = 0.3: +€211.7M; 0.4: +€82.6M;
0.6: −€60.8M; **0.7: −€83.7M**; 0.8: −€70.6M; 0.9: −€12.4M). τ = 0.7 scores
slightly better than τ\*, plausibly because the peaker is sometimes marginal
(where τ\* ≈ 0.44) and LightGBM already over-forecasts a little. τ\* is the
reported result because it was fixed from prices alone; choosing the best
sweep point would be selecting on the test year.

### Finding 3: storage value vs generation-fleet flexibility

With the baseline battery (1,000 MW / 4,000 MWh), realized cost falls by
€3.0–3.5M/year (about 0.04%) whichever forecast is used, all of it in
planned cost. With a fixed day-ahead plan, imbalance equals the forecast
error whether or not a battery exists, so the battery cannot absorb forecast
error here; its only lever is shifting energy toward cheaper hours of the
plan. `sensitivity.py` varies one parameter at a time (LightGBM forecast,
same 364 days):

| Parameter varied | Values tested | Battery savings (% of realized cost without battery) |
|---|---|---|
| CCGT ramp limit (MW/h) | 1,000 / 2,000 / 4,000 / **8,000** | €192.9M (2.24%) / €66.6M (0.80%) / €6.3M (0.08%) / **€3.5M (0.04%)** |
| CCGT capacity (MW) | 15,000\* / 20,000 / **25,000** | €45.9M (0.56%)\* / €22.5M (0.27%) / **€3.5M (0.04%)** |
| Peaker cost (€/MWh) | 100 / **120** / 180 / 250 | €2.2M / **€3.5M** / €7.2M / €11.5M |
| Battery power (MW) | 500 / **1,000** / 2,000 / 4,000 | €2.3M / **€3.5M** / €4.0M / €4.0M |
| Battery energy (MWh) | 1,000 / **4,000** / 8,000 / 16,000 | €1.3M / **€3.5M** / €3.9M / €4.1M |
| Balancing prices up/down (€/MWh) | 150/50 / **200/20** / 300/10 | €3.5M in all three cases |

Baseline in **bold**. \*The 15,000 MW fleet cannot meet demand on 21 days (the
LP has no load shedding), so that row covers 343 days and is not directly
comparable.

- **Ramp flexibility is the main driver.** At a 1,000 MW/h CCGT ramp limit
  the system needs 3.22 TWh of peaker output to follow demand; the battery
  cuts it to 1.22 TWh, a 55-fold increase in battery value.
- **Less CCGT capacity works the same way**: more peaker hours to displace.
- **Peaker cost scales the saving linearly**: the battery displaces the same
  ~61.5 GWh of peaker output at every price.
- **A bigger battery alone adds little**: the fleet, not battery size, limits
  the opportunity.
- **Balancing prices leave battery value unchanged**, as the model implies;
  this is a consistency check rather than an economic finding.

## Design notes

**Leakage-safe day-ahead features.** All ~24 hours of day D+1 are forecast
at one cutoff (10:00 on day D). A lag defined relative to the target hour
would reach past that cutoff for about half the hours, so every lag and
rolling window is anchored to the shared forecast origin instead.
`features.assert_no_leakage` checks this by timestamp bookkeeping, and the
tests poison future values with a sentinel and confirm none reaches a
feature. Calendar features are known in advance; weather uses observed
values as a proxy for a weather forecast (see limitations).

**Evaluation.** For each local (Europe/Madrid) day: plan from one forecast
plus renewable availability only, reveal actual demand, settle the
difference. Per hour: planned supply = Σ generation + renewables used +
discharge − charge (equal to the forecast by construction); imbalance =
actual − planned supply; imbalance cost = €200 × shortfall − €20 × surplus
(a surplus is sold, since its generation cost is already in the planned
cost); realized cost = planned cost + imbalance cost. Every scenario covers
the same days: a day counts only if it has all its local hours (23/24/25 on
DST days) and no missing input for any scenario; skipped days are reported.

**Significance.** Diebold-Mariano tests use a Newey-West variance over 168
hourly lags, because day-ahead errors share an origin within a day and
persist across days; ignoring that overstates significance.

**Engineering.** Configuration lives in `config.py` (single source of
truth). 156 tests run on synthetic data only, including hand-computed LP,
imbalance and planning cases, leakage checks and a headless render of every
dashboard page. `ruff` clean.

## Assumptions and limitations

- **Illustrative system.** Generation capacities, marginal costs, ramp
  limits, battery parameters and balancing prices are assumptions chosen to
  give a plausible single-node system (`config.py`), not calibrated Spanish
  market data. The euro amounts illustrate mechanisms, not market savings.
- **Weather proxy.** OPSD has no weather forecasts, so the model uses
  observed target-hour weather, which overstates achievable accuracy.
- **Perfect renewable foresight.** Dispatch planning uses observed solar and
  wind output; only demand is forecast. This isolates the demand question.
- **No real-time re-dispatch.** The day-ahead plan is fixed and every MWh of
  forecast error goes to the balancing market, even when the battery or
  curtailed renewables could have covered it. This is why the battery cannot
  absorb forecast error here.
- **Simple settlement.** Surplus is sold at a single down-price; some real
  balancing designs penalize deliberate over-scheduling, which would reduce
  the value of planning above the forecast.
- **Economic dispatch, not unit commitment.** No on/off decisions, start-up
  costs or minimum up-times; single node, no transmission.
- **One test year.** 2019 only; 2019-12-24 excluded (missing solar/wind
  data). 2020 was held out of the split because of COVID, but no stress
  test has been run on it yet.
- **Quantile interpolation.** Plans between q10, point and q90 assume a
  Gaussian-shaped error; dedicated quantile models at τ\* would be cleaner.

## Next steps

- **Holiday features** (bridge days, Easter week, regional holidays, August
  vacation): the largest identified source of error against the TSO.
- **Real-time re-dispatch** (two-stage planning), letting the battery and
  curtailed renewables respond to actual demand.
- **Hour-dependent planning quantile** based on the marginal unit (τ\* ≈ 0.78
  when CCGT is marginal, ≈ 0.44 when the peaker is).
- **Renewable forecast error**, a 2020 stress test, and unit commitment as a
  MILP.

## Reproduce

```bash
pip install -e ".[dev]"

python -m energy_dispatch.pipeline                  # everything, end to end
python -m energy_dispatch.pipeline --skip-download  # reuse data/raw/
python -m energy_dispatch.pipeline --max-days 7     # quick smoke run
python -m energy_dispatch.pipeline --from-step evaluate   # resume from a stage

streamlit run app/streamlit_app.py                  # dashboard
pytest                                              # tests (synthetic data only)
```

Each stage also runs alone: `python -m energy_dispatch.<data|forecast|evaluate|sensitivity|planning|diagnostics>`.
Most of the run time is the forecast backtest and about 16,000 day-ahead
LP solves (evaluation, sensitivity, planning and diagnostics).

**Data.** Two OPSD packages (not committed): the *Time series* 60-minute
CSV (hourly load, ENTSO-E day-ahead load forecast, solar and wind for Spain)
and the *Weather data* CSV (temperature, radiation). `python -m
energy_dispatch.data` downloads them into `data/raw/` and writes
`data/processed/spain_hourly.parquet`. OPSD has no "latest" URL, so the
release versions are pinned in `config.py`; if a download 404s, update
`OPSD_TIMESERIES_VERSION` / `OPSD_WEATHER_VERSION`, or place the two CSVs in
`data/raw/` yourself and use `--skip-download`. Spain was chosen as a proxy
for Morocco's grid (interconnected, similarly solar-heavy). Train
2015–2017, validation 2018, test 2019.

**Hosted dashboard.** Streamlit Community Cloud has no `data/processed/`, so
the app reads a committed copy of the results from `app/demo_results/`
(local results take precedence). After running the pipeline:

```bash
python -m energy_dispatch.dashboard --export-demo-results
git add app/demo_results && git commit -m "Update dashboard results" && git push
```

The hosted app installs only the light `requirements.txt` (no LightGBM or
raw data).

## Repository structure

```
energy-dispatch/
├── src/energy_dispatch/
│   ├── config.py        # paths, dates, system assumptions (single source of truth)
│   ├── data.py          # download, clean, merge OPSD data
│   ├── features.py      # leakage-safe calendar, lag, rolling, weather features
│   ├── forecast.py      # baselines, LightGBM, conformal intervals, backtest, DM/SHAP
│   ├── optimize.py      # day-ahead dispatch LP (PuLP/CBC)
│   ├── evaluate.py      # plan -> reveal -> settle; scenario comparison
│   ├── planning.py      # uncertainty-aware (quantile) planning
│   ├── sensitivity.py   # storage-value sensitivity sweeps
│   ├── diagnostics.py   # significance tests, SHAP, error breakdown
│   ├── dashboard.py     # data layer for the Streamlit app
│   └── pipeline.py      # end-to-end entry point
├── app/
│   ├── streamlit_app.py # dashboard (presentation only)
│   └── demo_results/    # results bundled for the hosted app
├── tests/               # 156 tests, synthetic fixtures only
├── data/                # raw/ and processed/ (gitignored)
├── requirements.txt     # hosted-dashboard dependencies
└── pyproject.toml       # full project dependencies
```
