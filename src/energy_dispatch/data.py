"""
Module 1 — data ingestion & cleaning.

Downloads (or loads locally cached) OPSD time series + weather data,
selects the Spain columns, cleans them, merges, and writes one clean
hourly parquet table to `config.PROCESSED_HOURLY_PARQUET`.

This module does NOT run automatically and does NOT ship any real OPSD
data. Run it yourself, locally:

    python -m energy_dispatch.data

which downloads the two CSVs into data/raw/ (skipping any that already
exist) and writes the cleaned parquet to data/processed/. See --help for
flags, and see "Manual download" below if you'd rather fetch the files
yourself.

Manual download
----------------
If the automatic download fails (network restrictions, a moved OPSD URL,
etc.), fetch these two files yourself and place them at the exact paths
below, then re-run with --skip-download:

  - config.OPSD_TIMESERIES_URL  ->  data/raw/time_series_60min_singleindex.csv
  - config.OPSD_WEATHER_URL     ->  data/raw/weather_data.csv

Both URLs point at dated OPSD release directories (OPSD has no "latest"
alias). If either 404s, check the current version listed at
https://data.open-power-system-data.org/time_series/ or
.../weather_data/ and update OPSD_TIMESERIES_VERSION / OPSD_WEATHER_VERSION
in config.py — everything downstream is derived from those two constants.

Pipeline (spec section 4):
    1. download_opsd_data()      -> raw CSVs in data/raw/
    2. load_raw_timeseries() /
       load_raw_weather()        -> raw DataFrames, Spain columns only
    3. clean_timeseries()        -> quality checks + imputation
    4. merge_weather()           -> join on timestamp
    5. build_processed_dataset() -> orchestrates 1-4, writes parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from energy_dispatch import config

logger = logging.getLogger(__name__)

# Chunk size for streaming downloads of the (large) raw CSVs.
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def _stream_download(url: str, dest: Path, timeout_seconds: int = 60) -> None:
    """Stream `url` to `dest`, writing to a .part file first so a failed
    download never leaves a truncated file at the final path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")

    logger.info("Downloading %s -> %s", url, dest)
    try:
        with requests.get(url, stream=True, timeout=timeout_seconds) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length", 0))
            written = 0
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    f.write(chunk)
                    written += len(chunk)
                    if total_bytes:
                        logger.info(
                            "  %s: %.0f%% (%.1f / %.1f MB)",
                            dest.name,
                            100 * written / total_bytes,
                            written / 1e6,
                            total_bytes / 1e6,
                        )
    except requests.RequestException as exc:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {url}: {exc}\n"
            f"Fetch it manually and save it to {dest} yourself, then re-run "
            "with --skip-download. If the URL 404s, OPSD may have moved to "
            "a newer version directory — see the module docstring in data.py."
        ) from exc

    part_path.replace(dest)
    logger.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def download_opsd_data(force: bool = False) -> None:
    """Download the OPSD time series and weather CSVs to data/raw/.

    No-ops per file if it already exists, unless force=True. This is the
    only function in the module that touches the network — everything
    downstream reads from data/raw/, so the rest of the pipeline can run
    fully offline once these two files are cached (or placed manually).
    """
    for url, dest in (
        (config.OPSD_TIMESERIES_URL, config.RAW_TIMESERIES_CSV),
        (config.OPSD_WEATHER_URL, config.RAW_WEATHER_CSV),
    ):
        if dest.exists() and not force:
            logger.info("%s already exists, skipping (use --force-download to re-fetch)", dest)
            continue
        _stream_download(url, dest)


# ---------------------------------------------------------------------------
# 2. Load raw CSVs (Spain columns only)
# ---------------------------------------------------------------------------

def _require_file(path: Path, url: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m energy_dispatch.data` to download it, "
            f"or fetch {url} yourself and save it to this path."
        )


def load_raw_timeseries(path: Path = config.RAW_TIMESERIES_CSV) -> pd.DataFrame:
    """Read the raw OPSD time series CSV and select Spain (ES_*) columns.

    Returns a DataFrame indexed by a UTC DatetimeIndex, with columns
    LOAD_ACTUAL_COL, LOAD_FORECAST_COL, SOLAR_GEN_COL, WIND_GEN_COL.

    `path` is overridable (e.g. to point at a small test fixture instead
    of the real download) — everything else about the parsing is fixed.
    """
    _require_file(path, config.OPSD_TIMESERIES_URL)

    df = pd.read_csv(
        path,
        usecols=list(config.TIMESERIES_USECOLS),
        parse_dates=[config.TIMESTAMP_COL],
    )
    df = df.set_index(config.TIMESTAMP_COL).sort_index()
    # OPSD's utc_timestamp is unambiguous UTC but arrives tz-naive; localize
    # explicitly rather than assuming pandas inferred it.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "utc_timestamp"
    return df


def load_raw_weather(path: Path = config.RAW_WEATHER_CSV) -> pd.DataFrame:
    """Read the raw OPSD weather CSV and select Spain columns.

    Returns a DataFrame indexed by UTC timestamp with columns
    WEATHER_TEMPERATURE_COL, WEATHER_RADIATION_DIRECT_COL,
    WEATHER_RADIATION_DIFFUSE_COL.
    """
    _require_file(path, config.OPSD_WEATHER_URL)

    df = pd.read_csv(
        path,
        usecols=list(config.WEATHER_USECOLS),
        parse_dates=[config.WEATHER_TIMESTAMP_COL],
    )
    df = df.set_index(config.WEATHER_TIMESTAMP_COL).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "utc_timestamp"
    return df


# ---------------------------------------------------------------------------
# 3. Quality checks + cleaning
# ---------------------------------------------------------------------------

def check_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Flag data quality issues without mutating values.

    Adds boolean flag columns (does not touch the original data columns):
      - is_duplicate_timestamp: index value repeated (a publisher data-quality
        glitch, not a DST artifact — DST folds only duplicate wall-clock
        time, and this index is UTC by construction)
      - is_missing_hour: added by reindexing to a complete hourly range;
        rows added this way have NaN for every original column
      - is_implausible_load: LOAD_ACTUAL_COL outside config.LOAD_PLAUSIBLE_RANGE_MW
      - is_spike: |diff| on LOAD_ACTUAL_COL beyond config.SPIKE_ROBUST_ZSCORE_THRESHOLD
        robust (MAD-based) standard deviations

    Cleaning/imputation happens in clean_timeseries(), not here — this
    function only reports.
    """
    out = df.copy()

    out["is_duplicate_timestamp"] = out.index.duplicated(keep=False)

    full_index = pd.date_range(out.index.min(), out.index.max(), freq="h", tz=out.index.tz)
    missing_hours = full_index.difference(out.index)
    if len(missing_hours):
        out = out.reindex(out.index.union(full_index))
    out["is_missing_hour"] = out.index.isin(missing_hours)

    if config.LOAD_ACTUAL_COL in out.columns:
        lo, hi = config.LOAD_PLAUSIBLE_RANGE_MW
        out["is_implausible_load"] = ~out[config.LOAD_ACTUAL_COL].between(lo, hi) & out[
            config.LOAD_ACTUAL_COL
        ].notna()

        diff = out[config.LOAD_ACTUAL_COL].diff()
        median_diff = diff.median()
        mad = (diff - median_diff).abs().median()
        # 1.4826 makes MAD a consistent estimator of the standard deviation
        # under normality; guard against mad == 0 (e.g. a constant series).
        robust_std = 1.4826 * mad if mad > 0 else diff.std()
        if robust_std:
            robust_z = (diff - median_diff) / robust_std
        else:
            robust_z = pd.Series(0, index=diff.index)
        out["is_spike"] = robust_z.abs() > config.SPIKE_ROBUST_ZSCORE_THRESHOLD
    else:
        out["is_implausible_load"] = False
        out["is_spike"] = False

    return out


def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality checks then impute or exclude.

    - Duplicated UTC timestamps: keep the first occurrence. Since we index
      on utc_timestamp rather than the wall-clock cet_cest_timestamp, this
      is a data-quality anomaly rather than an inherent DST artifact (DST
      folds only duplicate local time) — still handled explicitly, not
      silently dropped.
    - Short gaps (<= config.MAX_INTERPOLATE_GAP_HOURS): time-interpolated.
    - Longer gaps: left as NaN, with `exclude_long_gap=True` on every row
      in the gap so downstream forecasting/evaluation code can drop them
      rather than train on interpolated fiction.
    - Implausible values and spikes (flagged by check_quality) are set to
      NaN before interpolation, so a bad reading doesn't get "smoothed"
      into a plausible-looking but wrong value.
    """
    flagged = check_quality(df)
    value_cols = [c for c in df.columns if c in flagged.columns]

    cleaned = flagged.copy()
    if config.LOAD_ACTUAL_COL in cleaned.columns:
        bad = cleaned["is_implausible_load"] | cleaned["is_spike"]
        cleaned.loc[bad, config.LOAD_ACTUAL_COL] = np.nan

    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]

    # Gap-length bookkeeping, per column with missing values, in terms of
    # consecutive NaN run length (in hours, since the index is hourly).
    exclude_long_gap = pd.Series(False, index=cleaned.index)
    for col in value_cols:
        is_na = cleaned[col].isna()
        if not is_na.any():
            continue
        run_id = (is_na != is_na.shift()).cumsum()
        run_lengths = is_na.groupby(run_id).transform("sum")
        long_gap = is_na & (run_lengths > config.MAX_INTERPOLATE_GAP_HOURS)
        exclude_long_gap |= long_gap

        short_gap = is_na & ~long_gap
        if short_gap.any():
            cleaned[col] = cleaned[col].interpolate(method="time").where(
                ~long_gap, cleaned[col]
            )

    cleaned["exclude_long_gap"] = exclude_long_gap
    return cleaned


# ---------------------------------------------------------------------------
# 4. Merge
# ---------------------------------------------------------------------------

def merge_weather(load_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join weather onto the load/generation table on UTC timestamp."""
    return load_df.join(weather_df, how="left")


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------

def build_processed_dataset(
    force_download: bool = False,
    skip_download: bool = False,
    timeseries_path: Path = config.RAW_TIMESERIES_CSV,
    weather_path: Path = config.RAW_WEATHER_CSV,
) -> pd.DataFrame:
    """Run the full ingestion pipeline and write the processed parquet.

    Orchestrates download -> load -> clean -> merge -> write, and returns
    the resulting DataFrame. This is the function notebooks/01_eda.ipynb
    and the rest of the pipeline should call rather than reimplementing
    any of the steps above.

    `timeseries_path` / `weather_path` are overridable so tests (or an
    ad-hoc run against a sample) can point at fixtures instead of the raw
    OPSD download without touching config.py.
    """
    if not skip_download:
        download_opsd_data(force=force_download)

    load_df = load_raw_timeseries(timeseries_path)
    weather_df = load_raw_weather(weather_path)

    cleaned = clean_timeseries(load_df)
    merged = merge_weather(cleaned, weather_df)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(config.PROCESSED_HOURLY_PARQUET)
    logger.info(
        "Wrote %s (%d rows, %s to %s)",
        config.PROCESSED_HOURLY_PARQUET,
        len(merged),
        merged.index.min(),
        merged.index.max(),
    )
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and clean the OPSD data for the energy-dispatch project."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the raw CSVs even if they already exist in data/raw/.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Don't attempt any network access; use whatever is already in data/raw/ "
        "(fail with a clear error if a file is missing).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_arg_parser().parse_args(argv)
    build_processed_dataset(force_download=args.force_download, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
