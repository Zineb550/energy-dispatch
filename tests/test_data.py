"""
Tests for energy_dispatch.data — run entirely against small, checked-in
fixtures (tests/fixtures/) or synthetic in-memory frames, never against
the real OPSD download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_dispatch import config, data

# ---------------------------------------------------------------------------
# Loading raw fixtures
# ---------------------------------------------------------------------------

def test_load_raw_timeseries_reads_fixture():
    df = data.load_raw_timeseries(config.SAMPLE_TIMESERIES_CSV)

    assert list(df.columns) == [
        config.LOAD_ACTUAL_COL,
        config.LOAD_FORECAST_COL,
        config.SOLAR_GEN_COL,
        config.WIND_GEN_COL,
    ]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    # The fixture deliberately contains one duplicated timestamp.
    assert df.index.duplicated().sum() == 1


def test_load_raw_weather_reads_fixture():
    df = data.load_raw_weather(config.SAMPLE_WEATHER_CSV)

    assert list(df.columns) == [
        config.WEATHER_TEMPERATURE_COL,
        config.WEATHER_RADIATION_DIRECT_COL,
        config.WEATHER_RADIATION_DIFFUSE_COL,
    ]
    assert str(df.index.tz) == "UTC"


def test_load_raw_timeseries_missing_file_raises_with_actionable_message(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="python -m energy_dispatch.data"):
        data.load_raw_timeseries(missing)


# ---------------------------------------------------------------------------
# Quality checks (synthetic frames — precise control over the anomaly)
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_hourly_frame() -> pd.DataFrame:
    idx = pd.date_range("2019-01-01", periods=48, freq="h", tz="UTC")
    return pd.DataFrame(
        {config.LOAD_ACTUAL_COL: 30_000 + np.sin(np.linspace(0, 4 * np.pi, 48)) * 1000},
        index=idx,
    )


def test_check_quality_flags_missing_hour(clean_hourly_frame):
    df = clean_hourly_frame.drop(clean_hourly_frame.index[10])
    flagged = data.check_quality(df)
    assert flagged["is_missing_hour"].sum() == 1
    assert flagged.loc[flagged["is_missing_hour"], config.LOAD_ACTUAL_COL].isna().all()


def test_check_quality_flags_duplicate_timestamp(clean_hourly_frame):
    dup = pd.concat([clean_hourly_frame, clean_hourly_frame.iloc[[0]]])
    flagged = data.check_quality(dup)
    assert flagged["is_duplicate_timestamp"].sum() == 2  # both copies flagged


def test_check_quality_flags_implausible_load(clean_hourly_frame):
    df = clean_hourly_frame.copy()
    df.iloc[5, df.columns.get_loc(config.LOAD_ACTUAL_COL)] = -100
    flagged = data.check_quality(df)
    assert flagged["is_implausible_load"].iloc[5]
    assert flagged["is_implausible_load"].sum() == 1


def test_check_quality_flags_spike(clean_hourly_frame):
    df = clean_hourly_frame.copy()
    df.iloc[20, df.columns.get_loc(config.LOAD_ACTUAL_COL)] = 500_000
    flagged = data.check_quality(df)
    assert flagged["is_spike"].iloc[20]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_clean_timeseries_interpolates_short_gap(clean_hourly_frame):
    df = clean_hourly_frame.copy()
    df.iloc[10:12, df.columns.get_loc(config.LOAD_ACTUAL_COL)] = np.nan  # 2h gap

    cleaned = data.clean_timeseries(df)

    assert not cleaned[config.LOAD_ACTUAL_COL].iloc[10:12].isna().any()
    assert not cleaned["exclude_long_gap"].iloc[10:12].any()


def test_clean_timeseries_excludes_long_gap(clean_hourly_frame):
    df = clean_hourly_frame.copy()
    df.iloc[10:16, df.columns.get_loc(config.LOAD_ACTUAL_COL)] = np.nan  # 6h gap

    cleaned = data.clean_timeseries(df)

    assert cleaned["exclude_long_gap"].iloc[10:16].all()
    # Long gaps are left NaN, not interpolated over.
    assert cleaned[config.LOAD_ACTUAL_COL].iloc[10:16].isna().all()


def test_clean_timeseries_drops_duplicate_keeping_first(clean_hourly_frame):
    dup = pd.concat([clean_hourly_frame, clean_hourly_frame.iloc[[0]]])
    cleaned = data.clean_timeseries(dup)
    assert not cleaned.index.duplicated().any()
    assert len(cleaned) == len(clean_hourly_frame)


def test_clean_timeseries_nans_out_implausible_and_spike_before_interpolating(clean_hourly_frame):
    df = clean_hourly_frame.copy()
    df.iloc[10, df.columns.get_loc(config.LOAD_ACTUAL_COL)] = -999  # implausible
    cleaned = data.clean_timeseries(df)
    # Should have been NaN'd out then interpolated (isolated single bad
    # point = a 1h gap, well under the interpolation threshold) rather
    # than kept as -999.
    assert cleaned[config.LOAD_ACTUAL_COL].iloc[10] > 0


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def test_merge_weather_left_joins_on_timestamp(clean_hourly_frame):
    weather = pd.DataFrame(
        {config.WEATHER_TEMPERATURE_COL: 20.0}, index=clean_hourly_frame.index[:10]
    )
    merged = data.merge_weather(clean_hourly_frame, weather)
    assert len(merged) == len(clean_hourly_frame)
    assert merged[config.WEATHER_TEMPERATURE_COL].notna().sum() == 10


# ---------------------------------------------------------------------------
# End-to-end against fixtures
# ---------------------------------------------------------------------------

def test_build_processed_dataset_end_to_end_on_fixtures(tmp_path, monkeypatch):
    out_parquet = tmp_path / "processed.parquet"
    monkeypatch.setattr(config, "PROCESSED_HOURLY_PARQUET", out_parquet)
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)

    result = data.build_processed_dataset(
        skip_download=True,
        timeseries_path=config.SAMPLE_TIMESERIES_CSV,
        weather_path=config.SAMPLE_WEATHER_CSV,
    )

    assert out_parquet.exists()
    assert config.WEATHER_TEMPERATURE_COL in result.columns
    assert config.LOAD_ACTUAL_COL in result.columns
    assert "exclude_long_gap" in result.columns
    assert not result.index.duplicated().any()


def test_build_processed_dataset_skip_download_without_files_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_TIMESERIES_CSV", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        data.build_processed_dataset(skip_download=True, timeseries_path=tmp_path / "nope.csv")
