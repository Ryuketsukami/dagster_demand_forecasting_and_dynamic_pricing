"""
Shared pytest fixtures for the quickstart_etl test suite.

Fixture hierarchy:
  Session-scoped  — dagster_home, env_vars
  Function-scoped — DataFrames, mock resources, mock GCS/BQ clients
"""

from __future__ import annotations

import io
import json
import os
import pickle
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------------------------
# Session-scoped: Dagster instance + minimal required env vars
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def dagster_home(tmp_path_factory):
    """Isolated DAGSTER_HOME for the whole test session."""
    home = tmp_path_factory.mktemp("dagster_home")
    os.environ["DAGSTER_HOME"] = str(home)
    yield home


@pytest.fixture(scope="session", autouse=True)
def env_vars():
    """Set minimal environment variables required by all assets."""
    overrides = {
        "GCP_PROJECT_ID": "test-project",
        "BIGQUERY_DATASET": "test_dataset",
        "BIGQUERY_LOCATION": "EU",
        "GCS_BUCKET_NAME": "test-bucket",
        "GCS_BRONZE_PREFIX": "bronze",
        "GCS_CHAMPION_PREFIX": "champion",
        "GCS_MONITORING_PREFIX": "monitoring",
        "SERVING_HOST": "0.0.0.0",
        "SERVING_PORT": "8080",
    }
    original = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    yield overrides
    # Restore
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Raw DataFrames (enough rows for all rolling indicators: 60+ trading days)
# ---------------------------------------------------------------------------

TICKERS = ["DAL", "UAL", "AAL", "LUV"]
N_DAYS = 80  # enough for EMA26 + BB20 warmup
BASE_DATE = pd.Timestamp("2024-01-02")  # Tuesday — a trading day


def _date_range(n: int) -> list[str]:
    """Generate n weekday dates starting from BASE_DATE."""
    dates = pd.bdate_range(start=BASE_DATE, periods=n)
    return [d.strftime("%Y-%m-%d") for d in dates]


@pytest.fixture
def trading_dates() -> list[str]:
    return _date_range(N_DAYS)


@pytest.fixture
def sample_ohlcv_df(trading_dates) -> pd.DataFrame:
    """Long-form OHLCV DataFrame with all 4 tickers over N_DAYS trading days.

    Prices follow a simple random walk seeded for reproducibility.
    """
    rng = np.random.default_rng(42)
    rows = []
    for ticker in TICKERS:
        price = 50.0
        vol_base = 1_000_000
        for date in trading_dates:
            ret = rng.normal(0.001, 0.02)
            price = max(1.0, price * (1 + ret))
            high = price * (1 + abs(rng.normal(0, 0.005)))
            low = price * (1 - abs(rng.normal(0, 0.005)))
            open_ = price * (1 + rng.normal(0, 0.003))
            volume = float(int(vol_base * rng.uniform(0.8, 1.2)))
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": round(open_, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(price, 4),
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def single_ticker_ohlcv(sample_ohlcv_df) -> pd.DataFrame:
    """OHLCV for a single ticker (DAL), sorted by date."""
    return sample_ohlcv_df[sample_ohlcv_df["ticker"] == "DAL"].copy().reset_index(drop=True)


@pytest.fixture
def sample_weather_df(trading_dates) -> pd.DataFrame:
    """Wide weather DataFrame: one row per date, 25 columns (5 cities × 5 vars)."""
    cities = ["atl", "lax", "ord", "dfw", "jfk"]
    rng = np.random.default_rng(7)
    rows = []
    for date in trading_dates:
        row: dict = {"date": date}
        for city in cities:
            row[f"{city}_temp_max"] = round(float(rng.uniform(5, 35)), 1)
            row[f"{city}_temp_min"] = round(float(rng.uniform(-5, 20)), 1)
            row[f"{city}_precip"] = round(float(rng.uniform(0, 10)), 2)
            row[f"{city}_wind"] = round(float(rng.uniform(0, 50)), 1)
            row[f"{city}_weather_code"] = int(rng.choice([0, 1, 3, 61, 80]))
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def sample_currency_df(trading_dates) -> pd.DataFrame:
    """Currency rates DataFrame: one row per date, USD→EUR/GBP/BRL."""
    rng = np.random.default_rng(13)
    rows = []
    for date in trading_dates:
        rows.append(
            {
                "date": date,
                "actual_date": date,
                "eur_usd": round(float(rng.uniform(0.85, 0.95)), 4),
                "gbp_usd": round(float(rng.uniform(0.72, 0.82)), 4),
                "brl_usd": round(float(rng.uniform(4.8, 5.5)), 4),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_gold_df(sample_ohlcv_df, sample_weather_df, sample_currency_df) -> pd.DataFrame:
    """Minimal gold_features DataFrame built from the raw fixtures.

    Applies feature_logic functions directly — used for training/monitoring tests.
    """
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from quickstart_etl.lib.feature_logic import (
        add_weather_lag_features,
        compute_calendar_features,
        compute_currency_features,
        compute_market_features,
    )

    weather_cols = [
        f"{city}_{var}"
        for city in ["atl", "lax", "ord", "dfw", "jfk"]
        for var in ["temp_max", "temp_min", "precip", "wind", "weather_code"]
    ]
    currency_cols = ["eur_usd", "gbp_usd", "brl_usd"]

    # Currency features over full window
    currency_df = compute_currency_features(sample_currency_df.copy(), currency_cols)

    frames = []
    for ticker, tdf in sample_ohlcv_df.groupby("ticker"):
        feat_df = compute_market_features(tdf.copy())
        dates = feat_df["date"].tolist()
        weather_rows = sample_weather_df.set_index("date")
        for i, date in enumerate(dates):
            row_df = feat_df[feat_df["date"] == date].copy()
            if row_df.empty:
                continue
            w_today = weather_rows.loc[date] if date in weather_rows.index else None
            prev_date = (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            w_prev = weather_rows.loc[prev_date] if prev_date in weather_rows.index else None
            row_df = add_weather_lag_features(row_df, w_today, w_prev, weather_cols)
            cur_row = currency_df[currency_df["date"] == date]
            if not cur_row.empty:
                for col in [c for c in cur_row.columns if c not in ("date", "actual_date")]:
                    row_df[col] = cur_row[col].values[0]
            frames.append(row_df)

    gold = pd.concat(frames, ignore_index=True)
    gold = compute_calendar_features(gold)
    gold["target_return"] = None
    return gold


# ---------------------------------------------------------------------------
# Trained reference model (tiny, for serving tests)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "log_return",
    "high_low_range",
    "overnight_gap",
    "vol_5d",
    "vol_20d",
    "return_5d",
    "return_20d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "vol_ratio_5d",
]


@pytest.fixture
def tiny_model_and_cols():
    """A trained HGBR model on 50 synthetic rows + the feature column list."""
    rng = np.random.default_rng(99)
    n = 50
    X = rng.standard_normal((n, len(FEATURE_COLS))).astype(np.float32)
    y = rng.standard_normal(n)
    model = HistGradientBoostingRegressor(max_iter=10, random_state=0)
    model.fit(X, y)
    return model, list(FEATURE_COLS)


# ---------------------------------------------------------------------------
# GCS mock
# ---------------------------------------------------------------------------


class FakeGCSBlob:
    """In-memory GCS blob."""

    def __init__(self):
        self._data: bytes | None = None

    def exists(self) -> bool:
        return self._data is not None

    def upload_from_file(self, fh, content_type=None):
        self._data = fh.read()

    def upload_from_string(self, data: bytes | str, content_type=None):
        self._data = data.encode() if isinstance(data, str) else data

    def download_as_bytes(self) -> bytes:
        if self._data is None:
            raise Exception("Blob does not exist")
        return self._data


class FakeGCSBucket:
    def __init__(self):
        self._blobs: dict[str, FakeGCSBlob] = {}

    def blob(self, path: str) -> FakeGCSBlob:
        if path not in self._blobs:
            self._blobs[path] = FakeGCSBlob()
        return self._blobs[path]

    def list_blobs(self, prefix: str = "") -> list[FakeGCSBlob]:
        """Return all blobs whose key starts with prefix, with a .name attribute."""
        results = []
        for name, blob in self._blobs.items():
            if name.startswith(prefix) and blob.exists():
                blob.name = name
                results.append(blob)
        return results


class FakeGCSClient:
    def __init__(self):
        self._buckets: dict[str, FakeGCSBucket] = {}

    def bucket(self, name: str) -> FakeGCSBucket:
        if name not in self._buckets:
            self._buckets[name] = FakeGCSBucket()
        return self._buckets[name]


@pytest.fixture
def fake_gcs_client() -> FakeGCSClient:
    return FakeGCSClient()


@pytest.fixture
def mock_gcs_resource(fake_gcs_client) -> MagicMock:
    """GCSResource mock that uses the in-memory FakeGCSClient."""
    resource = MagicMock()
    resource.get_client.return_value = fake_gcs_client
    return resource


# ---------------------------------------------------------------------------
# BigQuery mock
# ---------------------------------------------------------------------------


class FakeBQQueryJob:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


class FakeBQClient:
    """In-memory BigQuery client that resolves queries against registered DataFrames."""

    def __init__(self):
        self._tables: dict[str, pd.DataFrame] = {}

    def register(self, table_id: str, df: pd.DataFrame):
        """Register a DataFrame as the result for queries mentioning table_id."""
        self._tables[table_id] = df.copy()

    def query(self, sql: str) -> FakeBQQueryJob:
        for table_id, df in self._tables.items():
            # Simple substring match on the table name
            if table_id in sql:
                return FakeBQQueryJob(df)
        return FakeBQQueryJob(pd.DataFrame())

    def insert_rows_json(self, table_id, rows) -> list:
        """No-op: accept serving log inserts."""
        return []

    def create_table(self, table, exists_ok=False):
        """No-op table creation."""


@pytest.fixture
def fake_bq_client() -> FakeBQClient:
    return FakeBQClient()


@pytest.fixture
def mock_bigquery_resource(fake_bq_client) -> MagicMock:
    """BigQueryResource mock that returns a FakeBQClient as context manager."""
    resource = MagicMock()
    resource.get_client.return_value.__enter__ = lambda s: fake_bq_client
    resource.get_client.return_value.__exit__ = MagicMock(return_value=False)
    return resource


# ---------------------------------------------------------------------------
# Parquet helpers — write/read from FakeGCSClient
# ---------------------------------------------------------------------------


def write_parquet_to_fake_gcs(fake_gcs: FakeGCSClient, bucket: str, path: str, df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    fake_gcs.bucket(bucket).blob(path).upload_from_string(buf.getvalue())


def write_json_to_fake_gcs(fake_gcs: FakeGCSClient, bucket: str, path: str, data: dict):
    fake_gcs.bucket(bucket).blob(path).upload_from_string(
        json.dumps(data).encode(), content_type="application/json"
    )


def write_pickle_to_fake_gcs(fake_gcs: FakeGCSClient, bucket: str, path: str, obj):
    fake_gcs.bucket(bucket).blob(path).upload_from_string(pickle.dumps(obj))
