"""
Bronze layer — raw data ingestion to GCS Parquet.

Assets:
    raw_airline_market_data  — yfinance OHLCV for DAL, UAL, AAL, LUV
    raw_weather_data         — Open-Meteo daily weather for 5 hub cities
    raw_currency_rates       — Frankfurter USD → EUR/GBP/BRL exchange rates
"""

import io
import os
import time
from datetime import timedelta

import httpx
import pandas as pd
import yfinance as yf
from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset
from dagster_gcp import GCSResource

from quickstart_etl.partitions import daily_partitions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AIRLINE_TICKERS = ["DAL", "UAL", "AAL", "LUV"]

HUB_CITIES = {
    "ATL": {"latitude": 33.749, "longitude": -84.388, "timezone": "America/New_York"},
    "LAX": {"latitude": 33.942, "longitude": -118.408, "timezone": "America/Los_Angeles"},
    "ORD": {"latitude": 41.978, "longitude": -87.905, "timezone": "America/Chicago"},
    "DFW": {"latitude": 32.897, "longitude": -97.038, "timezone": "America/Chicago"},
    "JFK": {"latitude": 40.641, "longitude": -73.779, "timezone": "America/New_York"},
}

WEATHER_VARS = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet_to_gcs(
    df: pd.DataFrame,
    gcs_client,
    bucket_name: str,
    blob_path: str,
) -> None:
    """Serialise df to Parquet in memory and upload to GCS."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_file(buf, content_type="application/octet-stream")


def _http_get_with_retry(
    client: httpx.Client,
    url: str,
    params: dict,
    max_retries: int = 3,
    backoff_base: float = 2.0,
) -> dict:
    """GET request with simple exponential backoff on 429 / 5xx."""
    for attempt in range(max_retries):
        resp = client.get(url, params=params)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
            time.sleep(backoff_base**attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # should not reach here
    return {}


# ---------------------------------------------------------------------------
# Asset 1: Airline market data (yfinance)
# ---------------------------------------------------------------------------


@asset(
    group_name="ingestion",
    partitions_def=daily_partitions,
    kinds=["python", "gcs", "yfinance"],
)
def raw_airline_market_data(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> MaterializeResult:
    """Pull daily OHLCV for DAL, UAL, AAL, LUV from Yahoo Finance and write to GCS Parquet.

    Weekend / market-holiday partitions produce an empty file (0 rows) — this is expected
    and handled downstream by silver_airline_market.
    """
    partition_date = context.partition_key
    next_date = (pd.Timestamp(partition_date) + timedelta(days=1)).strftime("%Y-%m-%d")

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bronze_prefix = os.environ.get("GCS_BRONZE_PREFIX", "bronze")
    blob_path = f"{bronze_prefix}/airline_market/{partition_date}.parquet"

    # yfinance: end is exclusive so we get exactly partition_date
    df_raw = yf.download(
        tickers=AIRLINE_TICKERS,
        start=partition_date,
        end=next_date,
        interval="1d",
        auto_adjust=True,
        multi_level_index=True,
        progress=False,
        threads=True,
    )

    if df_raw.empty:
        df_long = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    else:
        # Stack MultiIndex (metric, Ticker) → long form
        df_long = df_raw.stack(level="Ticker").reset_index()
        df_long.rename(
            columns={
                "Date": "date",
                "Ticker": "ticker",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )
        df_long["date"] = pd.to_datetime(df_long["date"]).dt.strftime("%Y-%m-%d")
        # Keep only expected columns (yfinance may include extras)
        df_long = df_long[["date", "ticker", "open", "high", "low", "close", "volume"]]

    gcs_client = gcs_resource.get_client()
    _write_parquet_to_gcs(df_long, gcs_client, bucket_name, blob_path)

    return MaterializeResult(
        metadata={
            "gcs_path": MetadataValue.text(f"gs://{bucket_name}/{blob_path}"),
            "rows": MetadataValue.int(len(df_long)),
            "tickers": MetadataValue.text(", ".join(AIRLINE_TICKERS)),
            "partition_date": MetadataValue.text(partition_date),
            "market_open": MetadataValue.bool(len(df_long) > 0),
        }
    )


# ---------------------------------------------------------------------------
# Asset 2: Weather data (Open-Meteo)
# ---------------------------------------------------------------------------


@asset(
    group_name="ingestion",
    partitions_def=daily_partitions,
    kinds=["python", "gcs", "openmeteo"],
)
def raw_weather_data(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> MaterializeResult:
    """Pull daily weather for 5 hub cities from Open-Meteo and write wide Parquet to GCS.

    One row per partition date, 25 columns (5 cities × 5 weather variables).
    """
    partition_date = context.partition_key
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bronze_prefix = os.environ.get("GCS_BRONZE_PREFIX", "bronze")
    blob_path = f"{bronze_prefix}/weather/{partition_date}.parquet"

    row: dict = {"date": partition_date}

    with httpx.Client(timeout=30.0) as client:
        for city, coords in HUB_CITIES.items():
            city_key = city.lower()
            params = {
                "latitude": coords["latitude"],
                "longitude": coords["longitude"],
                "start_date": partition_date,
                "end_date": partition_date,
                "daily": WEATHER_VARS,
                "timezone": coords["timezone"],
            }
            data = _http_get_with_retry(
                client, "https://archive-api.open-meteo.com/v1/archive", params
            )
            daily = data.get("daily", {})
            row[f"{city_key}_temp_max"] = (daily.get("temperature_2m_max") or [None])[0]
            row[f"{city_key}_temp_min"] = (daily.get("temperature_2m_min") or [None])[0]
            row[f"{city_key}_precip"] = (daily.get("precipitation_sum") or [None])[0]
            row[f"{city_key}_wind"] = (daily.get("wind_speed_10m_max") or [None])[0]
            row[f"{city_key}_weather_code"] = (daily.get("weather_code") or [None])[0]

    df = pd.DataFrame([row])

    gcs_client = gcs_resource.get_client()
    _write_parquet_to_gcs(df, gcs_client, bucket_name, blob_path)

    return MaterializeResult(
        metadata={
            "gcs_path": MetadataValue.text(f"gs://{bucket_name}/{blob_path}"),
            "rows": MetadataValue.int(len(df)),
            "cities": MetadataValue.text(", ".join(HUB_CITIES.keys())),
            "partition_date": MetadataValue.text(partition_date),
        }
    )


# ---------------------------------------------------------------------------
# Asset 3: Currency rates (Frankfurter)
# ---------------------------------------------------------------------------


@asset(
    group_name="ingestion",
    partitions_def=daily_partitions,
    kinds=["python", "gcs", "frankfurter"],
)
def raw_currency_rates(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> MaterializeResult:
    """Pull daily USD → EUR/GBP/BRL rates from Frankfurter and write to GCS Parquet.

    Frankfurter returns the nearest prior business day for weekends/holidays.
    The `actual_date` column records what date the API actually returned.
    """
    partition_date = context.partition_key
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bronze_prefix = os.environ.get("GCS_BRONZE_PREFIX", "bronze")
    blob_path = f"{bronze_prefix}/currency/{partition_date}.parquet"

    with httpx.Client(timeout=30.0) as client:
        data = _http_get_with_retry(
            client,
            f"https://api.frankfurter.app/{partition_date}",
            {"from": "USD", "to": "EUR,GBP,BRL"},
        )

    rates = data.get("rates", {})
    df = pd.DataFrame(
        [
            {
                "date": partition_date,
                "actual_date": data.get("date", partition_date),
                "eur_usd": rates.get("EUR"),
                "gbp_usd": rates.get("GBP"),
                "brl_usd": rates.get("BRL"),
            }
        ]
    )

    gcs_client = gcs_resource.get_client()
    _write_parquet_to_gcs(df, gcs_client, bucket_name, blob_path)

    eur_val = rates.get("EUR", 0.0) or 0.0
    return MaterializeResult(
        metadata={
            "gcs_path": MetadataValue.text(f"gs://{bucket_name}/{blob_path}"),
            "rows": MetadataValue.int(len(df)),
            "actual_date": MetadataValue.text(data.get("date", partition_date)),
            "eur_usd": MetadataValue.float(float(eur_val)),
            "partition_date": MetadataValue.text(partition_date),
        }
    )
