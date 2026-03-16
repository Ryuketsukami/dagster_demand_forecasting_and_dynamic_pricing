"""
Silver layer — Pandera-validated DataFrames written to BigQuery via IO manager.

Assets:
    silver_airline_market  — validated OHLCV (long form, one row per date × ticker)
    silver_weather         — validated wide weather (one row per date, 25 cols)
    silver_currency        — validated FX rates (one row per date)
"""

import io
import os

import pandas as pd
import pandera.pandas as pa
from dagster import AssetExecutionContext, MetadataValue, Output, asset
from dagster_gcp import GCSResource

from quickstart_etl.partitions import daily_partitions

# ---------------------------------------------------------------------------
# Pandera schemas
# ---------------------------------------------------------------------------

_AIRLINE_TICKERS = ["DAL", "UAL", "AAL", "LUV"]
_HUB_CITIES = ["atl", "lax", "ord", "dfw", "jfk"]
_WEATHER_VARS = ["temp_max", "temp_min", "precip", "wind", "weather_code"]

AirlineMarketSchema = pa.DataFrameSchema(
    columns={
        "date": pa.Column(str, nullable=False),
        "ticker": pa.Column(str, pa.Check.isin(_AIRLINE_TICKERS), nullable=False),
        "open": pa.Column(float, pa.Check.ge(0), nullable=True),
        "high": pa.Column(float, pa.Check.ge(0), nullable=True),
        "low": pa.Column(float, pa.Check.ge(0), nullable=True),
        "close": pa.Column(float, pa.Check.ge(0), nullable=True),
        "volume": pa.Column(float, pa.Check.ge(0), nullable=True),
    },
    coerce=True,
    strict=False,
)

WeatherSchema = pa.DataFrameSchema(
    columns={
        "date": pa.Column(str, nullable=False),
        **{
            f"{city}_{var}": pa.Column(float, nullable=True)
            for city in _HUB_CITIES
            for var in _WEATHER_VARS
        },
    },
    coerce=True,
    strict=False,
)

CurrencySchema = pa.DataFrameSchema(
    columns={
        "date": pa.Column(str, nullable=False),
        "actual_date": pa.Column(str, nullable=True),
        "eur_usd": pa.Column(float, pa.Check.between(0.3, 3.0), nullable=True),
        "gbp_usd": pa.Column(float, pa.Check.between(0.3, 3.0), nullable=True),
        "brl_usd": pa.Column(float, pa.Check.between(0.5, 15.0), nullable=True),
    },
    coerce=True,
    strict=False,
)

# ---------------------------------------------------------------------------
# GCS helper
# ---------------------------------------------------------------------------


def _read_parquet_from_gcs(gcs_client, bucket_name: str, blob_path: str) -> pd.DataFrame | None:
    """Download a Parquet blob from GCS; return None if the blob does not exist."""
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None
    return pd.read_parquet(io.BytesIO(blob.download_as_bytes()))


# ---------------------------------------------------------------------------
# Silver assets
# ---------------------------------------------------------------------------


@asset(
    group_name="validation",
    partitions_def=daily_partitions,
    deps=["raw_airline_market_data"],
    io_manager_key="io_manager",
    kinds=["python", "bigquery", "pandera"],
)
def silver_airline_market(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> Output:
    """Validate Bronze airline OHLCV and write to BigQuery silver_airline_market.

    Market-closed partitions (weekends / holidays) are stored as empty tables —
    downstream assets check for emptiness before computing features.
    """
    partition_date = context.partition_key
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bronze_prefix = os.environ.get("GCS_BRONZE_PREFIX", "bronze")
    blob_path = f"{bronze_prefix}/airline_market/{partition_date}.parquet"

    gcs_client = gcs_resource.get_client()
    df = _read_parquet_from_gcs(gcs_client, bucket_name, blob_path)

    # Market-closed or missing Bronze file → empty partition
    if df is None or df.empty:
        empty = pd.DataFrame(
            {
                "date": pd.Series(dtype="str"),
                "ticker": pd.Series(dtype="str"),
                "open": pd.Series(dtype="float64"),
                "high": pd.Series(dtype="float64"),
                "low": pd.Series(dtype="float64"),
                "close": pd.Series(dtype="float64"),
                "volume": pd.Series(dtype="float64"),
            }
        )
        return Output(
            empty,
            metadata={
                "rows": MetadataValue.int(0),
                "market_open": MetadataValue.bool(False),
                "partition_date": MetadataValue.text(partition_date),
            },
        )

    validated = AirlineMarketSchema.validate(df)

    return Output(
        validated,
        metadata={
            "rows": MetadataValue.int(len(validated)),
            "market_open": MetadataValue.bool(True),
            "tickers": MetadataValue.text(", ".join(sorted(validated["ticker"].unique()))),
            "partition_date": MetadataValue.text(partition_date),
        },
    )


@asset(
    group_name="validation",
    partitions_def=daily_partitions,
    deps=["raw_weather_data"],
    io_manager_key="io_manager",
    kinds=["python", "bigquery", "pandera"],
)
def silver_weather(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> Output:
    """Validate Bronze weather data and write to BigQuery silver_weather."""
    partition_date = context.partition_key
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bronze_prefix = os.environ.get("GCS_BRONZE_PREFIX", "bronze")
    blob_path = f"{bronze_prefix}/weather/{partition_date}.parquet"

    gcs_client = gcs_resource.get_client()
    df = _read_parquet_from_gcs(gcs_client, bucket_name, blob_path)

    if df is None or df.empty:
        return Output(
            pd.DataFrame({"date": pd.Series(dtype="str")}),
            metadata={
                "rows": MetadataValue.int(0),
                "partition_date": MetadataValue.text(partition_date),
            },
        )

    validated = WeatherSchema.validate(df)

    # Derive a simple "any severe weather" flag for metadata visibility
    severe_cities = [
        c
        for c in _HUB_CITIES
        if validated[f"{c}_weather_code"].iloc[0] is not None
        and validated[f"{c}_weather_code"].iloc[0] >= 80
    ]

    return Output(
        validated,
        metadata={
            "rows": MetadataValue.int(len(validated)),
            "partition_date": MetadataValue.text(partition_date),
            "severe_weather_cities": MetadataValue.text(", ".join(severe_cities) or "none"),
        },
    )


@asset(
    group_name="validation",
    partitions_def=daily_partitions,
    deps=["raw_currency_rates"],
    io_manager_key="io_manager",
    kinds=["python", "bigquery", "pandera"],
)
def silver_currency(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> Output:
    """Validate Bronze currency rates and write to BigQuery silver_currency."""
    partition_date = context.partition_key
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    bronze_prefix = os.environ.get("GCS_BRONZE_PREFIX", "bronze")
    blob_path = f"{bronze_prefix}/currency/{partition_date}.parquet"

    gcs_client = gcs_resource.get_client()
    df = _read_parquet_from_gcs(gcs_client, bucket_name, blob_path)

    if df is None or df.empty:
        return Output(
            pd.DataFrame({"date": pd.Series(dtype="str")}),
            metadata={
                "rows": MetadataValue.int(0),
                "partition_date": MetadataValue.text(partition_date),
            },
        )

    validated = CurrencySchema.validate(df)

    eur = validated["eur_usd"].iloc[0]
    actual = (
        validated["actual_date"].iloc[0] if "actual_date" in validated.columns else partition_date
    )

    return Output(
        validated,
        metadata={
            "rows": MetadataValue.int(len(validated)),
            "partition_date": MetadataValue.text(partition_date),
            "actual_date": MetadataValue.text(str(actual)),
            "eur_usd": MetadataValue.float(float(eur) if eur is not None else 0.0),
        },
    )
