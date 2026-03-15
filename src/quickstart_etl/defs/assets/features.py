"""
Gold layer — feature-engineered DataFrame written to BigQuery via IO manager.

Asset:
    gold_features  — one row per (date, ticker), 60+ feature columns + target_return
"""

import os

import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, Output, asset
from dagster_gcp import BigQueryResource

from quickstart_etl.lib.feature_logic import (
    add_weather_lag_features,
    compute_calendar_features,
    compute_currency_features,
    compute_market_features,
)
from quickstart_etl.partitions import daily_partitions

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

WEATHER_COLS = [
    f"{city}_{var}"
    for city in ["atl", "lax", "ord", "dfw", "jfk"]
    for var in ["temp_max", "temp_min", "precip", "wind", "weather_code"]
]

CURRENCY_COLS = ["eur_usd", "gbp_usd", "brl_usd"]

# Days of lookback needed to compute longest rolling window (EMA26 + BB20 + buffer)
_LOOKBACK_DAYS = 90


# ---------------------------------------------------------------------------
# Asset
# ---------------------------------------------------------------------------


@asset(
    group_name="features",
    partitions_def=daily_partitions,
    deps=["silver_airline_market", "silver_weather", "silver_currency"],
    io_manager_key="io_manager",
    kinds=["python", "bigquery", "feature-engineering"],
)
def gold_features(
    context: AssetExecutionContext,
    bigquery: BigQueryResource,
) -> Output:
    """Join Silver tables and compute the full feature set for each (date, ticker).

    Reads a 90-day lookback window of market + currency data to compute rolling
    indicators, then filters down to the single partition date before writing.
    Weather lags use yesterday's Silver weather row.

    Returns an empty DataFrame on market-closed partitions.
    """
    partition_date = context.partition_key
    project = os.environ["GCP_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET"]

    lookback_date = (pd.Timestamp(partition_date) - pd.Timedelta(days=_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )
    yesterday_date = (pd.Timestamp(partition_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    with bigquery.get_client() as bq:
        market_df = bq.query(
            f"SELECT * FROM `{project}.{dataset}.silver_airline_market`"
            f" WHERE date >= '{lookback_date}' AND date <= '{partition_date}'"
            f" ORDER BY date, ticker"
        ).to_dataframe()

        weather_df = bq.query(
            f"SELECT * FROM `{project}.{dataset}.silver_weather`"
            f" WHERE date IN ('{yesterday_date}', '{partition_date}')"
            f" ORDER BY date"
        ).to_dataframe()

        currency_df = bq.query(
            f"SELECT * FROM `{project}.{dataset}.silver_currency`"
            f" WHERE date >= '{lookback_date}' AND date <= '{partition_date}'"
            f" ORDER BY date"
        ).to_dataframe()

    # Market closed — nothing to compute
    if market_df.empty:
        context.log.info(f"No market data for {partition_date} (market closed).")
        return Output(
            pd.DataFrame(),
            metadata={
                "rows": MetadataValue.int(0),
                "market_open": MetadataValue.bool(False),
                "partition_date": MetadataValue.text(partition_date),
            },
        )

    # ------------------------------------------------------------------ #
    # Currency: compute rolling features over lookback, then grab partition date
    # ------------------------------------------------------------------ #
    currency_partition_row = None
    currency_feature_cols: list[str] = []

    if not currency_df.empty:
        currency_df = compute_currency_features(currency_df, CURRENCY_COLS)
        partition_currency = currency_df[currency_df["date"] == partition_date]
        if not partition_currency.empty:
            # All currency feature cols except date / actual_date
            currency_feature_cols = [
                c for c in partition_currency.columns if c not in ("date", "actual_date")
            ]
            currency_partition_row = partition_currency.iloc[0]

    # ------------------------------------------------------------------ #
    # Weather: today + yesterday rows
    # ------------------------------------------------------------------ #
    weather_today_row = None
    weather_yesterday_row = None

    if not weather_df.empty:
        today_rows = weather_df[weather_df["date"] == partition_date]
        yesterday_rows = weather_df[weather_df["date"] == yesterday_date]
        weather_today_row = today_rows.iloc[0] if not today_rows.empty else None
        weather_yesterday_row = yesterday_rows.iloc[0] if not yesterday_rows.empty else None

    # ------------------------------------------------------------------ #
    # Per-ticker market features + join external signals
    # ------------------------------------------------------------------ #
    feature_frames: list[pd.DataFrame] = []

    for ticker, ticker_df in market_df.groupby("ticker"):
        ticker_features = compute_market_features(ticker_df.copy())

        # Keep only the partition-date row
        partition_row_df = ticker_features[ticker_features["date"] == partition_date]
        if partition_row_df.empty:
            continue

        # Weather features (today + lag1)
        partition_row_df = add_weather_lag_features(
            partition_row_df,
            weather_today=weather_today_row,
            weather_yesterday=weather_yesterday_row,
            weather_cols=WEATHER_COLS,
        )

        # Currency features
        if currency_partition_row is not None:
            for col in currency_feature_cols:
                partition_row_df = partition_row_df.copy()
                partition_row_df[col] = currency_partition_row[col]

        feature_frames.append(partition_row_df)

    if not feature_frames:
        return Output(
            pd.DataFrame(),
            metadata={
                "rows": MetadataValue.int(0),
                "market_open": MetadataValue.bool(False),
                "partition_date": MetadataValue.text(partition_date),
            },
        )

    gold_df = pd.concat(feature_frames, ignore_index=True)
    gold_df = compute_calendar_features(gold_df)

    # target_return is unknown at Gold time (future close not yet available).
    # Training asset computes and back-fills this from the full gold_features table.
    gold_df["target_return"] = None

    tickers_present = sorted(gold_df["ticker"].unique().tolist())
    n_features = gold_df.shape[1]

    return Output(
        gold_df,
        metadata={
            "rows": MetadataValue.int(len(gold_df)),
            "tickers": MetadataValue.text(", ".join(tickers_present)),
            "feature_columns": MetadataValue.int(n_features),
            "market_open": MetadataValue.bool(True),
            "partition_date": MetadataValue.text(partition_date),
        },
    )
