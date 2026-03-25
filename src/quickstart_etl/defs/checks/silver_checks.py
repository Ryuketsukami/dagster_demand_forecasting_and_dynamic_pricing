"""
Asset checks for Silver layer data quality.

These checks run after each Silver asset materialises and surface results
as named, queryable checks in the Dagster+ UI — separate from the asset
materialisation itself.

Checks:
    silver_market_partition_completeness  — row count in {0, 4} per trading day
    silver_weather_null_rate              — weather columns < 20% null
    silver_currency_rate_bounds           — FX rates within plausible ranges
"""

import os

from dagster import AssetCheckResult, AssetCheckSeverity, MetadataValue, asset_check
from dagster_gcp import BigQueryResource


def _DATASET() -> str:
    return os.environ["BIGQUERY_DATASET"]


def _PROJECT() -> str:
    return os.environ["GCP_PROJECT_ID"]


# Expected number of tickers per trading day (0 = market closed, 4 = open)
_EXPECTED_TICKERS = 4

# Weather columns that must not be predominantly null
_WEATHER_COLS = [
    f"{city}_{var}"
    for city in ["atl", "lax", "ord", "dfw", "jfk"]
    for var in ["temp_max", "temp_min", "precip", "wind", "weather_code"]
]
_MAX_NULL_RATE = 0.20  # 20% null threshold


# ---------------------------------------------------------------------------
# silver_airline_market checks
# ---------------------------------------------------------------------------


@asset_check(asset="silver_airline_market", blocking=True)
def silver_market_partition_completeness(bigquery: BigQueryResource) -> AssetCheckResult:
    """Verify each trading-day partition has exactly 0 or 4 rows (one per ticker).

    0 rows = market closed (weekend / holiday) — valid.
    4 rows = one row per ticker (DAL, UAL, AAL, LUV) — valid.
    Any other count indicates a partial ingest or duplicate rows.
    """
    project = _PROJECT()
    dataset = _DATASET()

    with bigquery.get_client() as bq:
        result = bq.query(
            f"SELECT COUNT(*) AS row_count"
            f" FROM `{project}.{dataset}.silver_airline_market`"
            f" WHERE date = (SELECT MAX(date) FROM `{project}.{dataset}.silver_airline_market`)"
        ).to_dataframe()

    row_count = int(result["row_count"].iloc[0])
    passed = row_count in (0, _EXPECTED_TICKERS)

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "row_count": MetadataValue.int(row_count),
            "expected": MetadataValue.text(f"0 (closed) or {_EXPECTED_TICKERS} (open)"),
        },
        description=(
            f"Latest partition has {row_count} rows — "
            + ("OK" if passed else f"expected 0 or {_EXPECTED_TICKERS}")
        ),
    )


# ---------------------------------------------------------------------------
# silver_weather checks
# ---------------------------------------------------------------------------


@asset_check(asset="silver_weather", blocking=False)
def silver_weather_null_rate(bigquery: BigQueryResource) -> AssetCheckResult:
    """Warn if any weather column has > 20% null values in the last 7 days.

    Open-Meteo occasionally returns nulls for weather_code on some dates.
    A high null rate indicates an upstream API issue rather than a closed
    market day.
    """
    project = _PROJECT()
    dataset = _DATASET()

    col_null_checks = ", ".join(
        f"COUNTIF({col} IS NULL) / COUNT(*) AS {col}_null_rate" for col in _WEATHER_COLS
    )
    with bigquery.get_client() as bq:
        df = bq.query(
            f"SELECT {col_null_checks}"
            f" FROM `{project}.{dataset}.silver_weather`"
            f" WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)"
        ).to_dataframe()

    if df.empty:
        return AssetCheckResult(
            passed=True,
            description="No weather data in the last 7 days — skipping null check.",
        )

    offending = {
        col: float(df[f"{col}_null_rate"].iloc[0])
        for col in _WEATHER_COLS
        if float(df[f"{col}_null_rate"].iloc[0]) > _MAX_NULL_RATE
    }
    passed = len(offending) == 0

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.WARN,
        metadata={
            "high_null_columns": MetadataValue.text(
                ", ".join(f"{c}={v:.1%}" for c, v in offending.items()) or "none"
            ),
            "null_threshold": MetadataValue.float(_MAX_NULL_RATE),
        },
        description=(
            "All weather columns within null threshold."
            if passed
            else f"{len(offending)} column(s) exceed {_MAX_NULL_RATE:.0%} null rate."
        ),
    )


# ---------------------------------------------------------------------------
# silver_currency checks
# ---------------------------------------------------------------------------


@asset_check(asset="silver_currency", blocking=False)
def silver_currency_rate_bounds(bigquery: BigQueryResource) -> AssetCheckResult:
    """Warn if any FX rate falls outside plausible historical bounds.

    Bounds are generous to handle long-term drift but catch obvious data errors:
      eur_usd: 0.7 – 1.6
      gbp_usd: 0.8 – 2.0
      brl_usd: 1.5 – 10.0
    """
    project = _PROJECT()
    dataset = _DATASET()

    with bigquery.get_client() as bq:
        df = bq.query(
            f"SELECT"
            f"  MIN(eur_usd) AS eur_min, MAX(eur_usd) AS eur_max,"
            f"  MIN(gbp_usd) AS gbp_min, MAX(gbp_usd) AS gbp_max,"
            f"  MIN(brl_usd) AS brl_min, MAX(brl_usd) AS brl_max"
            f" FROM `{project}.{dataset}.silver_currency`"
            f" WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)"
            f"   AND eur_usd IS NOT NULL"
        ).to_dataframe()

    if df.empty:
        return AssetCheckResult(passed=True, description="No currency data in last 30 days.")

    violations = []
    row = df.iloc[0]
    if not (0.7 <= row["eur_min"] and row["eur_max"] <= 1.6):
        violations.append(f"eur_usd range [{row['eur_min']:.4f}, {row['eur_max']:.4f}]")
    if not (0.8 <= row["gbp_min"] and row["gbp_max"] <= 2.0):
        violations.append(f"gbp_usd range [{row['gbp_min']:.4f}, {row['gbp_max']:.4f}]")
    if not (1.5 <= row["brl_min"] and row["brl_max"] <= 10.0):
        violations.append(f"brl_usd range [{row['brl_min']:.4f}, {row['brl_max']:.4f}]")

    passed = len(violations) == 0

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.WARN,
        metadata={"violations": MetadataValue.text("; ".join(violations) or "none")},
        description=(
            "All FX rates within expected bounds."
            if passed
            else f"Out-of-bounds FX rates detected: {'; '.join(violations)}"
        ),
    )
