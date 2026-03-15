"""
Standalone FastAPI serving application — Airline Return Predictor.

Start with:
    uvicorn quickstart_etl.lib.serving_app:app --host 0.0.0.0 --port 8080

The app loads the champion model from GCS on the first request (cached in
memory) and queries gold_features from BigQuery for the requested (date, ticker).
All configuration comes from environment variables (same .env as the Dagster pipeline).

Prediction logs are written asynchronously to BigQuery table `serving_logs`
after each successful prediction. This table feeds the concept-drift monitoring
layer (Evidently RegressionPreset in drift_report).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from google.cloud import bigquery as bq_client
from google.cloud import storage
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Background executor for fire-and-forget BigQuery log writes
_log_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="serving_log")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Airline Return Predictor",
    description="Predicts next-day percentage return for DAL, UAL, AAL, LUV.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

_VALID_TICKERS = {"DAL", "UAL", "AAL", "LUV"}


class PredictRequest(BaseModel):
    date: str  # YYYY-MM-DD — date for which features exist in gold_features
    ticker: str

    @field_validator("ticker")
    @classmethod
    def ticker_must_be_valid(cls, v: str) -> str:
        v = v.upper()
        if v not in _VALID_TICKERS:
            raise ValueError(f"ticker must be one of {sorted(_VALID_TICKERS)}")
        return v

    @field_validator("date")
    @classmethod
    def date_must_be_iso(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")
        return v


class PredictResponse(BaseModel):
    ticker: str
    prediction_date: str
    predicted_return: float
    model_version: str
    prediction_timestamp: str


# ---------------------------------------------------------------------------
# Lazy-loaded singletons (loaded once per process, cached in memory)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_champion_model():
    """Load the champion model pickle from GCS."""
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    champion_prefix = os.environ.get("GCS_CHAMPION_PREFIX", "champion")
    gcs = storage.Client()
    blob = gcs.bucket(bucket_name).blob(f"{champion_prefix}/model.pkl")
    if not blob.exists():
        raise RuntimeError("Champion model not found in GCS — run the training pipeline first.")
    return pickle.loads(blob.download_as_bytes())


@lru_cache(maxsize=1)
def _load_feature_cols() -> list[str]:
    """Load the feature column list saved alongside the champion model."""
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    champion_prefix = os.environ.get("GCS_CHAMPION_PREFIX", "champion")
    gcs = storage.Client()
    blob = gcs.bucket(bucket_name).blob(f"{champion_prefix}/feature_cols.json")
    if not blob.exists():
        raise RuntimeError("Champion feature_cols.json not found in GCS.")
    return json.loads(blob.download_as_bytes())


@lru_cache(maxsize=1)
def _load_champion_version() -> str:
    """Return a version string from champion metrics (RMSE as proxy)."""
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    champion_prefix = os.environ.get("GCS_CHAMPION_PREFIX", "champion")
    gcs = storage.Client()
    blob = gcs.bucket(bucket_name).blob(f"{champion_prefix}/metrics.json")
    if not blob.exists():
        return "unknown"
    metrics = json.loads(blob.download_as_bytes())
    rmse = metrics.get("rmse", 0)
    return f"rmse={rmse:.5f}"


# ---------------------------------------------------------------------------
# Serving log writer (fire-and-forget background write to BigQuery)
# ---------------------------------------------------------------------------


def _write_serving_log(
    date: str,
    ticker: str,
    predicted_return: float,
    model_version: str,
    prediction_timestamp: str,
) -> None:
    """Insert one prediction record into BigQuery serving_logs (non-blocking).

    Called via ThreadPoolExecutor — errors are logged but never propagate to the
    caller, so a BigQuery issue never fails a live prediction response.

    Table schema (auto-created on first write if it doesn't exist):
        date STRING, ticker STRING, predicted_return FLOAT64,
        model_version STRING, prediction_timestamp TIMESTAMP
    """
    try:
        project = os.environ["GCP_PROJECT_ID"]
        dataset = os.environ["BIGQUERY_DATASET"]
        table_id = f"{project}.{dataset}.serving_logs"

        bq = bq_client.Client(project=project)

        # Create the table if it doesn't exist yet
        schema = [
            bq_client.SchemaField("date", "STRING"),
            bq_client.SchemaField("ticker", "STRING"),
            bq_client.SchemaField("predicted_return", "FLOAT64"),
            bq_client.SchemaField("model_version", "STRING"),
            bq_client.SchemaField("prediction_timestamp", "TIMESTAMP"),
        ]
        table_ref = bq_client.Table(table_id, schema=schema)
        bq.create_table(table_ref, exists_ok=True)

        rows = [
            {
                "date": date,
                "ticker": ticker,
                "predicted_return": predicted_return,
                "model_version": model_version,
                "prediction_timestamp": prediction_timestamp,
            }
        ]
        errors = bq.insert_rows_json(table_id, rows)
        if errors:
            logger.warning("BigQuery serving_log insert errors: %s", errors)
    except Exception as exc:
        logger.warning("Failed to write serving log (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict next-day return for a given (date, ticker) using gold_features from BigQuery."""
    project = os.environ["GCP_PROJECT_ID"]
    dataset = os.environ["BIGQUERY_DATASET"]

    # Load cached artifacts
    try:
        model = _load_champion_model()
        feature_cols = _load_feature_cols()
        model_version = _load_champion_version()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Query gold_features for the requested (date, ticker)
    bq = bq_client.Client(project=project)
    query = (
        f"SELECT * FROM `{project}.{dataset}.gold_features`"
        f" WHERE date = '{request.date}' AND ticker = '{request.ticker}'"
        f" LIMIT 1"
    )
    result = bq.query(query).to_dataframe()

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No gold_features found for date={request.date}, ticker={request.ticker}. "
                "Ensure the daily ingestion pipeline has run for this date."
            ),
        )

    # Select feature columns in training order
    missing = [c for c in feature_cols if c not in result.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Feature schema mismatch — missing columns: {missing}",
        )

    import numpy as np

    X = result[feature_cols].to_numpy(dtype=np.float32)
    predicted_return = float(model.predict(X)[0])
    prediction_timestamp = datetime.now(timezone.utc).isoformat()

    # Fire-and-forget async log write — does NOT block the response
    _log_executor.submit(
        _write_serving_log,
        request.date,
        request.ticker,
        predicted_return,
        model_version,
        prediction_timestamp,
    )

    return PredictResponse(
        ticker=request.ticker,
        prediction_date=request.date,
        predicted_return=predicted_return,
        model_version=model_version,
        prediction_timestamp=prediction_timestamp,
    )
