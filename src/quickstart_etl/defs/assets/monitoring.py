"""
Monitoring layer — data drift and concept drift detection via Evidently.

Asset:
    drift_report  — runs daily and writes a JSON report to GCS containing:

    Data drift (always):
        Compares last 7 days of gold_features feature distributions against
        the training baseline using Evidently DataDriftPreset.

    Concept drift (when serving_logs data is available):
        Joins serving_logs predictions against actual next-day returns computed
        from silver_airline_market and runs Evidently RegressionPreset.
        This detects model degradation independent of feature distribution shifts.

The drift_retrain_sensor (sensors/drift_sensors.py) fires on each materialisation
and triggers retrain_job when share_of_drifted_columns > 0.3.
"""

import io
import json
import os
from datetime import datetime, timezone

import pandas as pd
from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset
from dagster_gcp import BigQueryResource, GCSResource

from quickstart_etl.partitions import daily_partitions

# Fraction of features that must drift before we consider it "significant"
_DRIFT_THRESHOLD = 0.3

_TRAIN_PATH = "training/splits/train.parquet"
_MONITORING_PREFIX = "monitoring"

# Module-level counter — must NOT be instantiated inside the asset function.
# Prometheus registers counters globally; re-instantiation in the same process
# raises ValueError: Duplicated timeseries in CollectorRegistry.
try:
    from prometheus_client import Counter as _Counter

    _DRIFT_COUNTER = _Counter(
        "drift_detected_total",
        "Number of times significant feature drift was detected",
        ["report_date"],
    )
except Exception:
    _DRIFT_COUNTER = None  # prometheus_client not available — non-fatal


@asset(
    group_name="monitoring",
    partitions_def=daily_partitions,
    deps=["gold_features"],
    kinds=["python", "gcs", "evidently"],
)
def drift_report(
    context: AssetExecutionContext,
    bigquery: BigQueryResource,
    gcs_resource: GCSResource,
) -> MaterializeResult:
    """Run Evidently drift detection and write a JSON report to GCS.

    Data drift (always): DataDriftPreset on last 7 days of gold_features
    vs training baseline.

    Concept drift (opportunistic): RegressionPreset on serving_logs predictions
    vs actual next-day returns from silver_airline_market. Skipped gracefully
    if no serving logs exist for the monitoring window.
    """
    partition_date = context.partition_key
    project = os.environ["GCP_PROJECT_ID"]
    dataset_id = os.environ["BIGQUERY_DATASET"]
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    monitoring_prefix = os.environ.get("GCS_MONITORING_PREFIX", _MONITORING_PREFIX)

    window_start = (pd.Timestamp(partition_date) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    gcs_client = gcs_resource.get_client()

    # ---- Load reference (training baseline) from GCS ----
    ref_blob = gcs_client.bucket(bucket_name).blob(_TRAIN_PATH)
    if not ref_blob.exists():
        context.log.warning("Training baseline not found — skipping drift report.")
        return MaterializeResult(
            metadata={
                "skipped": MetadataValue.bool(True),
                "reason": MetadataValue.text("training baseline not found in GCS"),
                "partition_date": MetadataValue.text(partition_date),
            }
        )

    reference_df = pd.read_parquet(io.BytesIO(ref_blob.download_as_bytes()))

    # ---- Load current window from BigQuery ----
    with bigquery.get_client() as bq:
        current_df = bq.query(
            f"SELECT * FROM `{project}.{dataset_id}.gold_features`"
            f" WHERE date >= '{window_start}' AND date <= '{partition_date}'"
            f" ORDER BY date, ticker"
        ).to_dataframe()

    if current_df.empty:
        context.log.warning(f"No gold_features data for {window_start}–{partition_date}.")
        return MaterializeResult(
            metadata={
                "skipped": MetadataValue.bool(True),
                "reason": MetadataValue.text("no current data in window"),
                "partition_date": MetadataValue.text(partition_date),
            }
        )

    # ---- Select numeric feature columns (exclude identifiers + label) ----
    _EXCLUDE = {"date", "ticker", "target_return", "actual_date"}
    numeric_cols = [
        c
        for c in reference_df.columns
        if c not in _EXCLUDE and pd.api.types.is_numeric_dtype(reference_df[c])
    ]

    ref_features = reference_df[numeric_cols]
    cur_features = current_df[[c for c in numeric_cols if c in current_df.columns]]

    # ---- DATA DRIFT: Run Evidently DataDriftPreset ----
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features, current_data=cur_features)
        report_dict = report.as_dict()

        drift_result = report_dict["metrics"][0]["result"]
        share_drifted = float(drift_result.get("share_of_drifted_columns", 0.0))
        n_drifted = int(drift_result.get("number_of_drifted_columns", 0))
        n_cols = int(drift_result.get("number_of_columns", len(numeric_cols)))
        column_details = drift_result.get("drift_by_columns", {})
    except Exception as exc:
        context.log.warning(f"Evidently DataDriftPreset failed ({exc}); using z-score fallback.")
        ref_means = ref_features.mean()
        ref_stds = ref_features.std().replace(0, 1)
        cur_means = cur_features.mean()
        z_scores = ((cur_means - ref_means) / ref_stds).abs()
        drifted_cols = z_scores[z_scores > 3].index.tolist()
        n_drifted = len(drifted_cols)
        n_cols = len(numeric_cols)
        share_drifted = n_drifted / n_cols if n_cols > 0 else 0.0
        column_details = {c: {"drifted": True} for c in drifted_cols}
        report_dict = None

    drift_detected = share_drifted > _DRIFT_THRESHOLD

    # ---- CONCEPT DRIFT: RegressionPreset on serving_logs vs actuals ----
    concept_drift_result: dict = {}
    concept_mae: float | None = None
    concept_rmse: float | None = None
    n_predictions: int = 0

    try:
        with bigquery.get_client() as bq:
            # Check if serving_logs table exists before querying
            logs_query = (
                f"SELECT date, ticker, predicted_return FROM `{project}.{dataset_id}.serving_logs`"
                f" WHERE date >= '{window_start}' AND date <= '{partition_date}'"
                f" ORDER BY date, ticker"
            )
            logs_df = bq.query(logs_query).to_dataframe()

        if not logs_df.empty:
            context.log.info(f"Found {len(logs_df)} serving log rows for concept drift evaluation.")

            # Fetch actual next-day returns from silver_airline_market.
            # Prediction for date=T predicts return = (close_{T+1} - close_T) / close_T.
            # We need market data for window_start-1 through partition_date+1 to compute returns.
            lookback = (pd.Timestamp(window_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            lookahead = (pd.Timestamp(partition_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            with bigquery.get_client() as bq:
                market_df = bq.query(
                    f"SELECT date, ticker, close FROM `{project}.{dataset_id}.silver_airline_market`"
                    f" WHERE date >= '{lookback}' AND date <= '{lookahead}'"
                    f" ORDER BY ticker, date"
                ).to_dataframe()

            if not market_df.empty:
                # Compute actual daily return per (date, ticker)
                market_df = market_df.sort_values(["ticker", "date"])
                market_df["actual_return"] = market_df.groupby("ticker")["close"].transform(
                    lambda s: s.shift(-1).sub(s).div(s)
                )
                actuals_df = market_df[["date", "ticker", "actual_return"]].dropna(
                    subset=["actual_return"]
                )

                # Join predictions to actuals on (date, ticker)
                eval_df = logs_df.merge(actuals_df, on=["date", "ticker"], how="inner")

                if not eval_df.empty:
                    n_predictions = len(eval_df)
                    try:
                        from evidently.metric_preset import RegressionPreset
                        from evidently.report import Report

                        reg_report = Report(metrics=[RegressionPreset()])
                        reg_report.run(
                            reference_data=None,
                            current_data=eval_df.rename(
                                columns={
                                    "predicted_return": "prediction",
                                    "actual_return": "target",
                                }
                            )[["prediction", "target"]],
                        )
                        reg_dict = reg_report.as_dict()
                        concept_drift_result = reg_dict["metrics"][0].get("result", {})
                        concept_mae = concept_drift_result.get("mean_abs_error")
                        concept_rmse = concept_drift_result.get("rmse")
                    except Exception as exc:
                        # Fallback: compute MAE/RMSE manually
                        context.log.warning(
                            f"Evidently RegressionPreset failed ({exc}); computing manually."
                        )
                        import numpy as np

                        preds = eval_df["predicted_return"].to_numpy()
                        actuals = eval_df["actual_return"].to_numpy()
                        concept_mae = float(np.mean(np.abs(preds - actuals)))
                        concept_rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
                        concept_drift_result = {
                            "mean_abs_error": concept_mae,
                            "rmse": concept_rmse,
                            "n_predictions": n_predictions,
                        }
    except Exception as exc:
        context.log.info(
            f"Concept drift evaluation skipped (serving_logs unavailable or empty): {exc}"
        )

    # ---- Write JSON report to GCS ----
    output = {
        "report_date": partition_date,
        "window_start": window_start,
        "window_end": partition_date,
        # Data drift
        "share_of_drifted_columns": share_drifted,
        "n_drifted_columns": n_drifted,
        "n_columns": n_cols,
        "drift_detected": drift_detected,
        "drift_threshold": _DRIFT_THRESHOLD,
        "column_details": column_details,
        # Concept drift
        "concept_drift": {
            "available": n_predictions > 0,
            "n_predictions": n_predictions,
            "mae": concept_mae,
            "rmse": concept_rmse,
            "details": concept_drift_result,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    blob_path = f"{monitoring_prefix}/{partition_date}_drift_report.json"
    gcs_client.bucket(bucket_name).blob(blob_path).upload_from_string(
        json.dumps(output, indent=2),
        content_type="application/json",
    )

    # ---- Prometheus counter ----
    if drift_detected and _DRIFT_COUNTER is not None:
        try:
            _DRIFT_COUNTER.labels(report_date=partition_date).inc()
        except Exception:
            pass  # non-fatal if Prometheus push fails

    context.log.info(
        f"Drift report — data drift: {n_drifted}/{n_cols} features ({share_drifted:.1%}), "
        f"drift_detected={drift_detected}"
        + (
            f" | concept drift: MAE={concept_mae:.6f}, RMSE={concept_rmse:.6f} "
            f"({n_predictions} predictions)"
            if concept_mae is not None
            else " | concept drift: no serving logs"
        )
    )

    metadata: dict = {
        "share_of_drifted_columns": MetadataValue.float(share_drifted),
        "n_drifted_columns": MetadataValue.int(n_drifted),
        "n_columns": MetadataValue.int(n_cols),
        "drift_detected": MetadataValue.bool(drift_detected),
        "window": MetadataValue.text(f"{window_start} → {partition_date}"),
        "gcs_report_path": MetadataValue.text(f"gs://{bucket_name}/{blob_path}"),
        "partition_date": MetadataValue.text(partition_date),
        "concept_drift_available": MetadataValue.bool(n_predictions > 0),
        "concept_n_predictions": MetadataValue.int(n_predictions),
    }
    if concept_mae is not None:
        metadata["concept_mae"] = MetadataValue.float(concept_mae)
    if concept_rmse is not None:
        metadata["concept_rmse"] = MetadataValue.float(concept_rmse)

    return MaterializeResult(metadata=metadata)
