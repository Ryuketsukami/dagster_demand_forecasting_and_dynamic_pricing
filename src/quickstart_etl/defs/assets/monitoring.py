"""
Monitoring layer — data drift and prediction drift detection via Evidently.

Asset:
    drift_report  — compares last 7 days of gold_features against the training
                    baseline; writes a JSON drift report to GCS; emits a
                    Prometheus counter when significant drift is detected.

The drift_retrain_sensor (sensors/drift_sensors.py) polls these reports and
triggers retrain_job when the share of drifted columns exceeds a threshold.
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
    """Run Evidently DataDriftPreset on the last 7 days vs training baseline.

    Writes a JSON report to gs://{bucket}/monitoring/{date}_drift_report.json.
    Increments the Prometheus counter drift_detected_total if significant drift
    is found (> 30% of numeric features drifted).
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

    # ---- Run Evidently DataDriftPreset ----
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_features, current_data=cur_features)
        report_dict = report.as_dict()

        # Extract summary from Evidently output
        drift_result = report_dict["metrics"][0]["result"]
        share_drifted = float(drift_result.get("share_of_drifted_columns", 0.0))
        n_drifted = int(drift_result.get("number_of_drifted_columns", 0))
        n_cols = int(drift_result.get("number_of_columns", len(numeric_cols)))
        column_details = drift_result.get("drift_by_columns", {})
    except Exception as exc:
        context.log.warning(f"Evidently run failed ({exc}); computing basic drift stats.")
        # Fallback: simple mean shift detection (z-score > 3 on column means)
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

    # ---- Write JSON report to GCS ----
    output = {
        "report_date": partition_date,
        "window_start": window_start,
        "window_end": partition_date,
        "share_of_drifted_columns": share_drifted,
        "n_drifted_columns": n_drifted,
        "n_columns": n_cols,
        "drift_detected": drift_detected,
        "drift_threshold": _DRIFT_THRESHOLD,
        "column_details": column_details,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    blob_path = f"{monitoring_prefix}/{partition_date}_drift_report.json"
    gcs_client.bucket(bucket_name).blob(blob_path).upload_from_string(
        json.dumps(output, indent=2),
        content_type="application/json",
    )

    # ---- Prometheus counter (fire-and-forget; no crash if unavailable) ----
    if drift_detected:
        try:
            from prometheus_client import Counter

            _DRIFT_COUNTER = Counter(
                "drift_detected_total",
                "Number of times significant feature drift was detected",
                ["report_date"],
            )
            _DRIFT_COUNTER.labels(report_date=partition_date).inc()
        except Exception:
            pass  # Prometheus not configured — non-fatal

    context.log.info(
        f"Drift report — {n_drifted}/{n_cols} features drifted "
        f"({share_drifted:.1%}), drift_detected={drift_detected}"
    )

    return MaterializeResult(
        metadata={
            "share_of_drifted_columns": MetadataValue.float(share_drifted),
            "n_drifted_columns": MetadataValue.int(n_drifted),
            "n_columns": MetadataValue.int(n_cols),
            "drift_detected": MetadataValue.bool(drift_detected),
            "window": MetadataValue.text(f"{window_start} → {partition_date}"),
            "gcs_report_path": MetadataValue.text(f"gs://{bucket_name}/{blob_path}"),
            "partition_date": MetadataValue.text(partition_date),
        }
    )
