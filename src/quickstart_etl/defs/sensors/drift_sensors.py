"""
Drift sensor — polls GCS monitoring reports and triggers retrain_job on significant drift.
"""

import json
import os

from dagster import RunRequest, SensorEvaluationContext, sensor

# Fraction of features that triggers a retrain (must match monitoring.py threshold)
_DRIFT_THRESHOLD = 0.3


@sensor(
    job_name="retrain_job",
    description=(
        "Polls gs://{bucket}/monitoring/ for new drift reports. "
        "Triggers retrain_job when share_of_drifted_columns > 0.3."
    ),
    minimum_interval_seconds=3600,  # at most once per hour
)
def drift_retrain_sensor(context: SensorEvaluationContext):
    """Read the latest unprocessed drift report from GCS.

    Uses the sensor cursor to track the last processed report date so we
    don't trigger retrain repeatedly for the same drift event.
    """
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    monitoring_prefix = os.environ.get("GCS_MONITORING_PREFIX", "monitoring")

    if not bucket_name:
        context.log.warning("GCS_BUCKET_NAME not set — drift sensor is a no-op.")
        return

    try:
        from google.cloud import storage

        gcs = storage.Client()
        bucket = gcs.bucket(bucket_name)

        # List all drift report blobs, sorted by name (ISO dates sort correctly)
        blobs = sorted(
            [
                b
                for b in bucket.list_blobs(prefix=f"{monitoring_prefix}/")
                if b.name.endswith("_drift_report.json")
            ],
            key=lambda b: b.name,
        )

        if not blobs:
            return

        latest_blob = blobs[-1]

        # Extract date from blob name: monitoring/YYYY-MM-DD_drift_report.json
        blob_date = latest_blob.name.split("/")[-1].replace("_drift_report.json", "")

        # Skip if we've already processed this report
        last_processed = context.cursor or ""
        if blob_date <= last_processed:
            return

        report = json.loads(latest_blob.download_as_bytes())
        share_drifted = report.get("share_of_drifted_columns", 0.0)
        drift_detected = report.get("drift_detected", False)

        if drift_detected or share_drifted > _DRIFT_THRESHOLD:
            context.log.info(
                f"Drift detected on {blob_date}: {share_drifted:.1%} of features drifted. "
                "Triggering retrain_job."
            )
            context.update_cursor(blob_date)
            yield RunRequest(
                run_key=f"drift_retrain_{blob_date}",
                tags={
                    "triggered_by": "drift_sensor",
                    "drift_report_date": blob_date,
                    "share_drifted": str(round(share_drifted, 4)),
                },
            )
        else:
            # Advance cursor even when no retrain is needed
            context.update_cursor(blob_date)

    except Exception as exc:
        context.log.warning(f"drift_retrain_sensor error (non-fatal): {exc}")
