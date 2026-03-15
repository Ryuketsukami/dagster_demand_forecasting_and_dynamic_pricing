"""
Drift sensor — triggers retrain_job when drift_report materialises with significant drift.

Uses @asset_sensor so Dagster fires this directly on each drift_report materialisation,
eliminating the need to poll GCS, manage a blob cursor, or wait out a minimum interval.
"""

from dagster import AssetKey, EventLogEntry, RunRequest, SensorEvaluationContext, asset_sensor

from quickstart_etl.defs.jobs import retrain_job

# Must match the threshold used in monitoring.py
_DRIFT_THRESHOLD = 0.3


@asset_sensor(
    asset_key=AssetKey("drift_report"),
    job=retrain_job,
    description=(
        "Fires on every drift_report materialisation. "
        "Triggers retrain_job when share_of_drifted_columns > 0.3."
    ),
)
def drift_retrain_sensor(context: SensorEvaluationContext, asset_event: EventLogEntry):
    """Inspect the drift_report materialisation metadata and conditionally trigger a retrain."""
    mat = asset_event.dagster_event.asset_materialization
    metadata = {k: v.value for k, v in mat.metadata.items()}

    share_drifted = float(metadata.get("share_of_drifted_columns", 0.0))
    drift_detected = bool(metadata.get("drift_detected", False))
    report_date = str(metadata.get("partition_date", "unknown"))

    if drift_detected or share_drifted > _DRIFT_THRESHOLD:
        context.log.info(
            f"Drift detected on {report_date}: {share_drifted:.1%} of features drifted. "
            "Triggering retrain_job."
        )
        yield RunRequest(
            run_key=f"drift_retrain_{asset_event.run_id}",
            tags={
                "triggered_by": "drift_sensor",
                "drift_report_date": report_date,
                "share_drifted": str(round(share_drifted, 4)),
            },
        )
    else:
        context.log.info(
            f"No significant drift on {report_date}: {share_drifted:.1%} drifted "
            f"(threshold {_DRIFT_THRESHOLD:.0%}). No retrain triggered."
        )
