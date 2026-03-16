"""
Drift sensor — triggers retrain_job when drift_report materialises with significant drift.

Uses @asset_sensor so Dagster fires this directly on each drift_report materialisation,
eliminating the need to poll GCS, manage a blob cursor, or wait out a minimum interval.

Triggers retrain on either:
  1. Data drift  — share_of_drifted_columns > _DRIFT_THRESHOLD (0.30)
  2. Concept drift — concept_rmse > _CONCEPT_RMSE_THRESHOLD (0.025)
     (only fires when serving_logs data is available, i.e. concept_n_predictions > 0)
"""

from dagster import AssetKey, EventLogEntry, RunRequest, SensorEvaluationContext, asset_sensor

from quickstart_etl.defs.jobs import retrain_job

# Data drift: fraction of features that must shift before triggering a retrain
_DRIFT_THRESHOLD = 0.3

# Concept drift: live RMSE above this level indicates model degradation.
# Roughly 2.5% average absolute error on daily return predictions is a
# reasonable upper bound — tune this once baseline serving RMSE is established.
_CONCEPT_RMSE_THRESHOLD = 0.025


@asset_sensor(
    asset_key=AssetKey("drift_report"),
    job=retrain_job,
    description=(
        "Fires on every drift_report materialisation. "
        "Triggers retrain_job when data drift > 30% OR concept RMSE > 0.025."
    ),
)
def drift_retrain_sensor(context: SensorEvaluationContext, asset_event: EventLogEntry):
    """Inspect drift_report metadata and trigger retrain on data or concept drift."""
    mat = asset_event.dagster_event.asset_materialization
    metadata = {k: v.value for k, v in mat.metadata.items()}

    report_date = str(metadata.get("partition_date", "unknown"))
    share_drifted = float(metadata.get("share_of_drifted_columns", 0.0))
    drift_detected = bool(metadata.get("drift_detected", False))

    # Concept drift fields — only present when serving_logs has data
    concept_available = bool(metadata.get("concept_drift_available", False))
    n_predictions = int(metadata.get("concept_n_predictions", 0))
    concept_rmse = float(metadata.get("concept_rmse", 0.0)) if concept_available else 0.0

    # ---- Evaluate trigger conditions ----
    data_drift_trigger = drift_detected or share_drifted > _DRIFT_THRESHOLD
    concept_drift_trigger = (
        concept_available
        and n_predictions >= 10  # require at least 10 predictions before judging
        and concept_rmse > _CONCEPT_RMSE_THRESHOLD
    )

    if data_drift_trigger or concept_drift_trigger:
        trigger_reason = []
        if data_drift_trigger:
            trigger_reason.append(f"data_drift={share_drifted:.1%}")
        if concept_drift_trigger:
            trigger_reason.append(f"concept_rmse={concept_rmse:.6f}")

        context.log.info(
            f"Retrain triggered on {report_date}: {', '.join(trigger_reason)}. "
            "Yielding RunRequest for retrain_job."
        )
        yield RunRequest(
            run_key=f"drift_retrain_{asset_event.run_id}",
            tags={
                "triggered_by": "drift_sensor",
                "drift_report_date": report_date,
                "share_drifted": str(round(share_drifted, 4)),
                "concept_rmse": str(round(concept_rmse, 6)),
                "trigger_reason": ",".join(trigger_reason),
            },
        )
    else:
        context.log.info(
            f"No retrain triggered on {report_date}: "
            f"data_drift={share_drifted:.1%} (threshold {_DRIFT_THRESHOLD:.0%})"
            + (
                f", concept_rmse={concept_rmse:.6f} (threshold {_CONCEPT_RMSE_THRESHOLD})"
                if concept_available
                else ", concept drift: no serving logs yet"
            )
        )
