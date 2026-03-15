from dagster import AssetSelection, ScheduleDefinition, define_asset_job

from ..jobs import daily_ingestion_job, retrain_job

monitoring_job = define_asset_job(
    name="monitoring_job",
    description="Run daily drift reports for all airline tickers.",
    selection=AssetSelection.groups("monitoring"),
)

# Mon–Fri at 06:00 UTC — runs after US market close (previous day's data is settled)
daily_schedule = ScheduleDefinition(
    name="daily_ingestion_schedule",
    job=daily_ingestion_job,
    cron_schedule="0 6 * * 1-5",
)

# Sundays at 08:00 UTC — weekly full retrain on latest gold_features history
retraining_schedule = ScheduleDefinition(
    name="weekly_retraining_schedule",
    job=retrain_job,
    cron_schedule="0 8 * * 0",
)

# Mon–Fri at 07:00 UTC — runs after daily ingestion to detect feature drift
monitoring_schedule = ScheduleDefinition(
    name="daily_monitoring_schedule",
    job=monitoring_job,
    cron_schedule="0 7 * * 1-5",
)
