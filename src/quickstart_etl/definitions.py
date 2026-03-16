from dagster import Definitions, load_assets_from_modules

from .defs.assets import features, ingestion, monitoring, serving, training, validation
from .defs.checks.silver_checks import (
    silver_currency_rate_bounds,
    silver_market_partition_completeness,
    silver_weather_null_rate,
)
from .defs.checks.training_checks import (
    training_feature_count_check,
    training_split_sizes_check,
    training_target_range_check,
)
from .defs.jobs import daily_ingestion_job, monitoring_job, retrain_job
from .defs.resources.storage import bigquery_io_manager, bigquery_resource, gcs_resource
from .defs.schedules import daily_schedule, monitoring_schedule, retraining_schedule
from .defs.sensors.drift_sensors import drift_retrain_sensor

all_assets = load_assets_from_modules(
    [ingestion, validation, features, training, serving, monitoring]
)

defs = Definitions(
    assets=all_assets,
    asset_checks=[
        # Silver data quality
        silver_market_partition_completeness,
        silver_weather_null_rate,
        silver_currency_rate_bounds,
        # Training data quality
        training_feature_count_check,
        training_target_range_check,
        training_split_sizes_check,
    ],
    resources={
        "bigquery": bigquery_resource,
        "io_manager": bigquery_io_manager,
        "gcs_resource": gcs_resource,
    },
    schedules=[daily_schedule, retraining_schedule, monitoring_schedule],
    sensors=[drift_retrain_sensor],
    jobs=[daily_ingestion_job, retrain_job, monitoring_job],
)
