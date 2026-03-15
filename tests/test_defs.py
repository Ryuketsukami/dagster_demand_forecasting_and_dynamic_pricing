"""
Infrastructure tests — Dagster Definitions, asset graph topology, jobs, schedules, sensors.

These tests verify the orchestration layer is correctly wired without executing any
real I/O. They catch mis-named keys, missing deps, wrong job targets, and bad crons.
"""

from __future__ import annotations

import pytest
from dagster import AssetKey, DailyPartitionsDefinition, Definitions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _asset_keys(defs: Definitions) -> set[str]:
    return {a.key.to_user_string() for a in defs.assets}


def _job_names(defs: Definitions) -> set[str]:
    return {j.name for j in defs.jobs}


def _schedule_names(defs: Definitions) -> set[str]:
    return {s.name for s in defs.schedules}


def _sensor_names(defs: Definitions) -> set[str]:
    return {s.name for s in defs.sensors}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def defs():
    from quickstart_etl.definitions import defs as _defs

    return _defs


# ---------------------------------------------------------------------------
# 1. Definitions load
# ---------------------------------------------------------------------------


def test_definitions_load(defs):
    """Definitions object is importable and not None."""
    assert defs is not None
    assert isinstance(defs, Definitions)


# ---------------------------------------------------------------------------
# 2. Asset inventory
# ---------------------------------------------------------------------------

EXPECTED_ASSETS = [
    # Bronze
    "raw_airline_market_data",
    "raw_weather_data",
    "raw_currency_rates",
    # Silver
    "silver_airline_market",
    "silver_weather",
    "silver_currency",
    # Gold
    "gold_features",
    # Training
    "training_dataset",
    "trained_model",
    "model_evaluation",
    "champion_model",
    # Serving
    "serving_endpoint",
    # Monitoring
    "drift_report",
]


@pytest.mark.parametrize("asset_name", EXPECTED_ASSETS)
def test_asset_exists(defs, asset_name):
    assert asset_name in _asset_keys(defs), f"Missing asset: {asset_name}"


# ---------------------------------------------------------------------------
# 3. Asset group assignments
# ---------------------------------------------------------------------------

ASSET_GROUPS = {
    "raw_airline_market_data": "ingestion",
    "raw_weather_data": "ingestion",
    "raw_currency_rates": "ingestion",
    "silver_airline_market": "validation",
    "silver_weather": "validation",
    "silver_currency": "validation",
    "gold_features": "features",
    "training_dataset": "training",
    "trained_model": "training",
    "model_evaluation": "training",
    "champion_model": "training",
    "serving_endpoint": "serving",
    "drift_report": "monitoring",
}


@pytest.mark.parametrize("asset_name,expected_group", ASSET_GROUPS.items())
def test_asset_group(defs, asset_name, expected_group):
    asset = next((a for a in defs.assets if a.key.to_user_string() == asset_name), None)
    assert asset is not None
    group_names = set(asset.group_names_by_key.values())
    assert expected_group in group_names, (
        f"{asset_name} is in groups {group_names}, expected '{expected_group}'"
    )


# ---------------------------------------------------------------------------
# 4. Partition assignments — partitioned vs non-partitioned
# ---------------------------------------------------------------------------

PARTITIONED_ASSETS = {
    "raw_airline_market_data",
    "raw_weather_data",
    "raw_currency_rates",
    "silver_airline_market",
    "silver_weather",
    "silver_currency",
    "gold_features",
    "drift_report",
}

NON_PARTITIONED_ASSETS = {
    "training_dataset",
    "trained_model",
    "model_evaluation",
    "champion_model",
    "serving_endpoint",
}


@pytest.mark.parametrize("asset_name", sorted(PARTITIONED_ASSETS))
def test_asset_is_partitioned(defs, asset_name):
    asset = next((a for a in defs.assets if a.key.to_user_string() == asset_name), None)
    assert asset is not None
    assert asset.partitions_def is not None, f"{asset_name} should be partitioned"
    assert isinstance(asset.partitions_def, DailyPartitionsDefinition)
    # All daily partitions must start at 2020-01-01
    assert asset.partitions_def.start.strftime("%Y-%m-%d") == "2020-01-01", (
        f"{asset_name} partition start date should be 2020-01-01"
    )


@pytest.mark.parametrize("asset_name", sorted(NON_PARTITIONED_ASSETS))
def test_asset_is_not_partitioned(defs, asset_name):
    asset = next((a for a in defs.assets if a.key.to_user_string() == asset_name), None)
    assert asset is not None
    assert asset.partitions_def is None, (
        f"{asset_name} should NOT be partitioned (training assets run on full dataset)"
    )


# ---------------------------------------------------------------------------
# 5. Critical upstream dependencies (B1 fix verification)
# ---------------------------------------------------------------------------


def _all_upstream_keys(asset) -> set:
    """Flatten all upstream AssetKeys from asset_deps dict values."""
    keys = set()
    for dep_set in asset.asset_deps.values():
        keys.update(dep_set)
    return keys


def test_training_dataset_depends_on_gold_features(defs):
    """B1 fix: training_dataset must declare gold_features as upstream dep."""
    asset = next((a for a in defs.assets if a.key.to_user_string() == "training_dataset"), None)
    assert asset is not None
    gold_key = AssetKey("gold_features")
    assert gold_key in _all_upstream_keys(asset), (
        "training_dataset must declare deps=['gold_features'] — see audit issue B1"
    )


def test_trained_model_depends_on_training_dataset(defs):
    asset = next((a for a in defs.assets if a.key.to_user_string() == "trained_model"), None)
    assert asset is not None
    assert AssetKey("training_dataset") in _all_upstream_keys(asset)


def test_champion_model_depends_on_model_evaluation(defs):
    asset = next((a for a in defs.assets if a.key.to_user_string() == "champion_model"), None)
    assert asset is not None
    assert AssetKey("model_evaluation") in _all_upstream_keys(asset)


def test_drift_report_depends_on_gold_features(defs):
    asset = next((a for a in defs.assets if a.key.to_user_string() == "drift_report"), None)
    assert asset is not None
    assert AssetKey("gold_features") in _all_upstream_keys(asset)


# ---------------------------------------------------------------------------
# 6. Job inventory and asset selection coverage
# ---------------------------------------------------------------------------


EXPECTED_JOBS = ["daily_ingestion_job", "retrain_job", "monitoring_job"]


@pytest.mark.parametrize("job_name", EXPECTED_JOBS)
def test_job_exists(defs, job_name):
    assert job_name in _job_names(defs), f"Missing job: {job_name}"


def test_monitoring_job_not_in_schedules_module(defs):
    """A1 fix: monitoring_job must be importable from defs.jobs, not schedules."""
    from quickstart_etl.defs.jobs import monitoring_job

    assert monitoring_job is not None
    assert monitoring_job.name == "monitoring_job"


def test_daily_ingestion_job_covers_ingestion_validation_features(defs):
    """daily_ingestion_job selection must cover all three medallion layers."""
    job = next((j for j in defs.jobs if j.name == "daily_ingestion_job"), None)
    assert job is not None
    selection_str = str(job.selection)
    for expected_group in ("ingestion", "validation", "features"):
        assert expected_group in selection_str, (
            f"daily_ingestion_job selection should include group '{expected_group}'; got: {selection_str}"
        )


def test_retrain_job_covers_all_training_assets(defs):
    job = next((j for j in defs.jobs if j.name == "retrain_job"), None)
    assert job is not None
    selection_str = str(job.selection)
    assert "training" in selection_str, (
        f"retrain_job selection should include group 'training'; got: {selection_str}"
    )


# ---------------------------------------------------------------------------
# 7. Schedule inventory and cron expressions
# ---------------------------------------------------------------------------

EXPECTED_SCHEDULES = {
    "daily_ingestion_schedule": "0 6 * * 1-5",
    "weekly_retraining_schedule": "0 8 * * 0",
    "daily_monitoring_schedule": "0 7 * * 1-5",
}


@pytest.mark.parametrize("schedule_name,expected_cron", EXPECTED_SCHEDULES.items())
def test_schedule_exists_with_correct_cron(defs, schedule_name, expected_cron):
    schedule = next((s for s in defs.schedules if s.name == schedule_name), None)
    assert schedule is not None, f"Missing schedule: {schedule_name}"
    assert schedule.cron_schedule == expected_cron, (
        f"{schedule_name}: expected cron '{expected_cron}', got '{schedule.cron_schedule}'"
    )


def test_monitoring_runs_after_ingestion():
    """Monitoring schedule (07:00) must be strictly after ingestion (06:00)."""
    from quickstart_etl.definitions import defs as _defs

    schedules = {s.name: s for s in _defs.schedules}
    ingestion_hour = int(schedules["daily_ingestion_schedule"].cron_schedule.split()[1])
    monitoring_hour = int(schedules["daily_monitoring_schedule"].cron_schedule.split()[1])
    assert monitoring_hour > ingestion_hour, (
        "Monitoring must run after ingestion to see fresh gold_features"
    )


def test_schedules_target_correct_jobs(defs):
    """Each schedule must point to the right job object (not a stale name)."""
    from quickstart_etl.defs.jobs import daily_ingestion_job, monitoring_job, retrain_job

    schedule_jobs = {s.name: s.job for s in defs.schedules}
    assert schedule_jobs["daily_ingestion_schedule"].name == daily_ingestion_job.name
    assert schedule_jobs["weekly_retraining_schedule"].name == retrain_job.name
    assert schedule_jobs["daily_monitoring_schedule"].name == monitoring_job.name


# ---------------------------------------------------------------------------
# 8. Sensor configuration
# ---------------------------------------------------------------------------


def test_drift_retrain_sensor_exists(defs):
    assert "drift_retrain_sensor" in _sensor_names(defs)


def test_drift_retrain_sensor_targets_retrain_job(defs):
    """B3/A2 fix: sensor must target retrain_job via object reference."""
    sensor = next((s for s in defs.sensors if s.name == "drift_retrain_sensor"), None)
    assert sensor is not None
    # The job target should be retrain_job
    assert sensor.job.name == "retrain_job"


def test_drift_retrain_sensor_monitors_drift_report(defs):
    """A2 fix: sensor must be an asset_sensor targeting drift_report."""
    from dagster import AssetKey
    from dagster._core.definitions.asset_sensor_definition import AssetSensorDefinition

    sensor = next((s for s in defs.sensors if s.name == "drift_retrain_sensor"), None)
    assert sensor is not None
    assert isinstance(sensor, AssetSensorDefinition), (
        "drift_retrain_sensor should be an @asset_sensor"
    )
    assert sensor.asset_key == AssetKey("drift_report"), (
        f"drift_retrain_sensor should watch 'drift_report', got: {sensor.asset_key}"
    )


# ---------------------------------------------------------------------------
# 9. Resource bindings
# ---------------------------------------------------------------------------


def test_resources_registered(defs):
    """All resource keys referenced by assets must be registered in Definitions."""
    resource_keys = set(defs.resources.keys())
    assert "io_manager" in resource_keys
    assert "bigquery" in resource_keys
    assert "gcs_resource" in resource_keys


def test_gcs_resource_type(defs):
    from dagster_gcp import GCSResource

    assert isinstance(defs.resources["gcs_resource"], GCSResource)


def test_bigquery_io_manager_type(defs):
    from dagster_gcp_pandas import BigQueryPandasIOManager

    assert isinstance(defs.resources["io_manager"], BigQueryPandasIOManager)
