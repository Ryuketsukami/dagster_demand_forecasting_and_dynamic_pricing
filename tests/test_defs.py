from quickstart_etl.definitions import defs


def test_defs_load():
    """Definitions object loads without error."""
    assert defs is not None


def test_expected_jobs_exist():
    job_names = {j.name for j in defs.jobs}
    assert "daily_ingestion_job" in job_names
    assert "retrain_job" in job_names
    assert "monitoring_job" in job_names


def test_expected_assets_exist():
    asset_keys = {a.key.to_user_string() for a in defs.assets}
    for expected in [
        "raw_airline_market_data",
        "raw_weather_data",
        "raw_currency_rates",
        "silver_airline_market",
        "silver_weather",
        "silver_currency",
        "gold_features",
        "training_dataset",
        "trained_model",
        "model_evaluation",
        "champion_model",
        "serving_endpoint",
        "drift_report",
    ]:
        assert expected in asset_keys, f"Missing asset: {expected}"


def test_expected_schedules_exist():
    schedule_names = {s.name for s in defs.schedules}
    assert "daily_ingestion_schedule" in schedule_names
    assert "weekly_retraining_schedule" in schedule_names
    assert "daily_monitoring_schedule" in schedule_names
