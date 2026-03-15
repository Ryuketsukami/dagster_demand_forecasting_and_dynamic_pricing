from dagster import AssetSelection, define_asset_job

daily_ingestion_job = define_asset_job(
    name="daily_ingestion_job",
    description="Ingest raw data (Bronze), validate (Silver), and engineer features (Gold) for one partition.",
    selection=(
        AssetSelection.groups("ingestion")
        | AssetSelection.groups("validation")
        | AssetSelection.groups("features")
    ),
)

retrain_job = define_asset_job(
    name="retrain_job",
    description="Re-train the LightGBM model on the full gold_features history and promote champion if RMSE improves.",
    selection=AssetSelection.groups("training"),
)
