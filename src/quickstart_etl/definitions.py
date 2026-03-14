# from .assets import ingestion, validation, features, training, serving, monitoring
# from .resources.storage import bigquery_io_manager, gcs_resource
# from .resources.dask_resource import dask_resource
# from .resources.mlflow_resource import mlflow_resource

from dagster import Definitions, load_assets_from_modules

from .assets import features, ingestion, monitoring, serving, training, validation
from .jobs import retrain_job
from .schedules import daily_schedule, retraining_schedule
from .sensors.drift_sensor import drift_retrain_sensor

all_assets = load_assets_from_modules(
    [ingestion, validation, features, training, serving, monitoring]
)

defs = Definitions(
    assets=all_assets,
    resources={...},
    schedules=[daily_schedule, retraining_schedule],
    sensors=[drift_retrain_sensor],
    jobs=[retrain_job],
)


# from pathlib import Path

# from dagster import (
#     Definitions,
#     ScheduleDefinition,
#     define_asset_job,
#     graph_asset,
#     link_code_references_to_git,
#     op,
#     with_source_code_references,
# )
# from dagster._core.definitions.metadata.source_code import AnchorBasedFilePathMapping

# from .defs.assets import most_frequent_words, topstories, topstory_ids

# daily_refresh_schedule = ScheduleDefinition(
#     job=define_asset_job(name="all_assets_job"), cron_schedule="0 0 * * *"
# )


# @op
# def foo_op():
#     return 5


# @graph_asset
# def my_asset():
#     return foo_op()


# my_assets = with_source_code_references(
#     [
#         my_asset,
#         topstory_ids,
#         topstories,
#         most_frequent_words,
#     ]
# )

# my_assets = link_code_references_to_git(
#     assets_defs=my_assets,
#     git_url="https://github.com/dagster-io/dagster/",
#     git_branch="master",
#     file_path_mapping=AnchorBasedFilePathMapping(
#         local_file_anchor=Path(__file__).parent,
#         file_anchor_path_in_repository="examples/quickstart_etl/src/quickstart_etl/",
#     ),
# )

# defs = Definitions(
#     assets=my_assets,
#     schedules=[daily_refresh_schedule],
# )
