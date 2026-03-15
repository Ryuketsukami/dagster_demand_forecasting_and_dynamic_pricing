from dagster import EnvVar
from dagster_gcp import BigQueryResource, GCSResource
from dagster_gcp_pandas import BigQueryPandasIOManager

bigquery_resource = BigQueryResource(
    project=EnvVar("GCP_PROJECT_ID"),
    location=EnvVar("BIGQUERY_LOCATION"),
)

bigquery_io_manager = BigQueryPandasIOManager(
    project=EnvVar("GCP_PROJECT_ID"),
    location=EnvVar("BIGQUERY_LOCATION"),
    dataset=EnvVar("BIGQUERY_DATASET"),
    timeout=30.0,
)

gcs_resource = GCSResource(project=EnvVar("GCP_PROJECT_ID"))
