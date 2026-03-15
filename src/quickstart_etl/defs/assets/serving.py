"""
Serving layer — validates champion model and writes serving config to GCS.

Asset:
    serving_endpoint  — verifies champion model is present and loadable;
                        writes serving_config.json to GCS with version info.

The actual inference server (FastAPI) lives in lib/serving_app.py and is
started externally with:
    uvicorn quickstart_etl.lib.serving_app:app --host 0.0.0.0 --port 8080
"""

import json
import os
import pickle
from datetime import datetime, timezone

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset
from dagster_gcp import GCSResource

_CHAMPION_MODEL_PATH = "champion/model.pkl"
_CHAMPION_METRICS_PATH = "champion/metrics.json"
_CHAMPION_FEATURE_COLS_PATH = "champion/feature_cols.json"
_SERVING_CONFIG_PATH = "serving/serving_config.json"


@asset(
    group_name="serving",
    deps=["champion_model"],
    kinds=["python", "gcs", "fastapi"],
)
def serving_endpoint(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> MaterializeResult:
    """Verify champion model is loadable and publish serving_config.json to GCS.

    This asset does NOT start a server. It acts as a readiness gate: if it
    materialises successfully, the serving app can safely be (re-)deployed.
    """
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    serving_host = os.environ.get("SERVING_HOST", "0.0.0.0")
    serving_port = os.environ.get("SERVING_PORT", "8080")

    gcs_client = gcs_resource.get_client()
    bucket = gcs_client.bucket(bucket_name)

    # ---- 1. Check champion model blob exists ----
    model_blob = bucket.blob(_CHAMPION_MODEL_PATH)
    if not model_blob.exists():
        raise ValueError("Champion model not found in GCS — run the training pipeline first.")

    # ---- 2. Smoke-test: load model + feature cols ----
    model = pickle.loads(model_blob.download_as_bytes())

    feat_blob = bucket.blob(_CHAMPION_FEATURE_COLS_PATH)
    if not feat_blob.exists():
        raise ValueError("champion/feature_cols.json missing from GCS.")
    feature_cols: list[str] = json.loads(feat_blob.download_as_bytes())

    # Quick shape check
    import numpy as np

    dummy_X = np.zeros((1, len(feature_cols)), dtype=np.float32)
    _ = model.predict(dummy_X)  # raises if model is corrupt
    context.log.info(f"Champion model smoke-test passed ({len(feature_cols)} features)")

    # ---- 3. Load champion metrics ----
    metrics_blob = bucket.blob(_CHAMPION_METRICS_PATH)
    metrics = json.loads(metrics_blob.download_as_bytes()) if metrics_blob.exists() else {}
    champion_rmse = metrics.get("rmse", "unknown")

    # ---- 4. Write serving_config.json ----
    config = {
        "model_path": f"gs://{bucket_name}/{_CHAMPION_MODEL_PATH}",
        "feature_cols_path": f"gs://{bucket_name}/{_CHAMPION_FEATURE_COLS_PATH}",
        "champion_rmse": champion_rmse,
        "n_features": len(feature_cols),
        "serving_host": serving_host,
        "serving_port": serving_port,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "start_command": (
            f"uvicorn quickstart_etl.lib.serving_app:app "
            f"--host {serving_host} --port {serving_port}"
        ),
    }
    bucket.blob(_SERVING_CONFIG_PATH).upload_from_string(
        json.dumps(config, indent=2),
        content_type="application/json",
    )

    return MaterializeResult(
        metadata={
            "champion_rmse": MetadataValue.float(
                float(champion_rmse) if isinstance(champion_rmse, int | float) else 0.0
            ),
            "n_features": MetadataValue.int(len(feature_cols)),
            "serving_config_path": MetadataValue.text(f"gs://{bucket_name}/{_SERVING_CONFIG_PATH}"),
            "start_command": MetadataValue.text(config["start_command"]),
            "last_updated": MetadataValue.text(config["last_updated"]),
        }
    )
