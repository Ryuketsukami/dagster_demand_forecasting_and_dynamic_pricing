"""
Training layer — HistGradientBoostingRegressor with Optuna HPO.

Assets (not partitioned — each runs on the full dataset):
    training_dataset   — build + split features, write train/val/test Parquets to GCS
    trained_model      — Optuna HPO (50 trials) → best HGBR, saved to GCS
    model_evaluation   — MAE / RMSE / R² / directional accuracy on test set, logged as Dagster metadata
    champion_model     — compare vs current champion by RMSE; promote if better
"""

import io
import json
import os
import pickle

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset
from dagster_gcp import BigQueryResource, GCSResource

# ---------------------------------------------------------------------------
# Fixed GCS artifact paths (overwritten each training run)
# ---------------------------------------------------------------------------

_TRAIN_PATH = "training/splits/train.parquet"
_VAL_PATH = "training/splits/val.parquet"
_TEST_PATH = "training/splits/test.parquet"
_MODEL_PATH = "models/latest/model.pkl"
_FEATURE_COLS_PATH = "models/latest/feature_cols.json"
_EVAL_REPORT_PATH = "models/latest/eval_report.json"
_CHAMPION_MODEL_PATH = "champion/model.pkl"
_CHAMPION_FEATURE_COLS_PATH = "champion/feature_cols.json"
_CHAMPION_METRICS_PATH = "champion/metrics.json"

# Chronological split boundaries (from CLAUDE.md)
_TRAIN_END = "2024-12-31"
_VAL_START = "2025-01-01"
_VAL_END = "2025-06-30"
_TEST_START = "2025-07-01"

# Columns excluded from the feature matrix
_EXCLUDE_COLS = frozenset({"date", "ticker", "target_return", "actual_date"})

# HistGradientBoostingRegressor fixed hyper-parameters (non-tunable)
_HGBR_FIXED = {
    "loss": "squared_error",
    "random_state": 42,
    "early_stopping": False,  # we evaluate manually against held-out val split
}

# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------


def _upload(
    gcs_client,
    bucket_name: str,
    blob_path: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> None:
    gcs_client.bucket(bucket_name).blob(blob_path).upload_from_string(
        data, content_type=content_type
    )


def _download(gcs_client, bucket_name: str, blob_path: str) -> bytes | None:
    blob = gcs_client.bucket(bucket_name).blob(blob_path)
    return blob.download_as_bytes() if blob.exists() else None


# ---------------------------------------------------------------------------
# Asset 1: Build training splits
# ---------------------------------------------------------------------------


@asset(
    group_name="training",
    deps=["gold_features"],
    kinds=["python", "bigquery", "gcs"],
)
def training_dataset(
    context: AssetExecutionContext,
    bigquery: BigQueryResource,
    gcs_resource: GCSResource,
) -> MaterializeResult:
    """Query all gold_features from BigQuery, compute target_return, split chronologically.

    target_return is computed here (not at Gold time) because it requires the
    next day's close price, which isn't available when the day's partition runs.
    Splits are written to GCS as Parquet for downstream training assets.
    """
    project = os.environ["GCP_PROJECT_ID"]
    dataset_id = os.environ["BIGQUERY_DATASET"]
    bucket_name = os.environ["GCS_BUCKET_NAME"]

    context.log.info("Querying gold_features from BigQuery …")
    with bigquery.get_client() as bq:
        df = bq.query(
            f"SELECT * FROM `{project}.{dataset_id}.gold_features` ORDER BY date, ticker"
        ).to_dataframe()

    if df.empty:
        raise ValueError("gold_features table is empty — run the daily ingestion pipeline first.")

    # Compute target_return: (close_{t+1} - close_t) / close_t per ticker
    df = df.sort_values(["ticker", "date"])
    df["target_return"] = df.groupby("ticker")["close"].transform(
        lambda s: s.shift(-1).sub(s).div(s)
    )
    df = df.dropna(subset=["target_return"])

    # Chronological split — no shuffle, strictly ordered by time
    train_df = df[df["date"] <= _TRAIN_END]
    val_df = df[(df["date"] >= _VAL_START) & (df["date"] <= _VAL_END)]
    test_df = df[df["date"] >= _TEST_START]

    context.log.info(
        f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    # --- Data sanity assertions before writing splits ---
    target_999 = df["target_return"].abs().quantile(0.999)
    if target_999 >= 0.5:
        raise ValueError(
            f"target_return 99.9th percentile is {target_999:.2%} — exceeds 50%. "
            "Possible data corruption or join bug in gold_features."
        )

    n_feature_cols = len([c for c in df.columns if c not in _EXCLUDE_COLS])
    if n_feature_cols < 60:
        raise ValueError(
            f"Expected ≥60 feature columns, got {n_feature_cols}. "
            "feature_logic.py may have changed or silver tables are missing columns."
        )

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Train ({len(train_df)}) or val ({len(val_df)}) split is empty — "
            "check gold_features date range coverage."
        )

    gcs_client = gcs_resource.get_client()
    for split_df, path in [
        (train_df, _TRAIN_PATH),
        (val_df, _VAL_PATH),
        (test_df, _TEST_PATH),
    ]:
        buf = io.BytesIO()
        split_df.to_parquet(buf, index=False)
        _upload(gcs_client, bucket_name, path, buf.getvalue())

    n_features = len([c for c in df.columns if c not in _EXCLUDE_COLS])

    return MaterializeResult(
        metadata={
            "total_rows": MetadataValue.int(len(df)),
            "train_rows": MetadataValue.int(len(train_df)),
            "val_rows": MetadataValue.int(len(val_df)),
            "test_rows": MetadataValue.int(len(test_df)),
            "feature_columns": MetadataValue.int(n_features),
            "train_date_range": MetadataValue.text(f"2020-01-01 → {_TRAIN_END}"),
            "val_date_range": MetadataValue.text(f"{_VAL_START} → {_VAL_END}"),
            "test_date_range": MetadataValue.text(f"{_TEST_START} → present"),
            "gcs_train_path": MetadataValue.text(f"gs://{bucket_name}/{_TRAIN_PATH}"),
        }
    )


# ---------------------------------------------------------------------------
# Asset 2: Train LightGBM with Optuna HPO
# ---------------------------------------------------------------------------


@asset(
    group_name="training",
    deps=["training_dataset"],
    kinds=["sklearn", "optuna", "gcs"],
)
def trained_model(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> MaterializeResult:
    """Run 50-trial Optuna HPO on the training split; train final model on best params.

    Uses HistGradientBoostingRegressor — equivalent to LightGBM but pure Python/C++,
    no OpenMP system library required (compatible with Dagster+ Serverless).
    Best model + feature column list are saved to GCS under models/latest/.
    """
    import optuna  # lazy — avoid import-time overhead on Serverless
    from sklearn.ensemble import HistGradientBoostingRegressor  # lazy
    from sklearn.metrics import mean_squared_error  # lazy

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    gcs_client = gcs_resource.get_client()

    train_bytes = _download(gcs_client, bucket_name, _TRAIN_PATH)
    val_bytes = _download(gcs_client, bucket_name, _VAL_PATH)
    if not train_bytes or not val_bytes:
        raise ValueError("Training splits not found in GCS — run training_dataset first.")

    train_df = pd.read_parquet(io.BytesIO(train_bytes))
    val_df = pd.read_parquet(io.BytesIO(val_bytes))

    feature_cols = [c for c in train_df.columns if c not in _EXCLUDE_COLS]
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["target_return"].to_numpy(dtype=np.float64)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["target_return"].to_numpy(dtype=np.float64)

    context.log.info(
        f"Starting Optuna HPO — {len(feature_cols)} features, {len(X_train)} train rows"
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial: optuna.Trial) -> float:
        params = {
            **_HGBR_FIXED,
            "max_iter": trial.suggest_int("max_iter", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, preds)))

    study = optuna.create_study(direction="minimize", study_name="hgbr_airline_return")
    study.optimize(_objective, n_trials=50, show_progress_bar=False)

    context.log.info(f"Best val RMSE: {study.best_value:.6f}  params: {study.best_params}")

    # Train final model with best params on full train+val combined for max data
    best_params = {**_HGBR_FIXED, **study.best_params}
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    final_model = HistGradientBoostingRegressor(**best_params)
    final_model.fit(X_full, y_full)

    # Persist model + feature column list
    _upload(gcs_client, bucket_name, _MODEL_PATH, pickle.dumps(final_model))
    _upload(
        gcs_client,
        bucket_name,
        _FEATURE_COLS_PATH,
        json.dumps(feature_cols).encode(),
        content_type="application/json",
    )

    bp = study.best_params
    return MaterializeResult(
        metadata={
            "val_rmse": MetadataValue.float(study.best_value),
            "n_trials": MetadataValue.int(50),
            "n_features": MetadataValue.int(len(feature_cols)),
            "best_max_iter": MetadataValue.int(bp.get("max_iter", 0)),
            "best_learning_rate": MetadataValue.float(bp.get("learning_rate", 0.0)),
            "best_max_leaf_nodes": MetadataValue.int(bp.get("max_leaf_nodes", 0)),
            "best_max_depth": MetadataValue.int(bp.get("max_depth", 0)),
            "best_l2_regularization": MetadataValue.float(bp.get("l2_regularization", 0.0)),
            "best_max_features": MetadataValue.float(bp.get("max_features", 0.0)),
            "gcs_model_path": MetadataValue.text(f"gs://{bucket_name}/{_MODEL_PATH}"),
        }
    )


# ---------------------------------------------------------------------------
# Asset 3: Evaluate on held-out test set
# ---------------------------------------------------------------------------


@asset(
    group_name="training",
    deps=["trained_model"],
    kinds=["python", "gcs"],
)
def model_evaluation(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> MaterializeResult:
    """Evaluate trained model on the test set; write eval_report.json to GCS.

    Metrics logged as Dagster asset metadata (visible in Dagster+ UI per run).
    """
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    gcs_client = gcs_resource.get_client()

    model_bytes = _download(gcs_client, bucket_name, _MODEL_PATH)
    test_bytes = _download(gcs_client, bucket_name, _TEST_PATH)
    feature_cols_bytes = _download(gcs_client, bucket_name, _FEATURE_COLS_PATH)

    if not model_bytes or not test_bytes or not feature_cols_bytes:
        raise ValueError("Model or test artifacts missing in GCS — run trained_model first.")

    from sklearn.inspection import permutation_importance  # lazy
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # lazy

    model = pickle.loads(model_bytes)
    test_df = pd.read_parquet(io.BytesIO(test_bytes))
    feature_cols: list[str] = json.loads(feature_cols_bytes)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["target_return"].to_numpy(dtype=np.float64)

    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    directional_accuracy = float(np.mean(np.sign(preds) == np.sign(y_test)))

    # HistGradientBoostingRegressor has no feature_importances_; use permutation importance
    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    importances = dict(zip(feature_cols, perm.importances_mean.tolist()))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:20]

    eval_report = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "test_rows": int(len(y_test)),
        "top_features": top_features,
    }
    _upload(
        gcs_client,
        bucket_name,
        _EVAL_REPORT_PATH,
        json.dumps(eval_report).encode(),
        content_type="application/json",
    )

    context.log.info(
        f"Test — MAE: {mae:.6f}  RMSE: {rmse:.6f}  R²: {r2:.4f}  "
        f"Directional accuracy: {directional_accuracy:.2%}"
    )

    return MaterializeResult(
        metadata={
            "test_mae": MetadataValue.float(mae),
            "test_rmse": MetadataValue.float(rmse),
            "test_r2": MetadataValue.float(r2),
            "directional_accuracy": MetadataValue.float(directional_accuracy),
            "test_rows": MetadataValue.int(len(y_test)),
            "top_feature_1": MetadataValue.text(top_features[0][0] if top_features else ""),
            "top_feature_2": MetadataValue.text(
                top_features[1][0] if len(top_features) > 1 else ""
            ),
            "gcs_report_path": MetadataValue.text(f"gs://{bucket_name}/{_EVAL_REPORT_PATH}"),
        }
    )


# ---------------------------------------------------------------------------
# Asset 4: Promote champion if RMSE improves
# ---------------------------------------------------------------------------


@asset(
    group_name="training",
    deps=["model_evaluation"],
    kinds=["python", "gcs"],
)
def champion_model(
    context: AssetExecutionContext,
    gcs_resource: GCSResource,
) -> MaterializeResult:
    """Compare newly trained model against the current champion by test RMSE.

    Promotes the new model (copies to champion/) only when its RMSE is strictly lower.
    The champion model and its feature_cols.json are the artifacts consumed by serving.
    """
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    gcs_client = gcs_resource.get_client()

    new_eval_bytes = _download(gcs_client, bucket_name, _EVAL_REPORT_PATH)
    if not new_eval_bytes:
        raise ValueError("Eval report not found — run model_evaluation first.")

    new_metrics: dict = json.loads(new_eval_bytes)
    new_rmse: float = new_metrics["rmse"]

    # Load existing champion metrics (infinity if no champion exists yet)
    champion_bytes = _download(gcs_client, bucket_name, _CHAMPION_METRICS_PATH)
    champion_rmse = float("inf")
    if champion_bytes:
        champion_rmse = json.loads(champion_bytes).get("rmse", float("inf"))

    promoted = new_rmse < champion_rmse

    if promoted:
        context.log.info(
            f"Promoting new champion: RMSE {new_rmse:.6f} < previous {champion_rmse:.6f}"
        )
        model_bytes = _download(gcs_client, bucket_name, _MODEL_PATH)
        feat_bytes = _download(gcs_client, bucket_name, _FEATURE_COLS_PATH)

        if not model_bytes or not feat_bytes:
            raise ValueError("Latest model artifacts missing — cannot promote.")

        _upload(gcs_client, bucket_name, _CHAMPION_MODEL_PATH, model_bytes)
        _upload(
            gcs_client,
            bucket_name,
            _CHAMPION_FEATURE_COLS_PATH,
            feat_bytes,
            content_type="application/json",
        )
        _upload(
            gcs_client,
            bucket_name,
            _CHAMPION_METRICS_PATH,
            json.dumps(new_metrics).encode(),
            content_type="application/json",
        )
    else:
        context.log.info(
            f"New model not promoted: RMSE {new_rmse:.6f} >= champion {champion_rmse:.6f}"
        )

    return MaterializeResult(
        metadata={
            "promoted": MetadataValue.bool(promoted),
            "new_rmse": MetadataValue.float(new_rmse),
            "champion_rmse": MetadataValue.float(
                champion_rmse if champion_rmse != float("inf") else 0.0
            ),
            "rmse_delta": MetadataValue.float(
                champion_rmse - new_rmse if champion_rmse != float("inf") else 0.0
            ),
            "champion_model_path": MetadataValue.text(f"gs://{bucket_name}/{_CHAMPION_MODEL_PATH}"),
        }
    )
