"""
Asset checks for the Training layer.

These checks run after training_dataset materialises and surface results
in the Dagster+ UI without crashing the asset.

Checks:
    training_feature_count_check  — ≥60 feature columns present (blocking)
    training_target_range_check   — target_return 99.9th pct < 50% (warn)
    training_split_sizes_check    — train/val/test splits are non-empty (blocking)
"""

import io
import json
import os

from dagster import AssetCheckResult, AssetCheckSeverity, MetadataValue, asset_check
from dagster_gcp import GCSResource

_TRAIN_PATH = "training/splits/train.parquet"
_VAL_PATH = "training/splits/val.parquet"
_TEST_PATH = "training/splits/test.parquet"
_FEATURE_COLS_PATH = "models/latest/feature_cols.json"

_MIN_FEATURE_COLS = 60
_MAX_TARGET_QUANTILE = 0.5  # 50% return at 99.9th pct is suspicious


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_bytes(gcs_client, bucket_name: str, path: str) -> bytes | None:
    blob = gcs_client.bucket(bucket_name).blob(path)
    return blob.download_as_bytes() if blob.exists() else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@asset_check(asset="training_dataset", blocking=True)
def training_feature_count_check(gcs_resource: GCSResource) -> AssetCheckResult:
    """Fail if the feature column list saved alongside the model has < 60 columns.

    A low count indicates that feature_logic.py changed shape, or that Silver
    tables were missing columns when gold_features was computed.
    """
    bucket_name = os.environ["GCS_BUCKET_NAME"]
    gcs_client = gcs_resource.get_client()

    feat_bytes = _download_bytes(gcs_client, bucket_name, _FEATURE_COLS_PATH)
    if feat_bytes is None:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description=f"feature_cols.json not found at {_FEATURE_COLS_PATH}",
        )

    feature_cols: list[str] = json.loads(feat_bytes)
    n = len(feature_cols)
    passed = n >= _MIN_FEATURE_COLS

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={
            "feature_count": MetadataValue.int(n),
            "minimum_required": MetadataValue.int(_MIN_FEATURE_COLS),
        },
        description=(
            f"{n} feature columns found — {'OK' if passed else f'below minimum of {_MIN_FEATURE_COLS}'}."
        ),
    )


@asset_check(asset="training_dataset", blocking=False)
def training_target_range_check(gcs_resource: GCSResource) -> AssetCheckResult:
    """Warn if the 99.9th percentile of |target_return| in the train split exceeds 50%.

    Extreme values suggest a data corruption issue or a join bug in gold_features
    that accidentally leaked future prices into the feature row.
    """
    import pandas as pd

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    gcs_client = gcs_resource.get_client()

    train_bytes = _download_bytes(gcs_client, bucket_name, _TRAIN_PATH)
    if train_bytes is None:
        return AssetCheckResult(
            passed=False,
            severity=AssetCheckSeverity.ERROR,
            description=f"train.parquet not found at {_TRAIN_PATH}",
        )

    train_df = pd.read_parquet(io.BytesIO(train_bytes))
    quantile_999 = float(train_df["target_return"].abs().quantile(0.999))
    passed = quantile_999 < _MAX_TARGET_QUANTILE

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.WARN,
        metadata={
            "target_return_999th_pct": MetadataValue.float(quantile_999),
            "threshold": MetadataValue.float(_MAX_TARGET_QUANTILE),
        },
        description=(
            f"|target_return| 99.9th pct = {quantile_999:.2%} "
            + (
                "— within expected range."
                if passed
                else f"— exceeds {_MAX_TARGET_QUANTILE:.0%} threshold!"
            )
        ),
    )


@asset_check(asset="training_dataset", blocking=True)
def training_split_sizes_check(gcs_resource: GCSResource) -> AssetCheckResult:
    """Fail if train or val splits are empty.

    An empty train split means gold_features has no data before 2025-01-01,
    which indicates the backfill has not been run.
    """
    import pandas as pd

    bucket_name = os.environ["GCS_BUCKET_NAME"]
    gcs_client = gcs_resource.get_client()

    sizes: dict[str, int] = {}
    for name, path in [("train", _TRAIN_PATH), ("val", _VAL_PATH), ("test", _TEST_PATH)]:
        b = _download_bytes(gcs_client, bucket_name, path)
        sizes[name] = len(pd.read_parquet(io.BytesIO(b))) if b else 0

    passed = sizes["train"] > 0 and sizes["val"] > 0

    return AssetCheckResult(
        passed=passed,
        severity=AssetCheckSeverity.ERROR,
        metadata={k: MetadataValue.int(v) for k, v in sizes.items()},
        description=(
            f"Split sizes — train: {sizes['train']}, val: {sizes['val']}, test: {sizes['test']}. "
            + (
                "OK."
                if passed
                else "train or val split is empty — run the daily ingestion backfill."
            )
        ),
    )
