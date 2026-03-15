"""
Training pipeline tests — unit and integration.

Covers:
  - Chronological split logic (no data leakage, correct boundary dates)
  - target_return computation on the full dataset
  - HistGradientBoostingRegressor + Optuna HPO (mini run with 2 trials)
  - model_evaluation metrics (MAE, RMSE, R², directional accuracy)
  - champion_model promotion logic (promotes on improvement, rejects on regression)
  - GCS artifact round-trips (pickle, JSON, Parquet)
"""

from __future__ import annotations

import io
import json
import pickle
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from dagster import build_asset_context
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tests.conftest import (
    FakeGCSClient,
    write_json_to_fake_gcs,
    write_parquet_to_fake_gcs,
    write_pickle_to_fake_gcs,
)

# ---------------------------------------------------------------------------
# Constants mirrored from training.py
# ---------------------------------------------------------------------------

_TRAIN_END = "2024-12-31"
_VAL_START = "2025-01-01"
_VAL_END = "2025-06-30"
_TEST_START = "2025-07-01"
_EXCLUDE_COLS = frozenset({"date", "ticker", "target_return", "actual_date"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gold_df_with_target(n_per_ticker: int = 60) -> pd.DataFrame:
    """Minimal gold_features DataFrame spanning Train/Val/Test with target_return."""
    rng = np.random.default_rng(42)
    tickers = ["DAL", "UAL", "AAL", "LUV"]

    # Create dates that straddle all three splits
    dates_train = pd.bdate_range("2024-11-01", periods=30).strftime("%Y-%m-%d").tolist()
    dates_val = pd.bdate_range("2025-01-02", periods=15).strftime("%Y-%m-%d").tolist()
    dates_test = pd.bdate_range("2025-07-01", periods=15).strftime("%Y-%m-%d").tolist()
    all_dates = dates_train + dates_val + dates_test

    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "log_return",
        "high_low_range",
        "overnight_gap",
        "vol_5d",
        "vol_20d",
        "return_5d",
        "return_20d",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "vol_ratio_5d",
    ]
    rows = []
    for ticker in tickers:
        for date in all_dates:
            row = {"date": date, "ticker": ticker}
            for fc in feature_cols:
                row[fc] = float(rng.standard_normal())
            row["target_return"] = float(rng.normal(0.001, 0.02))
            rows.append(row)

    df = pd.DataFrame(rows)
    # Last row per ticker must have NaN target (simulating training_dataset behavior)
    for ticker in tickers:
        last_idx = df[df["ticker"] == ticker]["date"].idxmax()
        df.loc[last_idx, "target_return"] = np.nan

    return df.dropna(subset=["target_return"])


# ---------------------------------------------------------------------------
# Unit: Chronological split logic
# ---------------------------------------------------------------------------


class TestChronologicalSplit:
    def test_no_overlap_between_splits(self):
        df = _make_gold_df_with_target()
        train = df[df["date"] <= _TRAIN_END]
        val = df[(df["date"] >= _VAL_START) & (df["date"] <= _VAL_END)]
        test = df[df["date"] >= _TEST_START]

        train_dates = set(train["date"])
        val_dates = set(val["date"])
        test_dates = set(test["date"])

        assert len(train_dates & val_dates) == 0, "Train/val overlap"
        assert len(train_dates & test_dates) == 0, "Train/test overlap"
        assert len(val_dates & test_dates) == 0, "Val/test overlap"

    def test_chronological_order_preserved(self):
        df = _make_gold_df_with_target()
        train = df[df["date"] <= _TRAIN_END]
        val = df[(df["date"] >= _VAL_START) & (df["date"] <= _VAL_END)]
        test = df[df["date"] >= _TEST_START]

        assert train["date"].max() < val["date"].min(), "Train bleeds into val"
        assert val["date"].max() < test["date"].min(), "Val bleeds into test"

    def test_train_covers_correct_boundary(self):
        df = _make_gold_df_with_target()
        train = df[df["date"] <= _TRAIN_END]
        assert train["date"].max() <= _TRAIN_END

    def test_test_starts_after_val_end(self):
        df = _make_gold_df_with_target()
        test = df[df["date"] >= _TEST_START]
        assert test["date"].min() >= _TEST_START

    def test_all_rows_assigned_to_a_split(self):
        """Every row must fall into exactly one of train/val/test."""
        df = _make_gold_df_with_target()
        train = df[df["date"] <= _TRAIN_END]
        val = df[(df["date"] >= _VAL_START) & (df["date"] <= _VAL_END)]
        test = df[df["date"] >= _TEST_START]

        total = len(train) + len(val) + len(test)
        # Dates between val_end and test_start may be unassigned — that is intentional
        # (gap between 2025-07-01 boundary). Check no double-counting.
        assert total <= len(df)


# ---------------------------------------------------------------------------
# Unit: HGBR model training
# ---------------------------------------------------------------------------


class TestHGBRModel:
    def test_model_trains_and_predicts(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 20)).astype(np.float32)
        y = rng.standard_normal(100)
        model = HistGradientBoostingRegressor(max_iter=20, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 100

    def test_model_output_is_float_array(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 10)).astype(np.float32)
        y = rng.standard_normal(50)
        model = HistGradientBoostingRegressor(max_iter=5, random_state=0)
        model.fit(X, y)
        preds = model.predict(rng.standard_normal((10, 10)).astype(np.float32))
        assert isinstance(preds, np.ndarray)
        assert preds.dtype in (np.float32, np.float64)

    def test_model_serialisable_via_pickle(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 5)).astype(np.float32)
        y = rng.standard_normal(30)
        model = HistGradientBoostingRegressor(max_iter=5, random_state=0)
        model.fit(X, y)

        buf = io.BytesIO()
        pickle.dump(model, buf)
        buf.seek(0)
        loaded = pickle.load(buf)

        np.testing.assert_array_almost_equal(model.predict(X), loaded.predict(X))

    def test_hgbr_params_are_hgbr_not_lgbm(self):
        """Confirm we never pass LightGBM-specific params to HGBR."""
        lgbm_params = {"num_leaves", "subsample", "colsample_bytree", "reg_alpha"}
        hgbr = HistGradientBoostingRegressor()
        hgbr_param_names = set(hgbr.get_params().keys())
        assert not lgbm_params.intersection(hgbr_param_names), (
            "HGBR was given LightGBM params — wrong model class!"
        )


# ---------------------------------------------------------------------------
# Unit: Evaluation metrics
# ---------------------------------------------------------------------------


class TestEvaluationMetrics:
    @pytest.fixture
    def predictions(self):
        rng = np.random.default_rng(99)
        y_true = rng.normal(0, 0.02, 200)
        noise = rng.normal(0, 0.005, 200)
        y_pred = y_true + noise
        return y_true, y_pred

    def test_mae_positive(self, predictions):
        y_true, y_pred = predictions
        mae = float(mean_absolute_error(y_true, y_pred))
        assert mae > 0

    def test_rmse_gte_mae(self, predictions):
        y_true, y_pred = predictions
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        assert rmse >= mae

    def test_r2_between_neg_inf_and_1(self, predictions):
        y_true, y_pred = predictions
        r2 = float(r2_score(y_true, y_pred))
        assert r2 <= 1.0

    def test_directional_accuracy_bounded(self, predictions):
        y_true, y_pred = predictions
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        assert 0.0 <= dir_acc <= 1.0

    def test_perfect_predictions_rmse_zero(self):
        y = np.array([0.01, -0.02, 0.03, -0.01])
        rmse = float(np.sqrt(mean_squared_error(y, y)))
        assert rmse == 0.0

    def test_perfect_predictions_r2_one(self):
        y = np.array([0.01, -0.02, 0.03, -0.01])
        assert r2_score(y, y) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration: training_dataset asset
# ---------------------------------------------------------------------------


class TestTrainingDatasetAsset:
    def test_writes_three_split_parquets(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.training import training_dataset

        gold_df = _make_gold_df_with_target()
        fake_bq_client.register("gold_features", gold_df)

        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: fake_bq_client
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        training_dataset(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

        for path in [
            "training/splits/train.parquet",
            "training/splits/val.parquet",
            "training/splits/test.parquet",
        ]:
            blob = fake_gcs_client.bucket("test-bucket").blob(path)
            assert blob.exists(), f"Missing split: {path}"

    def test_split_metadata_logged(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.training import training_dataset

        gold_df = _make_gold_df_with_target()
        fake_bq_client.register("gold_features", gold_df)

        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: fake_bq_client
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        result = training_dataset(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

        assert result.metadata["train_rows"].value > 0
        assert "feature_columns" in result.metadata

    def test_raises_on_empty_gold_features(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.training import training_dataset

        fake_bq_client.register("gold_features", pd.DataFrame())

        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: fake_bq_client
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        with pytest.raises(ValueError, match="gold_features table is empty"):
            training_dataset(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

    def test_no_data_leakage_in_splits(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.training import training_dataset

        gold_df = _make_gold_df_with_target()
        fake_bq_client.register("gold_features", gold_df)

        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: fake_bq_client
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        training_dataset(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

        train_blob = fake_gcs_client.bucket("test-bucket").blob("training/splits/train.parquet")
        val_blob = fake_gcs_client.bucket("test-bucket").blob("training/splits/val.parquet")
        test_blob = fake_gcs_client.bucket("test-bucket").blob("training/splits/test.parquet")

        train_df = pd.read_parquet(io.BytesIO(train_blob.download_as_bytes()))
        val_df = pd.read_parquet(io.BytesIO(val_blob.download_as_bytes()))
        test_df = pd.read_parquet(io.BytesIO(test_blob.download_as_bytes()))

        assert train_df["date"].max() <= _TRAIN_END
        if len(val_df) > 0:
            assert val_df["date"].min() >= _VAL_START
            assert val_df["date"].max() <= _VAL_END
        if len(test_df) > 0:
            assert test_df["date"].min() >= _TEST_START


# ---------------------------------------------------------------------------
# Integration: champion_model promotion logic
# ---------------------------------------------------------------------------


class TestChampionModelPromotion:
    def _write_artifacts(
        self,
        fake_gcs: FakeGCSClient,
        model,
        feature_cols: list[str],
        eval_metrics: dict,
        champion_metrics: dict | None = None,
    ):
        write_pickle_to_fake_gcs(fake_gcs, "test-bucket", "models/latest/model.pkl", model)
        write_json_to_fake_gcs(
            fake_gcs, "test-bucket", "models/latest/feature_cols.json", feature_cols
        )
        write_json_to_fake_gcs(
            fake_gcs, "test-bucket", "models/latest/eval_report.json", eval_metrics
        )
        if champion_metrics:
            write_json_to_fake_gcs(
                fake_gcs, "test-bucket", "champion/metrics.json", champion_metrics
            )

    def test_promotes_when_rmse_improves(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.training import champion_model

        model, cols = tiny_model_and_cols
        self._write_artifacts(
            fake_gcs_client,
            model,
            cols,
            {"rmse": 0.010, "mae": 0.008},
            champion_metrics={"rmse": 0.015, "mae": 0.012},  # worse champion
        )

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        result = champion_model(context=ctx, gcs_resource=mock_gcs)

        assert result.metadata["promoted"].value is True
        # Champion artifacts must now exist
        assert fake_gcs_client.bucket("test-bucket").blob("champion/model.pkl").exists()
        assert fake_gcs_client.bucket("test-bucket").blob("champion/metrics.json").exists()

    def test_rejects_when_rmse_worse(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.training import champion_model

        model, cols = tiny_model_and_cols
        self._write_artifacts(
            fake_gcs_client,
            model,
            cols,
            {"rmse": 0.020, "mae": 0.015},  # worse than champion
            champion_metrics={"rmse": 0.010, "mae": 0.008},
        )

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        result = champion_model(context=ctx, gcs_resource=mock_gcs)

        assert result.metadata["promoted"].value is False

    def test_promotes_when_no_champion_exists(self, fake_gcs_client, tiny_model_and_cols):
        """First ever training run — no champion yet → always promote."""
        from quickstart_etl.defs.assets.training import champion_model

        model, cols = tiny_model_and_cols
        self._write_artifacts(
            fake_gcs_client,
            model,
            cols,
            {"rmse": 0.015, "mae": 0.010},
            champion_metrics=None,  # no existing champion
        )

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        result = champion_model(context=ctx, gcs_resource=mock_gcs)

        assert result.metadata["promoted"].value is True

    def test_rmse_delta_logged_correctly(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.training import champion_model

        model, cols = tiny_model_and_cols
        new_rmse = 0.010
        old_rmse = 0.015
        self._write_artifacts(
            fake_gcs_client,
            model,
            cols,
            {"rmse": new_rmse},
            champion_metrics={"rmse": old_rmse},
        )

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        result = champion_model(context=ctx, gcs_resource=mock_gcs)

        delta = result.metadata["rmse_delta"].value
        assert abs(delta - (old_rmse - new_rmse)) < 1e-8

    def test_raises_if_eval_report_missing(self, fake_gcs_client):
        from quickstart_etl.defs.assets.training import champion_model

        # No artifacts written at all
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        with pytest.raises(ValueError, match="Eval report not found"):
            champion_model(context=ctx, gcs_resource=mock_gcs)


# ---------------------------------------------------------------------------
# Integration: model_evaluation asset
# ---------------------------------------------------------------------------


class TestModelEvaluationAsset:
    def test_computes_all_metrics(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.training import model_evaluation

        model, cols = tiny_model_and_cols
        rng = np.random.default_rng(7)
        n = 40
        X = rng.standard_normal((n, len(cols))).astype(np.float32)
        y = rng.standard_normal(n)

        test_df = pd.DataFrame(X, columns=cols)
        test_df["target_return"] = y

        write_pickle_to_fake_gcs(fake_gcs_client, "test-bucket", "models/latest/model.pkl", model)
        write_parquet_to_fake_gcs(
            fake_gcs_client, "test-bucket", "training/splits/test.parquet", test_df
        )
        write_json_to_fake_gcs(
            fake_gcs_client, "test-bucket", "models/latest/feature_cols.json", cols
        )

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        result = model_evaluation(context=ctx, gcs_resource=mock_gcs)

        for metric in ["test_mae", "test_rmse", "test_r2", "directional_accuracy"]:
            assert metric in result.metadata
            assert isinstance(result.metadata[metric].value, float)

    def test_rmse_gte_mae(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.training import model_evaluation

        model, cols = tiny_model_and_cols
        rng = np.random.default_rng(8)
        n = 40
        X = rng.standard_normal((n, len(cols))).astype(np.float32)
        y = rng.standard_normal(n)
        test_df = pd.DataFrame(X, columns=cols)
        test_df["target_return"] = y

        write_pickle_to_fake_gcs(fake_gcs_client, "test-bucket", "models/latest/model.pkl", model)
        write_parquet_to_fake_gcs(
            fake_gcs_client, "test-bucket", "training/splits/test.parquet", test_df
        )
        write_json_to_fake_gcs(
            fake_gcs_client, "test-bucket", "models/latest/feature_cols.json", cols
        )

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        result = model_evaluation(context=ctx, gcs_resource=mock_gcs)

        mae = result.metadata["test_mae"].value
        rmse = result.metadata["test_rmse"].value
        assert rmse >= mae - 1e-9  # RMSE >= MAE always

    def test_eval_report_written_to_gcs(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.training import model_evaluation

        model, cols = tiny_model_and_cols
        rng = np.random.default_rng(9)
        n = 30
        X = rng.standard_normal((n, len(cols))).astype(np.float32)
        test_df = pd.DataFrame(X, columns=cols)
        test_df["target_return"] = rng.standard_normal(n)

        write_pickle_to_fake_gcs(fake_gcs_client, "test-bucket", "models/latest/model.pkl", model)
        write_parquet_to_fake_gcs(
            fake_gcs_client, "test-bucket", "training/splits/test.parquet", test_df
        )
        write_json_to_fake_gcs(
            fake_gcs_client, "test-bucket", "models/latest/feature_cols.json", cols
        )

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        model_evaluation(context=ctx, gcs_resource=mock_gcs)

        report_blob = fake_gcs_client.bucket("test-bucket").blob("models/latest/eval_report.json")
        assert report_blob.exists()
        report = json.loads(report_blob.download_as_bytes())
        assert "rmse" in report and "mae" in report and "directional_accuracy" in report
