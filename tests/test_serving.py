"""
Serving layer tests — serving_endpoint asset and FastAPI /predict endpoint.

Covers:
  - serving_endpoint: smoke-test passes with a valid champion model
  - serving_endpoint: raises when champion model is missing from GCS
  - serving_endpoint: writes serving_config.json to GCS
  - serving_endpoint: metadata contains correct values
  - FastAPI /health: returns 200 ok
  - FastAPI /predict: correct response schema and values
  - FastAPI /predict: 404 for unknown (date, ticker)
  - FastAPI /predict: 422 for invalid ticker
  - FastAPI /predict: 422 for bad date format
  - FastAPI /predict: prediction logs written to BigQuery (serving_logs)
  - Serving log writer: gracefully handles BigQuery failure (non-fatal)
"""

from __future__ import annotations

import json
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from dagster import build_asset_context

from tests.conftest import (
    FakeGCSClient,
    write_json_to_fake_gcs,
    write_pickle_to_fake_gcs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
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


def _make_gold_row(date: str = "2024-03-15", ticker: str = "DAL") -> pd.DataFrame:
    """Single-row gold_features DataFrame for serving."""
    rng = np.random.default_rng(42)
    row = {col: float(rng.standard_normal()) for col in _FEATURE_COLS}
    row["date"] = date
    row["ticker"] = ticker
    row["target_return"] = None
    return pd.DataFrame([row])


def _write_champion_artifacts(
    fake_gcs: FakeGCSClient,
    model,
    feature_cols: list[str],
    metrics: dict | None = None,
):
    write_pickle_to_fake_gcs(fake_gcs, "test-bucket", "champion/model.pkl", model)
    write_json_to_fake_gcs(fake_gcs, "test-bucket", "champion/feature_cols.json", feature_cols)
    if metrics:
        write_json_to_fake_gcs(fake_gcs, "test-bucket", "champion/metrics.json", metrics)


# ---------------------------------------------------------------------------
# Integration: serving_endpoint Dagster asset
# ---------------------------------------------------------------------------


class TestServingEndpointAsset:
    def test_passes_smoke_test_with_valid_model(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.serving import serving_endpoint

        model, cols = tiny_model_and_cols
        _write_champion_artifacts(fake_gcs_client, model, cols, {"rmse": 0.012, "mae": 0.009})

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        result = serving_endpoint(context=ctx, gcs_resource=mock_gcs)

        assert result.metadata["n_features"].value == len(cols)

    def test_raises_if_champion_model_missing(self, fake_gcs_client):
        from quickstart_etl.defs.assets.serving import serving_endpoint

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        with pytest.raises(ValueError, match="Champion model not found"):
            serving_endpoint(context=ctx, gcs_resource=mock_gcs)

    def test_writes_serving_config_to_gcs(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.serving import serving_endpoint

        model, cols = tiny_model_and_cols
        _write_champion_artifacts(fake_gcs_client, model, cols, {"rmse": 0.01})

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        serving_endpoint(context=ctx, gcs_resource=mock_gcs)

        config_blob = fake_gcs_client.bucket("test-bucket").blob("serving/serving_config.json")
        assert config_blob.exists()

        config = json.loads(config_blob.download_as_bytes())
        assert "model_path" in config
        assert "n_features" in config
        assert "start_command" in config

    def test_champion_rmse_in_metadata(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.serving import serving_endpoint

        model, cols = tiny_model_and_cols
        expected_rmse = 0.01234
        _write_champion_artifacts(fake_gcs_client, model, cols, {"rmse": expected_rmse})

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        result = serving_endpoint(context=ctx, gcs_resource=mock_gcs)

        assert abs(result.metadata["champion_rmse"].value - expected_rmse) < 1e-6

    def test_raises_if_feature_cols_missing(self, fake_gcs_client, tiny_model_and_cols):
        from quickstart_etl.defs.assets.serving import serving_endpoint

        model, _ = tiny_model_and_cols
        # Write model but not feature_cols.json
        write_pickle_to_fake_gcs(fake_gcs_client, "test-bucket", "champion/model.pkl", model)

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context()
        with pytest.raises(ValueError, match="feature_cols.json missing"):
            serving_endpoint(context=ctx, gcs_resource=mock_gcs)


# ---------------------------------------------------------------------------
# FastAPI app tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    """FastAPI TestClient with mocked GCS (lru_cache-busted between tests)."""
    from starlette.testclient import TestClient

    from quickstart_etl.lib import serving_app as sa

    # Clear all lru_cache singletons before each module test run
    sa._load_champion_model.cache_clear()
    sa._load_feature_cols.cache_clear()
    sa._load_champion_version.cache_clear()

    return TestClient(sa.app)


class TestServingAppHealth:
    def test_health_returns_200(self, app_client):
        response = app_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestServingAppPredict:
    DATE = "2024-03-15"
    TICKER = "DAL"

    @pytest.fixture(autouse=True)
    def clear_caches(self):
        from quickstart_etl.lib import serving_app as sa

        sa._load_champion_model.cache_clear()
        sa._load_feature_cols.cache_clear()
        sa._load_champion_version.cache_clear()
        sa._bq_client_singleton = None  # reset BQ singleton so each test gets a fresh mock
        yield
        sa._load_champion_model.cache_clear()
        sa._load_feature_cols.cache_clear()
        sa._load_champion_version.cache_clear()
        sa._bq_client_singleton = None

    def _mock_gcs_and_bq(self, mocker_patches, model, feature_cols: list[str]):
        """Patch GCS and BQ clients so the app finds a champion model."""
        # Patch GCS storage.Client
        mock_gcs_client = MagicMock()
        mock_bucket = MagicMock()
        mock_gcs_client.bucket.return_value = mock_bucket

        def blob_side_effect(path):
            blob = MagicMock()
            blob.exists.return_value = True
            if path.endswith("model.pkl"):
                blob.download_as_bytes.return_value = pickle.dumps(model)
            elif path.endswith("feature_cols.json"):
                blob.download_as_bytes.return_value = json.dumps(feature_cols).encode()
            elif path.endswith("metrics.json"):
                blob.download_as_bytes.return_value = json.dumps({"rmse": 0.012}).encode()
            return blob

        mock_bucket.blob.side_effect = blob_side_effect

        # Patch BQ client
        gold_row = _make_gold_row(self.DATE, self.TICKER)
        mock_bq_client = MagicMock()
        mock_bq_result = MagicMock()
        mock_bq_result.to_dataframe.return_value = gold_row
        mock_bq_client.query.return_value = mock_bq_result
        mock_bq_client.insert_rows_json.return_value = []
        mock_bq_client.create_table = MagicMock()

        return mock_gcs_client, mock_bq_client

    def test_valid_request_returns_200(self, app_client, tiny_model_and_cols):
        model, cols = tiny_model_and_cols
        gcs_mock, bq_mock = self._mock_gcs_and_bq(None, model, cols)

        with (
            patch("quickstart_etl.lib.serving_app.storage.Client", return_value=gcs_mock),
            patch("quickstart_etl.lib.serving_app.bq_client.Client", return_value=bq_mock),
        ):
            response = app_client.post("/predict", json={"date": self.DATE, "ticker": self.TICKER})

        assert response.status_code == 200

    def test_response_schema(self, app_client, tiny_model_and_cols):
        model, cols = tiny_model_and_cols
        gcs_mock, bq_mock = self._mock_gcs_and_bq(None, model, cols)

        with (
            patch("quickstart_etl.lib.serving_app.storage.Client", return_value=gcs_mock),
            patch("quickstart_etl.lib.serving_app.bq_client.Client", return_value=bq_mock),
        ):
            response = app_client.post("/predict", json={"date": self.DATE, "ticker": self.TICKER})

        body = response.json()
        assert "ticker" in body
        assert "prediction_date" in body
        assert "predicted_return" in body
        assert "model_version" in body
        assert "prediction_timestamp" in body
        assert isinstance(body["predicted_return"], float)

    def test_response_ticker_matches_request(self, app_client, tiny_model_and_cols):
        model, cols = tiny_model_and_cols
        gcs_mock, bq_mock = self._mock_gcs_and_bq(None, model, cols)

        with (
            patch("quickstart_etl.lib.serving_app.storage.Client", return_value=gcs_mock),
            patch("quickstart_etl.lib.serving_app.bq_client.Client", return_value=bq_mock),
        ):
            response = app_client.post("/predict", json={"date": self.DATE, "ticker": self.TICKER})

        assert response.json()["ticker"] == self.TICKER

    def test_404_on_unknown_date_ticker(self, app_client, tiny_model_and_cols):
        model, cols = tiny_model_and_cols
        gcs_mock, bq_mock = self._mock_gcs_and_bq(None, model, cols)

        # BQ returns empty DataFrame for the query
        empty_result = MagicMock()
        empty_result.to_dataframe.return_value = pd.DataFrame()
        bq_mock.query.return_value = empty_result

        with (
            patch("quickstart_etl.lib.serving_app.storage.Client", return_value=gcs_mock),
            patch("quickstart_etl.lib.serving_app.bq_client.Client", return_value=bq_mock),
        ):
            response = app_client.post("/predict", json={"date": "2010-01-01", "ticker": "DAL"})

        assert response.status_code == 404

    def test_422_on_invalid_ticker(self, app_client):
        response = app_client.post("/predict", json={"date": self.DATE, "ticker": "INVALID"})
        assert response.status_code == 422

    def test_422_on_bad_date_format(self, app_client):
        response = app_client.post("/predict", json={"date": "15-03-2024", "ticker": "DAL"})
        assert response.status_code == 422

    def test_ticker_upcased_automatically(self, app_client, tiny_model_and_cols):
        """Lowercase ticker should be accepted and upcased."""
        model, cols = tiny_model_and_cols
        gcs_mock, bq_mock = self._mock_gcs_and_bq(None, model, cols)

        with (
            patch("quickstart_etl.lib.serving_app.storage.Client", return_value=gcs_mock),
            patch("quickstart_etl.lib.serving_app.bq_client.Client", return_value=bq_mock),
        ):
            response = app_client.post("/predict", json={"date": self.DATE, "ticker": "dal"})

        assert response.status_code == 200
        assert response.json()["ticker"] == "DAL"

    @pytest.mark.parametrize("ticker", ["DAL", "UAL", "AAL", "LUV"])
    def test_all_valid_tickers_accepted(self, app_client, tiny_model_and_cols, ticker):
        model, cols = tiny_model_and_cols
        gcs_mock, bq_mock = self._mock_gcs_and_bq(None, model, cols)
        gold_row = _make_gold_row(self.DATE, ticker)
        bq_mock.query.return_value.to_dataframe.return_value = gold_row

        with (
            patch("quickstart_etl.lib.serving_app.storage.Client", return_value=gcs_mock),
            patch("quickstart_etl.lib.serving_app.bq_client.Client", return_value=bq_mock),
        ):
            response = app_client.post("/predict", json={"date": self.DATE, "ticker": ticker})

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Serving log writer
# ---------------------------------------------------------------------------


class TestServingLogWriter:
    def test_writes_row_to_bigquery(self):
        from quickstart_etl.lib.serving_app import _write_serving_log

        mock_bq = MagicMock()
        mock_bq.insert_rows_json.return_value = []
        mock_bq.create_table = MagicMock()

        with patch("quickstart_etl.lib.serving_app.bq_client.Client", return_value=mock_bq):
            _write_serving_log(
                date="2024-03-15",
                ticker="DAL",
                predicted_return=0.0142,
                model_version="rmse=0.01234",
                prediction_timestamp="2024-03-15T12:00:00Z",
            )

        mock_bq.insert_rows_json.assert_called_once()
        call_args = mock_bq.insert_rows_json.call_args
        rows = call_args[0][1]
        assert len(rows) == 1
        assert rows[0]["ticker"] == "DAL"
        assert rows[0]["predicted_return"] == pytest.approx(0.0142)

    def test_does_not_raise_on_bq_failure(self):
        """Log write failures must be swallowed — never propagate to the API response."""
        from quickstart_etl.lib.serving_app import _write_serving_log

        with patch(
            "quickstart_etl.lib.serving_app._get_bq_client",
            side_effect=Exception("BQ connection refused"),
        ):
            # Should not raise
            _write_serving_log(
                date="2024-03-15",
                ticker="DAL",
                predicted_return=0.005,
                model_version="rmse=0.01",
                prediction_timestamp="2024-03-15T12:00:00Z",
            )

    def test_row_schema_matches_serving_logs_table(self):
        from quickstart_etl.lib.serving_app import _write_serving_log

        captured_rows = []
        mock_bq = MagicMock()
        mock_bq.insert_rows_json.side_effect = lambda tbl, rows: captured_rows.extend(rows) or []
        mock_bq.create_table = MagicMock()

        with patch("quickstart_etl.lib.serving_app._get_bq_client", return_value=mock_bq):
            _write_serving_log("2024-03-15", "UAL", 0.008, "rmse=0.01", "2024-03-15T12:00:00Z")

        assert len(captured_rows) == 1
        row = captured_rows[0]
        expected_keys = {
            "date",
            "ticker",
            "predicted_return",
            "model_version",
            "prediction_timestamp",
        }
        assert set(row.keys()) == expected_keys
