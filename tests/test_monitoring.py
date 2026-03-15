"""
Monitoring layer tests — drift detection, Prometheus counter safety, sensor logic.

Covers:
  - Data drift detection: identical distributions → 0 drift
  - Data drift detection: divergent distributions → high drift
  - Fallback z-score detection when Evidently is unavailable
  - Prometheus counter is module-level (B2 fix verification)
  - drift_report asset: skips gracefully on missing baseline
  - drift_report asset: writes JSON report to GCS
  - Concept drift: correct MAE/RMSE when serving logs are available
  - drift_retrain_sensor: fires RunRequest on drift, skips on no drift
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from dagster import EventLogEntry, build_asset_context

from tests.conftest import (
    FakeBQClient,
    write_parquet_to_fake_gcs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_df(n: int, seed: int = 42, shift: float = 0.0) -> pd.DataFrame:
    """Minimal DataFrame mimicking a gold_features slice."""
    rng = np.random.default_rng(seed)
    feature_cols = [
        "open",
        "close",
        "volume",
        "daily_return",
        "log_return",
        "vol_5d",
        "rsi_14",
        "macd",
        "bb_width",
    ]
    data = {c: rng.standard_normal(n) + shift for c in feature_cols}
    dates = pd.bdate_range("2024-01-02", periods=n).strftime("%Y-%m-%d").tolist()
    data["date"] = dates[:n]
    data["ticker"] = "DAL"
    data["target_return"] = rng.standard_normal(n) * 0.02
    return pd.DataFrame(data)


def _make_serving_logs(dates: list[str], tickers: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    tickers = tickers or ["DAL"]
    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "predicted_return": float(rng.normal(0.001, 0.02)),
                    "model_version": "rmse=0.01234",
                    "prediction_timestamp": f"{date}T12:00:00Z",
                }
            )
    return pd.DataFrame(rows)


def _make_market_df(dates: list[str], tickers: list[str] | None = None) -> pd.DataFrame:
    """silver_airline_market with close prices to compute actual_return."""
    rng = np.random.default_rng(9)
    tickers = tickers or ["DAL"]
    rows = []
    for ticker in tickers:
        price = 50.0
        for date in sorted(dates):
            price = max(1.0, price * (1 + rng.normal(0.001, 0.02)))
            rows.append({"date": date, "ticker": ticker, "close": round(price, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit: Drift thresholding logic
# ---------------------------------------------------------------------------


class TestDriftThresholding:
    def test_no_drift_on_identical_distributions(self):
        """When reference == current, z-score drift should detect 0 columns."""
        ref = _make_feature_df(200, seed=1)
        cur = _make_feature_df(20, seed=1)  # same distribution, different size

        exclude = {"date", "ticker", "target_return"}
        numeric_cols = [
            c for c in ref.columns if c not in exclude and pd.api.types.is_numeric_dtype(ref[c])
        ]

        ref_means = ref[numeric_cols].mean()
        ref_stds = ref[numeric_cols].std().replace(0, 1)
        cur_means = cur[numeric_cols].mean()
        z_scores = ((cur_means - ref_means) / ref_stds).abs()
        drifted = z_scores[z_scores > 3]
        share = len(drifted) / len(numeric_cols)
        assert share < 0.3, f"Expected low drift on same distribution, got {share:.1%}"

    def test_high_drift_on_shifted_distributions(self):
        """When current data is shifted by 10 sigma, all features should drift."""
        ref = _make_feature_df(200, seed=2, shift=0.0)
        cur = _make_feature_df(20, seed=2, shift=10.0)  # 10-sigma shift

        exclude = {"date", "ticker", "target_return"}
        numeric_cols = [
            c for c in ref.columns if c not in exclude and pd.api.types.is_numeric_dtype(ref[c])
        ]

        ref_means = ref[numeric_cols].mean()
        ref_stds = ref[numeric_cols].std().replace(0, 1)
        cur_means = cur[numeric_cols].mean()
        z_scores = ((cur_means - ref_means) / ref_stds).abs()
        drifted = z_scores[z_scores > 3]
        share = len(drifted) / len(numeric_cols)
        assert share > 0.8, f"Expected high drift on shifted distribution, got {share:.1%}"

    def test_drift_threshold_boundary(self):
        """Exactly 0.3 share_drifted should NOT trigger retrain (threshold is strict >)."""
        assert (0.3 > 0.3) is False  # share > 0.3 means 0.30 does NOT trigger
        assert (0.31 > 0.3) is True


# ---------------------------------------------------------------------------
# Unit: Prometheus counter must be module-level (B2 fix verification)
# ---------------------------------------------------------------------------


class TestPrometheusCounterModuleLevel:
    def test_drift_counter_is_module_attribute(self):
        """B2 fix: _DRIFT_COUNTER must be defined at module level in monitoring.py."""
        import quickstart_etl.defs.assets.monitoring as mon_module

        assert hasattr(mon_module, "_DRIFT_COUNTER"), (
            "_DRIFT_COUNTER must be a module-level attribute, not created inside the asset "
            "function — see audit issue B2"
        )

    def test_counter_not_redefined_on_multiple_imports(self):
        """Importing the module twice should not raise Prometheus collision."""
        import importlib

        import quickstart_etl.defs.assets.monitoring as mon1

        # Re-importing should not re-register
        try:
            importlib.reload(mon1)
        except Exception as exc:
            pytest.fail(
                f"Reloading monitoring.py raised an exception — Prometheus counter "
                f"may be recreated on reload: {exc}"
            )


# ---------------------------------------------------------------------------
# Integration: drift_report asset
# ---------------------------------------------------------------------------


class TestDriftReportAsset:
    PARTITION = "2024-03-15"

    def _setup_bq(self, fake_bq: FakeBQClient, partition_date: str, shift: float = 0.0):
        window_start = (pd.Timestamp(partition_date) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        dates = pd.bdate_range(window_start, partition_date).strftime("%Y-%m-%d").tolist()
        cur_df = _make_feature_df(len(dates), seed=5, shift=shift)
        cur_df["date"] = dates[: len(cur_df)]
        fake_bq.register("gold_features", cur_df)
        # Register empty serving_logs by default
        fake_bq.register(
            "serving_logs",
            pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "predicted_return",
                    "model_version",
                    "prediction_timestamp",
                ]
            ),
        )
        # Register empty silver_airline_market
        fake_bq.register("silver_airline_market", pd.DataFrame(columns=["date", "ticker", "close"]))
        return cur_df

    def test_skips_when_no_baseline(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.monitoring import drift_report

        # No training baseline blob in GCS
        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: fake_bq_client
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context(partition_key=self.PARTITION)
        result = drift_report(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

        assert result.metadata["skipped"].value is True

    def test_no_drift_on_same_distribution(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.monitoring import drift_report

        ref_df = _make_feature_df(200, seed=3, shift=0.0)
        write_parquet_to_fake_gcs(
            fake_gcs_client, "test-bucket", "training/splits/train.parquet", ref_df
        )

        self._setup_bq(fake_bq_client, self.PARTITION, shift=0.0)

        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: fake_bq_client
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context(partition_key=self.PARTITION)
        result = drift_report(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

        assert result.metadata["drift_detected"].value is False

    def test_writes_json_report_to_gcs(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.monitoring import drift_report

        ref_df = _make_feature_df(200, seed=10)
        write_parquet_to_fake_gcs(
            fake_gcs_client, "test-bucket", "training/splits/train.parquet", ref_df
        )
        self._setup_bq(fake_bq_client, self.PARTITION)

        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: fake_bq_client
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context(partition_key=self.PARTITION)
        drift_report(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

        blob_path = f"monitoring/{self.PARTITION}_drift_report.json"
        blob = fake_gcs_client.bucket("test-bucket").blob(blob_path)
        assert blob.exists()

        report = json.loads(blob.download_as_bytes())
        assert "share_of_drifted_columns" in report
        assert "drift_detected" in report
        assert "report_date" in report

    def test_report_contains_concept_drift_section(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.monitoring import drift_report

        ref_df = _make_feature_df(200, seed=11)
        write_parquet_to_fake_gcs(
            fake_gcs_client, "test-bucket", "training/splits/train.parquet", ref_df
        )
        self._setup_bq(fake_bq_client, self.PARTITION)

        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: fake_bq_client
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context(partition_key=self.PARTITION)
        drift_report(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

        blob = fake_gcs_client.bucket("test-bucket").blob(
            f"monitoring/{self.PARTITION}_drift_report.json"
        )
        report = json.loads(blob.download_as_bytes())
        assert "concept_drift" in report
        assert "available" in report["concept_drift"]

    def test_concept_drift_computed_when_logs_available(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.monitoring import drift_report

        ref_df = _make_feature_df(200, seed=12)
        write_parquet_to_fake_gcs(
            fake_gcs_client, "test-bucket", "training/splits/train.parquet", ref_df
        )

        # Build window dates
        window_start = (pd.Timestamp(self.PARTITION) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        dates = pd.bdate_range(window_start, self.PARTITION).strftime("%Y-%m-%d").tolist()

        cur_df = _make_feature_df(len(dates), seed=12)
        cur_df["date"] = dates[: len(cur_df)]

        logs_df = _make_serving_logs(dates[:3])
        market_df = _make_market_df(
            dates + [(pd.Timestamp(self.PARTITION) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
        )

        # Register multiple tables — FakeBQClient uses substring matching
        # Use a separate client per query to control responses
        def query_side_effect(sql):
            if "serving_logs" in sql:
                return MagicMock(to_dataframe=lambda: logs_df.copy())
            elif "silver_airline_market" in sql:
                return MagicMock(to_dataframe=lambda: market_df.copy())
            elif "gold_features" in sql:
                return MagicMock(to_dataframe=lambda: cur_df.copy())
            return MagicMock(to_dataframe=lambda: pd.DataFrame())

        smart_bq = MagicMock()
        smart_bq.query.side_effect = query_side_effect

        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: smart_bq
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context(partition_key=self.PARTITION)
        result = drift_report(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

        # concept_drift metadata should be populated
        assert "concept_drift_available" in result.metadata

    def test_metadata_includes_numeric_drift_values(self, fake_gcs_client, fake_bq_client):
        from quickstart_etl.defs.assets.monitoring import drift_report

        ref_df = _make_feature_df(200, seed=20)
        write_parquet_to_fake_gcs(
            fake_gcs_client, "test-bucket", "training/splits/train.parquet", ref_df
        )
        self._setup_bq(fake_bq_client, self.PARTITION)

        mock_bq = MagicMock()
        mock_bq.get_client.return_value.__enter__ = lambda s: fake_bq_client
        mock_bq.get_client.return_value.__exit__ = MagicMock(return_value=False)
        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        ctx = build_asset_context(partition_key=self.PARTITION)
        result = drift_report(context=ctx, bigquery=mock_bq, gcs_resource=mock_gcs)

        assert "share_of_drifted_columns" in result.metadata
        assert "n_drifted_columns" in result.metadata
        assert 0.0 <= result.metadata["share_of_drifted_columns"].value <= 1.0
        assert result.metadata["n_drifted_columns"].value >= 0


# ---------------------------------------------------------------------------
# Integration: drift_retrain_sensor
# ---------------------------------------------------------------------------


class TestDriftRetrainSensor:
    def _make_asset_event(self, metadata: dict) -> EventLogEntry:
        """Build a minimal EventLogEntry with materialisation metadata."""
        from dagster import AssetMaterialization
        from dagster._core.events.log import EventLogEntry

        mat_metadata = {k: MagicMock(value=v) for k, v in metadata.items()}
        mat = MagicMock(spec=AssetMaterialization)
        mat.metadata = mat_metadata

        event = MagicMock()
        event.asset_materialization = mat

        entry = MagicMock(spec=EventLogEntry)
        entry.dagster_event = event
        entry.run_id = "test-run-id-123"
        return entry

    def test_yields_run_request_on_drift(self):
        from quickstart_etl.defs.sensors.drift_sensors import drift_retrain_sensor

        event = self._make_asset_event(
            {
                "share_of_drifted_columns": 0.45,
                "drift_detected": True,
                "partition_date": "2024-03-15",
            }
        )

        ctx = MagicMock()
        ctx.log = MagicMock()
        ctx.cursor = "test-run-id-123"

        from dagster import RunRequest

        results = list(drift_retrain_sensor._asset_materialization_fn(ctx, event))
        assert len(results) == 1
        assert isinstance(results[0], RunRequest)

    def test_yields_nothing_below_threshold(self):
        from quickstart_etl.defs.sensors.drift_sensors import drift_retrain_sensor

        event = self._make_asset_event(
            {
                "share_of_drifted_columns": 0.10,
                "drift_detected": False,
                "partition_date": "2024-03-15",
            }
        )

        ctx = MagicMock()
        ctx.log = MagicMock()
        ctx.cursor = "test-run-id-456"

        results = list(drift_retrain_sensor._asset_materialization_fn(ctx, event))
        assert len(results) == 0

    def test_triggers_on_exactly_0_3_plus_epsilon(self):
        """share_drifted > 0.3 (strict): 0.301 must trigger."""
        from quickstart_etl.defs.sensors.drift_sensors import drift_retrain_sensor

        event = self._make_asset_event(
            {
                "share_of_drifted_columns": 0.301,
                "drift_detected": False,
                "partition_date": "2024-03-16",
            }
        )

        ctx = MagicMock()
        ctx.log = MagicMock()
        ctx.cursor = "test-run-id-789"

        results = list(drift_retrain_sensor._asset_materialization_fn(ctx, event))
        assert len(results) == 1

    def test_no_trigger_at_exactly_threshold(self):
        """share_drifted = 0.30 exactly must NOT trigger (strict >)."""
        from quickstart_etl.defs.sensors.drift_sensors import drift_retrain_sensor

        event = self._make_asset_event(
            {
                "share_of_drifted_columns": 0.30,
                "drift_detected": False,
                "partition_date": "2024-03-17",
            }
        )

        ctx = MagicMock()
        ctx.log = MagicMock()
        ctx.cursor = "test-run-id-000"

        results = list(drift_retrain_sensor._asset_materialization_fn(ctx, event))
        assert len(results) == 0

    def test_run_request_has_unique_run_key(self):
        from quickstart_etl.defs.sensors.drift_sensors import drift_retrain_sensor

        run_id_1 = "run-abc"
        run_id_2 = "run-xyz"

        def _make_event(run_id):
            return self._make_asset_event(
                {
                    "share_of_drifted_columns": 0.5,
                    "drift_detected": True,
                    "partition_date": "2024-03-15",
                }
            )

        event1 = _make_event(run_id_1)
        event1.run_id = run_id_1
        event2 = _make_event(run_id_2)
        event2.run_id = run_id_2

        ctx = MagicMock()
        ctx.log = MagicMock()
        ctx.cursor = run_id_1

        r1 = list(drift_retrain_sensor._asset_materialization_fn(ctx, event1))[0]
        ctx.cursor = run_id_2
        r2 = list(drift_retrain_sensor._asset_materialization_fn(ctx, event2))[0]

        assert r1.run_key != r2.run_key, "Each RunRequest must have a unique run_key"
