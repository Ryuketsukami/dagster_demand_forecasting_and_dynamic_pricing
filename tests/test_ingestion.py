"""
Integration tests for Bronze ingestion assets.

All external I/O is mocked:
  - yfinance.download  → returns a controlled synthetic DataFrame
  - httpx.Client.get   → returns controlled JSON for Open-Meteo and Frankfurter
  - GCSResource        → uses FakeGCSClient from conftest
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dagster import build_asset_context

from tests.conftest import FakeGCSClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_yfinance_df(date: str, tickers: list[str]) -> pd.DataFrame:
    """Return a minimal multi-index DataFrame that mimics yf.download output."""
    import numpy as np

    prices = {t: 50.0 + i * 5 for i, t in enumerate(tickers)}
    tuples = [
        (metric, ticker)
        for metric in ["Close", "High", "Low", "Open", "Volume"]
        for ticker in tickers
    ]
    columns = pd.MultiIndex.from_tuples(tuples, names=[None, "Ticker"])
    idx = pd.DatetimeIndex([pd.Timestamp(date)], name="Date")
    data = {}
    rng = np.random.default_rng(0)
    for metric, ticker in tuples:
        p = prices[ticker]
        if metric == "Volume":
            data[(metric, ticker)] = [float(rng.integers(1_000_000, 5_000_000))]
        else:
            data[(metric, ticker)] = [round(p * rng.uniform(0.99, 1.01), 4)]
    return pd.DataFrame(data, columns=columns, index=idx)


def _open_meteo_response(date: str) -> dict:
    return {
        "latitude": 33.749,
        "longitude": -84.388,
        "daily": {
            "time": [date],
            "temperature_2m_max": [22.5],
            "temperature_2m_min": [10.1],
            "precipitation_sum": [0.0],
            "wind_speed_10m_max": [15.2],
            "weather_code": [3],
        },
    }


def _frankfurter_response(date: str) -> dict:
    return {
        "amount": 1.0,
        "base": "USD",
        "date": date,
        "rates": {"EUR": 0.9123, "GBP": 0.7812, "BRL": 5.2341},
    }


# ---------------------------------------------------------------------------
# raw_airline_market_data
# ---------------------------------------------------------------------------


class TestRawAirlineMarketData:
    TICKERS = ["DAL", "UAL", "AAL", "LUV"]
    PARTITION = "2024-03-15"  # a Friday — market open

    def _run(self, partition_date: str, fake_gcs: FakeGCSClient, yf_df: pd.DataFrame):
        from quickstart_etl.defs.assets.ingestion import raw_airline_market_data

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs

        with patch("quickstart_etl.defs.assets.ingestion.yf.download", return_value=yf_df):
            ctx = build_asset_context(partition_key=partition_date)
            result = raw_airline_market_data(context=ctx, gcs_resource=mock_gcs)
        return result

    def test_market_open_writes_parquet(self, fake_gcs_client):
        yf_df = _make_yfinance_df(self.PARTITION, self.TICKERS)
        self._run(self.PARTITION, fake_gcs_client, yf_df)

        blob = fake_gcs_client.bucket("test-bucket").blob(
            f"bronze/airline_market/{self.PARTITION}.parquet"
        )
        assert blob.exists(), "Parquet blob should have been written"

        written = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        assert set(written.columns) == {"date", "ticker", "open", "high", "low", "close", "volume"}
        assert len(written) == len(self.TICKERS)
        assert set(written["ticker"]) == set(self.TICKERS)

    def test_market_open_metadata_rows(self, fake_gcs_client):
        yf_df = _make_yfinance_df(self.PARTITION, self.TICKERS)
        result = self._run(self.PARTITION, fake_gcs_client, yf_df)
        assert result.metadata["rows"].value == len(self.TICKERS)

    def test_market_open_metadata_market_flag(self, fake_gcs_client):
        yf_df = _make_yfinance_df(self.PARTITION, self.TICKERS)
        result = self._run(self.PARTITION, fake_gcs_client, yf_df)
        assert result.metadata["market_open"].value is True

    def test_market_closed_writes_empty_parquet(self, fake_gcs_client):
        """yfinance returns empty DataFrame on weekends/holidays."""
        empty_df = pd.DataFrame()
        self._run("2024-03-16", fake_gcs_client, empty_df)  # Saturday

        blob = fake_gcs_client.bucket("test-bucket").blob(
            "bronze/airline_market/2024-03-16.parquet"
        )
        assert blob.exists()
        written = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        assert len(written) == 0

    def test_market_closed_metadata_zero_rows(self, fake_gcs_client):
        empty_df = pd.DataFrame()
        result = self._run("2024-03-16", fake_gcs_client, empty_df)
        assert result.metadata["rows"].value == 0
        assert result.metadata["market_open"].value is False

    def test_gcs_path_in_metadata(self, fake_gcs_client):
        yf_df = _make_yfinance_df(self.PARTITION, self.TICKERS)
        result = self._run(self.PARTITION, fake_gcs_client, yf_df)
        gcs_path = result.metadata["gcs_path"].value
        assert gcs_path.startswith("gs://test-bucket/bronze/airline_market/")
        assert self.PARTITION in gcs_path

    def test_date_column_matches_partition_key(self, fake_gcs_client):
        yf_df = _make_yfinance_df(self.PARTITION, self.TICKERS)
        self._run(self.PARTITION, fake_gcs_client, yf_df)
        blob = fake_gcs_client.bucket("test-bucket").blob(
            f"bronze/airline_market/{self.PARTITION}.parquet"
        )
        written = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        assert (written["date"] == self.PARTITION).all()


# ---------------------------------------------------------------------------
# raw_weather_data
# ---------------------------------------------------------------------------


class TestRawWeatherData:
    PARTITION = "2024-03-15"
    HUB_CITIES = ["ATL", "LAX", "ORD", "DFW", "JFK"]

    def _make_mock_httpx(self, date: str):
        """Return a mock httpx.Client whose .get() always returns Open-Meteo JSON."""
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = _open_meteo_response(date)

        client_instance = MagicMock()
        client_instance.get.return_value = response
        client_instance.__enter__ = MagicMock(return_value=client_instance)
        client_instance.__exit__ = MagicMock(return_value=False)
        return client_instance

    def test_writes_parquet_with_25_weather_cols(self, fake_gcs_client):
        from quickstart_etl.defs.assets.ingestion import raw_weather_data

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        with patch(
            "quickstart_etl.defs.assets.ingestion.httpx.Client",
            return_value=self._make_mock_httpx(self.PARTITION),
        ):
            ctx = build_asset_context(partition_key=self.PARTITION)
            raw_weather_data(context=ctx, gcs_resource=mock_gcs)

        blob = fake_gcs_client.bucket("test-bucket").blob(
            f"bronze/weather/{self.PARTITION}.parquet"
        )
        assert blob.exists()
        written = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        assert len(written) == 1
        assert "date" in written.columns
        # 5 cities × 5 vars = 25 weather columns
        weather_cols = [c for c in written.columns if c != "date"]
        assert len(weather_cols) == 25

    def test_all_cities_in_output(self, fake_gcs_client):
        from quickstart_etl.defs.assets.ingestion import raw_weather_data

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        with patch(
            "quickstart_etl.defs.assets.ingestion.httpx.Client",
            return_value=self._make_mock_httpx(self.PARTITION),
        ):
            ctx = build_asset_context(partition_key=self.PARTITION)
            raw_weather_data(context=ctx, gcs_resource=mock_gcs)

        blob = fake_gcs_client.bucket("test-bucket").blob(
            f"bronze/weather/{self.PARTITION}.parquet"
        )
        written = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        for city in [c.lower() for c in self.HUB_CITIES]:
            assert f"{city}_temp_max" in written.columns, f"Missing {city}_temp_max"

    def test_correct_values_from_api(self, fake_gcs_client):
        from quickstart_etl.defs.assets.ingestion import raw_weather_data

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        with patch(
            "quickstart_etl.defs.assets.ingestion.httpx.Client",
            return_value=self._make_mock_httpx(self.PARTITION),
        ):
            ctx = build_asset_context(partition_key=self.PARTITION)
            raw_weather_data(context=ctx, gcs_resource=mock_gcs)

        blob = fake_gcs_client.bucket("test-bucket").blob(
            f"bronze/weather/{self.PARTITION}.parquet"
        )
        written = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        # API always returns temp_max=22.5 for every city in our mock
        assert written["atl_temp_max"].iloc[0] == 22.5

    def test_metadata_row_count(self, fake_gcs_client):
        from quickstart_etl.defs.assets.ingestion import raw_weather_data

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client

        with patch(
            "quickstart_etl.defs.assets.ingestion.httpx.Client",
            return_value=self._make_mock_httpx(self.PARTITION),
        ):
            ctx = build_asset_context(partition_key=self.PARTITION)
            result = raw_weather_data(context=ctx, gcs_resource=mock_gcs)

        assert result.metadata["rows"].value == 1


# ---------------------------------------------------------------------------
# raw_currency_rates
# ---------------------------------------------------------------------------


class TestRawCurrencyRates:
    PARTITION = "2024-03-15"

    def _make_mock_httpx(self, response_data: dict):
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status = MagicMock()
        response.json.return_value = response_data

        client_instance = MagicMock()
        client_instance.get.return_value = response
        client_instance.__enter__ = MagicMock(return_value=client_instance)
        client_instance.__exit__ = MagicMock(return_value=False)
        return client_instance

    def test_writes_parquet_with_correct_columns(self, fake_gcs_client):
        from quickstart_etl.defs.assets.ingestion import raw_currency_rates

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client
        resp = _frankfurter_response(self.PARTITION)

        with patch(
            "quickstart_etl.defs.assets.ingestion.httpx.Client",
            return_value=self._make_mock_httpx(resp),
        ):
            ctx = build_asset_context(partition_key=self.PARTITION)
            raw_currency_rates(context=ctx, gcs_resource=mock_gcs)

        blob = fake_gcs_client.bucket("test-bucket").blob(
            f"bronze/currency/{self.PARTITION}.parquet"
        )
        assert blob.exists()
        written = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        assert set(written.columns) == {"date", "actual_date", "eur_usd", "gbp_usd", "brl_usd"}

    def test_rates_correctly_parsed(self, fake_gcs_client):
        from quickstart_etl.defs.assets.ingestion import raw_currency_rates

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client
        resp = _frankfurter_response(self.PARTITION)

        with patch(
            "quickstart_etl.defs.assets.ingestion.httpx.Client",
            return_value=self._make_mock_httpx(resp),
        ):
            ctx = build_asset_context(partition_key=self.PARTITION)
            raw_currency_rates(context=ctx, gcs_resource=mock_gcs)

        blob = fake_gcs_client.bucket("test-bucket").blob(
            f"bronze/currency/{self.PARTITION}.parquet"
        )
        written = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        assert abs(written["eur_usd"].iloc[0] - 0.9123) < 1e-6
        assert abs(written["gbp_usd"].iloc[0] - 0.7812) < 1e-6
        assert abs(written["brl_usd"].iloc[0] - 5.2341) < 1e-6

    def test_weekend_actual_date_preserved(self, fake_gcs_client):
        """Frankfurter returns prior business day for weekends; actual_date must differ."""
        from quickstart_etl.defs.assets.ingestion import raw_currency_rates

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client
        # Request Saturday 2024-03-16, API returns Friday 2024-03-15
        resp = _frankfurter_response("2024-03-15")

        with patch(
            "quickstart_etl.defs.assets.ingestion.httpx.Client",
            return_value=self._make_mock_httpx(resp),
        ):
            ctx = build_asset_context(partition_key="2024-03-16")
            raw_currency_rates(context=ctx, gcs_resource=mock_gcs)

        blob = fake_gcs_client.bucket("test-bucket").blob("bronze/currency/2024-03-16.parquet")
        written = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        assert written["date"].iloc[0] == "2024-03-16"
        assert written["actual_date"].iloc[0] == "2024-03-15"

    def test_eur_usd_in_metadata(self, fake_gcs_client):
        from quickstart_etl.defs.assets.ingestion import raw_currency_rates

        mock_gcs = MagicMock()
        mock_gcs.get_client.return_value = fake_gcs_client
        resp = _frankfurter_response(self.PARTITION)

        with patch(
            "quickstart_etl.defs.assets.ingestion.httpx.Client",
            return_value=self._make_mock_httpx(resp),
        ):
            ctx = build_asset_context(partition_key=self.PARTITION)
            result = raw_currency_rates(context=ctx, gcs_resource=mock_gcs)

        assert abs(result.metadata["eur_usd"].value - 0.9123) < 1e-4


# ---------------------------------------------------------------------------
# Retry logic helper
# ---------------------------------------------------------------------------


class TestHttpRetryHelper:
    """Tests for _http_get_with_retry behavior."""

    def test_retries_on_429_then_succeeds(self):
        from quickstart_etl.defs.assets.ingestion import _http_get_with_retry

        call_count = 0

        def side_effect(url, params):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            if call_count < 2:
                mock_resp.status_code = 429
                mock_resp.raise_for_status.side_effect = Exception("Rate limited")
            else:
                mock_resp.status_code = 200
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json.return_value = {"ok": True}
            return mock_resp

        client = MagicMock()
        client.get.side_effect = side_effect

        with patch("quickstart_etl.defs.assets.ingestion.time.sleep"):
            result = _http_get_with_retry(client, "http://test", {}, max_retries=3)

        assert result == {"ok": True}
        assert call_count == 2

    def test_raises_after_max_retries(self):
        from quickstart_etl.defs.assets.ingestion import _http_get_with_retry

        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.raise_for_status.side_effect = Exception("Service unavailable")

        client = MagicMock()
        client.get.return_value = mock_resp

        with patch("quickstart_etl.defs.assets.ingestion.time.sleep"):
            with pytest.raises(Exception, match="Service unavailable"):
                _http_get_with_retry(client, "http://test", {}, max_retries=2)
