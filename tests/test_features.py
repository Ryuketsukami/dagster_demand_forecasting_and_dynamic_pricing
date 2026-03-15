"""
Unit tests for lib/feature_logic.py — pure pandas feature computation functions.

All tests are isolated from I/O. They verify mathematical correctness, boundary
conditions, NaN propagation rules, and no-look-ahead guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quickstart_etl.lib.feature_logic import (
    add_weather_lag_features,
    compute_bollinger,
    compute_calendar_features,
    compute_currency_features,
    compute_macd,
    compute_market_features,
    compute_rsi,
    compute_target_return,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rising_prices(n: int = 50, start: float = 100.0, step: float = 1.0) -> pd.Series:
    return pd.Series([start + i * step for i in range(n)], dtype=float)


def _falling_prices(n: int = 50, start: float = 150.0, step: float = 1.0) -> pd.Series:
    return pd.Series([start - i * step for i in range(n)], dtype=float)


def _flat_prices(n: int = 50, value: float = 100.0) -> pd.Series:
    return pd.Series([value] * n, dtype=float)


def _make_ohlcv(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic single-ticker OHLCV DataFrame with n rows."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    price = 50.0
    rows = []
    for d in dates:
        ret = rng.normal(0.001, 0.02)
        price = max(1.0, price * (1 + ret))
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "ticker": "DAL",
                "open": price * (1 + rng.normal(0, 0.003)),
                "high": price * (1 + abs(rng.normal(0, 0.005))),
                "low": price * (1 - abs(rng.normal(0, 0.005))),
                "close": price,
                "volume": float(rng.integers(500_000, 2_000_000)),
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# compute_rsi
# ===========================================================================


class TestComputeRsi:
    def test_output_bounded_0_to_100(self):
        prices = _rising_prices(60)
        rsi = compute_rsi(prices)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_nan_for_insufficient_data(self):
        prices = _rising_prices(10)
        rsi = compute_rsi(prices, period=14)
        # RSI(14) needs at least 14 rows; first 14 should be NaN
        assert rsi.iloc[:14].isna().all()

    def test_rising_prices_give_high_rsi(self):
        """Consistently rising prices should push RSI well above 50."""
        prices = _rising_prices(60)
        rsi = compute_rsi(prices)
        assert rsi.iloc[-1] > 70

    def test_falling_prices_give_low_rsi(self):
        prices = _falling_prices(60)
        rsi = compute_rsi(prices)
        assert rsi.iloc[-1] < 30

    def test_flat_prices_rsi_near_50(self):
        """Flat price (no gains or losses) → RSI should be 50 or NaN (0/0)."""
        prices = _flat_prices(60)
        rsi = compute_rsi(prices)
        # With all zeros, rs = NaN → RSI is NaN; that is acceptable
        # (no gain and no loss means undefined RSI)
        valid = rsi.dropna()
        if len(valid) > 0:
            assert ((valid >= 45) & (valid <= 55)).all() or valid.isna().all()

    def test_returns_series_same_length(self):
        prices = _rising_prices(30)
        rsi = compute_rsi(prices, period=14)
        assert len(rsi) == len(prices)


# ===========================================================================
# compute_macd
# ===========================================================================


class TestComputeMacd:
    def test_histogram_equals_macd_minus_signal(self):
        prices = _rising_prices(60)
        macd, signal, hist = compute_macd(prices)
        pd.testing.assert_series_equal(hist, macd - signal, check_names=False)

    def test_output_lengths_match_input(self):
        prices = _rising_prices(50)
        macd, signal, hist = compute_macd(prices)
        assert len(macd) == len(signal) == len(hist) == len(prices)

    def test_macd_fast_minus_slow_ema(self):
        """MACD line = EMA(fast) - EMA(slow)."""
        prices = _rising_prices(60)
        macd, _, _ = compute_macd(prices, fast=12, slow=26)
        ema_fast = prices.ewm(span=12, adjust=False).mean()
        ema_slow = prices.ewm(span=26, adjust=False).mean()
        pd.testing.assert_series_equal(macd, ema_fast - ema_slow, check_names=False)

    def test_rising_market_positive_macd(self):
        prices = _rising_prices(60)
        macd, _, _ = compute_macd(prices)
        assert macd.iloc[-1] > 0

    def test_falling_market_negative_macd(self):
        prices = _falling_prices(60)
        macd, _, _ = compute_macd(prices)
        assert macd.iloc[-1] < 0


# ===========================================================================
# compute_bollinger
# ===========================================================================


class TestComputeBollinger:
    def test_upper_always_gte_lower(self):
        prices = _rising_prices(60)
        upper, lower, width = compute_bollinger(prices)
        valid_u = upper.dropna()
        valid_l = lower.dropna()
        assert (valid_u.values >= valid_l.values).all()

    def test_band_width_non_negative(self):
        prices = _rising_prices(60)
        _, _, width = compute_bollinger(prices)
        valid = width.dropna()
        assert (valid >= 0).all()

    def test_nan_for_insufficient_data(self):
        prices = _rising_prices(25)
        upper, lower, _ = compute_bollinger(prices, window=20)
        assert upper.iloc[:19].isna().all()

    def test_flat_prices_zero_width(self):
        """Flat prices → std=0 → bands collapse to SMA → width=0."""
        prices = _flat_prices(60)
        upper, lower, width = compute_bollinger(prices)
        valid = width.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)

    def test_returns_three_series_same_length(self):
        prices = _rising_prices(50)
        upper, lower, width = compute_bollinger(prices)
        assert len(upper) == len(lower) == len(width) == len(prices)


# ===========================================================================
# compute_market_features
# ===========================================================================


class TestComputeMarketFeatures:
    EXPECTED_COLS = [
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

    def test_all_feature_columns_present(self, single_ticker_ohlcv):
        result = compute_market_features(single_ticker_ohlcv)
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_preserves_original_ohlcv_columns(self, single_ticker_ohlcv):
        result = compute_market_features(single_ticker_ohlcv)
        for col in ["date", "ticker", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_row_count_unchanged(self, single_ticker_ohlcv):
        result = compute_market_features(single_ticker_ohlcv)
        assert len(result) == len(single_ticker_ohlcv)

    def test_daily_return_formula(self, single_ticker_ohlcv):
        """daily_return = (close - prev_close) / prev_close."""
        result = compute_market_features(single_ticker_ohlcv)
        closes = single_ticker_ohlcv["close"].values
        expected = (closes[1] - closes[0]) / closes[0]
        actual = result["daily_return"].iloc[1]
        assert abs(actual - expected) < 1e-8

    def test_high_low_range_non_negative(self, single_ticker_ohlcv):
        result = compute_market_features(single_ticker_ohlcv)
        valid = result["high_low_range"].dropna()
        assert (valid >= 0).all()

    def test_rsi_bounded_0_to_100(self, single_ticker_ohlcv):
        result = compute_market_features(single_ticker_ohlcv)
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_hist_equals_macd_minus_signal(self, single_ticker_ohlcv):
        result = compute_market_features(single_ticker_ohlcv)
        diff = (result["macd"] - result["macd_signal"] - result["macd_hist"]).dropna().abs()
        assert (diff < 1e-8).all()

    def test_vol_ratio_5d_positive(self, single_ticker_ohlcv):
        result = compute_market_features(single_ticker_ohlcv)
        valid = result["vol_ratio_5d"].dropna()
        assert (valid > 0).all()

    def test_no_lookahead_bias(self, single_ticker_ohlcv):
        """Row i's features must not depend on data from rows i+1 onward.

        We verify by replacing all data from row k onward with NaN and confirming
        that feature values before k are unchanged.
        """
        full = compute_market_features(single_ticker_ohlcv.copy())
        cutoff = 40  # compute on first 40 rows
        truncated = compute_market_features(single_ticker_ohlcv.iloc[:cutoff].copy())

        # Features at row 39 (last row of truncated) must match full
        for col in ["daily_return", "rsi_14", "macd", "vol_5d", "return_20d"]:
            v_full = full[col].iloc[cutoff - 1]
            v_trunc = truncated[col].iloc[cutoff - 1]
            if pd.isna(v_full) and pd.isna(v_trunc):
                continue
            assert abs(v_full - v_trunc) < 1e-6, (
                f"Lookahead detected in {col}: full={v_full}, truncated={v_trunc}"
            )

    def test_output_sorted_by_date(self, single_ticker_ohlcv):
        shuffled = single_ticker_ohlcv.sample(frac=1, random_state=1).reset_index(drop=True)
        result = compute_market_features(shuffled)
        dates = result["date"].tolist()
        assert dates == sorted(dates)


# ===========================================================================
# compute_currency_features
# ===========================================================================


class TestComputeCurrencyFeatures:
    CURRENCY_COLS = ["eur_usd", "gbp_usd", "brl_usd"]

    @pytest.fixture
    def currency_df(self, sample_currency_df):
        return sample_currency_df.copy()

    def test_lag_columns_created(self, currency_df):
        result = compute_currency_features(currency_df, self.CURRENCY_COLS)
        for col in self.CURRENCY_COLS:
            assert f"lag1_{col}" in result.columns
            assert f"{col}_chg" in result.columns
            assert f"lag1_{col}_chg" in result.columns

    def test_lag1_matches_shifted_value(self, currency_df):
        result = compute_currency_features(currency_df, self.CURRENCY_COLS)
        # lag1_eur_usd[i] == eur_usd[i-1] for i >= 1
        for i in range(1, len(result)):
            expected = result["eur_usd"].iloc[i - 1]
            actual = result["lag1_eur_usd"].iloc[i]
            assert abs(actual - expected) < 1e-8

    def test_first_row_lag_is_nan(self, currency_df):
        result = compute_currency_features(currency_df, self.CURRENCY_COLS)
        assert pd.isna(result["lag1_eur_usd"].iloc[0])

    def test_pct_change_formula(self, currency_df):
        result = compute_currency_features(currency_df, self.CURRENCY_COLS)
        # eur_usd_chg[i] == (eur_usd[i] - eur_usd[i-1]) / eur_usd[i-1]
        for i in range(1, min(5, len(result))):
            prev = result["eur_usd"].iloc[i - 1]
            curr = result["eur_usd"].iloc[i]
            expected = (curr - prev) / prev
            actual = result["eur_usd_chg"].iloc[i]
            assert abs(actual - expected) < 1e-8

    def test_missing_col_skipped_gracefully(self, currency_df):
        """If a column in currency_cols is absent, it should be skipped without error."""
        result = compute_currency_features(currency_df, ["eur_usd", "nonexistent_col"])
        assert "lag1_eur_usd" in result.columns
        assert "lag1_nonexistent_col" not in result.columns

    def test_row_count_preserved(self, currency_df):
        result = compute_currency_features(currency_df, self.CURRENCY_COLS)
        assert len(result) == len(currency_df)


# ===========================================================================
# add_weather_lag_features
# ===========================================================================


class TestAddWeatherLagFeatures:
    WEATHER_COLS = ["atl_temp_max", "atl_temp_min", "atl_precip", "atl_wind", "atl_weather_code"]

    @pytest.fixture
    def single_row_df(self, single_ticker_ohlcv):
        return single_ticker_ohlcv.iloc[[0]].copy()

    @pytest.fixture
    def weather_series(self, sample_weather_df):
        row = sample_weather_df.iloc[0]
        return row[self.WEATHER_COLS]

    def test_today_columns_added(self, single_row_df, weather_series):
        result = add_weather_lag_features(single_row_df, weather_series, None, self.WEATHER_COLS)
        for col in self.WEATHER_COLS:
            assert col in result.columns
            assert result[col].iloc[0] == weather_series[col]

    def test_lag1_columns_added(self, single_row_df, weather_series):
        result = add_weather_lag_features(single_row_df, None, weather_series, self.WEATHER_COLS)
        for col in self.WEATHER_COLS:
            assert f"lag1_{col}" in result.columns
            assert result[f"lag1_{col}"].iloc[0] == weather_series[col]

    def test_none_today_gives_nan(self, single_row_df):
        result = add_weather_lag_features(single_row_df, None, None, self.WEATHER_COLS)
        for col in self.WEATHER_COLS:
            assert pd.isna(result[col].iloc[0])

    def test_none_yesterday_lag1_is_nan(self, single_row_df, weather_series):
        result = add_weather_lag_features(single_row_df, weather_series, None, self.WEATHER_COLS)
        for col in self.WEATHER_COLS:
            assert pd.isna(result[f"lag1_{col}"].iloc[0])

    def test_does_not_mutate_input(self, single_row_df, weather_series):
        original_cols = set(single_row_df.columns)
        _ = add_weather_lag_features(
            single_row_df, weather_series, weather_series, self.WEATHER_COLS
        )
        assert set(single_row_df.columns) == original_cols


# ===========================================================================
# compute_calendar_features
# ===========================================================================


class TestComputeCalendarFeatures:
    @pytest.fixture
    def dates_df(self):
        """DataFrame with known dates spanning a US holiday."""
        dates = [
            "2024-01-15",  # MLK Day (US holiday)
            "2024-01-16",  # Tuesday
            "2024-07-04",  # Independence Day
            "2024-11-28",  # Thanksgiving
            "2024-12-25",  # Christmas
            "2024-01-01",  # New Year's Day
        ]
        return pd.DataFrame({"date": dates, "ticker": "DAL"})

    def test_required_columns_present(self, dates_df):
        result = compute_calendar_features(dates_df)
        for col in [
            "day_of_week",
            "month",
            "quarter",
            "week_of_year",
            "is_us_holiday",
            "days_to_holiday",
        ]:
            assert col in result.columns

    def test_day_of_week_range(self):
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-02", periods=5).strftime("%Y-%m-%d"),
                "ticker": "DAL",
            }
        )
        result = compute_calendar_features(df)
        assert result["day_of_week"].between(0, 4).all()

    def test_month_range(self, dates_df):
        result = compute_calendar_features(dates_df)
        assert result["month"].between(1, 12).all()

    def test_quarter_range(self, dates_df):
        result = compute_calendar_features(dates_df)
        assert result["quarter"].between(1, 4).all()

    def test_mlk_day_is_holiday(self, dates_df):
        result = compute_calendar_features(dates_df)
        mlk_row = result[result["date"] == "2024-01-15"]
        assert mlk_row["is_us_holiday"].iloc[0] == 1

    def test_regular_tuesday_is_not_holiday(self, dates_df):
        result = compute_calendar_features(dates_df)
        tue_row = result[result["date"] == "2024-01-16"]
        assert tue_row["is_us_holiday"].iloc[0] == 0

    def test_days_to_holiday_zero_on_holiday(self, dates_df):
        result = compute_calendar_features(dates_df)
        mlk_row = result[result["date"] == "2024-01-15"]
        assert mlk_row["days_to_holiday"].iloc[0] == 0

    def test_days_to_holiday_positive_before_holiday(self):
        # 2024-01-13 is a Saturday before MLK day (01-15); use a weekday
        df = pd.DataFrame({"date": ["2024-01-12"], "ticker": "DAL"})  # Friday before MLK
        result = compute_calendar_features(df)
        # 3 days until MLK (Mon 01-15)
        assert result["days_to_holiday"].iloc[0] == 3

    def test_row_count_preserved(self, dates_df):
        result = compute_calendar_features(dates_df)
        assert len(result) == len(dates_df)


# ===========================================================================
# compute_target_return
# ===========================================================================


class TestComputeTargetReturn:
    def test_last_row_per_ticker_is_nan(self, sample_ohlcv_df):
        result = compute_target_return(sample_ohlcv_df.copy())
        for ticker in sample_ohlcv_df["ticker"].unique():
            ticker_df = result[result["ticker"] == ticker].sort_values("date")
            assert pd.isna(ticker_df["target_return"].iloc[-1])

    def test_formula_correctness(self, sample_ohlcv_df):
        """target_return[t] = (close[t+1] - close[t]) / close[t] per ticker."""
        result = compute_target_return(sample_ohlcv_df.copy())
        dal = result[result["ticker"] == "DAL"].sort_values("date").reset_index(drop=True)
        for i in range(len(dal) - 1):
            c_t = dal["close"].iloc[i]
            c_t1 = dal["close"].iloc[i + 1]
            expected = (c_t1 - c_t) / c_t
            actual = dal["target_return"].iloc[i]
            assert abs(actual - expected) < 1e-8, f"Mismatch at row {i}"

    def test_all_tickers_processed(self, sample_ohlcv_df):
        result = compute_target_return(sample_ohlcv_df.copy())
        for ticker in sample_ohlcv_df["ticker"].unique():
            ticker_result = result[result["ticker"] == ticker]
            non_nan = ticker_result["target_return"].dropna()
            assert len(non_nan) == len(ticker_result) - 1

    def test_no_cross_ticker_contamination(self, sample_ohlcv_df):
        """target_return for DAL must not use UAL's closing prices."""
        result = compute_target_return(sample_ohlcv_df.copy())
        dal = result[result["ticker"] == "DAL"].sort_values("date").reset_index(drop=True)
        ual = (
            sample_ohlcv_df[sample_ohlcv_df["ticker"] == "UAL"]
            .sort_values("date")
            .reset_index(drop=True)
        )
        # DAL's target_return should NOT equal what you'd get if you used UAL prices
        ual_contaminated = (ual["close"].shift(-1) - dal["close"]) / dal["close"]
        # They should differ (in general) — check first non-NaN row
        assert not np.isclose(dal["target_return"].iloc[0], ual_contaminated.iloc[0])
