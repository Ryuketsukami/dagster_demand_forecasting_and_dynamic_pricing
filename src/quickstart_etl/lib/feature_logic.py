"""
Shared feature computation library.

Used by:
    defs/assets/features.py   — Gold feature engineering (training path)
    defs/assets/serving.py    — Real-time prediction path (Phase 6)

All functions are pure pandas transforms; no I/O, no Dagster imports.
"""

from __future__ import annotations

import warnings
from datetime import timedelta

import holidays
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Market technical indicators
# ---------------------------------------------------------------------------


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's exponential smoothing (standard implementation)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 and avg_gain > 0, RS is infinite → RSI = 100
    # When both are 0 (flat prices), leave as NaN
    rsi = rsi.where((avg_loss != 0) | (avg_gain == 0), other=100.0)
    return rsi


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_bollinger(
    prices: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (upper_band, lower_band, band_width_pct).

    band_width = (upper - lower) / middle; NaN when middle == 0.
    """
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma.replace(0, np.nan)
    return upper, lower, width


# ---------------------------------------------------------------------------
# Market feature set
# ---------------------------------------------------------------------------


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all market-based technical features for a single ticker.

    Args:
        df: DataFrame sorted ascending by date with columns:
            date, ticker, open, high, low, close, volume

    Returns:
        Input df with additional feature columns appended.
    """
    df = df.sort_values("date").copy()
    prev_close = df["close"].shift(1)

    # Price / return features
    df["daily_return"] = (df["close"] - prev_close) / prev_close
    df["log_return"] = np.log(df["close"] / prev_close)
    df["high_low_range"] = (df["high"] - df["low"]) / prev_close
    df["overnight_gap"] = (df["open"] - prev_close) / prev_close

    # Rolling volatility / momentum
    df["vol_5d"] = df["daily_return"].rolling(5).std()
    df["vol_20d"] = df["daily_return"].rolling(20).std()
    df["return_5d"] = df["daily_return"].rolling(5).mean()
    df["return_20d"] = df["daily_return"].rolling(20).mean()

    # RSI
    df["rsi_14"] = compute_rsi(df["close"])

    # MACD
    macd_line, signal_line, hist = compute_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    # Bollinger Bands
    bb_upper, bb_lower, bb_width = compute_bollinger(df["close"])
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width

    # Volume ratio
    vol_ma = df["volume"].rolling(5).mean()
    df["vol_ratio_5d"] = df["volume"] / vol_ma.replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Currency features
# ---------------------------------------------------------------------------


def compute_currency_features(df: pd.DataFrame, currency_cols: list[str]) -> pd.DataFrame:
    """Add daily % change and t-1 lag columns for each currency rate.

    Args:
        df: DataFrame sorted ascending by date with FX rate columns.
        currency_cols: list of column names (e.g. ['eur_usd', 'gbp_usd', 'brl_usd'])

    Returns:
        df with additional {col}_chg, lag1_{col}, lag1_{col}_chg columns.
    """
    df = df.sort_values("date").copy()
    for col in currency_cols:
        if col not in df.columns:
            continue
        df[f"{col}_chg"] = df[col].pct_change()
        df[f"lag1_{col}"] = df[col].shift(1)
        df[f"lag1_{col}_chg"] = df[f"{col}_chg"].shift(1)
    return df


# ---------------------------------------------------------------------------
# Weather lag features
# ---------------------------------------------------------------------------


def add_weather_lag_features(
    df: pd.DataFrame,
    weather_today: pd.Series | None,
    weather_yesterday: pd.Series | None,
    weather_cols: list[str],
) -> pd.DataFrame:
    """Merge today's and yesterday's weather into df (one row per ticker).

    Args:
        df: Gold DataFrame for the partition date (one row per ticker).
        weather_today: Single-row Series with today's weather cols (or None).
        weather_yesterday: Single-row Series with yesterday's weather cols (or None).
        weather_cols: Weather column names to merge.
    """
    df = df.copy()
    for col in weather_cols:
        df[col] = (
            weather_today[col]
            if (weather_today is not None and col in weather_today.index)
            else np.nan
        )
        df[f"lag1_{col}"] = (
            weather_yesterday[col]
            if (weather_yesterday is not None and col in weather_yesterday.index)
            else np.nan
        )
    return df


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add US-calendar-aware features to df (must have 'date' col as str YYYY-MM-DD).

    Features added:
        day_of_week (0=Mon, 4=Fri), month, quarter, week_of_year,
        is_us_holiday (int 0/1), days_to_holiday (signed int)
    """
    us_holidays = holidays.US()
    dates = pd.to_datetime(df["date"])
    df = df.copy()

    df["day_of_week"] = dates.dt.dayofweek
    df["month"] = dates.dt.month
    df["quarter"] = dates.dt.quarter
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["is_us_holiday"] = dates.dt.date.apply(lambda d: int(d in us_holidays))

    def _days_to_nearest_holiday(d) -> int:
        """Positive = days until next holiday; negative = days since last holiday."""
        for delta in range(0, 31):
            if (d + timedelta(days=delta)) in us_holidays:
                return delta
        for delta in range(1, 31):
            if (d - timedelta(days=delta)) in us_holidays:
                return -delta
        return 30  # no holiday within ±30 days

    df["days_to_holiday"] = dates.dt.date.apply(_days_to_nearest_holiday)
    return df


# ---------------------------------------------------------------------------
# Target label
# ---------------------------------------------------------------------------


def compute_target_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add target_return = (close_{t+1} - close_t) / close_t per ticker.

    Call on the full historical DataFrame (all dates, all tickers).
    The last row per ticker will have NaN — expected.

    Args:
        df: Sorted ascending by date; must have 'date', 'ticker', 'close'.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["target_return"] = df.groupby("ticker")["close"].transform(
        lambda s: s.shift(-1).sub(s).div(s)
    )
    return df
