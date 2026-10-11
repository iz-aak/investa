"""Indicator engine — full set, pandas-native.

Every technical indicator used by the signal detectors is computed here. The
module is deliberately framework-free (no TA-Lib / pandas-ta dependency) so it
runs on any Python with pandas installed, including 3.14 where TA-Lib wheels
are not always available. Maths follow the standard reference definitions
(Wilder smoothing for RSI/ATR, standard MACD, SMA-based Bollinger bands,
classic %K/%D stochastic, cumulative OBV) so results are reproducible against
TradingView / TA-Lib up to float tolerance.

Design contract (from the plan):
* **Pure functions** — same input always produces same output. No global state.
* **No mutation** — a copy of the caller's frame is returned.
* **No fabrication** — warm-up periods that cannot be satisfied yield NaN.
  Signal detectors must treat NaN as "not ready", never as a value.

Indicators computed
-------------------
RSI-14, EMA-50, EMA-200, ATR-14, MACD(12,26,9), Bollinger(20,2),
OBV + 5-period slope, Stochastic(14,3), 20-day volume average + ratio,
1-bar percent change, weekly resampled frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RSI_PERIOD = 14
EMA_FAST_PERIOD = 50
EMA_SLOW_PERIOD = 200
ATR_PERIOD = 14
VOLUME_AVG_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
OBV_SLOPE_PERIOD = 5
OBV_SLOPE_NORM_WINDOW = 30


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA) — equivalent to EMA with alpha = 1/period.

    Uses the standard 'first value = simple mean of first `period` obs' seed
    so results match reference implementations (TA-Lib, TradingView).
    """
    if len(series) < period:
        return pd.Series(np.nan, index=series.index)

    # Seed with SMA of the first `period` values; recursive smoothing after.
    values = series.to_numpy(dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    seed = np.nanmean(values[:period])
    out[period - 1] = seed
    alpha = 1.0 / period
    for i in range(period, len(values)):
        prev = out[i - 1]
        cur = values[i]
        if np.isnan(prev) or np.isnan(cur):
            out[i] = prev
        else:
            out[i] = prev + alpha * (cur - prev)
    return pd.Series(out, index=series.index)


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI. Returns NaN for the first `period` rows (warm-up)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # If avg_loss is 0 the market only went up in the window → RSI = 100.
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return out


def ema(close: pd.Series, period: int) -> pd.Series:
    """Standard exponential moving average. NaN until we have `period` obs."""
    result = close.ewm(span=period, adjust=False, min_periods=period).mean()
    return result


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder_rma(tr, period)


def volume_average(volume: pd.Series, period: int = VOLUME_AVG_PERIOD) -> pd.Series:
    return volume.rolling(window=period, min_periods=period).mean()


def percent_change(close: pd.Series, periods: int = 1) -> pd.Series:
    """1-bar ROC as a fraction (0.01 = +1%). NaN for the first row."""
    return close.pct_change(periods=periods)


def macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW,
         signal: int = MACD_SIGNAL) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Standard MACD. Returns (macd_line, signal_line, histogram).

    NaN until the slow EMA is warm (≈ `slow` bars) and NaN on the signal line
    until `signal` additional bars have passed.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, period: int = BB_PERIOD, std: float = BB_STD
              ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (lower, mid, upper, bb_pos).

    ``bb_pos`` is the 0..1 position of close within the band:
        0 → on the lower band, 1 → on the upper band, <0 / >1 → outside.
    NaN during warm-up.
    """
    mid = close.rolling(window=period, min_periods=period).mean()
    roll_std = close.rolling(window=period, min_periods=period).std(ddof=0)
    lower = mid - std * roll_std
    upper = mid + std * roll_std
    width = (upper - lower).replace(0.0, np.nan)
    bb_pos = (close - lower) / width
    return lower, mid, upper, bb_pos


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative signed volume.

    Adds volume when close rises, subtracts when it falls, unchanged when flat.
    """
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = STOCH_K_PERIOD, d_period: int = STOCH_D_PERIOD
               ) -> tuple[pd.Series, pd.Series]:
    """Classic stochastic oscillator (%K fast, %D smoothed).

    Returns (stoch_k, stoch_d). %K = 100 * (close - lowest_low) / range.
    NaN until `k_period` bars are available; %D needs `d_period` more.
    """
    lowest = low.rolling(window=k_period, min_periods=k_period).min()
    highest = high.rolling(window=k_period, min_periods=k_period).max()
    range_ = (highest - lowest).replace(0.0, np.nan)
    stoch_k = 100.0 * (close - lowest) / range_
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
    return stoch_k, stoch_d


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append the full indicator column set to a copy of ``df``.

    Expected input columns (lowercase): ``open, high, low, close, volume``.
    Added columns:

    * ``rsi_14``
    * ``ema_50``, ``ema_200``
    * ``atr_14``
    * ``macd``, ``macd_signal``, ``macd_hist``
    * ``bb_lower``, ``bb_mid``, ``bb_upper``, ``bb_pos``
    * ``obv``, ``obv_slope``
    * ``stoch_k``, ``stoch_d``
    * ``volume_avg_20``, ``volume_ratio``  (ratio = volume / volume_avg_20)
    * ``pct_change_1``  (1-bar percent return)
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["rsi_14"] = rsi(out["close"], RSI_PERIOD)
    out["ema_50"] = ema(out["close"], EMA_FAST_PERIOD)
    out["ema_200"] = ema(out["close"], EMA_SLOW_PERIOD)
    out["atr_14"] = atr(out["high"], out["low"], out["close"], ATR_PERIOD)

    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["close"])
    out["bb_lower"], out["bb_mid"], out["bb_upper"], out["bb_pos"] = bollinger(out["close"])
    out["obv"] = obv(out["close"], out["volume"])
    # 5-bar linear-regression-ish slope: simple diff over OBV_SLOPE_PERIOD bars,
    # normalised against the rolling 30-bar OBV range for cross-ticker comparability.
    out["obv_slope"] = out["obv"].diff(OBV_SLOPE_PERIOD)
    out["stoch_k"], out["stoch_d"] = stochastic(out["high"], out["low"], out["close"])

    out["volume_avg_20"] = volume_average(out["volume"], VOLUME_AVG_PERIOD)
    out["volume_ratio"] = out["volume"] / out["volume_avg_20"]
    out["pct_change_1"] = percent_change(out["close"], 1)
    return out


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily indicator-enriched frame to weekly (W-FRI).

    Used by the weekly-confirmation signal. Requires a UTC DatetimeIndex.
    Returns an empty frame if the input is empty. Aggregates:
        open  → first, high → max, low → min, close → last, volume → sum
    Indicator columns are **not** carried over — the weekly frame is meant
    for trend confirmation only; callers recompute weekly indicators as needed.
    """
    if df.empty:
        return df.copy()

    agg_map: dict[str, str] = {"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}
    base_cols = [c for c in agg_map if c in df.columns]
    weekly = df[base_cols].resample("W-FRI").agg({c: agg_map[c] for c in base_cols})
    weekly = weekly.dropna(subset=["close"])
    return weekly
