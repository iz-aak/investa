"""Unit tests for the pandas-native indicator engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from investa.pipeline.indicators import (
    ATR_PERIOD,
    EMA_SLOW_PERIOD,
    RSI_PERIOD,
    atr,
    compute_indicators,
    ema,
    rsi,
)


def _linear_frame(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = np.array([start + i * step for i in range(n)], dtype=float)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": np.full(n, 1_000_000)},
        index=idx,
    )


def test_rsi_warmup_returns_nan():
    close = _linear_frame(RSI_PERIOD)["close"]
    r = rsi(close, RSI_PERIOD)
    # First `period` rows have no valid RSI.
    assert r.iloc[: RSI_PERIOD - 1].isna().all()


def test_rsi_pegs_at_100_on_monotonic_uptrend():
    # 30 up days → all gains, RSI must converge to 100.
    close = _linear_frame(30, start=50, step=1)["close"]
    r = rsi(close, RSI_PERIOD).dropna()
    assert (r.iloc[-5:] > 99.0).all()


def test_rsi_below_30_on_monotonic_downtrend():
    close = _linear_frame(30, start=100, step=-1)["close"]
    r = rsi(close, RSI_PERIOD).dropna()
    assert (r.iloc[-5:] < 5.0).all()


def test_ema_shorter_than_period_is_nan():
    close = _linear_frame(10)["close"]
    e = ema(close, 20)
    assert e.isna().all()


def test_atr_positive_when_bars_have_range():
    df = _linear_frame(30, start=100, step=0.1)
    a = atr(df["high"], df["low"], df["close"], ATR_PERIOD).dropna()
    assert (a > 0).all()


def test_compute_indicators_appends_expected_columns():
    df = _linear_frame(EMA_SLOW_PERIOD + 5)
    enriched = compute_indicators(df)
    for col in ("rsi_14", "ema_50", "ema_200", "atr_14", "volume_avg_20", "volume_ratio"):
        assert col in enriched.columns
    # Warm-up: ema_200 must have valid tail values.
    assert not np.isnan(enriched["ema_200"].iloc[-1])


def test_compute_indicators_does_not_mutate_input():
    df = _linear_frame(60)
    original_cols = list(df.columns)
    _ = compute_indicators(df)
    assert list(df.columns) == original_cols
