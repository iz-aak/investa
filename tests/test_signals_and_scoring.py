"""Unit tests for the signal detectors, scorer, gate, and risk annotator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investa.config import ScanMode
from investa.pipeline.gate import evaluate_gate
from investa.pipeline.indicators import compute_indicators
from investa.pipeline.risk import annotate_risk
from investa.pipeline.scorer import (
    SCORE_BANDS,
    SIGNAL_WEIGHTS,
    WEEKLY_BONUS_MAX,
    band_for,
    compute_score,
)
from investa.pipeline.signals import (
    SignalReading,
    _bullish_divergence,
    _swing_lows,
    detect_all,
    detect_bollinger,
    detect_macd,
    detect_obv,
    detect_rsi_oversold,
    detect_stochastic,
    detect_volume,
    detect_weekly_confirmation,
)



# ── RSI signal ──────────────────────────────────────────────────────

def _frame_with_rsi(rsi_value: float | None) -> pd.DataFrame:
    return pd.DataFrame({"rsi_14": [50.0, 40.0, rsi_value]})


def test_rsi_signal_does_not_fire_at_neutral():
    r = detect_rsi_oversold(_frame_with_rsi(50.0))
    assert r.fired is False
    assert r.strength == 0.0


@pytest.mark.parametrize(
    "value,expected_strength",
    [
        (35.0, 0.5),   # threshold
        (30.0, 0.8),   # mid-band
        (20.0, 1.0),   # deep oversold
        (15.0, 1.0),   # capped at 1.0
    ],
)
def test_rsi_signal_strength_curve(value, expected_strength):
    r = detect_rsi_oversold(_frame_with_rsi(value))
    assert r.fired is True
    assert r.strength == pytest.approx(expected_strength, abs=1e-6)


def test_rsi_signal_handles_nan_warmup():
    r = detect_rsi_oversold(_frame_with_rsi(np.nan))
    assert r.fired is False
    assert r.strength == 0.0


# ── Scorer ─────────────────────────────────────────────────────────

def test_scorer_weights_sum_to_one():
    assert sum(SIGNAL_WEIGHTS.values()) == pytest.approx(1.0)


def test_scorer_empty_signals_returns_zero():
    assert compute_score({}) == 0.0


def test_scorer_single_rsi_max_strength_yields_25_pts():
    signals = {"rsi": SignalReading(fired=True, strength=1.0, value=15.0)}
    # 1.0 strength × 25% weight × 100 = 25.0
    assert compute_score(signals) == pytest.approx(25.0)


def test_scorer_all_signals_max_yields_100_pts():
    signals = {k: SignalReading(True, 1.0) for k in SIGNAL_WEIGHTS}
    assert compute_score(signals) == pytest.approx(100.0)


def test_scorer_clamps_out_of_range_strengths():
    signals = {"rsi": SignalReading(True, 5.0)}  # nonsense strength
    assert compute_score(signals) == pytest.approx(25.0)


def test_band_for_covers_full_range():
    assert band_for(0.0).label == "Weak"
    assert band_for(45.0).label == "Moderate"
    assert band_for(65.0).label == "Strong"
    assert band_for(80.0).label == "Very Strong"
    assert band_for(100.0).label == "Conviction"


def test_bands_are_contiguous():
    # Every score in [0, 100] must land in exactly one band.
    for s in np.linspace(0, 100, 501):
        matches = [b for b in SCORE_BANDS if b.contains(s)]
        assert len(matches) == 1, f"score {s} matched {len(matches)} bands"


# ── Gate ───────────────────────────────────────────────────────────

def _row(**vals) -> pd.Series:
    base = {"close": np.nan, "ema_50": np.nan, "ema_200": np.nan}
    base.update(vals)
    return pd.Series(base)


def test_deep_reversal_gate_passes_when_price_below_both_emas():
    result = evaluate_gate(_row(close=90, ema_50=100, ema_200=110), ScanMode.deep_reversal)
    assert result.passed is True


def test_deep_reversal_gate_fails_when_close_above_ema50():
    result = evaluate_gate(_row(close=105, ema_50=100, ema_200=110), ScanMode.deep_reversal)
    assert result.passed is False
    assert result.reason == "close_not_below_ema50"


def test_deep_reversal_gate_fails_when_no_death_cross():
    result = evaluate_gate(_row(close=90, ema_50=110, ema_200=100), ScanMode.deep_reversal)
    assert result.passed is False
    assert result.reason == "ema50_not_below_ema200"


def test_deep_reversal_gate_fails_on_warmup():
    result = evaluate_gate(_row(close=100), ScanMode.deep_reversal)
    assert result.passed is False
    assert result.reason.startswith("warmup")


def test_pullback_gate_passes_in_uptrend_pullback():
    """close above ema200, below ema50, ema50 above ema200 → passes."""
    result = evaluate_gate(
        _row(close=105, ema_50=110, ema_200=100), ScanMode.pullback
    )
    assert result.passed is True
    assert result.reason == "pullback_in_uptrend_confirmed"


def test_pullback_gate_fails_when_close_below_ema200():
    """No long-term uptrend → not a pullback, it's a downtrend."""
    result = evaluate_gate(
        _row(close=90, ema_50=100, ema_200=110), ScanMode.pullback
    )
    assert result.passed is False
    assert result.reason == "close_not_above_ema200"


def test_pullback_gate_fails_when_no_pullback_to_ema50():
    """close above ema50 → not pulled back, just trending up."""
    result = evaluate_gate(
        _row(close=115, ema_50=110, ema_200=100), ScanMode.pullback
    )
    assert result.passed is False
    assert result.reason == "close_not_below_ema50"


def test_pullback_gate_fails_on_warmup():
    result = evaluate_gate(_row(close=100), ScanMode.pullback)
    assert result.passed is False
    assert result.reason.startswith("warmup")


# ── Risk annotator ─────────────────────────────────────────────────

def test_risk_returns_none_when_atr_unavailable():
    df = pd.DataFrame({"close": [100.0], "high": [101.0], "atr_14": [np.nan]})
    r = annotate_risk(df)
    assert r.suggested_stop is None
    assert r.suggested_target is None


def test_risk_stop_is_two_atrs_below_close_with_fallback_target():
    """No high above close in the window → target falls back to close + 3*ATR."""
    df = pd.DataFrame(
        {
            "close": [100.0] * 25,
            "high": [99.5] * 25,  # all highs below close (unrealistic but forces fallback path)
            "atr_14": [2.0] * 25,
        }
    )
    r = annotate_risk(df)
    assert r.suggested_stop == pytest.approx(96.0)
    assert r.suggested_target == pytest.approx(106.0)
    # R:R = (106-100) / (100-96) = 1.5
    assert r.risk_reward == pytest.approx(1.5)


def test_risk_target_uses_nearest_swing_high_when_available():
    highs = [105.0] * 19 + [110.0]  # last bar has a high above close
    df = pd.DataFrame(
        {
            "close": [100.0] * 20,
            "high": highs,
            "atr_14": [2.0] * 20,
        }
    )
    r = annotate_risk(df)
    # min of highs above close = 105
    assert r.suggested_target == pytest.approx(105.0)


# ── Swing-low / divergence helpers ──────────────────────────────────

def test_swing_lows_detects_clean_valleys():
    # Two obvious valleys at index 2 and 6.
    s = pd.Series([5.0, 3.0, 1.0, 3.0, 5.0, 3.0, 1.0, 3.0, 5.0])
    pivots = _swing_lows(s, pivot_window=2)
    assert pivots == [2, 6]


def test_swing_lows_ignores_flat_plateaus():
    # A flat bottom of three equal lows should not register three pivots.
    s = pd.Series([5.0, 1.0, 1.0, 1.0, 5.0])
    pivots = _swing_lows(s, pivot_window=2)
    assert pivots == []  # the strict-unique rule rejects the plateau


def test_bullish_divergence_fires_on_price_lower_low_osc_higher_low():
    price = pd.Series([10, 8, 6, 8, 10, 7, 5, 7, 10], dtype=float)
    osc = pd.Series([30, 25, 20, 25, 30, 28, 24, 28, 30], dtype=float)
    found, strength = _bullish_divergence(price, osc, lookback=9)
    assert found is True
    assert strength > 0.0


def test_bullish_divergence_does_not_fire_when_both_make_lower_lows():
    price = pd.Series([10, 8, 6, 8, 10, 7, 5, 7, 10], dtype=float)
    osc = pd.Series([30, 25, 22, 25, 30, 25, 18, 25, 30], dtype=float)
    found, _ = _bullish_divergence(price, osc, lookback=9)
    assert found is False


# ── MACD / Bollinger / Volume / OBV / Stochastic detectors ──────────

def _enriched(n: int = 60, *, close: np.ndarray | None = None) -> pd.DataFrame:
    if close is None:
        close = np.linspace(100, 80, n)  # gentle downtrend
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": np.full(n, 1_000_000)},
        index=idx,
    )
    return compute_indicators(df)


def test_macd_signal_warmup_returns_not_fired():
    df = _enriched(n=20)
    r = detect_macd(df)
    assert r.fired is False
    assert r.strength == 0.0


def test_macd_crossover_detected_on_reversal():
    n = 60
    base = np.linspace(120, 90, n - 5)
    close = np.concatenate([base, np.linspace(90, 100, 5)])
    df = _enriched(n=n, close=close)
    r = detect_macd(df)
    assert isinstance(r, SignalReading)
    assert r.meta.keys() >= {"crossover", "divergence"}


def test_bollinger_does_not_fire_when_above_lower_band():
    df = _enriched(n=60)
    r = detect_bollinger(df)
    assert r.fired is False


def test_bollinger_fires_when_close_below_lower_band():
    n = 60
    close = np.full(n, 100.0)
    close[-3:] = 80.0
    df = _enriched(n=n, close=close)
    r = detect_bollinger(df)
    assert r.fired is True
    assert 0.0 < r.strength <= 1.0


def test_volume_does_not_fire_on_average_volume_flat_price():
    df = _enriched(n=60, close=np.full(60, 100.0))
    r = detect_volume(df)
    assert r.fired is False


def test_volume_fires_on_surge_with_stable_price():
    n = 60
    close = np.full(n, 100.0)
    close[-1] = 100.5  # +0.5% (above the -0.5% floor)
    df = _enriched(n=n, close=close)
    df.iloc[-1, df.columns.get_loc("volume")] = 4_000_000
    df["volume_avg_20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg_20"]
    df["pct_change_1"] = df["close"].pct_change(1)
    r = detect_volume(df)
    assert r.fired is True
    assert r.strength == pytest.approx(1.0)


def test_volume_does_not_fire_on_surge_with_sharp_drop():
    """Volume spike on a -3% day is distribution, not accumulation."""
    n = 60
    close = np.full(n, 100.0)
    close[-1] = 97.0
    df = _enriched(n=n, close=close)
    df.iloc[-1, df.columns.get_loc("volume")] = 4_000_000
    df["volume_avg_20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg_20"]
    df["pct_change_1"] = df["close"].pct_change(1)
    r = detect_volume(df)
    assert r.fired is False


def test_obv_fires_when_obv_rising_on_declining_price():
    n = 60
    rng = np.random.default_rng(0)
    close = np.linspace(100, 95, n) + rng.normal(0, 0.2, n)
    # Force the final bar to be a down-bar so the OBV divergence condition
    # (price declining on the latest bar) is satisfied.
    close[-1] = close[-2] - 0.5
    # Make up-bars carry 4× the volume of down-bars so OBV trends up
    # despite the overall price drift down.
    volume = np.where(np.diff(np.concatenate([[100], close])) >= 0, 4_000_000, 500_000)
    df = _enriched(n=n, close=close)
    df["volume"] = volume
    df = compute_indicators(df)
    r = detect_obv(df)
    assert r.fired is True
    assert r.strength > 0.0


def test_stochastic_does_not_fire_when_not_oversold():
    close = np.linspace(50, 100, 60)
    df = _enriched(n=60, close=close)
    r = detect_stochastic(df)
    assert r.fired is False


def test_stochastic_fires_in_oversold_zone():
    close = np.linspace(100, 60, 60)
    close[-10:] = 60.0  # plateau at the low
    df = _enriched(n=60, close=close)
    r = detect_stochastic(df)
    assert r.fired is True
    assert r.strength == 0.5


# ── Weekly confirmation ─────────────────────────────────────────────

def test_weekly_confirmation_returns_zero_without_weekly_data():
    df = _enriched(n=60)
    r = detect_weekly_confirmation(df, None)
    assert r.fired is False
    assert (r.value or 0.0) == 0.0


def test_weekly_confirmation_fires_on_oversold_weekly_rsi():
    n = 40
    close = np.linspace(200, 80, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="W-FRI", tz="UTC")
    weekly = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": np.full(n, 5_000_000)},
        index=idx,
    )
    df = _enriched(n=60)
    r = detect_weekly_confirmation(df, weekly)
    assert r.fired is True
    assert 3.0 <= (r.value or 0.0) <= WEEKLY_BONUS_MAX


# ── detect_all + scorer integration ─────────────────────────────────

def test_detect_all_returns_all_seven_keys():
    df = _enriched(n=60)
    readings = detect_all(df, weekly_df=None)
    assert set(readings.keys()) == {"rsi", "macd", "bollinger", "volume",
                                    "obv", "stochastic", "weekly"}


def test_scorer_applies_weekly_bonus():
    signals = {"rsi": SignalReading(True, 1.0)}
    base = compute_score(signals)
    signals_with_bonus = {**signals,
                          "weekly": SignalReading(True, 0.03, value=5.0)}
    boosted = compute_score(signals_with_bonus)
    assert boosted == pytest.approx(base + WEEKLY_BONUS_MAX)


def test_scorer_clamps_weekly_bonus_at_max():
    signals = {"weekly": SignalReading(True, 1.0, value=99.0)}
    score = compute_score(signals)
    assert score == pytest.approx(WEEKLY_BONUS_MAX)
