"""Signal detectors — full suite.

Each detector is a pure function that inspects an indicator-enriched frame
and returns a :class:`SignalReading` — a fired-or-not boolean plus a
0.0-to-1.0 strength score plus the raw reading(s) that drove the decision.

Implemented signals (keys must match :data:`investa.pipeline.scorer.SIGNAL_WEIGHTS`):

    rsi        — RSI oversold + optional bullish divergence       (25%)
    macd       — MACD bullish crossover and/or divergence         (25%)
    bollinger  — lower-band touch / mean-reversion pressure       (20%)
    volume     — accumulation volume anomaly                      (15%)
    obv        — OBV rising while price falls (hidden buying)     (10%)
    stochastic — secondary oversold confirmation                  (5%)

Plus a weekly-confirmation *bonus* (separate from the weighted signals,
applied as up to +5 pts on top of the composite by the scorer layer).

Divergence logic
----------------
Bullish divergence = price prints a *lower* low while the oscillator prints a
*higher* low. We locate swing lows via a simple pivot detector: a bar is a
swing low if its low is the minimum within a ±``pivot_window`` neighbourhood.
We then compare the two most recent swing lows' oscillator readings. This is
deterministic and reproducible — no curve-fitting, no smoothing tricks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .indicators import OBV_SLOPE_NORM_WINDOW

_PIVOT_WINDOW = 2  # bars on each side of a swing low


@dataclass
class SignalReading:
    """One signal's contribution to the composite score.

    * ``fired`` — did the signal trigger at all?
    * ``strength`` — 0.0..1.0. Feeds directly into the weighted scorer.
    * ``value`` — the primary raw indicator reading (e.g. RSI = 27.4) for the UI.
    * ``meta`` — free-form extra context (e.g. {"divergence": True}) for
      explainability.
    """

    fired: bool
    strength: float
    value: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fired": bool(self.fired),
            "strength": round(float(self.strength), 4),
            "value": None if self.value is None else round(float(self.value), 4),
            "meta": self.meta,
        }


# ── helpers ─────────────────────────────────────────────────────────


def _swing_lows(series: pd.Series, pivot_window: int = _PIVOT_WINDOW) -> list[int]:
    """Return integer positions of swing-low pivots in ``series``.

    A bar ``i`` is a swing low if ``series[i]`` is the minimum of the window
    ``[i - pivot_window, i + pivot_window]`` (inclusive). Boundary bars are
    never pivots. NaNs are ignored in the min comparison (a NaN bar cannot be
    a pivot).
    """
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    pivots: list[int] = []
    for i in range(pivot_window, n - pivot_window):
        window = vals[i - pivot_window : i + pivot_window + 1]
        if np.isnan(vals[i]):
            continue
        # Strictly-less-than to avoid flat-plateau double-counting.
        if np.nanmin(window) == vals[i] and np.nansum(window == vals[i]) == 1:
            pivots.append(i)
    return pivots


def _bullish_divergence(price: pd.Series, oscillator: pd.Series,
                        lookback: int, pivot_window: int = _PIVOT_WINDOW
                        ) -> tuple[bool, float]:
    """Detect bullish divergence within the last ``lookback`` bars.

    Returns ``(divergence_found, strength)``. Strength is 0.0 (no divergence)
    or 0.3 (weak) / 0.5 (strong) based on how clean the oscillator higher-low
    is relative to the price lower-low. Used as an *additive bonus* on top of
    a base signal strength by callers.
    """
    if len(price) < lookback or len(oscillator) < lookback:
        return False, 0.0

    p_recent = price.iloc[-lookback:]
    o_recent = oscillator.iloc[-lookback:]

    pivots = _swing_lows(p_recent, pivot_window)
    if len(pivots) < 2:
        return False, 0.0

    p1, p2 = pivots[-2], pivots[-1]
    price_lo1, price_lo2 = float(p_recent.iloc[p1]), float(p_recent.iloc[p2])
    osc_lo1, osc_lo2 = float(o_recent.iloc[p1]), float(o_recent.iloc[p2])

    if np.isnan(osc_lo1) or np.isnan(osc_lo2):
        return False, 0.0

    price_lower_low = price_lo2 < price_lo1
    osc_higher_low = osc_lo2 > osc_lo1

    if price_lower_low and osc_higher_low:
        # Cleaner divergence = bigger oscillator uplift relative to price drop.
        osc_uplift = (osc_lo2 - osc_lo1) / (abs(osc_lo1) + 1e-9)
        strength = 0.5 if osc_uplift > 0.1 else 0.3
        return True, strength
    return False, 0.0


def _clamp_strength(x: float) -> float:
    return max(0.0, min(1.0, x))


# ── RSI oversold ────────────────────────────────────────────────────
# Strength curve from the plan (piecewise linear):
#   RSI >  35        → strength 0.0 (not fired)
#   RSI =  35        → strength 0.5  (fired, threshold)
#   RSI =  30        → strength 0.8
#   RSI <= 20        → strength 1.0
# Plus a +0.3/+0.5 bonus if a bullish divergence is detected over the last
# 10 candles (price lower-low, RSI higher-low).
_RSI_THRESHOLD = 35.0
_RSI_DIVERGENCE_LOOKBACK = 10


def _rsi_strength(rsi_value: float) -> float:
    if rsi_value > _RSI_THRESHOLD:
        return 0.0
    if rsi_value >= 30.0:
        # 35 → 0.5, 30 → 0.8, linear between
        return 0.5 + (35.0 - rsi_value) * (0.3 / 5.0)
    if rsi_value >= 20.0:
        # 30 → 0.8, 20 → 1.0, linear between
        return 0.8 + (30.0 - rsi_value) * (0.2 / 10.0)
    return 1.0


def detect_rsi_oversold(df: pd.DataFrame) -> SignalReading:
    """Return the RSI-oversold reading for the latest bar of ``df``.

    ``df`` must include ``rsi_14`` and ``close`` columns. Warm-up (NaN)
    yields ``fired=False, strength=0.0``. A bullish divergence over the last
    10 candles adds 0.3–0.5 to the strength (capped at 1.0).
    """
    if "rsi_14" not in df.columns or df.empty:
        return SignalReading(False, 0.0, None)

    rsi_value = df["rsi_14"].iloc[-1]
    if pd.isna(rsi_value):
        return SignalReading(False, 0.0, None)

    base = _rsi_strength(float(rsi_value))
    div_found, div_strength = False, 0.0
    if "close" in df.columns and base > 0.0:
        div_found, div_strength = _bullish_divergence(
            df["close"], df["rsi_14"], _RSI_DIVERGENCE_LOOKBACK
        )

    strength = _clamp_strength(base + div_strength)
    return SignalReading(
        fired=strength > 0.0,
        strength=strength,
        value=float(rsi_value),
        meta={"divergence": div_found, "base_strength": round(base, 4)},
    )


# ── MACD bullish crossover / divergence ─────────────────────────────
_MACD_CROSSOVER_LOOKBACK = 3
_MACD_DIVERGENCE_LOOKBACK = 14


def detect_macd(df: pd.DataFrame) -> SignalReading:
    """Bullish MACD reading for the latest bar.

    * Crossover (MACD line crosses above signal within last 3 bars): +0.7
    * Divergence (price lower-low, MACD histogram higher-low over 14 bars): +0.9
    * Both → 1.0

    Warm-up (NaN on MACD line) yields ``fired=False``.
    """
    needed = ("macd", "macd_signal", "macd_hist", "close")
    if any(c not in df.columns for c in needed) or df.empty:
        return SignalReading(False, 0.0, None)

    macd_line = df["macd"]
    signal_line = df["macd_signal"]
    hist = df["macd_hist"]

    if pd.isna(macd_line.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
        return SignalReading(False, 0.0, None)

    # Crossover within the last N bars: diff flips from <=0 to >0.
    diff = (macd_line - signal_line).tail(_MACD_CROSSOVER_LOOKBACK + 1).dropna()
    crossover = False
    if len(diff) >= 2:
        signs = np.sign(diff.to_numpy())
        crossed = bool(np.any((signs[:-1] <= 0) & (signs[1:] > 0)))
        crossover = crossed and bool(signs[-1] > 0)

    div_found, _ = _bullish_divergence(
        df["close"], hist, _MACD_DIVERGENCE_LOOKBACK
    )
    div_found = bool(div_found)

    if crossover and div_found:
        strength = 1.0
    elif div_found:
        strength = 0.9
    elif crossover:
        strength = 0.7
    else:
        strength = 0.0

    return SignalReading(
        fired=strength > 0.0,
        strength=strength,
        value=float(hist.iloc[-1]),
        meta={"crossover": bool(crossover), "divergence": bool(div_found)},
    )


# ── Bollinger Band lower touch ──────────────────────────────────────
# Fires when close <= lower Bollinger Band. Strength is the ATR-multiple
# distance below the band — the further below, the stronger the
# mean-reversion pressure. Capped at 1.0 for distances ≥ 1 ATR below band.
def detect_bollinger(df: pd.DataFrame) -> SignalReading:
    """Lower-Bollinger-band touch / mean-reversion-pressure reading.

    Required columns: ``close``, ``bb_lower``, ``bb_upper``, ``bb_pos``,
    ``atr_14``. Warm-up (NaN) yields ``fired=False``.
    """
    needed = ("close", "bb_lower", "bb_upper", "bb_pos", "atr_14")
    if any(c not in df.columns for c in needed) or df.empty:
        return SignalReading(False, 0.0, None)

    close = float(df["close"].iloc[-1])
    bb_lower = df["bb_lower"].iloc[-1]
    bb_pos = df["bb_pos"].iloc[-1]
    atr_val = df["atr_14"].iloc[-1]

    if pd.isna(bb_lower) or pd.isna(atr_val) or atr_val <= 0:
        return SignalReading(False, 0.0, None)

    bb_lower = float(bb_lower)
    atr_val = float(atr_val)

    # Plan condition: close <= lower band.
    if close > bb_lower:
        return SignalReading(False, 0.0, value=float(bb_pos) if not pd.isna(bb_pos) else None)

    distance_below_atr = (bb_lower - close) / atr_val
    strength = _clamp_strength(distance_below_atr)  # 1 ATR below → 1.0

    return SignalReading(
        fired=True,
        strength=strength,
        value=float(bb_pos) if not pd.isna(bb_pos) else None,
        meta={"distance_below_band_atr": round(distance_below_atr, 4)},
    )


# ── Volume anomaly (accumulation) ───────────────────────────────────
# Fires when current volume > 1.5× 20-day average AND the bar's price
# change is >= -0.5% (price stabilising / up on big volume = accumulation).
_VOL_RATIO_THRESHOLD = 1.5
_VOL_PRICE_CHANGE_FLOOR = -0.005  # -0.5%
_VOL_RATIO_CAP = 3.0  # strength saturates at 3× average volume


def detect_volume(df: pd.DataFrame) -> SignalReading:
    """Accumulation volume-anomaly reading for the latest bar."""
    needed = ("volume", "volume_avg_20", "volume_ratio", "pct_change_1")
    if any(c not in df.columns for c in needed) or df.empty:
        return SignalReading(False, 0.0, None)

    vol_avg = df["volume_avg_20"].iloc[-1]
    vol_ratio = df["volume_ratio"].iloc[-1]
    pct = df["pct_change_1"].iloc[-1]

    if pd.isna(vol_ratio) or pd.isna(pct):
        return SignalReading(False, 0.0, None)

    vol_ratio = float(vol_ratio)
    pct = float(pct)

    # Skip pathological zero-volume / zero-average situations.
    if vol_ratio <= 0 or (not pd.isna(vol_avg) and float(vol_avg) <= 0):
        return SignalReading(False, 0.0, value=vol_ratio,
                             meta={"skipped": "zero_volume_or_average"})

    fired = (vol_ratio >= _VOL_RATIO_THRESHOLD) and (pct >= _VOL_PRICE_CHANGE_FLOOR)
    if not fired:
        return SignalReading(False, 0.0, value=vol_ratio,
                             meta={"ratio": round(vol_ratio, 4),
                                   "pct_change": round(pct, 4)})

    # Linear strength: 1.5×→0.0 start, 3×→1.0, capped above 3×.
    capped = min(vol_ratio, _VOL_RATIO_CAP)
    strength = _clamp_strength((capped - _VOL_RATIO_THRESHOLD)
                               / (_VOL_RATIO_CAP - _VOL_RATIO_THRESHOLD))
    return SignalReading(
        fired=True,
        strength=strength,
        value=vol_ratio,
        meta={"pct_change": round(pct, 4), "capped_ratio": round(capped, 4)},
    )


# ── OBV trend divergence ────────────────────────────────────────────
# Fires when OBV's 5-period slope is positive while the latest bar's price
# is still declining (pct_change_1 < 0). Strength scales with the OBV slope
# magnitude normalised against the rolling 30-bar OBV range.
_OBV_SLOPE_NORM_WINDOW = OBV_SLOPE_NORM_WINDOW  # 30


def detect_obv(df: pd.DataFrame) -> SignalReading:
    """Rising-OBV-on-falling-price (hidden accumulation) reading."""
    needed = ("close", "obv", "obv_slope", "pct_change_1")
    if any(c not in df.columns for c in needed) or df.empty:
        return SignalReading(False, 0.0, None)

    obv_slope = df["obv_slope"].iloc[-1]
    pct = df["pct_change_1"].iloc[-1]

    if pd.isna(obv_slope) or pd.isna(pct):
        return SignalReading(False, 0.0, None)

    obv_slope = float(obv_slope)
    pct = float(pct)

    # Price must still be falling (or flat) on the latest bar.
    price_declining = pct <= 0.0
    obv_rising = obv_slope > 0.0
    if not (price_declining and obv_rising):
        return SignalReading(False, 0.0, value=obv_slope,
                             meta={"obv_slope": round(obv_slope, 4),
                                   "pct_change": round(pct, 4)})

    # Normalise slope against the rolling range of OBV itself.
    obv_window = df["obv"].tail(_OBV_SLOPE_NORM_WINDOW).dropna()
    if obv_window.empty:
        return SignalReading(True, 0.5, value=obv_slope)
    rng = float(obv_window.max() - obv_window.min())
    if rng <= 0:
        return SignalReading(True, 0.5, value=obv_slope)
    strength = _clamp_strength(abs(obv_slope) / rng)
    return SignalReading(
        fired=True,
        strength=strength,
        value=obv_slope,
        meta={"pct_change": round(pct, 4), "normalised_slope": round(strength, 4)},
    )


# ── Stochastic oversold (confirmation only) ─────────────────────────
_STOCH_OVERSOLD = 20.0


def detect_stochastic(df: pd.DataFrame) -> SignalReading:
    """Secondary oversold confirmation. Binary 0.5 weight when both %K and %D
    are below 20; 0.0 otherwise."""
    needed = ("stoch_k", "stoch_d")
    if any(c not in df.columns for c in needed) or df.empty:
        return SignalReading(False, 0.0, None)

    k = df["stoch_k"].iloc[-1]
    d = df["stoch_d"].iloc[-1]
    if pd.isna(k) or pd.isna(d):
        return SignalReading(False, 0.0, None)

    k, d = float(k), float(d)
    if k < _STOCH_OVERSOLD and d < _STOCH_OVERSOLD:
        return SignalReading(True, 0.5, value=k, meta={"d": round(d, 4)})
    return SignalReading(False, 0.0, value=k, meta={"d": round(d, 4)})


# ── Weekly confirmation bonus ───────────────────────────────────────
# Distinct from the six weighted signals: this is a +0..+5 pt *bonus* the
# scorer applies on top of the composite. We detect it here and let the
# scorer layer fold it in.
_WEEKLY_CONFIRM_BONUS_MAX = 5.0


def detect_weekly_confirmation(df: pd.DataFrame, weekly_df: pd.DataFrame | None
                               ) -> SignalReading:
    """Bonus: is the weekly chart also showing reversal pressure?

    Confirmation logic (any one suffices, additive up to the cap):
    * Weekly RSI-14 < 40  → +3
    * Weekly close above weekly EMA-10  → +2  (early weekly momentum turn)

    Returns a :class:`SignalReading` whose ``value`` is the bonus points (0..5)
    rather than a 0..1 strength — the scorer knows to treat this signal
    specially. ``fired`` is True when the bonus > 0.
    """
    if weekly_df is None or weekly_df.empty or len(weekly_df) < 14:
        return SignalReading(False, 0.0, value=0.0, meta={"weekly": "insufficient_data"})

    from .indicators import ema as _ema, rsi as _rsi

    w_close = weekly_df["close"]
    w_rsi = _rsi(w_close, 14)
    w_ema10 = _ema(w_close, 10)

    bonus = 0.0
    meta: dict[str, Any] = {}

    latest_rsi = w_rsi.iloc[-1]
    if not pd.isna(latest_rsi) and float(latest_rsi) < 40.0:
        bonus += 3.0
        meta["weekly_rsi"] = round(float(latest_rsi), 4)

    latest_ema10 = w_ema10.iloc[-1]
    latest_close = float(w_close.iloc[-1])
    if not pd.isna(latest_ema10) and latest_close > float(latest_ema10):
        bonus += 2.0
        meta["weekly_above_ema10"] = True

    bonus = min(bonus, _WEEKLY_CONFIRM_BONUS_MAX)
    return SignalReading(
        fired=bonus > 0.0,
        strength=bonus / 100.0,  # stored as a fraction for as_dict consistency
        value=bonus,
        meta=meta,
    )


# ── aggregator ──────────────────────────────────────────────────────
# Keys must match the weight keys in :mod:`investa.pipeline.scorer`.
def detect_all(df: pd.DataFrame, weekly_df: pd.DataFrame | None = None
               ) -> dict[str, SignalReading]:
    """Run every implemented detector and return a dict keyed by signal name.

    The weekly-confirmation reading is included under the key ``weekly`` —
    the scorer treats it as a bonus, not a weighted signal.
    """
    readings: dict[str, SignalReading] = {
        "rsi": detect_rsi_oversold(df),
        "macd": detect_macd(df),
        "bollinger": detect_bollinger(df),
        "volume": detect_volume(df),
        "obv": detect_obv(df),
        "stochastic": detect_stochastic(df),
        "weekly": detect_weekly_confirmation(df, weekly_df),
    }
    return readings
