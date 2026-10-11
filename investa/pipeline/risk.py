"""Risk annotation for scored candidates.

Given a scored row we expose three fields, all derived from data already
computed by the indicator engine:

* ``suggested_stop``   = close − 2 × ATR-14  (standard volatility stop)
* ``suggested_target`` = nearest 20-day swing high above close, else
                         close + 3 × ATR-14 (fallback so we always have a
                         target)
* ``risk_reward``      = (target − close) / (close − stop)

These are informational only — Investa does not place orders. Exposing them
makes the screener output actionable rather than purely diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

STOP_ATR_MULTIPLE = 2.0
FALLBACK_TARGET_ATR_MULTIPLE = 3.0
SWING_HIGH_LOOKBACK = 20


@dataclass(frozen=True)
class RiskAnnotation:
    suggested_stop: float | None
    suggested_target: float | None
    risk_reward: float | None

    def as_dict(self) -> dict[str, float | None]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(float(x), 4)

        return {
            "suggested_stop": r(self.suggested_stop),
            "suggested_target": r(self.suggested_target),
            "risk_reward": r(self.risk_reward),
        }


def annotate_risk(df: pd.DataFrame) -> RiskAnnotation:
    """Compute stop / target / R:R for the latest bar of ``df``.

    Requires columns ``close``, ``high``, ``atr_14`` to be present. Returns
    all-None if ATR is not yet available (warm-up).
    """
    if df.empty or "atr_14" not in df.columns:
        return RiskAnnotation(None, None, None)

    latest = df.iloc[-1]
    close = float(latest["close"])
    atr = latest["atr_14"]
    if pd.isna(atr) or atr <= 0:
        return RiskAnnotation(None, None, None)
    atr = float(atr)

    stop = close - STOP_ATR_MULTIPLE * atr

    # Nearest recent swing high above current close, over the last N bars.
    recent = df["high"].tail(SWING_HIGH_LOOKBACK).dropna()
    highs_above = recent[recent > close]
    if not highs_above.empty:
        target = float(highs_above.min())
    else:
        target = close + FALLBACK_TARGET_ATR_MULTIPLE * atr

    denom = close - stop
    rr = (target - close) / denom if denom > 0 else None

    return RiskAnnotation(
        suggested_stop=stop,
        suggested_target=target,
        risk_reward=rr,
    )
